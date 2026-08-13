"""Convert API-Bank eval partition -> a ToolSearcher-free benchmark.

Pipeline
--------
1. HARVEST: scan the eight test-data JSON files, extract every inline
   ``{"name"|"apiCode", "description", "input_parameters"|"parameters",
   "output_parameters"|"response"}`` block, normalise to a single
   ``tool_catalog.json`` (one entry per unique real tool, NO ToolSearcher).

2. CONVERT per level:
   - Build a fixed system_prompt that lists EVERY real tool from the catalog
     in the API-Bank "API descriptions" format.  The LLM therefore never
     needs to call ToolSearcher to discover a tool — every tool is already
     advertised in the system prompt.
   - Rewrite the dialogue/trajectory, removing every ToolSearcher call and the
     ToolSearcher spec, while keeping the prior real-tool returns so the
     conversation still flows.
   - Emit a normalised JSONL where each row carries:
       {id, level, source_file, system_prompt, messages, gold_tool_calls,
        gold_final_response?}

Output dir: data/processed/
  tool_catalog.json
  level-1.jsonl  level-2.jsonl  level-3.jsonl
  manifest.json
"""
from __future__ import annotations
import json
import re
import os
import argparse
from collections import OrderedDict, defaultdict
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.normpath(os.path.join(HERE, "..", "data", "test-data"))
OUT = os.path.normpath(os.path.join(HERE, "..", "data", "processed"))

TOOLSEARCHER = "ToolSearcher"

# ---------------------------------------------------------------------------
# Tool-spec extraction (applies to lvl1/lvl2 ``instruction``/``input`` fields
# and lvl3-batch ``input`` fields: they embed flat-ish JSON dicts whose top
# level has ``name``|``apiCode`` and ``input_parameters``|``parameters``).
# ---------------------------------------------------------------------------

def _walk_json_objs(text: str):
    """Yield parsed JSON dicts found anywhere in ``text`` by brace matching."""
    if not text:
        return
    i, n = 0, len(text)
    while i < n:
        m = re.search(r'\{', text[i:])
        if not m:
            return
        start = i + m.start()
        depth = 0
        j = start
        in_str = False
        esc = False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        blob = text[start:j + 1]
                        try:
                            yield json.loads(blob)
                        except Exception:
                            pass
                        i = j + 1
                        break
            j += 1
        else:
            return
        if j == n and depth != 0:
            return


def _is_tool_spec(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    name = obj.get("name") or obj.get("apiCode")
    if not name or name == TOOLSEARCHER:
        # never put ToolSearcher itself in the catalog
        return name == TOOLSEARCHER and False
    has_in = ("input_parameters" in obj) or ("parameters" in obj)
    has_out = ("output_parameters" in obj) or ("response" in obj)
    return bool(has_in and has_out)


def _normalize_spec(obj: dict) -> dict:
    """Normalise an extracted spec to {name, description, input_parameters,
    output_parameters}."""
    name = obj.get("name") or obj.get("apiCode")
    desc = obj.get("description", "")
    if "input_parameters" in obj:
        in_p = obj["input_parameters"]
    else:
        in_p = obj.get("parameters", {})
    if "output_parameters" in obj:
        out_p = obj["output_parameters"]
    else:
        out_p = obj.get("response", {})
    return {
        "name": name,
        "description": desc,
        "input_parameters": in_p or {},
        "output_parameters": out_p or {},
    }


def harvest_catalog(files: list[str]) -> list[dict]:
    """Build the catalog: ordered dict name -> normalised spec, keeping
    insertion order of first appearance. ToolSearcher and obvious
    data-items (e.g. ``Influenza`` returned by EmergencyKnowledge) are
    excluded by _is_tool_spec (they lack input_parameters/output_parameters)."""
    catalog: "OrderedDict[str, dict]" = OrderedDict()
    for fn in files:
        with open(fn) as f:
            obj = json.load(f)
        rows = obj if isinstance(obj, list) else [obj]
        for row in rows:
            blob = (row.get("instruction", "") or "") + "\n" + (row.get("input", "") or "")
            blob += "\n" + json.dumps(row.get("expected_output", "") or row.get("output", "") or "")
            for sub in _walk_json_objs(blob):
                if _is_tool_spec(sub):
                    spec = _normalize_spec(sub)
                    name = spec["name"]
                    if name == TOOLSEARCHER:
                        continue
                    # keep first occurrence; if a later one has richer fields, prefer the
                    # one whose input/output_parameters are non-empty.
                    if name not in catalog or (
                        not catalog[name]["input_parameters"] and spec["input_parameters"]
                    ):
                        catalog[name] = spec
        # also harvest from lvl3-full apis[] step outputs that wrap a spec
        if fn.endswith("level-3.json"):
            for conv in rows:
                for step in conv.get("apis", []):
                    out = step.get("output")
                    if isinstance(out, dict):
                        inner = out.get("output")
                        if isinstance(inner, dict) and _is_tool_spec(inner):
                            spec = _normalize_spec(inner)
                            if spec["name"] != TOOLSEARCHER and (
                                spec["name"] not in catalog or not catalog[spec["name"]]["input_parameters"]
                            ):
                                catalog[spec["name"]] = spec
    return list(catalog.values())


# ---------------------------------------------------------------------------
# System-prompt builder (lists every real tool; no ToolSearcher).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_HEADER = (
    "You are an API-assistant. You can call any of the APIs listed below by "
    "replying with a single line in the exact format:\n"
    "API-Request: [ApiName(key1='value1', key2='value2', ...)]\n"
    "Choose the most appropriate API directly from the catalog. Do not search "
    "for tools — every available tool is already listed. After you make a call, "
    "you will receive its return value and may then make the next call or "
    "respond to the user.\n\n"
    "API descriptions:"
)


# ---------------------------------------------------------------------------
# Qwen3.6 native system-prompt builder.
#
# When the Qwen3.6 chat template (chat_template.jinja) detects a non-empty
# ``tools`` array it injects a system block shaped like:
#
#   <|im_start|>system
#   # Tools
#
#   You have access to the following functions:
#
#   <tools>
#   <one JSON object per tool>
#   </tools>
#
#   If you choose to call a function ONLY reply in the following format with NO suffix:
#
#   <tool>
#   <function=example_function_name>
#   <parameter=example_parameter_1>
#   value_1
#   </parameter>
#   <parameter=example_parameter_2>
#   This is the value for the second parameter
#   that can span
#   multiple lines
#   </parameter>
#   </function>
#   </tool>
#
#   <IMPORTANT>
#   Reminder:
#   - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool> XML tags
#   - Required parameters MUST be specified
#   - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
#   - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
#   </IMPORTANT>
#
# Below is the exact instruction string distilled from the upstream template
# (chat_template.jinja lines 47-53) so the grader can hand-build a system
# prompt for endpoints that expose Qwen3.6 but do NOT take the OpenAI
# ``tools=[...]`` array (or where passing it would double-inject the catalog).
# ``build_qwen_system_prompt`` then appends every tool from the catalog as a
# single JSON-per-line and any user-supplied preface/instructions after.
# ---------------------------------------------------------------------------
QWEN_TOOLS_HEADER = (
    "# Tools\n\n"
    "You have access to the following functions:\n\n"
    "<tools>"
)
QWEN_TOOLS_FOOTER = (
    "</tools>\n"
    "\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n"
    "\n<tool>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool>\n"
    "\n<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


def _qwen_tool_json(spec: dict) -> str:
    """Render a tool spec as the compact JSON object Qwen3.6's chat template
    emits inside ``<tools>``. We reuse the OpenAI function-shape (``type``,
    ``function: {name, description, parameters}``) since that is what the
    upstream template expects via ``tool | tojson``."""
    arr = build_openai_tools_array([spec])
    return json.dumps(arr[0], ensure_ascii=False)


def build_qwen_system_prompt(catalog: list[dict], max_tools: int | None = None,
                             preface: str | None = None) -> str:
    """Build a Qwen3.6-native system prompt listing every tool from the
    catalog inside the ``<tools>...</tools>`` block, followed by the exact
    format-string instruction Qwen3.6's chat_template.jinja injects.

    ``preface`` (optional) is appended after the ``</IMPORTANT>`` block and is
    the natural place to convey task-specific guidance (e.g. the API-Bank
    step-replay instruction)."""
    tools = catalog if max_tools is None else catalog[:max_tools]
    lines = [QWEN_TOOLS_HEADER]
    for spec in tools:
        lines.append(_qwen_tool_json(spec))
    lines.append(QWEN_TOOLS_FOOTER)
    if preface:
        lines.append(preface)
    return "\n".join(lines)


def build_system_prompt(catalog: list[dict], max_tools: int | None = None) -> str:
    """Build a single system prompt that lists every real tool's spec."""
    tools = catalog if max_tools is None else catalog[:max_tools]
    lines = [SYSTEM_PROMPT_HEADER]
    for spec in tools:
        lines.append(json.dumps(spec, ensure_ascii=False))
    return "\n".join(lines)


def build_openai_tools_array(catalog: list[dict]) -> list[dict]:
    """Build an OpenAI-style ``tools=[...]`` array from the catalog. Used by
    the grader for endpoints that accept function-calling."""
    arr = []
    for spec in catalog:
        props = spec.get("input_parameters", {}) or {}
        properties = {}
        required = []
        for pname, pinfo in props.items():
            if not isinstance(pinfo, dict):
                pinfo = {"type": "string", "description": str(pinfo)}
            ptype = pinfo.get("type", "string")
            jtype = {
                "str": "string", "string": "string", "int": "integer",
                "integer": "integer", "float": "number", "number": "number",
                "bool": "boolean", "boolean": "boolean", "list": "array",
                "array": "array", "dict": "object", "object": "object",
            }.get(str(ptype).lower(), "string")
            prop = {"type": jtype, "description": pinfo.get("description", "")}
            if "default" in pinfo:
                prop["default"] = pinfo["default"]
            properties[pname] = prop
            if pinfo.get("required") or pinfo.get("required", True) is True and "default" not in pinfo:
                # API-Bank flags required=True; only mark required when explicitly True
                if pinfo.get("required") is True:
                    required.append(pname)
        arr.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return arr


# ---------------------------------------------------------------------------
# API-call parser  (parses ``[ApiName(k='v', ...)]`` strings).
# ---------------------------------------------------------------------------

CALL_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\]", re.S)
EXPECTED_RE = re.compile(r"API-Request:\s*(\[.*?\])", re.S)


def parse_api_call(text: str) -> tuple[str, dict] | None:
    """Parse a single ``[ApiName(k1='v1', k2='v2')]`` (optionally preceded by
    'API-Request: '). Returns (tool_name, args_dict) or None."""
    if not text:
        return None
    me = EXPECTED_RE.search(text)
    if me:
        call = me.group(1)
        inner = CALL_RE.search(call)
        if not inner:
            return None
        name = inner.group(1)
        argstr = inner.group(2).strip()
    else:
        mi = CALL_RE.search(text)
        if not mi:
            return None
        name = mi.group(1)
        argstr = mi.group(2).strip()
    args: dict[str, Any] = {}
    if argstr:
        args = _parse_kwargs(argstr)
    return name, args


def _parse_kwargs(argstr: str) -> dict[str, Any]:
    """Parse ``k1='v1', k2=123, k3=['a','b'], k4={'x':1}`` into a dict.
    Conservative: uses ast.literal_eval per value when possible."""
    import ast
    out: dict[str, Any] = {}
    # split on commas that are at top level (depth 0 wrt quotes/brackets)
    parts = []
    depth = 0
    in_str = False
    esc = False
    cur = []
    for c in argstr:
        if in_str:
            cur.append(c)
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == "'" or c == '"':
                in_str = False
        else:
            if c in "'\"":
                in_str = True
                cur.append(c)
            elif c in "[{(":
                depth += 1
                cur.append(c)
            elif c in "]})":
                depth -= 1
                cur.append(c)
            elif c == "," and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
            else:
                cur.append(c)
    if cur:
        parts.append("".join(cur).strip())
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        # try literal eval; fall back to stripping quotes
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
                out[k] = v[1:-1]
            elif len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                out[k] = v[1:-1]
            else:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
# Converters per level.
# ---------------------------------------------------------------------------

def _strip_ts_returns(text: str) -> str:
    """Remove ``API-Request: [ToolSearcher(...)]->{...}`` and the bare
    ``[ToolSearcher(...)]->{...}`` fragments from a dialogue string so the
    Model does not see prior retrieval steps."""
    if not text:
        return text
    # greedy match from 'API-Request:' to the matching '->{' ... '}]' OR ->[ ... ]
    text = re.sub(
        r"API-Request:\s*\[ToolSearcher\([^)]*\)\]->.*?(?=\nAPI-Request:|\nUser:|\nAI:|\nGenerate|\Z)",
        "",
        text,
        flags=re.S,
    )
    text = re.sub(r"\[ToolSearcher\([^)]*\)\]->.*?(?=\nAPI-Request:|\nUser:|\nAI:|\nGenerate|\Z)",
                  "", text, flags=re.S)
    # any remaining bare ToolSearcher-only calls (no return appended)
    text = re.sub(r"API-Request:\s*\[ToolSearcher\([^)]*\)\]\s*\n?", "", text)
    text = re.sub(r"\[ToolSearcher\([^)]*\)\]\s*\n?", "", text)
    return text


def _dialogue_to_messages(input_text: str, expected_api: str | None,
                          expected_response: str | None) -> list[dict]:
    """Turn the API-Bank ``input`` string (which is a transcription of a
    multi-turn dialogue) into a list of role-tagged messages. We keep the
    user / assistant / api-request turns, but drop ToolSearcher turns and
    ToolSearcher return values."""
    msgs: list[dict] = []
    if not input_text:
        return msgs
    # Strip ToolSearcher artefacts (calls, returns, and the leading spec blob).
    txt = _clean_input(input_text)
    # Split into labelled turns. API-Bank format:
    #   User: ...\nAI: ...\nAPI-Request: [...]->...\nGenerate API Request:\n  (api file)
    #   User: ...\nAI: ...\nAPI-Request: [...]  (response file)
    # We carve chunks at labels.
    tokens = re.split(r"(?m)^(User:|AI:|API-Request:|Generate API Request:|Generate AI Response:)\s*",
                      txt)
    # tokens: [pre, label1, body1, label2, body2, ...]
    if tokens and tokens[0].strip():
        msgs.append({"role": "user", "content": tokens[0].strip()})
    i = 1
    pending_user = None
    while i < len(tokens):
        label = tokens[i].rstrip(":").strip() if i < len(tokens) else ""
        body = tokens[i + 1] if i + 1 < len(tokens) else ""
        body = body.strip() if body else ""
        if label == "User":
            pending_user = body
            msgs.append({"role": "user", "content": body})
        elif label == "AI":
            msgs.append({"role": "assistant", "content": body})
        elif label == "API-Request":
            parsed = parse_api_call(body)
            if parsed and parsed[0] != TOOLSEARCHER:
                msgs.append({"role": "assistant", "content": f"API-Request: [{_render_call(parsed)}]"})
                # if there's a ->return appended, expose it as a 'tool' role
                ret = body.split("->", 1)[1].strip() if "->" in body else ""
                if ret:
                    msgs.append({"role": "tool", "name": parsed[0], "content": ret})
        elif label in ("Generate API Request:", "Generate AI Response:"):
            # these are the model's cue lines — collapse to a user turn asking
            # the model to produce the next call/response.
            cue = "Generate the next API Request." if "API Request" in label else "Generate the AI response to the user."
            msgs.append({"role": "user", "content": cue})
        i += 2
    return msgs


def _render_call(parsed: tuple[str, dict]) -> str:
    name, args = parsed
    parts = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()]
    return f"{name}({', '.join(parts)})"


# ----- Level 1 -----------------------------------------------------------

def convert_level_1(system_prompt: str, files: dict) -> list[dict]:
    """Level-1: each row is a single-turn call task — tool specs ALREADY
    inline as real tools (no ToolSearcher). We re-emit with the global
    system_prompt and the cleaned gold call. Each row's ``input`` is shaped
    like ``User: <utterance>\\nAI: <reply>\\nGenerate API Request:\\n`` — so
    we tokenize with ``_dialogue_to_messages`` to surface distinct
    user / assistant turns."""
    out = []
    src = files["level-1-api"]
    src_resp = files.get("level-1-response")
    resp_by_key = {}
    if src_resp:
        for r in src_resp:
            resp_by_key[(r.get("file"), r.get("id"))] = r
    n = 0
    for row in src:
        parsed = parse_api_call(row.get("expected_output", ""))
        if not parsed or parsed[0] == TOOLSEARCHER:
            continue
        msgs = _dialogue_to_messages(row.get("input", "") or "", None, None)
        # Ensure there is a final user cue asking for the next API call.
        if not msgs or msgs[-1]["role"] != "user" or "Generate" not in msgs[-1]["content"]:
            msgs.append({"role": "user", "content": "Generate the next API Request."})
        nl = resp_by_key.get((row.get("file"), row.get("id") + 1))
        gold_resp = nl["expected_output"] if nl else None
        out.append({
            "id": n,
            "level": 1,
            "source_file": row.get("file"),
            "source_row_id": row.get("id"),
            "system_prompt": system_prompt,
            "messages": msgs,
            "gold_tool_calls": [{"tool_name": parsed[0], "arguments": parsed[1]}],
            "gold_final_response": gold_resp,
        })
        n += 1
    return out


# ----- Level 2 -----------------------------------------------------------

def _strip_ts_spec_json(text: str) -> str:
    """Remove any inline ``{"apiCode":"ToolSearcher",...}`` or
    ``{"name":"ToolSearcher",...}`` tool-spec blob from ``text``."""
    if not text or "ToolSearcher" not in text:
        return text
    def repl(m):
        blob = m.group(0)
        try:
            obj = json.loads(blob)
        except Exception:
            return blob
        nm = obj.get("name") or obj.get("apiCode")
        return "" if nm == TOOLSEARCHER else blob
    # match any balanced { ... } that contains a ToolSearcher name/apiCode
    out_parts = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "{":
            # find matching brace
            depth = 0
            j = i
            in_str = False
            esc = False
            while j < n:
                ch = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            break
                j += 1
            blob = text[i:j + 1]
            if '"ToolSearcher"' in blob:
                # only drop if the parsed object IS the ToolSearcher spec
                try:
                    obj = json.loads(blob)
                    if (obj.get("name") == TOOLSEARCHER) or (obj.get("apiCode") == TOOLSEARCHER):
                        i = j + 1
                        # also collapse a single trailing newline if present
                        if i < n and text[i] == "\n":
                            i += 1
                        continue
                except Exception:
                    pass
            out_parts.append(blob)
            i = j + 1
        else:
            out_parts.append(c)
            i += 1
    return "".join(out_parts)


def _strip_leading_user_label(text: str) -> str:
    """If ``text`` begins with a 'User:' or 'AI:' label, drop it so the
    message content reads naturally."""
    if not text:
        return text
    m = re.match(r"^\s*(User|AI)\s*:\s*", text)
    if m:
        return text[m.end():]
    return text


def _clean_input(text: str) -> str:
    """Strip (1) ToolSearcher API-return fragments, (2) bare TS calls, (3) the
    leading ``{"apiCode":"ToolSearcher",...}`` spec blob some batch files prefix,
    and (4) the AI's 'let me search for relevant tools' line that follows a TS
    call (it has no content without the search)."""
    t = _strip_ts_returns(text)
    t = _strip_ts_spec_json(t)
    return t


def _extract_real_call_returns(input_text: str) -> list[tuple[tuple[str, dict], str]]:
    """Walk the dialogue prefix ``input_text`` and return every
    ``API-Request: [<real tool>(...)] -> <return>`` pair in order, skipping
    ToolSearcher. Returns [(parsed, return_str), ...]."""
    out = []
    if not input_text:
        return out
    for m in re.finditer(r"API-Request:\s*(\[[^\]]+\])(?:->(.*?))?(?=\nAPI-Request:|\nUser:|\nAI:|\nGenerate|\Z)",
                         input_text, re.S):
        call_str = m.group(1)
        ret = (m.group(2) or "").strip()
        # strip trailing newlines brokers may have left
        ret = ret.rstrip("\n")
        parsed = parse_api_call(call_str)
        if parsed and parsed[0] != TOOLSEARCHER:
            out.append((parsed, ret))
    return out


def convert_level_2(system_prompt: str, files: dict) -> list[dict]:
    """Level-2: each ``file`` is a multi-turn dialogue. Row id=0 always calls
    ToolSearcher (drop it); subsequent rows call real tools. We collapse the
    dialogue to one datapoint per ``file`` with the full message history and
    the ordered list of non-ToolSearcher gold calls."""
    src = files["level-2-api"]
    src_resp = files.get("level-2-response")
    resp_by_file: dict = defaultdict(dict)
    if src_resp:
        for r in src_resp:
            resp_by_file[r.get("file")][r.get("id")] = r
    by_file: "OrderedDict[str, list]" = OrderedDict()
    for row in src:
        by_file.setdefault(row["file"], []).append(row)
    out = []
    n = 0
    for fn, rows in by_file.items():
        rows = sorted(rows, key=lambda r: r.get("id", 0))
        # --- Collect gold calls (dedupe consecutive identical) ---
        gold_calls: list[dict] = []
        for r in rows:
            parsed = parse_api_call(r.get("expected_output", ""))
            if not parsed or parsed[0] == TOOLSEARCHER:
                continue
            if gold_calls and gold_calls[-1] == {"tool_name": parsed[0], "arguments": parsed[1]}:
                continue
            gold_calls.append({"tool_name": parsed[0], "arguments": parsed[1]})
        # Build messages from the FIRST row's input ---
        # The first row's input is the smallest cumulative context, ending at
        # the first ``Generate API Request:`` cue. This is the prompt the model
        # sees for the FIRST gold call, with no ToolSearcher artefacts and —
        # critically — no gold actions leaked into the prompt. The full ordered
        # ``gold_tool_calls`` sequence is carried separately for step-replay
        # graders.
        first_input = rows[0].get("input", "") or ""
        msgs = _dialogue_to_messages(first_input, None, None)
        # Defensive: drop any assistant turn that starts with ``API-Request:``
        # (should be none here, since first row hasn't answered yet).
        while msgs and msgs[-1]["role"] == "assistant" and msgs[-1]["content"].startswith("API-Request:"):
            msgs.pop()
        if not msgs or not (msgs[-1]["role"] == "user" and "Generate" in msgs[-1]["content"]):
            msgs.append({"role": "user", "content": "Generate the next API Request."})
        # --- Capture the gold tool returns (parallel to gold_tool_calls) ---
        # Walk the longest cumulative input and pull each ``API-Request:
        # [<real tool>(...)]-><return>`` fragment so the grader can replay the
        # dialogue step-by-step (feeding the GOLD return after each predicted
        # call). ToolSearcher fragments are already stripped by _clean_input.
        longest_input = max((r.get("input", "") or "" for r in rows), key=len)
        gold_returns: list[str] = []
        for (parsed, ret) in _extract_real_call_returns(longest_input):
            gold_returns.append(ret)
        # Gold returns should align 1:1 with gold_tool_calls. If the upstream
        # dialogue left the LAST call's return unrecorded, pad with "".
        while len(gold_returns) < len(gold_calls):
            gold_returns.append("")
        gold_returns = gold_returns[:len(gold_calls)]
        # Attach the gold NL resultado from the response file (last NL row).
        gold_resp = None
        for rid in sorted(resp_by_file.get(fn, {}).keys()):
            r = resp_by_file[fn][rid]
            if r.get("expected_output"):
                gold_resp = r["expected_output"]
        out.append({
            "id": n,
            "level": 2,
            "source_file": fn,
            "system_prompt": system_prompt,
            "messages": msgs,
            "gold_tool_calls": gold_calls,
            "gold_tool_returns": gold_returns,
            "gold_final_response": gold_resp,
        })
        n += 1
    return out


# ----- Level 3 -----------------------------------------------------------

def convert_level_3(system_prompt: str, files: dict) -> list[dict]:
    """Level-3: 50 conversations {requirement, response, apis[]}. Each apis[i]
    step = {api_name, input, output:{api_name,input,output,exception}}. Drop
    ToolSearcher steps; keep real-tool steps as the gold trajectory. Build
    messages = [user requirement, then alternating assistant/tool per real
    step]."""
    src = files["level-3"]
    out = []
    for n, conv in enumerate(src):
        req = (conv.get("requirement") or "").strip()
        msgs: list[dict] = [{"role": "user", "content": req}]
        gold_calls: list[dict] = []
        gold_returns: list[str] = []
        for step in conv.get("apis", []):
            name = step.get("api_name")
            if not name or name == TOOLSEARCHER:
                continue
            args = step.get("input", {}) or {}
            if not isinstance(args, dict):
                continue
            gold_calls.append({"tool_name": name, "arguments": args})
            rendered = _render_call((name, args))
            msgs.append({"role": "assistant", "content": f"API-Request: [{rendered}]"})
            out_val = step.get("output")
            if isinstance(out_val, dict):
                inner = out_val.get("output")
            else:
                inner = out_val
            try:
                inner_s = json.dumps(inner, ensure_ascii=False)
            except Exception:
                inner_s = str(inner)
            gold_returns.append(inner_s)
            msgs.append({"role": "tool", "name": name, "content": inner_s})
        # final assistant NL response
        final_resp = (conv.get("response") or "").strip() or None
        if final_resp:
            msgs.append({"role": "assistant", "content": final_resp})
        out.append({
            "id": n,
            "level": 3,
            "source_file": "level-3.json",
            "system_prompt": system_prompt,
            "messages": msgs,
            "gold_tool_calls": gold_calls,
            "gold_tool_returns": gold_returns,
            "gold_final_response": final_resp,
        })
    return out


# ----- Level 3 (batch-inference per-step rows) ---------------------------

def convert_level_3_batch(system_prompt: str, files: dict, which: str) -> list[dict]:
    """level-3-batch-inf.json / level-3-batch-inf-icl.json are per-step rows
    (one row per api_id within each conversation). Each row ``input`` embeds
    the ToolSearcher spec + user requirement + previous returns. We strip TS
    and emit one datapoint per row whose gold call is the row's expected_output
    parsed tool. Used for fine-grained per-step grading."""
    src = files[which]
    out = []
    for n, row in enumerate(src):
        parsed = parse_api_call(row.get("output", ""))
        if not parsed or parsed[0] == TOOLSEARCHER:
            continue
        usr = _clean_input(row.get("input", "") or "")
        usr = _strip_leading_user_label(usr)
        msgs = []
        if usr:
            msgs.append({"role": "user", "content": usr})
        msgs.append({"role": "user", "content": "Generate the next API Request."})
        out.append({
            "id": n,
            "level": 3,
            "source_file": which,
            "sample_id": row.get("sample_id"),
            "api_id": row.get("api_id"),
            "system_prompt": system_prompt,
            "messages": msgs,
            "gold_tool_calls": [{"tool_name": parsed[0], "arguments": parsed[1]}],
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert API-Bank eval to a ToolSearcher-free benchmark")
    ap.add_argument("--test-data", default=TEST, help="source test-data dir")
    ap.add_argument("--out", default=OUT, help="output dir")
    ap.add_argument("--levels", default="1,2,3", help="comma list of levels to convert")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    levels = {int(x) for x in args.levels.split(",") if x.strip()}

    file_names = {
        "level-1-api": "level-1-api.json",
        "level-1-response": "level-1-response.json",
        "level-2-api": "level-2-api.json",
        "level-2-response": "level-2-response.json",
        "level-3": "level-3.json",
        "level-3-batch": "level-3-batch-inf.json",
        "level-3-batch-icl": "level-3-batch-inf-icl.json",
        "level-3-batch-resp": "level-3-batch-inf-response.json",
    }
    files = {}
    for k, fn in file_names.items():
        p = os.path.join(args.test_data, fn)
        if os.path.exists(p):
            with open(p) as f:
                files[k] = json.load(f)
        else:
            print(f"[warn] missing {p}")

    # 1. Catalog (harvest from ALL files we have, irrespective of selected levels)
    catalog = harvest_catalog([os.path.join(args.test_data, fn) for fn in file_names.values()
                               if os.path.exists(os.path.join(args.test_data, fn))])
    catalog_path = os.path.join(args.out, "tool_catalog.json")
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"[ok] catalog: {len(catalog)} tools -> {catalog_path}")
    # also dump the OpenAI-style tools array for the grader
    tools_arr = build_openai_tools_array(catalog)
    with open(os.path.join(args.out, "openai_tools.json"), "w") as f:
        json.dump(tools_arr, f, indent=2, ensure_ascii=False)
    print(f"[ok] openai tools array: {len(tools_arr)} entries")

    system_prompt = build_system_prompt(catalog)
    with open(os.path.join(args.out, "system_prompt.txt"), "w") as f:
        f.write(system_prompt)
    print(f"[ok] system_prompt.txt: {len(system_prompt)} chars")

    # also emit a Qwen3.6-native system prompt that lists tools in the
    # ``<tools>...</tools>`` shape with the exact format-string instruction
    # copied from chat_template.jinja. Useful for grading Qwen3.x models
    # served by endpoints that don't auto-inject the catalog from the
    # ``tools`` array (or where passing it would double-inject).
    qwen_prompt = build_qwen_system_prompt(catalog)
    with open(os.path.join(args.out, "qwen_system_prompt.txt"), "w") as f:
        f.write(qwen_prompt)
    print(f"[ok] qwen_system_prompt.txt: {len(qwen_prompt)} chars")

    manifest: dict = {"catalog_size": len(catalog), "levels": {}}

    # 2. Convert levels
    if 1 in levels:
        out1 = convert_level_1(system_prompt, files)
        _write_jsonl(out1, os.path.join(args.out, "level-1.jsonl"))
        manifest["levels"]["level-1"] = {"datapoints": len(out1)}
        # report how many ToolSearcher calls were present in source vs removed
        src_ts = sum(1 for r in files["level-1-api"] if "ToolSearcher" in (r.get("expected_output", "") or ""))
        manifest["levels"]["level-1"]["source_toolsearcher_calls"] = src_ts
        print(f"[ok] level-1: {len(out1)} datapoints (source had {src_ts} TS calls)")
    if 2 in levels:
        out2 = convert_level_2(system_prompt, files)
        _write_jsonl(out2, os.path.join(args.out, "level-2.jsonl"))
        src_ts = sum(1 for r in files["level-2-api"] if "ToolSearcher" in (r.get("expected_output", "") or ""))
        manifest["levels"]["level-2"] = {"datapoints": len(out2), "source_toolsearcher_calls": src_ts}
        print(f"[ok] level-2: {len(out2)} datapoints (source had {src_ts} TS calls)")
    if 3 in levels:
        out3 = convert_level_3(system_prompt, files)
        _write_jsonl(out3, os.path.join(args.out, "level-3.jsonl"))
        src_ts = sum(
            1 for c in files["level-3"] for s in c.get("apis", []) if s.get("api_name") == TOOLSEARCHER
        )
        manifest["levels"]["level-3"] = {"datapoints": len(out3), "source_toolsearcher_steps": src_ts}
        print(f"[ok] level-3: {len(out3)} datapoints (source had {src_ts} TS steps)")
        # also the per-step batch files for fine-grained grading
        for which in ("level-3-batch", "level-3-batch-icl"):
            if which in files:
                outb = convert_level_3_batch(system_prompt, files, which)
                tag = "level-3-batch" if which == "level-3-batch" else "level-3-batch-icl"
                _write_jsonl(outb, os.path.join(args.out, f"{tag}.jsonl"))
                manifest["levels"][tag] = {"datapoints": len(outb)}
                print(f"[ok] {tag}: {len(outb)} datapoints")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ok] manifest.json")


def _write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

"""Grade a custom LLM on the ToolSearcher-free API-Bank benchmark.

The LLM is expected to be OpenAI-compatible (``POST /chat/completions``).
Two evaluation modes are supported:

  --mode text   : the system prompt lists every tool in API-Bank's
                  ``API descriptions`` block; the model is expected to reply
                  with a single line ``API-Request: [ToolName(k='v', ...)]``.
                  We parse that line and compare against the gold call.
                  This is the mode that mirrors the original benchmark.

  --mode tools  : we additionally pass the OpenAI ``tools=[...]`` array built
                  by ``convert_apibank.py`` and parse the model's
                  ``tool_calls`` response.

Scoring (per gold tool call):
  - exact_name       : model picked the right tool
  - exact_args       : same tool AND identical argument dict (after JSON
                       normalisation: dict-equal, str strips, number coercion)
  - partial_args     : same tool AND >=50% argument keys match the gold value
  - wrong_tool / no_call : model did not emit a callable response

Usage:
  python scripts/grade_apibank.py \
      --ip 10.0.0.42 --model mymodel --port 8000 \
      --level 3 --mode text --limit 10 \
      --output data/results/l3.json

Outputs:
  <output>.json   - per-datapoint verdicts
  <output>.md     - aggregate markdown summary
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import ssl
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from convert_apibank import build_qwen_system_prompt  # noqa: E402

DEFAULT_PROCESSED = os.path.normpath(os.path.join(HERE, "..", "data", "processed"))

# ---------------------------------------------------------------------------
# Helpers shared with the converter (kept in sync).
# ---------------------------------------------------------------------------

CALL_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\]", re.S)
EXPECTED_RE = re.compile(r"API-Request:\s*(\[[^\]]+\])", re.S)
# Some models drop the brackets and emit ``API-Request: ApiName(k='v')`` — tolerate.
EXPECTED_NOBRACKET_RE = re.compile(r"API-Request:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:\n|$)", re.S)
# And some emit a bare ``ApiName(k='v')`` with no prefix at all — tolerate if it
# shows up as the assistant's entire reply.
BARE_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*$", re.S)


def parse_api_call(text: str) -> tuple[str, dict] | None:
    if not text:
        return None
    # Strategy order (most-specific to least):
    #   1. ``API-Request: [Tool(k='v')]`` — the canonical API-Bank format.
    #   2. ``API-Request: Tool(k='v')``  — brackets dropped (some Qwen models).
    #   3. ``[Tool(k='v')]``             — bare bracketed call (no prefix).
    #   4. ``Tool(k='v')``               — a bare call line, used by terse models.
    me = EXPECTED_RE.search(text)
    if me:
        call = me.group(1)
        inner = CALL_RE.search(call)
        if inner:
            name = inner.group(1)
            argstr = inner.group(2).strip()
        else:
            return None
    else:
        mn = EXPECTED_NOBRACKET_RE.search(text)
        if mn:
            name = mn.group(1)
            argstr = mn.group(2).strip()
        else:
            mi = CALL_RE.search(text)
            if mi:
                name = mi.group(1)
                argstr = mi.group(2).strip()
            else:
                # last resort: a bare single-line call
                mb = BARE_CALL_RE.search(text.strip().splitlines()[-1] if text.strip() else "")
                if mb:
                    name = mb.group(1)
                    argstr = mb.group(2).strip()
                else:
                    return None
    args: dict[str, Any] = {}
    if argstr:
        args = _parse_kwargs(argstr)
    return name, args


# ---------------------------------------------------------------------------
# Qwen3.6 native XML tool-call parser.
#
# Qwen3.6's chat_template.jinja instructs the model to emit tool calls in the
# following XML shape (one or more per assistant turn):
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
# Multiple tool calls inside a single assistant message are emitted as
# consecutive ``<tool>...</tool>`` blocks (the template separates them with a
# single ``\n``). ``parse_qwen_xml_calls`` walks the assistant text and
# returns every parsed call as a ``(tool_name, args_dict)`` tuple, in order.
# Values are parsed with ``ast.literal_eval`` so int/float/list/json-shape
# values round-trip to the same Python types the gold calls use; on failure
# the raw (stripped) string is kept.
# ---------------------------------------------------------------------------

QWEN_TOOL_BLOCK_RE = re.compile(r"<tool>\s*(.*?)\s*</tool>", re.S)
QWEN_FUNCTION_RE = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>", re.S)
QWEN_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>", re.S
)


def _coerce_value(v: str) -> Any:
    """Try to recover the typed value the model intended. Qwen3.6 emits
    parameter values as bare text on lines between the open/close tags; that
    means an int ``42`` arrives as ``"42"`` and a JSON object arrives as a
    multi-line string. We try ast.literal_eval, then json.loads, else
    keep the stripped string."""
    s = v.strip()
    if s == "":
        return s
    import ast
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    # JSON shapes (lists/dicts/true/false/null): ast.literal_eval already
    # covers ``True/False/None`` but models emit ``true/false/null``, so try
    # json.loads as a fallback.
    try:
        return json.loads(s)
    except Exception:
        return s


def parse_qwen_xml_calls(text: str) -> list[tuple[str, dict]]:
    """Pull every Qwen3.6 ``<tool><function=NAME>...</function></tool>`` block
    out of ``text``. Returns a list of ``(tool_name, args_dict)`` in order."""
    if not text:
        return []
    calls: list[tuple[str, dict]] = []
    for block_m in QWEN_TOOL_BLOCK_RE.finditer(text):
        block = block_m.group(1)
        fn_m = QWEN_FUNCTION_RE.search(block)
        if not fn_m:
            continue
        name = fn_m.group(1)
        body = fn_m.group(2)
        args: dict[str, Any] = {}
        # iterate in order so later duplicates overwrite earlier ones
        for pm in QWEN_PARAM_RE.finditer(body):
            pname = pm.group(1)
            pval = _coerce_value(pm.group(2))
            args[pname] = pval
        calls.append((name, args))
    return calls


def _parse_kwargs(argstr: str) -> dict[str, Any]:
    import ast
    out: dict[str, Any] = {}
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
            elif c == "\\":
                esc = True
            elif c in "'\"":
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
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            if len(v) >= 2 and v[0] in "'\"" and v[-1] == v[0]:
                out[k] = v[1:-1]
            else:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
# Argument comparison.
# ---------------------------------------------------------------------------

def _norm_arg(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        # try to coerce numeric strings
        try:
            return int(s)
        except Exception:
            try:
                return float(s)
            except Exception:
                return s
    if isinstance(v, list):
        return [_norm_arg(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm_arg(x) for k, x in v.items()}
    return v


def compare_args(gold: dict, pred: dict) -> tuple[bool, bool, float]:
    """Return (exact, partial, fraction)."""
    g = {k: _norm_arg(val) for k, val in (gold or {}).items()}
    p = {k: _norm_arg(val) for k, val in (pred or {}).items()}
    if g == p:
        return True, True, 1.0
    if not g:
        # gold has no args — exact iff pred has no args too
        return (not p, not p, 1.0 if not p else 0.0)
    matches = 0
    for k, gv in g.items():
        if k in p and p[k] == gv:
            matches += 1
    frac = matches / len(g)
    return (False, frac >= 0.5, frac)


# ---------------------------------------------------------------------------
# OpenAI-compatible client.
# ---------------------------------------------------------------------------

def call_llm(base_url: str, model: str, messages: list[dict], tools: list | None,
             api_key: str | None, timeout: int, retries: int = 2,
             max_tokens: int = 1024, insecure: bool = False,
             extra_payload: dict | None = None) -> dict:
    """POST /chat/completions. Returns the parsed JSON dict.

    ``insecure=True`` skips TLS certificate verification (needed for proxies
    that serve HTTPS with a self-signed cert).
    ``extra_payload`` lets callers add non-default request fields (e.g.
    ``reasoning_effort`` for reasoning models, ``enable_thinking=False`` ...).
    """
    payload: dict[str, Any] = {"model": model, "messages": messages,
                               "temperature": 0.0, "max_tokens": max_tokens}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if extra_payload:
        payload.update(extra_payload)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(base_url, data=json.dumps(payload).encode("utf-8"),
                                         headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 ** attempt)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_err}")


def extract_pred_calls(resp: dict, mode: str) -> list[tuple[str, dict]]:
    """Pull a list of (tool_name, args) from an OpenAI-style response.

    Three modes:
      * ``tools`` - parse the model's ``tool_calls`` array (OpenAI shape).
      * ``qwen``  - parse Qwen3.6 native ``<tool>...</tool>`` XML blocks from
                    the assistant content; falls back to the API-Request text
                    parser if no XML block was found.
      * ``text``  - parse ``API-Request: [...]`` lines (canonical API-Bank)."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    calls: list[tuple[str, dict]] = []
    if mode == "tools":
        for tc in msg.get("tool_calls") or []:
            fn = (tc.get("function") or {})
            name = fn.get("name")
            if not name:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            calls.append((name, args))
        # some servers echo the call as plain text; fall back to parsing text.
        if not calls and msg.get("content"):
            parsed = parse_api_call(msg["content"])
            if parsed:
                calls.append(parsed)
    elif mode == "qwen":
        # Qwen3.6 native XML tool-call format (see chat_template.jinja).
        content = msg.get("content") or ""
        calls = parse_qwen_xml_calls(content)
        if not calls and content:
            # fall back to the canonical API-Bank text parser in case the
            # server's chat template did not round-trip the XML shape.
            parsed = parse_api_call(content)
            if parsed:
                calls.append(parsed)
    else:  # text mode
        content = msg.get("content") or ""
        # Scan for every "API-Request: ...." line our parser handles (covers
        # both the ``[Tool(...)]`` and bare ``Tool(...)`` forms). Then fall
        # back to parsing any line that *only* contains a Tool(...) call.
        for line in re.split(r"(?m)^(.*API-Request:.*)$", content):
            if not line:
                continue
            parsed = parse_api_call(line)
            if parsed and parsed not in calls:
                calls.append(parsed)
        if not calls:
            parsed = parse_api_call(content)
            if parsed:
                calls.append(parsed)
    return calls


# ---------------------------------------------------------------------------
# Grading.
# ---------------------------------------------------------------------------

def _strip_sys_for_tools(sys_prompt: str) -> str:
    """For ``--mode tools`` we pass the catalog via the OpenAI ``tools``
    array; keep the system prompt's instructions but drop the duplicated
    ``API descriptions:`` block."""
    return sys_prompt.split("API descriptions:")[0].strip() + (
        "\n\nCall the right tool directly from the provided tools."
    )


def _render_call(name: str, args: dict) -> str:
    parts = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (args or {}).items()]
    return f"{name}({', '.join(parts)})"


def grade_one(dp: dict, base_url: str, model: str, mode: str,
              tools: list | None, api_key: str | None, timeout: int,
              max_steps: int | None = None, max_tokens: int = 1024,
              insecure: bool = False, extra_payload: dict | None = None,
              qwen_system_prompt: str | None = None) -> dict:
    """Grade a single datapoint with step-replay semantics.

    We feed the model the conversation up to the FIRST gold-call cue, parse its
    predicted call, COMPARE it to the first gold call, then ADVANCE the
    dialogue using the GOLD call + GOLD tool return (so a mistake at step i
    does not poison step i+1). Repeat for every gold call.
    """
    gold = dp.get("gold_tool_calls") or []
    gold_returns = dp.get("gold_tool_returns") or []
    if max_steps:
        gold = gold[:max_steps]
        gold_returns = gold_returns[:max_steps]
    # Build the running message list.
    sys_msg = dp["system_prompt"]
    if mode == "tools" and tools is not None:
        sys_msg = _strip_sys_for_tools(sys_msg)
    elif mode == "qwen":
        # Use the Qwen3.6-native system prompt: the catalog inside
        # ``<tools>...</tools>`` plus the exact format-string instruction
        # Qwen3.6's chat_template.jinja injects. Prefer ``dp``'s own
        # ``qwen_system_prompt`` (added by convert_apibank.py), else the
        # grader's recomputed copy (built from tool_catalog.json), else the
        # canonical API-Bank prompt.
        sys_msg = dp.get("qwen_system_prompt") or qwen_system_prompt or dp["system_prompt"]
    messages: list[dict] = [{"role": "system", "content": sys_msg}]
    messages.extend(dp["messages"])
    # Ensure the running list ends in a cue for the next call.
    if not messages or not (messages[-1]["role"] == "user" and "Generate" in messages[-1]["content"]):
        messages.append({"role": "user", "content": "Generate the next API Request."})

    verdicts: list[dict] = []
    pred_count = 0
    err = None
    for i, gc in enumerate(gold):
        # Ask the LLM for the i-th call.
        try:
            # ``qwen`` and ``text`` modes send NO tools array — the catalog
            # lives entirely in the system prompt.
            resp = call_llm(base_url, model, messages,
                            tools if mode == "tools" else None, api_key, timeout,
                            max_tokens=max_tokens, insecure=insecure,
                            extra_payload=extra_payload)
            preds = extract_pred_calls(resp, mode)
        except Exception as e:
            err = str(e)
            preds = []
        if preds:
            pname, pargs = preds[0]
            pred_count += 1
        else:
            pname, pargs = None, {}
        gname = gc["tool_name"]
        gargs = gc.get("arguments", {})
        name_ok = pname == gname
        if name_ok:
            exact, partial, frac = compare_args(gargs, pargs)
        else:
            exact, partial, frac = False, False, 0.0
        verdicts.append({
            "step": i,
            "gold": {"tool_name": gname, "arguments": gargs},
            "pred": {"tool_name": pname, "arguments": pargs},
            "name_ok": name_ok,
            "args_exact": exact,
            "args_partial": partial,
            "args_fraction": round(frac, 3),
            "error": err,
        })
        # Advance the dialogue using the GOLD call + return so per-step
        # scoring stays independent of the model's mistakes.
        messages.append({"role": "assistant",
                         "content": f"API-Request: [{_render_call(gname, gargs)}]"})
        if i < len(gold_returns) and gold_returns[i]:
            messages.append({"role": "tool", "name": gname, "content": gold_returns[i]})
        if i < len(gold) - 1:
            messages.append({"role": "user", "content": "Generate the next API Request."})
    no_call = pred_count == 0 and not err
    return {
        "id": dp["id"],
        "level": dp["level"],
        "source_file": dp.get("source_file"),
        "gold_call_count": len(gold),
        "pred_call_count": pred_count,
        "verdicts": verdicts,
        "no_call": no_call,
        "error": err,
    }


def aggregate(results: list[dict]) -> dict:
    total_calls = 0
    name_ok = 0
    args_exact = 0
    args_partial = 0
    no_call_dps = 0
    frac_sum = 0.0
    for r in results:
        if r["no_call"]:
            no_call_dps += 1
        for v in r["verdicts"]:
            total_calls += 1
            if v["name_ok"]:
                name_ok += 1
            if v["args_exact"]:
                args_exact += 1
            if v["args_partial"]:
                args_partial += 1
            frac_sum += v["args_fraction"]
    return {
        "datapoints": len(results),
        "total_gold_calls": total_calls,
        "tool_name_exact": name_ok,
        "tool_name_accuracy": round(name_ok / total_calls, 4) if total_calls else 0,
        "args_exact": args_exact,
        "args_exact_accuracy": round(args_exact / total_calls, 4) if total_calls else 0,
        "args_partial_accuracy": round(args_partial / total_calls, 4) if total_calls else 0,
        "mean_args_fraction": round(frac_sum / total_calls, 4) if total_calls else 0,
        "no_call_datapoints": no_call_dps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Grade a custom LLM on the API-Bank benchmark")
    ap.add_argument("--ip", required=True, help="LLM server IP (or hostname)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-base-path", default="/v1", help="base path (default /v1 -> /v1/chat/completions)")
    ap.add_argument("--model", required=True, help="model name as known to the server")
    ap.add_argument("--api-key", default=None, help="optional bearer token")
    ap.add_argument("--level", type=int, choices=[1, 2, 3], required=True,
                    help="API-Bank level to grade")
    ap.add_argument("--variant", default="full",
                    choices=["full", "batch", "batch-icl"],
                    help="level-3 only: 'full' (50 multi-call convs) or 'batch'/'batch-icl' (per-step rows)")
    ap.add_argument("--mode", choices=["text", "tools", "qwen"], default="text",
                    help="'text' = parse API-Request: [...] lines; "
                         "'tools' = use OpenAI tools=[...] array; "
                         "'qwen' = Qwen3.x native: tool catalog JSON in the "
                         "system prompt + parse <tool> XML calls")
    ap.add_argument("--limit", type=int, default=None, help="grade only the first N datapoints")
    ap.add_argument("--processed-dir", default=DEFAULT_PROCESSED,
                    help="dir with level-*.jsonl + openai_tools.json")
    ap.add_argument("--output", default=None,
                    help="output path (without extension). Writes <out>.json + <out>.md")
    ap.add_argument("--timeout", type=int, default=60, help="per-request timeout (s)")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep between requests (s)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap the number of gold-call steps graded per datapoint "
                         "(useful for the level-2/3 full variants)")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="max_tokens for the model's response; bump to 4096+ for "
                         "reasoning models that need budget before emitting the answer")
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS certificate verification (self-signed proxies)")
    ap.add_argument("--extra-payload", default=None,
                    help='JSON string merged into the chat-completions payload '
                         "(e.g. '{\"chat_template_kwargs\":{\"enable_thinking\":false}}')")
    ap.add_argument("--extra-payload-file", default=None,
                    help="path to a JSON file merged into the chat-completions payload")
    args = ap.parse_args()

    # Build endpoint URL
    scheme = "https" if args.insecure or args.port == 443 else "http"
    base = f"{scheme}://{args.ip}:{args.port}{args.api_base_path.rstrip('/')}/chat/completions"
    print(f"[info] endpoint: {base}")
    print(f"[info] model:    {args.model}")
    print(f"[info] mode:     {args.mode}")
    print(f"[info] level:    {args.level}" + (f" ({args.variant})" if args.level == 3 else ""))

    # Pick the right file
    if args.level == 1:
        fn = "level-1.jsonl"
    elif args.level == 2:
        fn = "level-2.jsonl"
    else:
        if args.variant == "full":
            fn = "level-3.jsonl"
        elif args.variant == "batch":
            fn = "level-3-batch.jsonl"
        else:
            fn = "level-3-batch-icl.jsonl"
    path = os.path.join(args.processed_dir, fn)
    if not os.path.exists(path):
        print(f"[error] missing {path}", file=sys.stderr)
        return 1
    dps = [json.loads(line) for line in open(path)]
    if args.limit:
        dps = dps[: args.limit]
    print(f"[info] datapoints: {len(dps)} from {path}")

    tools = None
    if args.mode == "tools":
        tp = os.path.join(args.processed_dir, "openai_tools.json")
        if not os.path.exists(tp):
            print(f"[error] --mode tools requires {tp}; run scripts/convert_apibank.py first",
                  file=sys.stderr)
            return 1
        tools = json.load(open(tp))
        print(f"[info] tools array: {len(tools)} entries")

    qwen_system_prompt = None
    if args.mode == "qwen":
        # Build the Qwen3.x-native system prompt from the catalog. This mirrors
        # what convert_apibank.py puts in qwen_system_prompt.txt / per-dp, but
        # the grader recomputes it so we don't have to re-run the converter.
        cp = os.path.join(args.processed_dir, "tool_catalog.json")
        if not os.path.exists(cp):
            print(f"[error] --mode qwen requires {cp}; run scripts/convert_apibank.py first",
                  file=sys.stderr)
            return 1
        catalog = json.load(open(cp))
        qwen_system_prompt = build_qwen_system_prompt(catalog)
        print(f"[info] qwen system prompt: {len(qwen_system_prompt)} chars "
              f"({len(catalog)} tools)")

    out_path = args.output or os.path.join(
        HERE, "..", "data", "results",
        f"l{args.level}{'-'+args.variant if args.level==3 else ''}_{args.mode}.json",
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results: list[dict] = []
    extra_pl = None
    if args.extra_payload_file:
        extra_pl = json.load(open(args.extra_payload_file))
    elif args.extra_payload:
        extra_pl = json.loads(args.extra_payload)
    for i, dp in enumerate(dps):
        if i and args.sleep:
            time.sleep(args.sleep)
        try:
            v = grade_one(dp, base, args.model, args.mode, tools, args.api_key,
                          args.timeout, max_steps=args.max_steps,
                          max_tokens=args.max_tokens, insecure=args.insecure,
                          extra_payload=extra_pl,
                          qwen_system_prompt=qwen_system_prompt)
        except KeyboardInterrupt:
            print("\n[interrupt] saving partial results")
            break
        results.append(v)
        # progress
        ok = sum(1 for x in v["verdicts"] if x["args_exact"])
        tot = v["gold_call_count"]
        err_flag = "ERR" if v["error"] else "ok"
        print(f"  [{i+1}/{len(dps)}] {err_flag} dp={dp['id']} "
              f"gold={tot} pred={v['pred_call_count']} exact={ok}/{tot}")

    # Write json + md
    with open(out_path, "w") as f:
        json.dump({"endpoint": base, "model": args.model, "mode": args.mode,
                   "level": args.level, "variant": args.variant,
                   "datapoints_graded": len(results),
                   "results": results}, f, indent=2)
    agg = aggregate(results)
    md = build_markdown(base, args, agg, results)
    md_path = out_path[:-5] + ".md" if out_path.endswith(".json") else out_path + ".md"
    with open(md_path, "w") as f:
        f.write(md)
    print("\n" + "=" * 60)
    print(md)
    print("=" * 60)
    print(f"[ok] results: {out_path}")
    print(f"[ok] summary:  {md_path}")
    return 0


def build_markdown(base: str, args, agg: dict, results: list[dict]) -> str:
    lines = [
        "# API-Bank Grading Report",
        "",
        f"- **Endpoint**: `{base}`",
        f"- **Model**: `{args.model}`",
        f"- **Mode**: `{args.mode}`",
        f"- **Level**: {args.level}" + (f" ({args.variant})" if args.level == 3 else ""),
        f"- **Datapoints graded**: {agg['datapoints']}",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total gold tool-calls | {agg['total_gold_calls']} |",
        f"| tool-name exact | {agg['tool_name_exact']} ({agg['tool_name_accuracy']:.1%}) |",
        f"| args exact (full-call match) | {agg['args_exact']} ({agg['args_exact_accuracy']:.1%}) |",
        f"| args partial (≥50% keys match) | {agg['args_partial_accuracy']:.1%} |",
        f"| mean args fraction | {agg['mean_args_fraction']:.3f} |",
        f"| no-call datapoints | {agg['no_call_datapoints']} |",
        "",
        "## Per-datapoint verdicts",
        "",
        "| id | gold | pred | name | args-exact | args-frac |",
        "|---:|---|---|:--:|:--:|---:|",
    ]
    for r in results[:50]:
        for v in r["verdicts"][:3]:
            lines.append(
                f"| {r['id']} | "
                f"{v['gold']['tool_name']}({json.dumps(v['gold']['arguments'], ensure_ascii=False)[:40]}) | "
                f"{v['pred']['tool_name']}(...) | "
                f"{'yes' if v['name_ok'] else 'no'} | "
                f"{'yes' if v['args_exact'] else 'no'} | "
                f"{v['args_fraction']:.2f} |"
            )
    if len(results) > 50:
        lines.append(f"| ... | _{len(results)-50} more rows truncated — see full JSON_ | | | | |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())

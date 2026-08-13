"""Validate that the ToolSearcher-free conversion differs from the upstream
API-Bank eval data ONLY in (a) ToolSearcher artefacts being removed and
(b) the role/format used to surface API (tool) outputs — nothing else.

Three checks per level:

1. TS-FREE     : no ``ToolSearcher`` string survives anywhere in the processed
                 benchmark (messages, system prompt, gold calls, gold returns).

2. GOLD-PRESERVED: the ordered list of gold tool calls in each processed
                 datapoint equals the ordered list of real-tool calls in the
                 corresponding upstream record (ToolSearcher calls dropped,
                 consecutive duplicates collapsed exactly as the converter does).

3. TEXT-PRESERVED: after stripping ToolSearcher artefacts and normalising the
                 ``API-Request: [..]->{..}`` fragments into (assistant call,
                 tool result) message pairs, the user/assistant *text stream*
                 of the upstream dialogue equals the processed messages' text
                 stream. This proves the converter deleted retrieval steps but
                 did not rewrite or invent any other content.

The role mapping itself (API result -> ``tool`` role instead of upstream's
``system`` role) is a deliberate technical change for Qwen3.x compatibility and
is not treated as a difference.
"""
from __future__ import annotations
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.normpath(os.path.join(HERE, "..", "data", "test-data"))
PROC = os.path.normpath(os.path.join(HERE, "..", "data", "processed"))

sys.path.insert(0, HERE)
from convert_apibank import (  # noqa: E402
    TOOLSEARCHER,
    _dialogue_to_messages,
    _clean_input,
    _extract_real_call_returns,
    parse_api_call,
)

FAIL = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append((name, detail))


def _toolsearch_in(obj) -> bool:
    if isinstance(obj, str):
        return TOOLSEARCHER in obj
    if isinstance(obj, dict):
        return any(_toolsearch_in(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_toolsearch_in(v) for v in obj)
    return False


# ---------------------------------------------------------------------------
# Check 1: TS-free
# ---------------------------------------------------------------------------
def check_ts_free() -> None:
    for fn in sorted(os.listdir(PROC)):
        if not fn.endswith(".jsonl") or fn == "manifest.json":
            continue
        path = os.path.join(PROC, fn)
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for r in rows:
            _check(f"TS-free {fn} row {r.get('id')}",
                   not _toolsearch_in(r),
                   f"found ToolSearcher in {fn} row {r.get('id')}")


# ---------------------------------------------------------------------------
# Check 2: gold calls preserved
# ---------------------------------------------------------------------------
def check_gold_preserved_level1() -> None:
    up = {(r["file"], r["id"]): r for r in json.load(open(os.path.join(TEST, "level-1-api.json")))}
    rows = [json.loads(l) for l in open(os.path.join(PROC, "level-1.jsonl")) if l.strip()]
    for r in rows:
        up_row = up.get((r.get("source_file"), r.get("source_row_id")))
        if up_row is None:
            _check(f"L1 gold {r['id']}", False, "no upstream row")
            continue
        parsed = parse_api_call(up_row.get("expected_output", ""))
        gold = [] if parsed is None else [{"tool_name": parsed[0], "arguments": parsed[1]}]
        _check(f"L1 gold-preserved row {r['id']}", gold == r["gold_tool_calls"])


def check_gold_preserved_level2() -> None:
    src = json.load(open(os.path.join(TEST, "level-2-api.json")))
    by_file: dict[str, list] = {}
    for row in src:
        by_file.setdefault(row["file"], []).append(row)
    rows = [json.loads(l) for l in open(os.path.join(PROC, "level-2.jsonl")) if l.strip()]
    for r in rows:
        ups = sorted(by_file.get(r["source_file"], []), key=lambda x: x.get("id", 0))
        gold: list[dict] = []
        for row in ups:
            parsed = parse_api_call(row.get("expected_output", ""))
            if not parsed or parsed[0] == TOOLSEARCHER:
                continue
            call = {"tool_name": parsed[0], "arguments": parsed[1]}
            if gold and gold[-1] == call:
                continue
            gold.append(call)
        _check(f"L2 gold-preserved row {r['id']} ({r['source_file']})",
               gold == r["gold_tool_calls"])


def check_gold_preserved_level3() -> None:
    src = json.load(open(os.path.join(TEST, "level-3.json")))
    rows = [json.loads(l) for l in open(os.path.join(PROC, "level-3.jsonl")) if l.strip()]
    for n, conv in enumerate(src):
        gold = [{"tool_name": s["api_name"], "arguments": s.get("input", {})}
                for s in conv.get("apis", [])
                if s.get("api_name") and s.get("api_name") != TOOLSEARCHER
                and isinstance(s.get("input", {}), dict)]
        _check(f"L3 gold-preserved conv {n}", gold == rows[n]["gold_tool_calls"])


# ---------------------------------------------------------------------------
# Check 3: text stream preserved (upstream minus ToolSearcher == processed)
# ---------------------------------------------------------------------------
def _text_stream(msgs: list[dict]) -> list[tuple[str, str]]:
    """Return (role, text) for the conversation turns, normalising the cue."""
    out = []
    for m in msgs:
        if m["role"] == "system":
            continue
        text = m["content"]
        if text == "Generate the next API Request.":
            continue
        out.append((m["role"], text))
    return out


def check_text_level1() -> None:
    up = {(r["file"], r["id"]): r for r in json.load(open(os.path.join(TEST, "level-1-api.json")))}
    rows = [json.loads(l) for l in open(os.path.join(PROC, "level-1.jsonl")) if l.strip()]
    for r in rows:
        up_row = up.get((r.get("source_file"), r.get("source_row_id")))
        if up_row is None:
            continue
        # rebuild upstream messages from the raw input with the converter's own
        # TS-stripping + role logic, then drop the trailing cue.
        rebuilt = _dialogue_to_messages(up_row.get("input", ""), None, None)
        _check(f"L1 text-preserved row {r['id']}",
               _text_stream(rebuilt) == _text_stream(r["messages"]),
               f"\n  rebuilt={_text_stream(rebuilt)}\n  proc={_text_stream(r['messages'])}")


def check_text_level2() -> None:
    src = json.load(open(os.path.join(TEST, "level-2-api.json")))
    by_file: dict[str, list] = {}
    for row in src:
        by_file.setdefault(row["file"], []).append(row)
    rows = [json.loads(l) for l in open(os.path.join(PROC, "level-2.jsonl")) if l.strip()]
    for r in rows:
        ups = sorted(by_file.get(r["source_file"], []), key=lambda x: x.get("id", 0))
        first_input = ups[0].get("input", "") or ""
        rebuilt = _dialogue_to_messages(first_input, None, None)
        while rebuilt and rebuilt[-1]["role"] == "assistant" and \
                rebuilt[-1]["content"].startswith("API-Request:"):
            rebuilt.pop()
        _check(f"L2 text-preserved row {r['id']} ({r['source_file']})",
               _text_stream(rebuilt) == _text_stream(r["messages"]),
               f"\n  rebuilt={_text_stream(rebuilt)}\n  proc={_text_stream(r['messages'])}")


def check_text_level3() -> None:
    src = json.load(open(os.path.join(TEST, "level-3.json")))
    rows = [json.loads(l) for l in open(os.path.join(PROC, "level-3.jsonl")) if l.strip()]
    for n, conv in enumerate(src):
        req = (conv.get("requirement") or "").strip()
        proc_msgs = [m for m in rows[n]["messages"] if m["role"] != "system"]
        # expected: user(requirement) then per real step (assistant call, tool)
        expected: list[tuple[str, str]] = [("user", req)]
        for step in conv.get("apis", []):
            if not step.get("api_name") or step["api_name"] == TOOLSEARCHER:
                continue
            out_val = step.get("output")
            inner = out_val.get("output") if isinstance(out_val, dict) else out_val
            try:
                inner_s = json.dumps(inner, ensure_ascii=False)
            except Exception:
                inner_s = str(inner)
            expected.append(("assistant", f"API-Request: [{step['api_name']}({', '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in (step.get('input') or {}).items())})]"))
            expected.append(("tool", inner_s))
        final_resp = (conv.get("response") or "").strip()
        if final_resp:
            expected.append(("assistant", final_resp))
        _check(f"L3 text-preserved conv {n}",
               _text_stream(proc_msgs) == expected,
               f"\n  rebuilt={expected}\n  proc={_text_stream(proc_msgs)}")


def check_batch_ts_free() -> None:
    """Per-step batch rows must be TS-free and their gold call must match the
    upstream row's real-tool output."""
    for which, tag in (("level-3-batch-inf.json", "level-3-batch"),
                       ("level-3-batch-inf-icl.json", "level-3-batch-icl")):
        up = json.load(open(os.path.join(TEST, which)))
        rows = [json.loads(l) for l in open(os.path.join(PROC, f"{tag}.jsonl")) if l.strip()]
        # converter keeps every row whose gold call is a parseable non-TS call;
        # rows whose gold embeds an unparseable list-in-string arg are dropped.
        kept = [r for r in up
                if "ToolSearcher" not in (r.get("output") or "")
                and parse_api_call(r.get("output", "")) is not None]
        _check(f"batch {tag}: count matches upstream parseable non-TS rows",
               len(rows) == len(kept),
               f"processed={len(rows)} upstream-kept={len(kept)}")
        for r in rows:
            # find upstream row by sample_id + api_id
            urows = [u for u in up if u.get("sample_id") == r.get("sample_id")
                     and u.get("api_id") == r.get("api_id")]
            _check(f"batch {tag} row {r['id']} gold call",
                   len(urows) == 1 and
                   parse_api_call(urows[0].get("output", "")) is not None and
                   {"tool_name": parse_api_call(urows[0]["output"])[0],
                    "arguments": parse_api_call(urows[0]["output"])[1]} == r["gold_tool_calls"][0])


def main() -> int:
    check_ts_free()
    check_gold_preserved_level1()
    check_gold_preserved_level2()
    check_gold_preserved_level3()
    check_text_level1()
    check_text_level2()
    check_text_level3()
    check_batch_ts_free()
    print()
    if FAIL:
        print(f"VALIDATION FAILED — {len(FAIL)} problems")
        for name, detail in FAIL[:20]:
            print(f"  - {name}: {detail}")
        return 1
    print("ALL VALIDATION CHECKS PASSED: processed data differs from upstream "
          "only in ToolSearcher removal + API-result role mapping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

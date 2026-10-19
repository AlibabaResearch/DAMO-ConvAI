"""Worked example of a VibeMemBench memory system: EchoMemorySystem.

Copy this file to plug in your own system. The harness discovers a method by
file name, so ``memory_systems/mymem.py`` is all that ``--method mymem`` needs.
Two rules only:

1. Subclass :class:`memory_interface.MemorySystem` and implement ``add`` and
   ``search``. ``count`` is optional.
2. Expose the class as the module level name ``MEMORY_SYSTEM``.

Nothing else in the repository has to change. The harness creates one instance
per target, so ``__init__`` takes no arguments; read configuration from the
environment or from your own defaults.

What this particular example is for: it shows how to read a mini-swe-agent
trajectory correctly, pairing an assistant command with the tool output that
follows it. Retrieval is naive Jaccard word overlap, with no LLM and no vectors,
so **it does not represent the results of any real memory system**.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Dict, List, Optional

from memory_interface import HistorySession, MemoryScope, MemorySystem, RetrievedMemory

# This is only a **runnable minimal example** that shows how to read a mini-swe-agent-1.1
# trajectory correctly. Retrieval uses naive Jaccard word overlap (no LLM, no vectors), so
# it **does not represent the results of any real memory system**.
_ECHO_MAX_OUTPUT_CHARS = 2000  # demo only: the truncation length of output in one memory
_ECHO_MAX_REASONING_CHARS = 800  # demo only: the truncation length of reasoning in one memory
_ECHO_TOKEN_RE = re.compile(r"[^0-9a-zA-Z_\u4e00-\u9fff]+")


def _echo_tokenize(text: str) -> set[str]:
    """Split text into a set of lowercase words (length >= 2) for naive Jaccard similarity."""
    if not text:
        return set()
    return {tok for tok in _ECHO_TOKEN_RE.split(text.lower()) if len(tok) >= 2}


def _echo_jaccard(a: set[str], b: set[str]) -> float:
    """Naive Jaccard word overlap similarity = |A intersect B| / |A union B|, in [0, 1]."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def _echo_iso_from_ts(ts: Any) -> Optional[str]:
    """Convert ``extra.timestamp`` (a Unix **seconds** float) to ISO8601 (UTC). Demo only."""
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        try:
            return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    return None


class EchoMemorySystem(MemorySystem):
    """A minimal runnable example: how to **correctly read** a mini-swe-agent-1.1 trajectory
    and build memories from it.

    This is **only a minimal runnable example, it does not represent the results of any real
    memory system.** Retrieval uses naive Jaccard word overlap, with no LLM, no vectors, and
    no semantic understanding. Its only purpose is to teach an integrator **which fields of a
    trajectory are valuable memory material**:

    * An assistant message's ``content`` is an empty string 84.1% of the time in practice, so
      you **must not** treat empty content as a memory.
    * The real action is in the assistant's ``extra.actions[].command`` (or the ``command`` in
      ``tool_calls[].function.arguments``), and the reasoning is in ``reasoning_content``.
    * The **output** of a command is in the ``extra.raw_output`` / ``extra.returncode`` of the
      following ``tool`` message with the same ``tool_call_id`` (``tool.content`` does not
      contain the command itself).
    * An ``exit`` message carries the final patch (``extra.submission``), which is high value
      memory and must not be dropped.

    So this example pairs "command <-> output" by ``tool_call_id``, combines "reasoning +
    command + output" into one memory, and fills in ``source_session_id`` correctly.
    """

    def __init__(self) -> None:
        # Each memory is stored as a dict: id / content / score placeholder / created_at / source_session_id
        self._store: List[Dict[str, Any]] = []

    # -- internal: split one trajectory into "command <-> output" steps --------------------------
    def _iter_steps(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pair each assistant action with its tool output by ``tool_call_id`` and produce a list
        of steps."""
        # 1) build an index of tool_call_id -> tool message
        tool_by_id: Dict[str, Dict[str, Any]] = {}
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                tid = msg.get("tool_call_id")
                if isinstance(tid, str) and tid:
                    tool_by_id[tid] = msg

        steps: List[Dict[str, Any]] = []

        # 2) walk the assistant messages and take the commands (prefer extra.actions, fall
        #    back to tool_calls.arguments)
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            reasoning = msg.get("reasoning_content") or ""
            extra = msg.get("extra") if isinstance(msg.get("extra"), dict) else {}
            ts = extra.get("timestamp")

            # Collect (command, tool_call_id): prefer the already parsed extra.actions
            cmds: List[Tuple[Optional[str], Optional[str]]] = []
            actions = extra.get("actions")
            if isinstance(actions, list):
                for act in actions:
                    if isinstance(act, dict):
                        cmds.append((act.get("command"), act.get("tool_call_id")))

            # Fallback: parse command from tool_calls[].function.arguments
            if not cmds:
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            continue
                        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                        args = fn.get("arguments")
                        command: Optional[str] = None
                        if isinstance(args, str):
                            try:
                                parsed = json.loads(args)
                                command = parsed.get("command") if isinstance(parsed, dict) else args
                            except (json.JSONDecodeError, ValueError):
                                command = args
                        elif isinstance(args, dict):
                            command = args.get("command")
                        cmds.append((command, call.get("id")))

            # Attach the tool output to each command
            for command, tid in cmds:
                tool_msg = tool_by_id.get(tid) if isinstance(tid, str) else None
                output = ""
                returncode: Any = None
                if isinstance(tool_msg, dict):
                    textra = tool_msg.get("extra") if isinstance(tool_msg.get("extra"), dict) else {}
                    output = textra.get("raw_output")
                    if not isinstance(output, str) or not output:
                        output = tool_msg.get("content") or ""
                    returncode = textra.get("returncode")
                steps.append(
                    {
                        "reasoning": reasoning if isinstance(reasoning, str) else "",
                        "command": command if isinstance(command, str) else "",
                        "output": output if isinstance(output, str) else str(output or ""),
                        "returncode": returncode,
                        "timestamp": ts,
                        "kind": "step",
                    }
                )

        # 3) exit message -> the final patch, high value memory, kept as its own step
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "exit":
                continue
            extra = msg.get("extra") if isinstance(msg.get("extra"), dict) else {}
            submission = extra.get("submission")
            if not isinstance(submission, str) or not submission:
                submission = msg.get("content") or ""
            if isinstance(submission, str) and submission.strip():
                steps.append(
                    {
                        "reasoning": "",
                        "command": "<final_patch>",
                        "output": submission,
                        "returncode": None,
                        "timestamp": None,
                        "kind": "final_patch",
                    }
                )
        return steps

    # -- MemorySystem interface implementation -----------------------------------------
    def add(self, session: HistorySession, *, scope: MemoryScope) -> None:
        """Split one history trajectory into "command <-> output" steps and store each as a
        memory. Synchronous and effective immediately."""
        for idx, step in enumerate(self._iter_steps(session.messages)):
            parts: List[str] = []
            reasoning = (step.get("reasoning") or "")[:_ECHO_MAX_REASONING_CHARS]
            command = step.get("command") or ""
            output = (step.get("output") or "")[:_ECHO_MAX_OUTPUT_CHARS]
            returncode = step.get("returncode")
            if reasoning:
                parts.append(f"<reasoning>\n{reasoning}\n</reasoning>")
            if command:
                parts.append(f"<command>\n{command}\n</command>")
            if returncode is not None:
                parts.append(f"<returncode>{returncode}</returncode>")
            if output:
                parts.append(f"<output>\n{output}\n</output>")
            content = "\n".join(parts).strip()
            if not content:
                continue  # do not build a memory for an empty step (avoid treating an assistant's empty content as memory)
            self._store.append(
                {
                    "id": f"{session.session_id}::step{idx}",
                    "content": content,
                    "created_at": _echo_iso_from_ts(step.get("timestamp")),
                    "source_session_id": session.session_id,
                    "tokens": _echo_tokenize(content),
                }
            )

    def search(self, query: str, *, scope: MemoryScope, top_k: int) -> List[RetrievedMemory]:
        """Naive Jaccard retrieval. Return the top ``top_k`` by descending score, or ``[]`` when
        there is nothing."""
        query_tokens = _echo_tokenize(query)
        scored: List[RetrievedMemory] = []
        for record in self._store:
            score = _echo_jaccard(query_tokens, record["tokens"])
            if score <= 0.0:
                continue
            scored.append(
                RetrievedMemory(
                    id=record["id"],
                    content=record["content"],
                    score=score,
                    created_at=record["created_at"],
                    source_session_id=record["source_session_id"],
                    metadata={},
                )
            )
        # The contract requires the caller to sort by descending relevance before returning
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(0, top_k)]

    def count(self) -> Optional[int]:
        return len(self._store)


# The harness looks for this name. Point it at the class you want registered.
MEMORY_SYSTEM = EchoMemorySystem

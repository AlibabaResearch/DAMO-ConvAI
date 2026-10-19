# VibeMemBench Memory System Integration Contract (CONTRACT)

> Language: **English** | [简体中文](./CONTRACT_zh.md)

> The single integration guide for **external teams**. Read this document plus one file,
> [`memory_interface.py`](./memory_interface.py), and you can plug your memory system into
> VibeMemBench and run the evaluation. You never need to touch any internal code of this
> repository.

The interface design follows the duty boundaries and hard constraints of the
[Agent Memory Leaderboard (AML)](https://agentmemoryleaderboard.ai/api-guide).

---

## 1. Duty boundary: you only do Add and Search

VibeMemBench splits one full "solve a task with memory" chain into 4 steps, and **only the
first two are yours**:

```
+----+--------------+----------+------------------------------------------+
| #  | Stage        | Action   | Owner                                    |
+----+--------------+----------+------------------------------------------+
| 01 | Store        | Add      | > You: write history trajectories into    |
|    |              |          |   your memory store                       |
| 02 | Retrieve     | Search   | > You: retrieve the most relevant memories|
|    |              |          |   by query                                |
+----+--------------+----------+------------------------------------------+
| 03 | Generate     | Answer   | # Platform fixed: how the retrieved       |
|    |              |          |   results are injected into the agent     |
| 04 | Score        | Eval     | # Platform fixed: how the final outcome   |
|    |              |          |   is scored                               |
+----+--------------+----------+------------------------------------------+
```

**What the boundary means**: `run_memory.py` only feeds you history (`add`) and asks you for
retrieval results (`search`). What happens to those results next, and how they are scored, is
the job of another platform pipeline and is fully decoupled from your integration code. You
**do not need** and **should not** care about Answer or Eval.

---

## 2. The interface you implement: `MemorySystem`

Only **2 required methods** plus 1 optional method. Everything is **synchronous** and
**in-process** (no HTTP for now).

```python
class MemorySystem(abc.ABC):
    def add(self, session: HistorySession, *, scope: MemoryScope) -> None: ...
    def search(self, query: str, *, scope: MemoryScope, top_k: int) -> List[RetrievedMemory]: ...
    def count(self) -> Optional[int]: ...   # optional, returns None by default
```

### Hard constraints inherited from AML (you must follow them)

| Constraint | Meaning |
|---|---|
| **Strictly synchronous** | Once `add()` returns, that trajectory **is already retrievable by `search()`**. **Do not** return a task id, **do not** poll, **do not** use any async/deferred/eventual-consistency semantics. |
| **Sort before you return** | `search()` must **return results already sorted by descending relevance**. The harness will not reorder them for you (out of order only warns). |
| **Length cap** | The length `search()` returns must be `<= top_k`. Anything over is truncated by the harness with a warning. |
| **Empty list when nothing matches** | When there is no relevant memory, return `[]`, **not** `None` and **not** an exception. |
| **Exact `user_id` match, zero leakage** | `search()` may only look inside the memory of `scope.user_id`. **Never return memories from another `user_id`** namespace. |

---

## 3. The three data contract classes (field by field)

All field names match the AML payload keys exactly. Every class implements
`to_payload() -> dict` and `from_payload(d) -> Self`, to leave room for a thin HTTP transport
later (the contract will not need to change then).

### 3.1 `MemoryScope` (frozen dataclass) - the isolation boundary, maps to the AML `user_id` semantics

| Field | Type | Required | Meaning |
|---|---|---|---|
| `user_id` | `str` | yes | The memory namespace key, shaped like `vmb:{run_id}:{repo_dirname}:{target_instance_id}`. **You may only read and write inside this namespace.** |
| `repo` | `str` | yes | The full repository name, such as `adamchainz/flake8-comprehensions`. |
| `repo_dirname` | `str` | yes | The repository directory name (`repo` with `/` replaced by `__`), such as `adamchainz__flake8-comprehensions`. |
| `language` | `str` | yes | The language, one of `python` / `js` / `go` / `ts` / `rust` / `kotlin` / `swift` / `java` / `php`. |
| `target_instance_id` | `str` | yes | The instance_id of the target to solve, such as `adamchainz__flake8-comprehensions-553`. |
| `base_commit` | `str` | yes | The base commit of the target (the anchor of the temporal cut point). |
| `created_at` | `str` | yes | The creation time string of the target. |

> **Warning: the isolation boundary is not a "user" but a "repository x temporal cut point".**
> This benchmark builds a separate memory for **each target**, and it may only feed history from
> the same repository that happened **before** that target. `user_id` encodes the `run_id` (this
> run) + `repo_dirname` (repository) + `target_instance_id` (cut point), so the memory namespace
> of any two targets never overlaps. The harness also creates a **fresh `MemorySystem` instance**
> for each target, which forms a double guarantee: **in-process instance isolation + `user_id`
> namespace isolation**.

### 3.2 `HistorySession` (frozen dataclass) - one history trajectory, maps to one AML Add

| Field | Type | Required | Meaning |
|---|---|---|---|
| `session_id` | `str` | yes | The instance_id of the history issue, such as `adamchainz__flake8-comprehensions-143`. |
| `messages` | `List[Dict[str, Any]]` | yes | **The list of stripped raw message dicts**, order preserved (see sections 4 and 5). |
| `trajectory_format` | `Optional[str]` | yes | `"mini-swe-agent-1.1"`, or `None` (legacy variant, see section 4.1). |
| `meta` | `Dict[str, Any]` | yes | The session level metadata whitelist (see the table below). |

**The whole trajectory is added in one call. Chunking is up to you.** The full messages of one
history issue (p50 = 113, max = 514) are handed to `add()` **in one call**, and the harness does
**no chunking**. The AML rule of "split when over 20 messages or 2000 words" is **not adopted**,
because splitting a coding trajectory at an arbitrary sentence boundary would tear an
`assistant(tool_calls)` message from its matching `tool` result, and both halves would become
useless. Whether to chunk, and how, is entirely your decision.

The `meta` whitelist (pick only these from the traj `info`, **never pass through `info.config`**):

| Key | Type | When present | Meaning |
|---|---|---|---|
| `num_messages` | `int` | always | The number of messages in this session (computed and added by the harness). |
| `issue_number` | `int` | always | The trailing PR/issue number of this history, that is, its position on the timeline. A smaller number means earlier. Use it directly to sort or to weight recency. |
| `target_issue_number` | `int` | always | The trailing number of the current target, that is, the causal cut point. Always satisfies `issue_number < target_issue_number`. |
| `exit_status` | `str` | standard variant only | Such as `"Submitted"`. |
| `mini_version` | `str` | standard variant only | The mini-swe-agent version. |
| `model_stats` | `{"instance_cost": float, "api_calls": int}` | standard variant only | Cost and call count. |

> For the legacy variant (`info` missing or empty), `exit_status` / `mini_version` /
> `model_stats` are **omitted directly**, and **no `None` placeholder is filled**, so `meta` may
> hold only the three keys `num_messages` / `issue_number` / `target_issue_number`.
>
> The timeline of this benchmark is the trailing number, and **real PR dates are not provided**:
> the released data has `created_at` only for the 111 targets, not for the 3634 histories. The
> `extra.timestamp` inside trajectory messages is the machine clock of when the agent ran, not the
> real PR time, so it is meaningless for sorting.

### 3.3 `RetrievedMemory` (dataclass) - one search result, maps to an item in the AML Search response `data[]`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | `str` | yes, non empty | The unique memory id (matches the AML `id`). |
| `content` | `str` | yes, non empty | The memory body (matches the AML `content`). |
| `score` | `float` | yes | The relevance score, **higher means more relevant**. The returned list must be sorted by it in descending order. |
| `created_at` | `Optional[str]` | no | An ISO8601 timestamp, `None` by default. |
| `source_session_id` | `Optional[str]` | no | **An extension field of this benchmark**: which history issue this memory came from, used for attribution analysis. |
| `metadata` | `Dict[str, Any]` | no | Any extra info, `{}` by default. |

---

## 4. Wire message schema (the exact structure of each dict in `HistorySession.messages`)

Each message `add()` receives is the **raw dict** from traj.json, with its native field names and
nesting kept. **We do no normalization**: we do not rename `exit` to `user`, we do not run
`content` through `json.dumps`, and we do not drop `tool_calls` / `reasoning_content` / `extra`.
What you get is exactly the measured schema below.

### 4.1 Trajectory top level: support both variants

| Variant | Top level structure |
|---|---|
| **standard** | `{"info": {...}, "messages": [...], "trajectory_format": "mini-swe-agent-1.1"}` |
| **legacy** | `{"messages": [...]}` - **no `trajectory_format`, `info` missing or an empty dict** |

> The harness supports both (and keeps a defensive branch for "the top level is directly a list").
> For the legacy variant, `HistorySession.trajectory_format == None` and `meta` holds only
> `num_messages`. **The current open source data snapshot is entirely the legacy variant**, but
> your implementation should be just as robust for the standard variant.

### 4.2 `role` values (5 of them, `exit` is a valid role, not an error)

`system` / `user` / `assistant` / `tool` / `exit`.

### 4.3 Top level key variants per role (measured, 4 shapes in total)

```
system / user : ('content', 'role')
tool          : ('content', 'extra', 'role', 'tool_call_id')
assistant     : ('content', 'extra', 'function_call', 'provider_specific_fields',
                 'reasoning_content', 'role', 'tool_calls')      # <- of the last three, function_call / provider_specific_fields are already stripped, see section 5
exit          : ('content', 'extra', 'role')
```

> Note: the raw `assistant` shape contains `function_call` and `provider_specific_fields`, but
> both are deleted at the strip stage (section 5). So the assistant top level keys you actually
> see in `add()` are `('content', 'extra', 'reasoning_content', 'role', 'tool_calls')`.

### 4.4 `extra` key variants (5 measured)

```
tool (the vast majority) : ('exception_info', 'raw_output', 'returncode', 'timestamp')
tool (exception)         : ('exception', 'exception_info', 'exception_type', 'raw_output',
                            'returncode', 'timestamp')
assistant                : ('actions', 'cost', 'response', 'timestamp')   # <- response is stripped and usage/model are promoted out, see section 5
exit                     : ('exit_status', 'submission')
exit (exception)         : ('exception_str', 'exit_status', 'submission', 'traceback')
```

> After stripping, the assistant `extra` is actually `('actions', 'cost', 'timestamp', 'usage',
> 'model')` (`response` is deleted, and `usage`/`model` are promoted from `response`).

### 4.5 Field examples (measured)

```pythonc
# -- assistant -----------------------------------------------------------
content           = ""                       # Warning: 84.1% of assistant content is an empty string in practice!
reasoning_content = "Let me analyze the issue..."   # the model's chain of thought, high value
tool_calls        = [{"index": 0, "id": "call_b95145f54569446ca6bd432d",
                      "type": "function",
                      "function": {"name": "bash",
                                   "arguments": "{\"command\": \"cd /testbed && ls -la\"}"}}]
extra.actions     = [{"command": "cd /testbed && ls -la",           # <- the already parsed command
                      "tool_call_id": "call_b95145f54569446ca6bd432d"}]
extra.cost        = 0.0                       # float
extra.timestamp   = 1782297212.8508334        # float, Unix **seconds** (not milliseconds! see section 5.3)
extra.usage       = {...}                     # promoted from response, see section 4.6
extra.model       = "deepseek-v4-flash"       # promoted from response, per call model provenance

# -- tool ----------------------------------------------------------------
content           = "<returncode>0</returncode>\n<output>\n...\n</output>"
                    # Warning: content does **not** contain the command itself; the command is in
                    # the previous assistant's tool_calls / extra.actions
tool_call_id      = "call_b95145f54569446ca6bd432d"   # <- use it to pair with the assistant action
extra.raw_output  = "..."
extra.returncode  = 0
extra.exception_info = ""

# -- exit ----------------------------------------------------------------
content           = "<the full final patch diff>"    # the real output of the past agent, high value memory
extra.exit_status = "Submitted"
extra.submission  = "<the full final patch diff>"
```

> **The correct way to read this** (the point the `EchoMemorySystem` example demonstrates):
> an assistant's `content` is often an empty string, so **do not treat it as a memory**. The real
> action is in `extra.actions[].command` (or the `command` in `tool_calls[].function.arguments`),
> and the reasoning is in `reasoning_content`. The **output** of a command is in the
> `extra.raw_output` / `extra.returncode` of the following `tool` message with the same
> `tool_call_id`. Use `tool_call_id` to pair "command <-> output", then combine them into one
> memory.

### 4.6 The shape of `extra.usage` (sub keys vary by provider, pass through as is)

```json
{"completion_tokens": 194, "prompt_tokens": 1962, "total_tokens": 2156,
 "completion_tokens_details": {"accepted_prediction_tokens": null, "audio_tokens": null,
   "reasoning_tokens": 144, "rejected_prediction_tokens": null},
 "prompt_tokens_details": {"audio_tokens": null, "cached_tokens": 0}}
```

> In another sample, `completion_tokens_details` also had `text_tokens` / `image_tokens` keys,
> which confirms that **`usage` must be passed through as a whole and field names must not be
> hardcoded**.

---

## 5. What we strip, and why (the only exception to "pass through as is")

On top of "pass through as is", the harness **only removes the following redundant items**
(measured to shrink the size significantly). The messages your `add()` receives are already
stripped this way.

### 5.1 Deleted fields

| Action | Field | Reason |
|---|---|---|
| **delete** | `extra.response` | The raw LLM API response object, about 35% of all characters. Its `choices` **fully duplicate** `content` / `tool_calls` / `reasoning_content`. |
| **delete** | `provider_specific_fields` | Its `reasoning_content` **duplicates the top level field 100%**. Its `refusal` is always `null` in practice. |
| **delete** | `function_call` | Always `null` in practice, an OpenAI legacy dead field. |

### 5.2 Fields promoted from `extra.response` up to `extra`

| Action | New field <- source | Reason |
|---|---|---|
| **promote** | `extra.usage` <- the original `extra.response.usage` | usage is the **only non redundant** info in response, so the whole dict is passed through as is. |
| **promote** | `extra.model` <- the original `extra.response.model` | A single string, keeps per call model provenance. |

### 5.3 What we **do not** do (important)

- **No field renaming of any kind.**
- **No timestamp unit conversion of any kind.** `extra.timestamp` stays a **Unix seconds `float`**
  as is (such as `1782297212.8508334`).
  > Warning: **this differs from AML**. The AML `created_at` / timestamps use **milliseconds
  > `int`**. In this benchmark's wire messages, `extra.timestamp` is **seconds `float`**. If your
  > system needs a milliseconds int, do the `int(ts * 1000)` conversion yourself.

### 5.4 Fields we **must keep**

`extra.cost`, `extra.timestamp`, `extra.actions`, `extra.raw_output`, `extra.returncode`,
`extra.exception_info`, `extra.exception`, `extra.exception_type`, `extra.exit_status`,
`extra.submission`, `extra.exception_str`, `extra.traceback`; and the top level `content`,
`reasoning_content`, `tool_calls`, `tool_call_id`, `role`.

---

## 6. Temporal causal filter (a harness side rule you need to understand)

When building memory for a target, the harness **only feeds same repository history that happened
before it**. Otherwise you would be solving a past issue with future experience (data leakage).
The rule:

> Under the same `repo_dirname`, only history whose `instance_id` **trailing number is < the
> target's trailing number** is fed to `add()`.

For example, for target `...-553`, only histories such as `...-143` / `...-179` / `...-205` with a
trailing number `< 553` are fed. `...-553` (the target itself) and anything numbered larger are
excluded. So every `HistorySession` you receive is **strictly earlier** than the current target.

Note that the target's own trajectory **does sit on disk** (that one file is right there in the
`history/<lang>/<repo>/` directory). The only thing blocking it is this filter, so it is a hard
constraint, not decoration. Both numbers are given to you explicitly as `meta["issue_number"]` and
`meta["target_issue_number"]`, so you do not need to dig the string out of `session_id`.

**Large file defense**: the harness offers a CLI argument `--max-session-mb` (default 64, decimal
MB). A history session over the threshold is skipped **before** `json.load`, logged as a WARNING,
and written into the result JSON's `skipped_sessions` (with `session_id` / `size_mb` / `reason`).
Otherwise that ~102 MB trajectory in the dataset would blow up process memory. **This means your
`add()` is not guaranteed to receive all history of a repository**; the ones blocked by size will
be absent.

---

## 7. The gold red line (never cross it)

In any data passed to `add()` / `search()`, and in any field written into the result JSON, the
**target's gold answer must never appear**. The gold list (the harness structurally never passes
these fields of the target row through to you):

```
patch, test_patch, pr_description, FAIL_TO_PASS, PASS_TO_PASS,
install_config, meta, and the target's own info.submission
```

- The `query` of `search()` **is only ever `target["problem_statement"]`**, not one character more.
- **Keep the distinction**: the `exit` message / `extra.submission` of the **history trajectory
  itself** inside a `HistorySession` is the **real output of a past agent**. It is **not** the
  target's gold. It is **legitimate memory material that must be kept and may be used**.
- Before `run_one()` finishes, the harness calls `_assert_no_gold_leakage()` to validate:
  - the `query` must equal `problem_statement` character for character;
  - the full text of the result JSON must not contain the **value** of the target's `patch` /
    `test_patch` / `pr_description` / `FAIL_TO_PASS`.
  > `PASS_TO_PASS` is not used as a substring fingerprint. They are the **existing** tests that
  > "pass both before and after the change", and an earlier history of the same repository already
  > ran the same test suite (such as `...::test_version` in pytest output). They belong to shared
  > repository vocabulary, not leakage. `install_config` / `meta` are not fingerprinted for the
  > same reason (they may embed repo/image identifiers that would collide with legitimate fields
  > in the result). They are covered instead by the structural guarantee that the harness never
  > reads or passes them through.

---

## 8. Output format (produced by the harness, for your reference)

Each target produces one directory `{out_root}/{method}/{language}/{instance_id}/` with two files:

| File | Role |
|---|---|
| `memory.txt` | **The experience text injected into the agent.** It joins `retrieved[]` into readable text in the already sorted descending order, and `run_agent.py --method <your name>` reads it directly. |
| `memory.json` | Full provenance, structured as below. It is not injected, it is used for audit and attribution. |

```json
{
  "method": "demo",
  "run_id": "20260913-142530",
  "instance_id": "adamchainz__flake8-comprehensions-553",
  "repo": "adamchainz/flake8-comprehensions",
  "language": "python",
  "user_id": "vmb:20260913-142530:adamchainz__flake8-comprehensions:adamchainz__flake8-comprehensions-553",
  "elapsed_seconds": 12.34,
  "num_histories_fed": 8,
  "num_messages_fed": 904,
  "skipped_sessions": [{"session_id": "...", "size_mb": 102.3, "reason": "exceeds --max-session-mb"}],
  "total_memories_stored": 123,
  "num_retrieved": 10,
  "retrieved": [
    {"id": "...", "content": "...", "score": 0.8712, "created_at": null,
     "source_session_id": "adamchainz__flake8-comprehensions-528", "metadata": {}}
  ]
}
```

- The top level key stays `retrieved` (more readable than the AML `data`), but **the field names of
  each item inside the array match the AML `data[]` item exactly** (`id` / `content` / `score` /
  `created_at` + the extensions `source_session_id` / `metadata`).
- `elapsed_seconds` = the total wall clock time of add + search for this target.
- `total_memories_stored` = the return value of your `count()` (`null` if not implemented).

### CLI arguments

| Argument | Default | Meaning |
|---|---|---|
| `--method` | `demo` | The memory system name, which also decides the output directory `{out_root}/{method}/`. |
| `--out-root` | `outputs/memory` | The output root directory. |
| `--top-k` | `1` | The max number of items `search()` returns, and the number written into `memory.txt`. The protocol injects one top-ranked experience, so raising this makes a run incomparable to the reported numbers. |
| `--filter-lang` | none | Run only one language (`python` / `js` / ...). |
| `--limit` | none | Run at most this many targets (for debugging). |
| `--overwrite` | off | Force a rerun when the result already exists (skip by default, supports resume). |
| `--run-id` | auto `YYYYMMDD-HHMMSS` | The run id for this run, encoded into `user_id`. |
| `--max-session-mb` | `64` | The size cap of one history trajectory (decimal MB), skipped when over. |

---

## 9. Integrate your own memory system in 30 lines

Subclass `MemorySystem` from `memory_interface.py` and save it as one file at
`memory_systems/<your name>.py`. The harness discovers a method by file name, so nothing
else in the repository has to change. Copy `memory_systems/demo.py` as a starting point.
Here is a minimal skeleton you can apply directly:

```python
from memory_interface import (
    MemorySystem, HistorySession, MemoryScope, RetrievedMemory,
)

class MyMemorySystem(MemorySystem):
    def __init__(self):
        self._db = {}                       # user_id -> your own store (vector db / graph db / ...)

    def add(self, session: HistorySession, *, scope: MemoryScope) -> None:
        # session.messages is the list of stripped raw mini-swe-agent-1.1 message dicts.
        # session.meta["issue_number"] is the temporal position of this history; use it to weight recency.
        # Tip: use tool_call_id to pair the assistant's extra.actions[].command with the following
        #      tool.extra.raw_output, then combine it with reasoning_content into a memory unit.
        store = self._db.setdefault(scope.user_id, [])      # <- must isolate by user_id
        for i, msg in enumerate(session.messages):
            text = msg.get("reasoning_content") or msg.get("content") or ""
            if text.strip():
                store.append({"id": f"{session.session_id}#{i}",
                              "content": text,
                              "source_session_id": session.session_id})
        # Warning: once this returns it must already be retrievable by search. No async, no task id.

    def search(self, query: str, *, scope: MemoryScope, top_k: int):
        store = self._db.get(scope.user_id, [])             # <- search only inside this user_id
        scored = [(self._relevance(query, m["content"]), m) for m in store]
        scored.sort(key=lambda x: x[0], reverse=True)        # <- you must sort descending yourself
        return [RetrievedMemory(id=m["id"], content=m["content"], score=s,
                                source_session_id=m["source_session_id"])
                for s, m in scored[:top_k]]                  # <- naturally returns [] when nothing matches

    def _relevance(self, query: str, content: str) -> float:
        return 0.0  # replace with your vector / graph / LLM scoring

    def count(self):
        return sum(len(v) for v in self._db.values())

# The harness looks for this exact name. The file name is what --method takes.
MEMORY_SYSTEM = MyMemorySystem
```

Run:

```bash
# 1) build memory, producing outputs/memory/mymem/<lang>/<iid>/memory.txt
python scripts/run_memory.py --method mymem --filter-lang python --limit 3

# 2) inject those memories into the agent and run the tasks
python scripts/run_agent.py --method mymem
```

> `EchoMemorySystem` (`--method demo`) is a **runnable minimal teaching example** that shows "how to
> read a trajectory correctly". Its retrieval uses naive Jaccard word overlap (no LLM, no vectors).
> **It does not represent the results of any real memory system**, it is only an integration demo.

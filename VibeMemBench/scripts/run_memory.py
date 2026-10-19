#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VibeMemBench external memory interface harness (fully self contained).

========================================================================
1. What this file is
========================================================================
VibeMemBench is a benchmark for the "memory systems" of coding agents. This file is
the **only** integration point exposed to the outside world. An external team reads
just this file plus ``CONTRACT.md`` in the same directory, and it can plug its own
memory system in and run the evaluation. It never needs to touch any internal code of
this repository.

The design follows the **duty boundary** rules of the Agent Memory Leaderboard (AML,
https://agentmemoryleaderboard.ai/api-guide):

    01 / Store     -> Add     -> integrator (write the history trajectories into your store)
    02 / Retrieve  -> Search  -> integrator (retrieve relevant memories by query)
    03 / Generate  -> Answer  -> platform fixed (how results are injected into the agent)
    04 / Score     -> Eval    -> platform fixed (how the outcome is scored)

In other words, **this harness only handles Add (feed history) and Search (retrieve)**.
What happens to the results afterwards, and how they are scored, belongs to another
pipeline of the platform and is decoupled from this file.

========================================================================
2. Hard constraints (you must follow them)
========================================================================
* Fully self contained: this file **does not import** any internal framework package or
  internal memory library of this repository (those are not open sourced with the
  benchmark). It depends only on the standard library plus ``pandas`` (``pyarrow`` is a
  lazy import inside :func:`load_targets`).
* Python >= 3.12; ``pyarrow<=17.0.0``, ``pandas<=2.2.3`` (compatible with CentOS 7 /
  glibc 2.17, do not raise these version caps on your own).
* Transport shape: **an in-process Python ABC only**. It starts no HTTP server and pulls
  in no fastapi/flask/uvicorn. But every data class field name matches the AML payload
  keys exactly, and each provides ``to_payload()`` / ``from_payload()`` to leave room for
  a thin HTTP wrapper later.
* Strictly synchronous: once ``add()`` returns, the record is already retrievable by
  ``search()``. No task id, no polling, no async or deferred semantics (AML states this
  explicitly).

========================================================================
3. Dataset facts (measured)
========================================================================
Directory layout::

    benchmark/
    ├── test.parquet                  # 111 rows, one row per target to solve
    └── history/{language}/{repo_dirname}/{instance_id}/{instance_id}.traj.json
                                      # repo_dirname = repo with '/' replaced by '__'

``test.parquet`` has 16 columns: ``base_commit, created_at, image_name,
instance_id, interface, language, license, patch, pr_description,
problem_statement, repo, test_patch, FAIL_TO_PASS, PASS_TO_PASS,
install_config, meta``. Among them **``problem_statement`` is the only source of the
retrieval query**. ``patch / test_patch / pr_description / FAIL_TO_PASS / PASS_TO_PASS /
install_config / meta`` are all **gold answers** and must never enter any add/search
call or the result JSON.

``traj.json`` comes in two variants, and this file supports both:

* standard variant: ``{"info": {...}, "messages": [...], "trajectory_format": "mini-swe-agent-1.1"}``
* legacy variant: ``{"messages": [...]}`` (no ``trajectory_format``, ``info`` missing or empty)
* (defensive branch) the top level is directly a ``list``: ``[...]``

A message ``role`` is one of: ``system / user / assistant / tool / exit``
(``exit`` is a **valid role, not an error**. It carries the final patch and is a high
value signal for coding memory).

========================================================================
4. Message fidelity policy
========================================================================
Each message ``add()`` receives is the **raw dict** from traj.json, with its native field
names and nesting kept. No normalization is done (``exit`` is not renamed to ``user``,
content is not run through json.dumps, ``tool_calls`` / ``reasoning_content`` / ``extra``
are not dropped). The only exception is stripping three redundant heavyweight fields per
Decision 2, see :func:`_strip_message`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd  # the only allowed third party top level dependency (pyarrow is lazy inside load_targets)

# ------------------------------------------------------------------------
# Path constants: everything resolves relative to this file's directory (scripts/), so
# the script runs from any cwd. The dataset is not here, it sits in benchmark/ at the
# repository root (downloaded from Hugging Face).
# ------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent  # scripts/
_REPO_ROOT = _HERE.parent  # repository root
_BENCH_ROOT = _REPO_ROOT / "benchmark"  # dataset directory
_TEST_TARGETS_PATH = _BENCH_ROOT / "test.parquet"
_HISTORY_ROOT = _BENCH_ROOT / "history"
# The experience directory defaults to outputs/ at the repository root (already git
# ignored). It is not written into benchmark/, which stays a pure dataset.
_DEFAULT_OUT_ROOT = _REPO_ROOT / "outputs" / "memory"

# Make memory_interface resolvable whether this file runs as a script or is imported.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Auto-load <repo>/.env so a registered memory system can read its own model and
# embedder keys from a file instead of exporting them every time. When
# python-dotenv is missing we warn rather than fail silently, otherwise a filled
# in .env looks like it works while every value is quietly ignored.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=False)
except ImportError:
    if (_REPO_ROOT / ".env").exists():
        print(
            f"[warn] {_REPO_ROOT / '.env'} exists but python-dotenv is not "
            "installed, so it is ignored. Run: pip install python-dotenv",
            file=sys.stderr,
        )

_LOG = logging.getLogger("vmb.benchmark")

# ------------------------------------------------------------------------
# Safety red line: the gold answer fields of a target (section 6).
# The **values** of these fields must never appear in the data of add()/search(), nor be
# written into the result JSON.
# Note: the target's own info.submission is also gold, but this harness never loads the
# target's own trajectory (the temporal causal filter already excludes it), so it is not
# listed as a separate parquet column.
# ------------------------------------------------------------------------
_GOLD_FIELDS: Tuple[str, ...] = (
    "patch",
    "test_patch",
    "pr_description",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "install_config",
    "meta",
)

# The gold subset actually used for **substring leak detection**.
#
# Why not all of :data:`_GOLD_FIELDS`? Because when substring matching against "earlier
# history trajectories of the same repo", some fields cause **false positives**. They are
# not the target's answer:
#
#   * PASS_TO_PASS and FAIL_TO_PASS are both lists of **test node ids**, which are shared
#     repository vocabulary rather than the target's answer.
#     PASS_TO_PASS holds tests that pass before and after the change, so an earlier PR of
#     the same repo ran them too and its pytest output legitimately names them.
#     FAIL_TO_PASS was originally assumed safe, on the theory that those tests are newly
#     added by test_patch and so cannot exist earlier. Measurement refuted that: over the
#     111 targets, FAIL_TO_PASS matched earlier history 167 times across 42 targets, while
#     patch, test_patch, and pr_description matched **zero** times. The reason is in the
#     name: a test that goes from fail to pass may well be a test that already existed and
#     was failing. For example ``tests/test_json.py::test_config`` is in the FAIL_TO_PASS of
#     systemd__mkosi-3346, and the trajectory of the earlier systemd__mkosi-2044 already
#     printed ``FAILED tests/test_json.py::test_config``. Knowing a test's name is not
#     knowing the fix, so neither list is used as a leak fingerprint.
#   * install_config / meta are harness config and annotations. They may embed repo/image
#     identifiers, and those identifiers legitimately appear in the repo/instance_id fields
#     of the result JSON, so substring matching would always raise a false positive.
#
# These groups are covered by a **structural guarantee**: the harness never reads the
# target row's PASS_TO_PASS / FAIL_TO_PASS / install_config / meta into any HistorySession
# or result field, so they have no way to leak. The primary defence against feeding a
# target its own trajectory is the temporal causal filter, not this substring check.
#
# The 3 fields kept in the fingerprint set are the answer content **unique** to the target:
# patch / test_patch / pr_description, that is the solution diff, the new tests, and the PR
# text. They are one of a kind, and they never appear in earlier history.
_GOLD_FINGERPRINT_FIELDS: Tuple[str, ...] = (
    "patch",
    "test_patch",
    "pr_description",
)

# Minimum length of a leak fingerprint: a very short string (such as
# meta.llm_metadata.code == "A") can easily hit by coincidence in unrelated text, so a
# length threshold filters it out and avoids false positives. patch/test_patch/
# pr_description/test node ids are all far longer than this threshold.
_GOLD_MIN_LEN = 16

# ------------------------------------------------------------------------
# Strip rules (Decision 2): expressed as **explicit sets**, not a recursive generic pruner
# ------------------------------------------------------------------------
# Top level message key blacklist:
#   * provider_specific_fields -- its reasoning_content duplicates the top level field 100%,
#                                 and its refusal is always null in practice
#   * function_call            -- always null in practice, an OpenAI legacy dead field
_MSG_DROP_KEYS: frozenset[str] = frozenset({"provider_specific_fields", "function_call"})
# Heavyweight keys to delete inside extra:
#   * response -- the raw LLM API response object, about 35% of all characters; its choices
#                 fully duplicate content/tool_calls/reasoning_content
_EXTRA_DROP_KEYS: frozenset[str] = frozenset({"response"})
# Keys **promoted** from extra.response up to extra (the only non redundant info in
# response, passed through as is):
#   * usage -- token usage; the sub keys vary by provider, so pass the whole dict through
#              as is and never hardcode field names
#   * model -- a single string, keeps per call model provenance
_EXTRA_PROMOTE_KEYS: Tuple[str, ...] = ("usage", "model")

# ------------------------------------------------------------------------
# HistorySession.meta whitelist: pick only these from the traj info, **never pass through
# info.config** (info.config.agent.system_template / instance_template are extremely long
# harness prompt templates, bloat with no memory value). num_messages is computed and
# added by the harness.
# ------------------------------------------------------------------------
_META_MODEL_STATS_KEYS: Tuple[str, ...] = ("instance_cost", "api_calls")

# ------------------------------------------------------------------------
# CLI defaults
# ------------------------------------------------------------------------
_DEFAULT_METHOD = "demo"
# The protocol injects **one** experience: "a declared target query selects one
# top-ranked experience, which the protocol injects into MiniSWEAgent". So the
# default asks search() for exactly the one item that will be used. Raising it
# injects more context than the protocol allows and makes a run incomparable to
# the reported numbers.
_DEFAULT_TOP_K = 1
_DEFAULT_MAX_SESSION_MB = 64.0  # decimal MB (bytes / 1_000_000), same unit as size_mb in the result JSON


# ========================================================================
# The data contract classes and the MemorySystem abstract interface were moved to
# memory_interface.py. An integrator only reads that one file to plug in. This script
# imports from it and no longer redefines them.
# ========================================================================
from memory_interface import (  # noqa: E402  (importable only after the path injection)
    HistorySession,
    MemoryScope,
    MemorySystem,
    RetrievedMemory,
)


# ========================================================================
# Wire layer: trajectory loading + stripping + meta whitelist
# ========================================================================
def _strip_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """On top of "pass through as is", strip the redundant heavyweight fields listed in
    Decision 2.

    Strip rules (expressed as explicit sets, one reason each):

    * Delete the top level ``provider_specific_fields``: its ``reasoning_content``
      duplicates the top level field 100%, and its ``refusal`` is always null in practice,
      so the whole block adds no information.
    * Delete the top level ``function_call``: always null in practice, an OpenAI legacy
      dead field.
    * Delete ``extra.response``: the raw LLM API response object, about 35% of all
      characters; its ``choices`` fully duplicate ``content`` / ``tool_calls`` /
      ``reasoning_content``.
    * **Promote** ``usage`` from ``extra.response`` to ``extra.usage``: token usage is the
      only non redundant info in response, so pass the whole dict through as is (the sub
      keys vary by provider, do not hardcode them).
    * **Promote** ``model`` from ``extra.response`` to ``extra.model``: a single string
      that keeps per call model provenance.

    **Must keep** (this function does not touch them): ``extra.cost`` / ``extra.timestamp``
    / ``extra.actions`` / ``extra.raw_output`` / ``extra.returncode`` /
    ``extra.exception_info`` / ``extra.exception`` / ``extra.exception_type`` /
    ``extra.exit_status`` / ``extra.submission`` / ``extra.exception_str`` /
    ``extra.traceback``; and the top level ``content`` / ``reasoning_content`` /
    ``tool_calls`` / ``tool_call_id`` / ``role``.

    **No renaming and no timestamp unit conversion**: ``extra.timestamp`` stays a Unix
    seconds float as is (note AML uses milliseconds int, the two differ, and the contract
    doc states this explicitly).

    Returns a **new dict** and does not modify the object passed in (it shallow copies the
    top level and the extra level).
    """
    if not isinstance(message, dict):
        # Defensive branch: a little dirty data may not be a dict, return it as is and let
        # the caller tolerate it
        return message

    out = dict(message)  # shallow copy the top level so the caller's original object stays clean

    # 1) delete the top level blacklist keys
    for key in _MSG_DROP_KEYS:
        out.pop(key, None)

    # 2) handle extra: delete response, but promote usage/model out first
    extra = out.get("extra")
    if isinstance(extra, dict):
        extra = dict(extra)  # copy one level so the original extra is not modified
        response = extra.get("response")
        # promote first (read the values before response is deleted)
        if isinstance(response, dict):
            for promote_key in _EXTRA_PROMOTE_KEYS:
                if promote_key in response:
                    extra[promote_key] = response[promote_key]
        # then delete the blacklist key (response)
        for key in _EXTRA_DROP_KEYS:
            extra.pop(key, None)
        out["extra"] = extra

    return out


def _load_trajectory(path: Path) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """Read one ``*.traj.json``, support all three shapes, and return
    ``(messages, trajectory_format, info)``.

    * standard variant: ``{"info": {...}, "messages": [...], "trajectory_format": "mini-swe-agent-1.1"}``
    * legacy variant: ``{"messages": [...]}`` -- no ``trajectory_format``, ``info`` missing or empty
    * defensive branch: the top level is directly a ``list`` (``[...]``) -- never seen in
      practice, kept for harmless compatibility

    ``info`` may contain ``config`` (with very long system_template / instance_template,
    which is bloat). This function returns ``info`` as is, and :func:`_build_session_meta`
    picks the fields by whitelist and **never passes through info.config**.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Defensive branch: the top level is a list
    if isinstance(data, list):
        return data, None, {}

    if not isinstance(data, dict):
        raise ValueError(f"unrecognized traj.json top level type: {type(data).__name__} ({path})")

    messages = data.get("messages")
    if messages is None:
        messages = []
    if not isinstance(messages, list):
        raise ValueError(f"traj.json messages is not a list: {type(messages).__name__} ({path})")

    trajectory_format = data.get("trajectory_format")  # None for the legacy variant
    info = data.get("info") or {}
    if not isinstance(info, dict):
        info = {}
    return messages, trajectory_format, info


def _build_session_meta(info: Dict[str, Any], num_messages: int) -> Dict[str, Any]:
    """Build ``HistorySession.meta`` by picking fields from ``info`` by whitelist.

    Whitelist: ``exit_status``(str) / ``mini_version``(str) /
    ``model_stats``({"instance_cost": float, "api_calls": int}) / ``num_messages``(int).

    * ``num_messages`` is computed and added by the harness (the message count), always present.
    * **Never pass through info.config** (it holds the huge system_template / instance_template).
    * For the legacy variant (empty ``info``), the matching keys are **omitted directly**,
      not filled with a None placeholder.
    """
    meta: Dict[str, Any] = {"num_messages": num_messages}

    if not isinstance(info, dict) or not info:
        return meta

    exit_status = info.get("exit_status")
    if isinstance(exit_status, str):
        meta["exit_status"] = exit_status

    mini_version = info.get("mini_version")
    if isinstance(mini_version, str):
        meta["mini_version"] = mini_version

    model_stats = info.get("model_stats")
    if isinstance(model_stats, dict):
        picked: Dict[str, Any] = {}
        instance_cost = model_stats.get("instance_cost")
        if isinstance(instance_cost, (int, float)) and not isinstance(instance_cost, bool):
            picked["instance_cost"] = instance_cost
        api_calls = model_stats.get("api_calls")
        if isinstance(api_calls, int) and not isinstance(api_calls, bool):
            picked["api_calls"] = api_calls
        if picked:
            meta["model_stats"] = picked

    return meta


# ========================================================================
# Temporal causal filter (section 5): only feed same repo history that happened before the target
# ========================================================================
def _repo_dirname(repo: str) -> str:
    """``repo`` -> directory name: replace ``'/'`` with ``'__'`` (e.g. ``a/b`` -> ``a__b``)."""
    return str(repo).replace("/", "__")


def _trailing_number(instance_id: str) -> Optional[int]:
    """Take the trailing PR/issue number of ``instance_id`` (e.g. ``...-553`` -> ``553``).

    The trailing number encodes **chronology**: a smaller number means earlier. Returns
    ``None`` when there is no trailing number.
    """
    match = re.search(r"(\d+)\s*$", str(instance_id).strip())
    return int(match.group(1)) if match else None


def _history_dir_for_target(target: Dict[str, Any]) -> Optional[Path]:
    """Return the history trajectory directory of the target's repository, or ``None`` if it
    does not exist.

    The path is ``history/{language}/{repo_dirname}/``.
    """
    language = target.get("language")
    repo = target.get("repo")
    if not language or not repo:
        return None
    candidate = _HISTORY_ROOT / str(language) / _repo_dirname(repo)
    return candidate if candidate.is_dir() else None


def _history_instance_dirs(repo_dir: Path) -> List[Path]:
    """List all instance subdirectories under a repo history directory (sorted by name for
    reproducibility)."""
    if not repo_dir.is_dir():
        return []
    return sorted((p for p in repo_dir.iterdir() if p.is_dir()), key=lambda p: p.name)


# ========================================================================
# Safety red line: gold leak validation (section 6)
# ========================================================================
# Lone UTF-16 surrogate characters (U+D800-U+DFFF) break json.dump and terminal encoding,
# so they must be cleaned
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _safe_text(value: Any) -> str:
    """Turn any value into a string that is safe to JSON serialize and write to disk, dropping
    lone surrogate characters."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return _SURROGATE_RE.sub("", value)


def _flatten_to_strings(value: Any) -> List[str]:
    """Recursively collect every string leaf inside a (possibly nested) value, used to build
    the gold leak fingerprint.

    * ``str`` -> collect directly
    * ``int`` / ``float`` / ``bool`` / ``None`` -> **do not collect** (numbers and booleans
      hit by coincidence far too often and cause false positives)
    * ``dict`` -> recurse into all its values
    * ``list`` / ``tuple`` -> recurse into all its elements
    * anything else (such as numpy.ndarray) -> prefer ``tolist()`` then recurse, to avoid
      importing numpy directly
    """
    out: List[str] = []
    if value is None or isinstance(value, bool):
        return out
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, (int, float)):
        return out
    elif isinstance(value, dict):
        for sub in value.values():
            out.extend(_flatten_to_strings(sub))
    elif isinstance(value, (list, tuple)):
        for sub in value:
            out.extend(_flatten_to_strings(sub))
    else:
        to_list = getattr(value, "tolist", None)
        if callable(to_list):
            out.extend(_flatten_to_strings(to_list()))
        else:
            out.append(str(value))
    return out


def _collect_gold_values(
    target: Dict[str, Any],
    fields: Tuple[str, ...] = _GOLD_FINGERPRINT_FIELDS,
) -> List[str]:
    """Collect the string fingerprints of the target's gold answers (length >=
    :data:`_GOLD_MIN_LEN`).

    By default it only covers :data:`_GOLD_FINGERPRINT_FIELDS` (answer content unique to the
    target). Array fields (such as ``FAIL_TO_PASS``) are expanded into individual test node
    ids. Values that are too short are filtered out to avoid false positives. Passing
    ``fields=_GOLD_FIELDS`` gives the full list (for docs/debugging only, not for matching
    against history).
    """
    values: List[str] = []
    for key in fields:
        values.extend(_flatten_to_strings(target.get(key)))
    return [v for v in values if v and len(v.strip()) >= _GOLD_MIN_LEN]


def _assert_text_has_no_gold(text: str, gold_values: List[str], *, context: str) -> None:
    """Assert that ``text`` contains no gold fingerprint; raise :class:`AssertionError` on a hit."""
    for gold in gold_values:
        if gold in text:
            snippet = gold[:60].replace("\n", "\\n")
            raise AssertionError(
                f"[GOLD LEAK] a target gold snippet (len={len(gold)}) appears in {context}: {snippet!r}..."
            )


def _assert_no_gold_leakage(
    target: Dict[str, Any],
    *,
    query: str,
    result_payload: Dict[str, Any],
) -> None:
    """The final validation called before ``run_one()`` returns (the section 6 safety red line).

    Two checks:

    1. **Query exactness**: the ``query`` passed to ``search()`` must equal
       ``target["problem_statement"]`` **character for character**, not one character more
       (this stops gold from being smuggled into the query).
    2. **Clean result JSON**: the full text written into the result JSON must never contain
       the **value** of any field in :data:`_GOLD_FINGERPRINT_FIELDS` (``patch`` /
       ``test_patch`` / ``pr_description`` / ``FAIL_TO_PASS``). ``PASS_TO_PASS`` /
       ``install_config`` / ``meta`` are not used as substring fingerprints (see the
       :data:`_GOLD_FINGERPRINT_FIELDS` comment for why); they are covered instead by the
       structural guarantee that the harness never reads or passes them through.

    Keep the distinction: the ``exit`` message / ``extra.submission`` of the **history
    trajectory itself** inside a ``HistorySession`` is the real output of a past agent. It
    is **not** the target's gold, it is legitimate memory material, and it is out of scope
    for this check.
    """
    problem_statement = target.get("problem_statement")
    if query != problem_statement:
        raise AssertionError(
            "[GOLD LEAK] the search query does not match target['problem_statement']. "
            "The query can only be problem_statement, not one character more."
        )

    gold_values = _collect_gold_values(target)
    result_text = json.dumps(result_payload, ensure_ascii=False, default=str)
    _assert_text_has_no_gold(result_text, gold_values, context="result_json")


# ========================================================================
# Load targets
# ========================================================================
def load_targets(
    *,
    filter_lang: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read ``test.parquet``, filter by language, sort by instance_id, and truncate to limit.

    ``pyarrow`` is a **lazy import inside this function** (reading parquet needs it as the
    engine), which honors the constraint that pyarrow is only imported inside load_targets().
    Each returned row is a plain dict. Columns such as ``FAIL_TO_PASS`` may be numpy arrays,
    and this function does not force a conversion (the gold columns are only used for leak
    validation, and :func:`_flatten_to_strings` handles numpy arrays).
    """
    import pyarrow  # noqa: F401  -- lazy import, declared explicitly as the engine dependency of pd.read_parquet

    if not _TEST_TARGETS_PATH.exists():
        raise FileNotFoundError(f"targets file not found: {_TEST_TARGETS_PATH}")

    df = pd.read_parquet(_TEST_TARGETS_PATH)
    records: List[Dict[str, Any]] = df.to_dict(orient="records")

    if filter_lang:
        records = [r for r in records if r.get("language") == filter_lang]

    # Sort by instance_id so the processing order is identical across runs and machines
    records.sort(key=lambda r: str(r.get("instance_id", "")))

    if limit is not None and limit > 0:
        records = records[:limit]

    return records


# ========================================================================
# Memory system registry
# ========================================================================
# Implementations live in memory_systems/ as one file each, discovered by file
# name, so plugging in a system never means editing this harness. The worked
# example is memory_systems/demo.py.
from memory_systems import available_methods, load_method  # noqa: E402


# ========================================================================
# Core: run one full evaluation for a single target (Add history -> Search)
# ========================================================================
def run_one(
    target: Dict[str, Any],
    *,
    method: str,
    memory_factory: Callable[[], MemorySystem],
    top_k: int,
    out_root: Path,
    run_id: str,
    max_session_mb: float,
    overwrite: bool,
) -> Optional[Dict[str, Any]]:
    """Build memory and retrieve for one target, write the result JSON, and return the result
    dict (return ``None`` when skipped).

    Flow:

    1. Compute the output path; if it already exists and ``--overwrite`` is off, skip
       (resume semantics).
    2. Build a :class:`MemoryScope` (``user_id`` encodes run_id / repo / target_instance_id).
    3. Temporal causal filter: under the same ``repo_dirname``, only feed history whose
       trailing number is **<** the target's.
    4. Large file defense: skip a session larger than ``--max-session-mb``, log a WARNING,
       and record it in ``skipped_sessions`` (otherwise that ~102 MB trajectory would blow up
       process memory).
    5. For each history: load -> strip -> build a :class:`HistorySession` -> gold check -> ``add()``.
    6. ``search(query=problem_statement)`` -> check descending order and length -> serialize.
    7. ``_assert_no_gold_leakage()`` final check -> write to disk.
    """
    instance_id = str(target["instance_id"])
    language = str(target["language"])
    repo = str(target["repo"])
    repo_dirname = _repo_dirname(repo)

    # Output directory: {out_root}/{method}/{language}/{instance_id}/
    #   memory.txt   the experience text injected into the agent, run_agent.py reads it directly
    #   memory.json  full provenance (user_id / which histories were fed / scores / skipped large files)
    out_dir = out_root / method / language / instance_id
    out_path = out_dir / "memory.txt"
    meta_path = out_dir / "memory.json"

    # ---- 1) resume: skip if the result already exists, --overwrite forces a rerun ----
    if out_path.exists() and not overwrite:
        _LOG.info("skip (result exists, use --overwrite to force a rerun): %s", out_path)
        return None

    # ---- 2) build the MemoryScope ----
    user_id = f"vmb:{run_id}:{repo_dirname}:{instance_id}"
    scope = MemoryScope(
        user_id=user_id,
        repo=repo,
        repo_dirname=repo_dirname,
        language=language,
        target_instance_id=instance_id,
        base_commit=str(target.get("base_commit", "")),
        created_at=str(target.get("created_at", "")),
    )

    # ---- 3) the only source of the retrieval query: problem_statement, not one char more ----
    query = target["problem_statement"]

    # ---- create a fresh MemorySystem instance for each target ----
    memory = memory_factory()

    gold_values = _collect_gold_values(target)  # leak fingerprint, checked against every fed history
    target_num = _trailing_number(instance_id)
    hist_dir = _history_dir_for_target(target)

    skipped_sessions: List[Dict[str, Any]] = []
    num_histories_fed = 0
    num_messages_fed = 0

    started = time.perf_counter()

    if hist_dir is None:
        _LOG.warning("target %s has no history directory (%s), feeding no history this run", instance_id, repo_dirname)
    else:
        for session_dir in _history_instance_dirs(hist_dir):
            session_id = session_dir.name

            # ---- temporal causal filter: only feed history whose trailing number < the target's ----
            hist_num = _trailing_number(session_id)
            if target_num is None or hist_num is None or hist_num >= target_num:
                # Anything numbered >= target (including the target's own trajectory) is never
                # fed, which rules out "solving a past issue with future experience"
                continue

            traj_path = session_dir / f"{session_id}.traj.json"
            if not traj_path.exists():
                _LOG.warning("history trajectory file missing, skip: %s", traj_path)
                continue

            # ---- 4) large file defense: measure the size first, block before json.load if over the cap ----
            size_mb = traj_path.stat().st_size / 1_000_000  # decimal MB
            if size_mb > max_session_mb:
                _LOG.warning(
                    "history session exceeds --max-session-mb=%.1f, skip: %s (%.2f MB)",
                    max_session_mb,
                    session_id,
                    size_mb,
                )
                skipped_sessions.append(
                    {
                        "session_id": session_id,
                        "size_mb": round(size_mb, 2),
                        "reason": "exceeds --max-session-mb",
                    }
                )
                continue

            # ---- 5) load -> strip -> build the HistorySession ----
            try:
                messages, trajectory_format, info = _load_trajectory(traj_path)
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                _LOG.warning("failed to parse history trajectory, skip: %s (%s)", traj_path, exc)
                continue

            stripped = [_strip_message(m) for m in messages]

            # Gold red line: a history session must never smuggle in the target's gold values
            session_text = json.dumps(stripped, ensure_ascii=False, default=str)
            _assert_text_has_no_gold(session_text, gold_values, context=f"history session {session_id}")

            session_meta = _build_session_meta(info, len(stripped))
            # The time keys are passed in explicitly: the trailing number is this benchmark's
            # timeline (a smaller number means earlier). The harness already hard filtered out
            # hist_num >= target_num above, so here it just hands the numbers to the memory
            # system and saves it from parsing the session_id string. Real PR dates are not in
            # the released data, so they are not provided.
            session_meta["issue_number"] = hist_num
            session_meta["target_issue_number"] = target_num
            session = HistorySession(
                session_id=session_id,
                messages=stripped,
                trajectory_format=trajectory_format,
                meta=session_meta,
            )

            memory.add(session, scope=scope)  # synchronous write, retrievable by search once it returns
            num_histories_fed += 1
            num_messages_fed += len(stripped)
            # the session object can be reclaimed after this scope ends, so no large object is held for long

    # ---- 6) retrieve ----
    retrieved = memory.search(query, scope=scope, top_k=top_k)
    elapsed_seconds = time.perf_counter() - started

    # Contract checks (no hard failure, only warn and truncate defensively, to tolerate a rough
    # external implementation)
    if len(retrieved) > top_k:
        _LOG.warning(
            "search() returned %d items > top_k=%d, truncated (%s)", len(retrieved), top_k, instance_id
        )
        retrieved = retrieved[:top_k]
    scores = [float(getattr(r, "score", 0.0)) for r in retrieved]
    if scores != sorted(scores, reverse=True):
        _LOG.warning("search() did not return descending by score -- the contract requires the integrator to sort (%s)", instance_id)

    retrieved_payload: List[Dict[str, Any]] = []
    for item in retrieved:
        if not isinstance(item, RetrievedMemory):
            _LOG.warning("search() returned a non RetrievedMemory object, skipped: %r", type(item).__name__)
            continue
        payload = item.to_payload()
        payload["content"] = _safe_text(payload.get("content", ""))  # clean surrogate characters
        payload["id"] = _safe_text(payload.get("id", ""))
        retrieved_payload.append(payload)

    total_memories_stored = memory.count()

    # ---- 7) assemble the result JSON (key names, see the output format section of CONTRACT.md) ----
    result_payload: Dict[str, Any] = {
        "method": method,
        "run_id": run_id,
        "instance_id": instance_id,
        "repo": repo,
        "language": language,
        "user_id": user_id,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "num_histories_fed": num_histories_fed,
        "num_messages_fed": num_messages_fed,
        "skipped_sessions": skipped_sessions,
        "total_memories_stored": total_memories_stored,
        "num_retrieved": len(retrieved_payload),
        "retrieved": retrieved_payload,
    }

    # ---- final safety red line check (before writing to disk) ----
    _assert_no_gold_leakage(target, query=query, result_payload=result_payload)

    # memory.txt is the injected text: join the retrieved memories into readable text in the
    # already sorted descending order.
    memory_text = "\n\n".join(
        f"[{idx}] (from history issue {item.get('source_session_id') or 'unknown'})\n"
        f"{str(item.get('content') or '').strip()}"
        for idx, item in enumerate(retrieved_payload, start=1)
        if str(item.get("content") or "").strip()
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(memory_text + ("\n" if memory_text else ""))
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(result_payload, fh, ensure_ascii=False, indent=2, default=str)

    _LOG.info(
        "done %s: fed %d histories / %d messages, skipped %d, retrieved %d -> %s",
        instance_id,
        num_histories_fed,
        num_messages_fed,
        len(skipped_sessions),
        len(retrieved_payload),
        out_path,
    )
    return result_payload


# ========================================================================
# CLI
# ========================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser (existing args plus the new --run-id / --max-session-mb)."""
    parser = argparse.ArgumentParser(
        prog="run_memory.py",
        description="VibeMemBench memory integration batch run: feed the earlier history trajectories of the same repo (Add), then retrieve with problem_statement (Search).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        default=_DEFAULT_METHOD,
        help=f"memory system name, resolved to memory_systems/<name>.py (available: {', '.join(available_methods())}); also decides the output directory {{out_root}}/{{method}}/",
    )
    parser.add_argument(
        "--out-root",
        default=str(_DEFAULT_OUT_ROOT),
        help="output root, the final path is {out_root}/{method}/{language}/{instance_id}/memory.txt",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=(
            "max number of memories search() returns, and the number written into "
            "memory.txt. The protocol injects one top-ranked experience, so raising "
            "this makes a run incomparable to the reported numbers"
        ),
    )
    parser.add_argument("--filter-lang", default=None, help="run only one language (such as python / js / go ...)")
    parser.add_argument("--limit", type=int, default=None, help="run at most this many targets (for debugging)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="force a rerun when the result file already exists (skip by default, supports resume)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="the run id for this run, encoded into the user_id namespace; defaults to an auto generated YYYYMMDD-HHMMSS",
    )
    parser.add_argument(
        "--max-session-mb",
        type=float,
        default=_DEFAULT_MAX_SESSION_MB,
        help="size cap of one history trajectory (decimal MB); over the cap it is skipped and recorded in skipped_sessions, to keep a large file from blowing up memory",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print DEBUG level logs")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: parse args -> load targets -> run_one for each -> summarize. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        memory_factory = load_method(args.method)
    except ValueError as exc:
        _LOG.error("%s", exc)
        return 2

    run_id = args.run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out_root).expanduser().resolve()

    try:
        targets = load_targets(filter_lang=args.filter_lang, limit=args.limit)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        _LOG.error("failed to load targets: %s", exc)
        return 1

    _LOG.info(
        "run_id=%s method=%s out_root=%s top_k=%d max_session_mb=%.1f targets=%d",
        run_id,
        args.method,
        out_root,
        args.top_k,
        args.max_session_mb,
        len(targets),
    )

    done = 0
    skipped = 0
    failed = 0
    for target in targets:
        try:
            result = run_one(
                target,
                method=args.method,
                memory_factory=memory_factory,
                top_k=args.top_k,
                out_root=out_root,
                run_id=run_id,
                max_session_mb=args.max_session_mb,
                overwrite=args.overwrite,
            )
            if result is None:
                skipped += 1
            else:
                done += 1
        except AssertionError as exc:
            # A safety red line assertion failure (such as a gold leak): log it and abort this
            # target, but keep running the rest
            failed += 1
            _LOG.error("target %s failed validation: %s", target.get("instance_id"), exc)
        except Exception as exc:  # noqa: BLE001 -- one target crashing must not drag down the whole batch
            failed += 1
            _LOG.exception("target %s raised an exception: %s", target.get("instance_id"), exc)

    _LOG.info("all done: %d succeeded, %d skipped, %d failed (%d total)", done, skipped, failed, len(targets))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

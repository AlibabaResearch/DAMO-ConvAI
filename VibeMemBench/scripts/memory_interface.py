"""VibeMemBench memory system interface. The one file an integrator must implement.

How to integrate: subclass :class:`MemorySystem`, implement :meth:`MemorySystem.add`
and :meth:`MemorySystem.search`, and save it as one file at
``memory_systems/<your name>.py`` that exposes the class as ``MEMORY_SYSTEM``. The
harness discovers a method by file name, so ``--method <your name>`` then works with
no change anywhere else. Copy ``memory_systems/demo.py`` as a starting point. The
full contract lives in ``CONTRACT.md`` next to this file.

This file depends only on the standard library. It imports no third party package and
can be imported on its own.

The field names of the three data contract classes match the AML payload keys exactly:

* :class:`MemoryScope` is the isolation boundary of one evaluation. It maps to the AML
  ``user_id`` semantics.
* :class:`HistorySession` is one history trajectory. It maps to one AML Add.
* :class:`RetrievedMemory` is one search result. It maps to an item in the ``data[]`` of
  an AML Search response.

Two hard rules on the harness side. You do not implement them, but you must understand
them:

1. **Temporal causality.** The history fed to a target only contains issues from the same
   repository that happened **before** it. The time key is the trailing number of
   ``instance_id`` (the PR/issue number, a smaller number means earlier). The harness uses
   it as a hard filter, so the target's own trajectory never enters memory. Each history
   reports its number in ``HistorySession.meta["issue_number"]``, and the cut point is
   ``meta["target_issue_number"]``. They always satisfy ``issue_number < target_issue_number``.
   Real PR dates are not in the released data, so they are not provided.
2. **Gold red line.** The values of the target's ``patch`` / ``test_patch`` /
   ``pr_description`` / ``FAIL_TO_PASS`` never appear in the fed data, and they must not
   appear in what you return.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Self


# ========================================================================
# Data contract classes (field names match the AML payload keys exactly)
# ========================================================================
@dataclass(frozen=True)
class MemoryScope:
    """The isolation boundary of one evaluation. Maps to the AML ``user_id`` semantics.

    **The boundary is not a "user" but a "repository x temporal cut point".** This
    benchmark builds a separate memory for each target, and it may only feed history from
    the same repository that happened **before** that target. So ``user_id`` encodes the
    ``run_id`` (this run), the ``repo_dirname`` (repository), and the
    ``target_instance_id`` (temporal cut point). This keeps the memory namespace of any
    two targets from overlapping. The harness also creates a fresh :class:`MemorySystem`
    instance for each target, which gives a double guarantee: in-process instance
    isolation plus ``user_id`` namespace isolation.

    ``search()`` may only look inside the memory of ``scope.user_id``. It must **never
    return records across user_id**.
    """

    user_id: str  # f"vmb:{run_id}:{repo_dirname}:{target_instance_id}"
    repo: str  # "adamchainz/flake8-comprehensions"
    repo_dirname: str  # "adamchainz__flake8-comprehensions"
    language: str  # "python"
    target_instance_id: str  # "adamchainz__flake8-comprehensions-553"
    base_commit: str
    created_at: str

    def to_payload(self) -> Dict[str, Any]:
        """Serialize to an AML style wire dict (keys match the field names)."""
        return {
            "user_id": self.user_id,
            "repo": self.repo,
            "repo_dirname": self.repo_dirname,
            "language": self.language,
            "target_instance_id": self.target_instance_id,
            "base_commit": self.base_commit,
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, d: Dict[str, Any]) -> Self:
        """Deserialize from a wire dict (a future HTTP transport restores args with it)."""
        return cls(
            user_id=d["user_id"],
            repo=d["repo"],
            repo_dirname=d["repo_dirname"],
            language=d["language"],
            target_instance_id=d["target_instance_id"],
            base_commit=d["base_commit"],
            created_at=d["created_at"],
        )


@dataclass(frozen=True)
class HistorySession:
    """The full solve trajectory of one history issue. Maps to one AML ``Add``.

    The whole trajectory is handed to :meth:`MemorySystem.add` in one call (p50 = 113
    messages). The harness does **no chunking**: splitting a coding trajectory at an
    arbitrary sentence boundary would tear an ``assistant(tool_calls)`` message from its
    matching ``tool`` result, and both halves would become useless. How to chunk, if at
    all, is up to the integrator.
    """

    session_id: str  # history instance_id, e.g. "adamchainz__flake8-comprehensions-143"
    messages: List[Dict[str, Any]]  # stripped raw message dicts, original order kept
    trajectory_format: Optional[str]  # "mini-swe-agent-1.1", or None (legacy variant)
    # meta always holds num_messages / issue_number / target_issue_number.
    # Keys such as exit_status appear only when the trajectory carries info (the legacy
    # variant omits them).
    # issue_number is the temporal position of this history. Use it to sort or to weight
    # recency.
    meta: Dict[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "trajectory_format": self.trajectory_format,
            "meta": self.meta,
        }

    @classmethod
    def from_payload(cls, d: Dict[str, Any]) -> Self:
        return cls(
            session_id=d["session_id"],
            messages=list(d.get("messages") or []),
            trajectory_format=d.get("trajectory_format"),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class RetrievedMemory:
    """One search result. Maps to an item in the ``data[]`` of an AML Search response.

    Field names match AML: ``id`` / ``content`` / ``score`` / ``created_at``.
    ``source_session_id`` is an **extension field** declared by this benchmark for
    attribution analysis (which history issue this memory came from).

    ``content`` is the experience text that finally gets injected into the agent, so make
    it self contained and readable.
    """

    id: str  # required, non empty
    content: str  # required, non empty
    score: float  # higher means more relevant; the caller **must sort descending** first
    created_at: Optional[str] = None  # ISO8601, optional
    source_session_id: Optional[str] = None  # which history issue it came from (extension)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "created_at": self.created_at,
            "source_session_id": self.source_session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, d: Dict[str, Any]) -> Self:
        return cls(
            id=d["id"],
            content=d["content"],
            score=float(d["score"]),
            created_at=d.get("created_at"),
            source_session_id=d.get("source_session_id"),
            metadata=dict(d.get("metadata") or {}),
        )


class MemorySystem(abc.ABC):
    """The abstract interface an external memory system implements. **Only Add and Search
    are required.**

    Answer (how the results are injected into the agent) and Eval (how they are scored)
    are fixed duties of the platform and are not part of this interface. An integrator
    only implements :meth:`add` and :meth:`search` (:meth:`count` is optional).

    The harness creates a fresh instance for each target, and the constructor takes no
    arguments. If you need configuration, read environment variables or hardcode defaults
    in your own subclass.
    """

    @abc.abstractmethod
    def add(self, session: HistorySession, *, scope: MemoryScope) -> None:
        """Write one history trajectory synchronously. **Once it returns, the trajectory is
        already retrievable by search.**

        No async or deferred semantics are allowed (AML is explicit: no task id, no
        polling). You may only write inside the ``scope.user_id`` namespace.
        """

    @abc.abstractmethod
    def search(self, query: str, *, scope: MemoryScope, top_k: int) -> List[RetrievedMemory]:
        """Return results in **descending** relevance, length ``<= top_k``. Return an empty
        list ``[]`` when there is nothing.

        ``query`` is only ever the target's ``problem_statement`` (not one character more).
        You may only search inside the memory of ``scope.user_id``. **Never return records
        across user_id.** The caller must sort the results itself, the harness will not
        reorder them for you.
        """

    def count(self) -> Optional[int]:
        """Optional: the current total number of memories. The default ``None`` means no
        counting."""
        return None


__all__ = [
    "MemoryScope",
    "HistorySession",
    "RetrievedMemory",
    "MemorySystem",
]

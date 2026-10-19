#!/usr/bin/env python3
"""Docker image lifecycle for the run_* scripts.

Two stages need the same thing. run_agent.py runs the agent in a container per
seed, and run_eval.py scores a patch in a container per seed. Both want to pull
an image once, run every task bound to it, then optionally drop it so local disk
stays bounded.

Why bounding disk matters here: every target carries its own image, tagged with
its PR number and commit, so the 111 targets need 111 distinct images and none
of them is ever reused. Without eviction the run accumulates all of them.

Eviction is free *within* one stage, because no target needs another target's
image. It is not free *across* stages: run_eval.py scores the same targets
later and needs the same images again, so a capped run pays for two passes of
pulls. Set ``pool_size=0`` when disk is ample and you plan to score right after
running, which keeps every image and skips the second pass.

The schedule is a pull-ahead pipeline with three independent knobs:

    pull_workers    how many `docker pull` run at once
    image_workers   how many distinct images have running tasks at once
    seed_workers    how many tasks of one image run at once

Concurrent containers therefore peak at ``image_workers * seed_workers``. The
seeds of one instance share an image and run in separate containers, so they do
not interfere beyond competing for CPU, memory, and disk throughput.

``pool_size`` caps how many images may sit on disk at any moment. A slot is
taken before a pull starts and released after the image is removed, so pulling
runs ahead of execution without letting disk grow without bound. Set it to 0 to
disable eviction entirely, which keeps every image and needs no slot accounting.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger("docker_pool")

# Defaults. Pulling the largest images (java or kotlin with a full SDK) can take
# far longer than docker's own patience, hence the wide timeout.
DEFAULT_PULL_WORKERS = 3
DEFAULT_IMAGE_WORKERS = 1
DEFAULT_SEED_WORKERS = 4
DEFAULT_POOL_SIZE = 10
DEFAULT_PULL_TIMEOUT = 1800
_RMI_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Image group
# ---------------------------------------------------------------------------


@dataclass
class ImageGroup:
    """One image and every task that needs it.

    ``image`` may be None, which means the tasks need no image and run right
    away without taking a disk slot. ``tasks`` are opaque payloads handed back
    to the caller's task function untouched.
    """

    image: str | None
    tasks: list[Any] = field(default_factory=list)


def group_by_image(
    tasks: Iterable[Any], image_of: Callable[[Any], str | None]
) -> list[ImageGroup]:
    """Bundle tasks by the image they need, preserving first-seen order."""
    order: list[str] = []
    bound: dict[str, list[Any]] = {}
    unbound: list[Any] = []
    for task in tasks:
        image = image_of(task)
        if not image:
            unbound.append(task)
            continue
        if image not in bound:
            bound[image] = []
            order.append(image)
        bound[image].append(task)
    groups = [ImageGroup(image=img, tasks=bound[img]) for img in order]
    if unbound:
        groups.append(ImageGroup(image=None, tasks=unbound))
    return groups


# ---------------------------------------------------------------------------
# Docker primitives
# ---------------------------------------------------------------------------

_image_locks: dict[str, threading.Lock] = {}
_image_locks_guard = threading.Lock()


def _image_lock(image: str) -> threading.Lock:
    """One lock per image, so two threads never pull or remove the same one."""
    with _image_locks_guard:
        return _image_locks.setdefault(image, threading.Lock())


def image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, check=False
    )
    return result.returncode == 0


def pull_image(image: str, *, timeout: int = DEFAULT_PULL_TIMEOUT) -> bool:
    """Pull one image unless it is already cached. True when it is available."""
    with _image_lock(image):
        if image_exists(image):
            LOG.info("[pull] cached: %s", image)
            return True
        LOG.info("[pull] fetching: %s", image)
        try:
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            LOG.error("[pull] timed out after %ds: %s", timeout, image)
            return False
        if result.returncode != 0:
            LOG.error("[pull] failed: %s: %s", image, (result.stderr or "")[-300:])
            return False
        LOG.info("[pull] done: %s", image)
        return True


def remove_image(image: str) -> None:
    """Drop one image to free its disk slot. Failure is logged, not fatal."""
    with _image_lock(image):
        try:
            result = subprocess.run(
                ["docker", "rmi", "-f", image],
                capture_output=True,
                text=True,
                check=False,
                timeout=_RMI_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            LOG.warning("[evict] timed out removing %s", image)
            return
        if result.returncode == 0:
            LOG.info("[evict] removed: %s", image)
        else:
            LOG.warning("[evict] failed: %s: %s", image, (result.stderr or "")[-200:])


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _settled(future: "Future[bool]") -> bool:
    """Read a pull future without letting an exception escape the callback."""
    try:
        return bool(future.result())
    except Exception as exc:  # a crashed pull is just an unavailable image
        LOG.error("[pull] crashed: %s", exc)
        return False


def _guarded(task_fn: Callable[[Any], dict], task: Any) -> dict:
    """Run one task so a raised exception cannot sink the whole batch."""
    try:
        return task_fn(task)
    except Exception as exc:
        LOG.error("[task] crashed: %r: %s", task, exc)
        return {"status": "error", "error": str(exc)}


def _default_failure(task: Any, image: str) -> dict:
    return {"status": "error", "error": f"failed to pull image {image}"}


def run_grouped(
    groups: list[ImageGroup],
    task_fn: Callable[[Any], dict],
    *,
    on_result: Callable[[dict, int, int], None] | None = None,
    failure_fn: Callable[[Any, str], dict] = _default_failure,
    pull_workers: int = DEFAULT_PULL_WORKERS,
    image_workers: int = DEFAULT_IMAGE_WORKERS,
    seed_workers: int = DEFAULT_SEED_WORKERS,
    pool_size: int = DEFAULT_POOL_SIZE,
    pull_timeout: int = DEFAULT_PULL_TIMEOUT,
) -> list[dict]:
    """Pull images ahead of execution, run each group's tasks, then evict.

    ``task_fn`` receives one opaque task and returns its result dict.
    ``failure_fn`` builds the result for a task whose image could not be pulled,
    so a pull failure is reported per task rather than silently dropped.
    ``on_result`` is called as ``(result, completed, total)`` for progress.

    Returns every result, in completion order.
    """
    total = sum(len(g.tasks) for g in groups)
    if not total:
        return []

    image_workers = max(1, image_workers)
    seed_workers = max(1, seed_workers)
    pull_workers = max(1, pull_workers)

    LOG.info(
        "[schedule] %d task(s) over %d image(s): pull=%d, images=%d, seeds=%d, "
        "pool=%s",
        total,
        sum(1 for g in groups if g.image),
        pull_workers,
        image_workers,
        seed_workers,
        pool_size if pool_size > 0 else "off",
    )

    results: list[dict] = []
    results_lock = threading.Lock()
    completed = 0

    # Caps images resident on disk. A slot is taken before the pull and released
    # after the eviction, so read-ahead cannot outgrow the cap.
    slots = threading.Semaphore(pool_size) if pool_size > 0 else None
    # Holds (group, image_available). One None per exec worker ends the stage.
    ready: queue.Queue = queue.Queue()

    def record(result: dict) -> None:
        nonlocal completed
        with results_lock:
            results.append(result)
            completed += 1
            seen = completed
        if on_result is not None:
            on_result(result, seen, total)

    def pull_stage() -> None:
        """Walk the groups in order, pulling under both caps."""
        pool = ThreadPoolExecutor(max_workers=pull_workers, thread_name_prefix="pull")
        try:
            for group in groups:
                if group.image is None:
                    ready.put((group, True))
                    continue
                if slots is not None:
                    slots.acquire()  # blocks once the disk cap is reached
                future = pool.submit(pull_image, group.image, timeout=pull_timeout)
                # Hand each image over the moment its pull settles, rather than
                # waiting for the whole batch, so execution overlaps pulling.
                future.add_done_callback(
                    lambda f, g=group: ready.put((g, _settled(f)))
                )
            pool.shutdown(wait=True)
        finally:
            for _ in range(image_workers):
                ready.put(None)

    def exec_stage() -> None:
        """Take pulled images and run their tasks, then free the slot."""
        while True:
            item = ready.get()
            if item is None:
                return
            group, available = item
            try:
                if not available:
                    LOG.error(
                        "[schedule] image unavailable, failing %d task(s): %s",
                        len(group.tasks),
                        group.image,
                    )
                    for task in group.tasks:
                        record(failure_fn(task, group.image or ""))
                    continue
                if seed_workers == 1 or len(group.tasks) == 1:
                    for task in group.tasks:
                        record(_guarded(task_fn, task))
                else:
                    with ThreadPoolExecutor(
                        max_workers=seed_workers, thread_name_prefix="seed"
                    ) as seed_pool:
                        futures = [
                            seed_pool.submit(_guarded, task_fn, task)
                            for task in group.tasks
                        ]
                        for future in as_completed(futures):
                            record(future.result())
            finally:
                if group.image is not None:
                    if pool_size > 0 and available:
                        remove_image(group.image)
                    if slots is not None:
                        slots.release()

    puller = threading.Thread(target=pull_stage, name="pull-stage", daemon=True)
    puller.start()

    workers = [
        threading.Thread(target=exec_stage, name=f"exec-{i}", daemon=True)
        for i in range(image_workers)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    puller.join()

    return results

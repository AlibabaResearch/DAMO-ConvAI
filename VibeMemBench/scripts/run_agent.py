#!/usr/bin/env python3
"""Parallel SWE-bench issue runner.

Reads instances from a parquet file, then runs mini-swe-agent in parallel
across (instance_id, seed) combinations. Docker images are pulled by
mini-swe-agent's own environment layer, not by this script.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MINI_SRC = REPO_ROOT / "mini-swe-agent" / "src"
DEFAULT_SWEBENCH_CONFIG = (
    MINI_SRC / "minisweagent" / "config" / "benchmarks" / "swebench.yaml"
)

# Defaults wired to the repository layout, so a plain
# `python scripts/run_agent.py --method base` runs the full target set.
DEFAULT_PARQUET = REPO_ROOT / "benchmark" / "test.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "agent_runs"
DEFAULT_MEMORY_ROOT = REPO_ROOT / "outputs" / "memory"

# Let docker_pool resolve whether this file runs as a script or is imported.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import docker_pool  # noqa: E402  (needs the path insert above)

# Auto-load <repo>/.env so MODEL/API_BASE/API_KEY can live in a local file
# instead of being exported each time. When python-dotenv is missing we warn
# rather than fail silently, otherwise a filled in .env looks like it works
# while every value is quietly ignored.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
except ImportError:
    if (REPO_ROOT / ".env").exists():
        print(
            f"[warn] {REPO_ROOT / '.env'} exists but python-dotenv is not "
            "installed, so it is ignored. Run: pip install python-dotenv",
            file=sys.stderr,
        )

LOG = logging.getLogger("run_agent")


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------


def _seed_dir(
    output: Path,
    instance_id: str,
    seed: int,
    *,
    lang: str,
    method: str,
) -> Path:
    """Resolve the on-disk directory for one (instance_id, seed) run.

    The layout is fixed:

        <output>/<language>/<instance_id>/<method>/seed<N>

    Putting the method below the instance keeps every condition for one issue
    side by side, which is what the paired comparison reads.
    """
    return output / lang / instance_id / method / f"seed{seed}"





# ---------------------------------------------------------------------------
# Memory methods
#
# Every run declares exactly one method. The agent, tools, sandbox image,
# budget, and seeds stay fixed across methods, so a difference in the outcome
# is attributable to the injected memory alone.
#
#   base        no experience block at all, the matched baseline
#   distractor  one fixed passage of everyday sentences with no software
#               content, injected through the same rule as a real memory. It
#               controls for "any injected text helps".
#   <anything>  a memory system name, e.g. mymem. Its per-instance memory is
#               read from <memory-root>/<method>/<lang>/<iid>/memory.txt,
#               which is what run_memory.py writes.
# ---------------------------------------------------------------------------

METHOD_BASE = "base"
METHOD_DISTRACTOR = "distractor"

_EXPERIENCE_BLOCK_TEMPLATE = (
    "<relevant_experiences>\n{experience}\n</relevant_experiences>\n\n"
)

# Fixed distractor passage. Deliberately mundane and free of any software,
# testing, or repository vocabulary, so it matches the injection shape of a
# real memory while carrying no transferable signal.
_DEFAULT_DISTRACTOR_TEXT = """The market on the corner opens a little after six in the morning. The owner
stacks the crates of pears by the door first, because that is where the light
falls until noon. Rain was expected all week, but the sky stayed pale and dry
and the awning never came down. An old cat sleeps on the third step and does
not move for anyone. In the evening the baker two doors along gives away
whatever is left, and by seven the shelves are bare."""

# Cache the base instance_template extracted from swebench.yaml so we don't
# re-parse the file for every (instance, seed) pair.
_BASE_INSTANCE_TEMPLATE_CACHE: str | None = None


def _load_memory_texts(memory_root: Path, method: str) -> dict[str, str]:
    """Index instance_id -> memory text for one method.

    Reads ``<memory_root>/<method>/<language>/<instance_id>/memory.txt``, the
    layout run_memory.py writes. The instance id comes from the directory name,
    so the index needs no dataset lookup.
    """
    method_dir = memory_root / method
    if not method_dir.is_dir():
        raise SystemExit(
            f"No memory directory for method {method!r}: {method_dir}\n"
            f"Run: python scripts/run_memory.py --method {method}"
        )

    files = sorted(method_dir.glob("*/*/memory.txt"))
    if not files:
        raise SystemExit(
            f"No memory.txt under {method_dir} (looked for */*/memory.txt). "
            f"Run: python scripts/run_memory.py --method {method}"
        )

    index: dict[str, str] = {}
    empty: list[str] = []
    for path in files:
        iid = path.parent.name
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"Cannot read memory file {path}: {exc}") from exc
        if not text:
            empty.append(iid)
            continue
        index[iid] = text

    LOG.info(
        "[memory] method=%s indexed %d instance(s) from %s",
        method,
        len(index),
        method_dir,
    )
    if empty:
        shown = ", ".join(sorted(empty)[:5]) + (" ..." if len(empty) > 5 else "")
        LOG.warning(
            "[memory] %d instance(s) have an empty memory.txt: %s", len(empty), shown
        )
    return index


def _experience_for(args: argparse.Namespace, instance_id: str) -> str:
    """Return the experience text to inject for one instance."""
    if args.method == METHOD_BASE:
        return ""
    if args.method == METHOD_DISTRACTOR:
        return args._distractor_text
    return args._memory_by_iid.get(instance_id, "")


def _get_base_instance_template() -> str:
    """Load and cache agent.instance_template from the bundled swebench.yaml."""
    global _BASE_INSTANCE_TEMPLATE_CACHE
    if _BASE_INSTANCE_TEMPLATE_CACHE is not None:
        return _BASE_INSTANCE_TEMPLATE_CACHE
    with DEFAULT_SWEBENCH_CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    template = ((cfg.get("agent") or {}).get("instance_template")) or ""
    if not template:
        raise SystemExit(
            f"Cannot find agent.instance_template in {DEFAULT_SWEBENCH_CONFIG}"
        )
    _BASE_INSTANCE_TEMPLATE_CACHE = template
    return template


def _write_experience_override_yaml(seed_dir: Path, experience: str) -> Path:
    """Generate a per-seed YAML that overrides agent.instance_template.

    The override file only contains the modified instance_template; mini-swe-agent's
    -c chain deep-merges it on top of the base swebench.yaml, so all other settings
    (system_template, cost_limit, environment, model, ...) are inherited intact.
    """
    base_template = _get_base_instance_template()
    block = _EXPERIENCE_BLOCK_TEMPLATE.format(experience=experience.strip())
    # Wrap experience block in {% raw %} so Jinja2 won't parse braces inside it.
    block = "{% raw %}" + block + "{% endraw %}"
    marker = "<pr_description>"
    if marker in base_template:
        injected = base_template.replace(marker, block + marker, 1)
    else:
        injected = block + base_template
    override = {"agent": {"instance_template": injected}}
    out_path = seed_dir / "_experience_override.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(override, f, allow_unicode=True, sort_keys=False)
    # Also persist the raw experience text for traceability/debugging.
    (seed_dir / "retrieved_experiences.txt").write_text(
        experience.strip() + "\n", encoding="utf-8"
    )
    return out_path


# Exit statuses that indicate the agent actually finished one full run
# (whether it solved the issue or hit cost/step limits). Any other value
# (e.g. exception class names like 'LLMError', 'KeyboardInterrupt') means
# the run was interrupted abnormally and should be re-executed.
_DONE_EXIT_STATUSES = frozenset({"Submitted", "LimitsExceeded"})


def _traj_exists(seed_path: Path, instance_id: str) -> bool:
    """Return True if a previous run for this seed is considered complete.

    A run counts as complete only when:
      1. ``<instance_id>.traj.json`` exists, AND
      2. its ``info.exit_status`` is one of ``_DONE_EXIT_STATUSES``.

    Trajectories with exit statuses outside that set (or whose JSON cannot
    be parsed) are treated as incomplete so ``--skip-existing`` will rerun
    them instead of silently skipping a half-finished run.
    """
    traj_path = seed_path / f"{instance_id}.traj.json"
    if not traj_path.exists():
        return False
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning(
            "[skip-existing] %s: failed to parse %s (%s); will rerun",
            instance_id,
            traj_path,
            exc,
        )
        return False
    info = data.get("info") if isinstance(data, dict) else None
    exit_status = (info or {}).get("exit_status")
    if exit_status in _DONE_EXIT_STATUSES:
        return True
    LOG.info(
        "[skip-existing] %s: exit_status=%r not in %s; will rerun",
        instance_id,
        exit_status,
        sorted(_DONE_EXIT_STATUSES),
    )
    return False


def run_instance(
    instance_id: str,
    seed: int,
    *,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    """Run mini-swe-agent for a single (instance_id, seed) pair."""
    lang = args._iid_to_lang.get(instance_id, "unknown")
    sd = _seed_dir(args.output, instance_id, seed, lang=lang, method=args.method)
    sd.mkdir(parents=True, exist_ok=True)
    traj_path = sd / f"{instance_id}.traj.json"

    start = dt.datetime.now()
    # The staging dir is prepared once in main() and contains a single
    # `<split>.parquet` file (symlink or copy of args.parquet). This avoids HF
    # datasets scanning a shared data directory and mis-classifying sibling
    # parquet/jsonl files into an unrelated split.
    subset_dir = args._subset_dir

    # mini-swe-agent v2 (Typer CLI): swebench-single takes one instance via
    # -i, writes one trajectory file via -o. Model/agent params go through
    # -c key=value, with the bundled swebench.yaml chained first (otherwise
    # the built-in defaults are dropped).
    cmd = [
        sys.executable,
        "-m",
        "minisweagent.run.utilities.mini_extra",
        "swebench-single",
        "--subset",
        str(subset_dir),
        "--split",
        args.split,
        "-i",
        instance_id,
        "-m",
        args.model,
        "-o",
        str(traj_path),
        "-y",  # yolo: no per-step confirmation
        "--exit-immediately",  # no confirmation when agent finishes
        "--agent-class",
        "default",  # non-interactive: avoids input() on LimitsExceeded under nohup
        "-c",
        str(DEFAULT_SWEBENCH_CONFIG),
        "-c",
        f"model.model_kwargs.api_base={args.api_base}",
        "-c",
        f"model.model_kwargs.api_key={args.api_key}",
        "-c",
        f"model.model_kwargs.temperature={args.temperature}",
        "-c",
        f"agent.cost_limit={args.cost_limit}",
        "-c",
        f"agent.step_limit={args.step_limit}",
        # Pulling is the environment layer's job. We only widen its budget,
        # because the 120s default times out on the largest images.
        "-c",
        f"environment.pull_timeout={args.pull_timeout}",
    ]

    # Prepend a <relevant_experiences> block to instance_template when the
    # declared method supplies one. We materialize a per-seed override yaml
    # and append it to the -c chain so mini-swe-agent's deep-merge keeps
    # everything else from swebench.yaml.
    experience_text = _experience_for(args, instance_id)
    if experience_text:
        override_yaml = _write_experience_override_yaml(sd, experience_text)
        cmd.extend(["-c", str(override_yaml)])

    log_file = sd / "run.log"
    if args.dry_run:
        # Everything above is real: the experience lookup, the injected prompt,
        # the override yaml, and the seed directory. Only the agent call is
        # skipped, so the wiring can be checked without Docker or an API key.
        log_file.write_text(f"CMD: {' '.join(cmd)}\n", encoding="utf-8")
        status, rc = "dry-run", 0
    else:
        try:
            with log_file.open("w", encoding="utf-8") as f:
                f.write(f"CMD: {' '.join(cmd)}\n\n")
                f.flush()
                proc = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                    env=env,
                    check=False,
                )
            status = "ok" if proc.returncode == 0 else "failed"
            rc = proc.returncode
        except Exception as exc:
            LOG.error("[error] %s seed%02d: %s", instance_id, seed, exc)
            status = "error"
            rc = -1

    end = dt.datetime.now()
    duration = (end - start).total_seconds()

    if status == "ok":
        LOG.info("[ok] %s seed%02d (%.1fs)", instance_id, seed, duration)
    else:
        LOG.error(
            "[%s] %s seed%02d rc=%d (%.1fs)", status, instance_id, seed, rc, duration
        )

    result = {
        "instance_id": instance_id,
        "seed": seed,
        "method": args.method,
        "status": status,
        "returncode": rc,
        "duration_s": round(duration, 1),
        "start_ts": start.timestamp(),
        "end_ts": end.timestamp(),
    }
    # Persist wall-clock metadata so downstream tooling (run_eval.py, stats)
    # doesn't have to reverse-engineer it from traj messages' extra.timestamp.
    try:
        (sd / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        LOG.warning(
            "[warn] %s seed%02d: failed to write result.json: %s",
            instance_id,
            seed,
            exc,
        )
    return result


# ---------------------------------------------------------------------------
# Scheduler
#
# Image lifecycle lives in docker_pool.py, shared with run_eval.py. Every target
# carries its own image and the 4 seeds of one target share it, so the natural
# unit of work is "one image, its seeds". docker_pool pulls ahead of execution
# and can cap how many images sit on disk, which matters because the 111 targets
# need 111 distinct multi-gigabyte images and none is ever reused.
# ---------------------------------------------------------------------------


def run_all(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    tasks: list[tuple[str, int]],
    iid_to_image: dict[str, str],
) -> list[dict[str, Any]]:
    """Group the (instance_id, seed) tasks by image and run them."""

    def image_of(task: tuple[str, int]) -> str | None:
        return iid_to_image.get(task[0])

    def run_one(task: tuple[str, int]) -> dict[str, Any]:
        iid, seed = task
        return run_instance(iid, seed, args=args, env=env)

    def on_pull_failure(task: tuple[str, int], image: str) -> dict[str, Any]:
        iid, seed = task
        return {
            "instance_id": iid,
            "seed": seed,
            "method": args.method,
            "status": "error",
            "returncode": -1,
            "duration_s": 0,
            "error": f"failed to pull image {image}",
        }

    def progress(result: dict[str, Any], done: int, total: int) -> None:
        LOG.info(
            "[%d/%d] %s seed%s -> %s",
            done,
            total,
            result.get("instance_id", "?"),
            result.get("seed", "?"),
            result.get("status", "?"),
        )

    groups = docker_pool.group_by_image(tasks, image_of)
    if args.dry_run:
        # No image is needed, so nothing is pulled or evicted. Everything else,
        # including the grouping and the seed concurrency, still runs.
        groups = docker_pool.group_by_image(tasks, lambda _t: None)
    return docker_pool.run_grouped(
        groups,
        run_one,
        on_result=progress,
        failure_fn=on_pull_failure,
        pull_workers=args.pull_workers,
        image_workers=args.image_workers,
        seed_workers=args.seed_workers,
        pool_size=args.image_pool_size,
        pull_timeout=args.pull_timeout,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_summary(results: list[dict[str, Any]], output: Path, method: str) -> None:
    """Write one summary CSV per method.

    The file name carries the method, because the three methods of a target are
    meant to be compared and a shared summary.csv would have each run overwrite
    the previous one.
    """
    csv_path = output / f"summary_{method}.csv"
    if not results:
        LOG.info("No results to summarize.")
        return
    fields = [
        "instance_id",
        "seed",
        "method",
        "status",
        "returncode",
        "duration_s",
        "start_ts",
        "end_ts",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    dry = sum(1 for r in results if r["status"] == "dry-run")
    failed = len(results) - ok - dry
    LOG.info(
        "[summary] total=%d ok=%d dry-run=%d failed=%d -> %s",
        len(results),
        ok,
        dry,
        failed,
        csv_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel SWE-bench issue runner using mini-swe-agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help=f"Path to the dataset parquet file (default: {DEFAULT_PARQUET}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "--pull-workers",
        type=int,
        default=docker_pool.DEFAULT_PULL_WORKERS,
        help=(
            "How many docker pulls run at once, ahead of execution "
            f"(default: {docker_pool.DEFAULT_PULL_WORKERS})."
        ),
    )
    p.add_argument(
        "--image-workers",
        type=int,
        default=docker_pool.DEFAULT_IMAGE_WORKERS,
        help=(
            "How many distinct images have running tasks at once. Concurrent "
            "containers peak at image-workers * seed-workers "
            f"(default: {docker_pool.DEFAULT_IMAGE_WORKERS})."
        ),
    )
    p.add_argument(
        "--seed-workers",
        type=int,
        default=docker_pool.DEFAULT_SEED_WORKERS,
        help=(
            "How many seeds of one instance run at once. They share the image "
            "but get separate containers, so they do not interfere beyond "
            f"competing for CPU, memory, and disk (default: "
            f"{docker_pool.DEFAULT_SEED_WORKERS})."
        ),
    )
    p.add_argument(
        "--image-pool-size",
        type=int,
        default=docker_pool.DEFAULT_POOL_SIZE,
        help=(
            "Cap on images resident on disk. Once every seed of an image "
            "finishes, the image is removed and its slot frees the next pull. "
            "No target reuses another's image, so eviction costs nothing "
            "within this stage, but run_eval.py needs the same images later, "
            "so a capped run pulls them twice overall. Set 0 to keep every "
            f"image (default: {docker_pool.DEFAULT_POOL_SIZE})."
        ),
    )
    p.add_argument(
        "--pull-timeout",
        type=int,
        default=1800,
        help=(
            "Seconds mini-swe-agent may spend pulling one docker image, passed "
            "through as environment.pull_timeout. Its own default of 120 times "
            "out on the largest swerebenchv2 images (default: 1800)."
        ),
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=4,
        help="Number of seeds per instance (default: 4).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL", "deepseek-v4-flash"),
        help="Model name (default: env MODEL or deepseek-v4-flash).",
    )
    p.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("API_BASE", ""),
        help="API base URL (default: env API_BASE).",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("API_KEY", ""),
        help="API key (default: env API_KEY).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (default: 0.5).",
    )
    p.add_argument(
        "--cost-limit",
        type=float,
        default=3.0,
        help="Cost limit in USD (default: 3.0).",
    )
    p.add_argument(
        "--step-limit", type=int, default=250, help="Agent max steps (default: 250)."
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="Parquet split to read (default: test).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip seeds whose .traj.json reports a finished run "
            "(info.exit_status in {Submitted, LimitsExceeded}). "
            "Trajectories with other exit statuses (e.g. exceptions) or "
            "unparseable files are treated as incomplete and rerun."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build everything but never call the agent and never touch Docker. "
            "Each seed directory still gets run.log with the exact command, and "
            "a memory-on run still gets its override yaml and "
            "retrieved_experiences.txt, so the whole wiring can be inspected "
            "without Docker or an API key."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N instances (0 = no limit). Useful for debugging.",
    )
    p.add_argument(
        "--method",
        type=str,
        default=METHOD_BASE,
        help=(
            "Declared memory method. 'base' injects nothing and is the matched "
            "baseline. 'distractor' injects one fixed passage of everyday "
            "sentences with no software content. Any other value names a memory "
            "system, e.g. mymem, whose per-instance memory is read from "
            "<memory-root>/<method>/<lang>/<iid>/memory.txt "
            f"(default: {METHOD_BASE})."
        ),
    )
    p.add_argument(
        "--memory-root",
        type=Path,
        default=DEFAULT_MEMORY_ROOT,
        help=(
            "Root of the memory directories written by run_memory.py "
            f"(default: {DEFAULT_MEMORY_ROOT})."
        ),
    )
    p.add_argument(
        "--distractor-file",
        type=Path,
        default=None,
        help=(
            "Replace the built-in distractor passage with the contents of this "
            "UTF-8 file. Only read when --method=distractor."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate inputs
    if not args.parquet.exists():
        LOG.error("Parquet file not found: %s", args.parquet)
        return 1

    # Defensive: litellm requires a provider prefix in the model name
    # (e.g. 'openai/deepseek-v4-flash'). If the user passed a bare name,
    # auto-prefix with 'openai/' since DashScope etc. use the OpenAI-
    # compatible protocol, which is the most common case.
    if "/" not in args.model:
        LOG.warning(
            "[model] %r has no litellm provider prefix; auto-prefixing as %r. "
            "Set MODEL=openai/<name> in .env (or pass --model openai/<name>) to silence this.",
            args.model,
            f"openai/{args.model}",
        )
        args.model = f"openai/{args.model}"

    args.output.mkdir(parents=True, exist_ok=True)

    # Build a clean staging dir so HF datasets sees exactly one parquet under
    # the user-supplied split name. Without this, passing a shared directory
    # like data/relevant_repo lets HF guess the split from sibling filenames
    # (e.g. docker_patch_validation.jsonl -> 'validation'), which causes
    # `Unknown split "train"` errors in swebench-single.
    #
    # IMPORTANT: the staging directory name embeds the parquet basename so
    # parallel runners that share the same --output (e.g. a target runner
    # and a history runner writing into outputs/relevant_repo2/baseline/)
    # don't clobber each other's `_hf_subset/train.parquet`.
    parquet_stem = args.parquet.stem  # e.g. 'target', 'history_only'
    staging = (args.output / f"_hf_subset__{parquet_stem}").resolve()
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    staged_parquet = staging / f"{args.split}.parquet"
    # NOTE: HuggingFace `datasets` (>=3.x) silently skips symlinked parquet
    # files when scanning a directory via `load_dataset(<dir>)`, raising
    # `EmptyDatasetError`. Copy the file instead of symlinking so it is always
    # picked up regardless of the datasets / fsspec version.
    shutil.copy2(args.parquet, staged_parquet)
    args._subset_dir = staging
    LOG.info(
        "[subset] staged %s -> %s (split=%s)",
        args.parquet,
        staged_parquet,
        args.split,
    )

    # Resolve the declared memory method once, so every run_instance call
    # reuses the same memory map without re-reading files.
    args._distractor_text = ""
    args._memory_by_iid = {}
    if args.method == METHOD_BASE:
        LOG.info("[method] base: no experience block is injected")
    elif args.method == METHOD_DISTRACTOR:
        if args.distractor_file:
            if not args.distractor_file.exists():
                LOG.error("--distractor-file not found: %s", args.distractor_file)
                return 1
            args._distractor_text = args.distractor_file.read_text(
                encoding="utf-8"
            ).strip()
        else:
            args._distractor_text = _DEFAULT_DISTRACTOR_TEXT.strip()
        LOG.info(
            "[method] distractor: one fixed passage of %d chars goes to every "
            "instance",
            len(args._distractor_text),
        )
    else:
        args._memory_by_iid = _load_memory_texts(
            args.memory_root.resolve(), args.method
        )

    # Load instances from parquet
    LOG.info("Loading instances from: %s", args.parquet)
    df = pd.read_parquet(args.parquet)
    if "instance_id" not in df.columns:
        LOG.error("Parquet file missing 'instance_id' column")
        return 1

    instances = df.drop_duplicates(subset=["instance_id"]).to_dict("records")
    LOG.info("Loaded %d unique instances", len(instances))

    # Optional: limit to first N instances (for debugging)
    if args.limit > 0 and args.limit < len(instances):
        instances = instances[: args.limit]
        LOG.info("[limit] truncated to first %d instances", len(instances))

    # Fail fast when a memory method has nothing for some instance. Falling back
    # to no injection would quietly turn those runs into base ones and corrupt
    # the paired comparison.
    if args.method not in (METHOD_BASE, METHOD_DISTRACTOR):
        missing = [
            inst["instance_id"]
            for inst in instances
            if not args._memory_by_iid.get(inst["instance_id"])
        ]
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            LOG.error(
                "[method] %s: no memory for %d of %d instance(s). Rerun "
                "run_memory.py --method %s so every target has a memory.txt, or "
                "narrow this run with --limit. Missing: %s",
                args.method,
                len(missing),
                len(instances),
                args.method,
                shown,
            )
            return 1

    # instance_id -> language drives the output layout. Stashed on args so
    # run_instance() need not re-read the parquet for every task.
    if "language" not in df.columns:
        LOG.error("Parquet file missing 'language' column: %s", args.parquet)
        return 1
    args._iid_to_lang = {
        inst["instance_id"]: (str(inst.get("language") or "").strip() or "unknown")
        for inst in instances
    }
    LOG.info(
        "[layout] %d instance(s) -> <output>/<lang>/<iid>/%s/seedN",
        len(args._iid_to_lang),
        args.method,
    )

    # Build task queue: (instance_id, seed) combinations
    tasks: list[tuple[str, int]] = []
    skipped = 0
    for inst in instances:
        iid = inst["instance_id"]
        lang = args._iid_to_lang[iid]
        for seed in range(args.seeds):
            sd = _seed_dir(args.output, iid, seed, lang=lang, method=args.method)
            if args.skip_existing and _traj_exists(sd, iid):
                skipped += 1
                continue
            tasks.append((iid, seed))

    LOG.info("Tasks scheduled: %d (skipped existing: %d)", len(tasks), skipped)
    if not tasks:
        LOG.info("Nothing to do \u2014 all tasks already complete.")
        return 0

    # instance_id -> image, so docker_pool can group the seeds of one target
    # under the single image they share.
    iid_to_image: dict[str, str] = {}
    if "image_name" in df.columns:
        iid_to_image = {
            str(rec["instance_id"]): str(rec["image_name"])
            for rec in df[["instance_id", "image_name"]].dropna().to_dict("records")
        }

    # Prepare environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(MINI_SRC) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    results = run_all(args=args, env=env, tasks=tasks, iid_to_image=iid_to_image)

    # Write summary
    for r in results:
        r.setdefault("method", args.method)
    results.sort(key=lambda r: (r["instance_id"], r["seed"]))
    write_summary(results, args.output, args.method)

    # A dry run never calls the agent, so its rows are not failures.
    failed_count = sum(1 for r in results if r["status"] not in ("ok", "dry-run"))
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

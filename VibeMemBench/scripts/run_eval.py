"""Evaluate completed SWE-rebench runs.

Recursively scans --runs-dir for seed directories containing *.traj.json,
invokes the vendored scripts/vendor/eval.py for each, and produces a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# The evaluator is vendored under scripts/vendor, so no external checkout is
# needed. The dataset carries install_config.test_cmd and log_parser per
# instance, which is everything eval.py needs to score a patch.
DEFAULT_EVAL_SCRIPT = SCRIPT_DIR / "vendor" / "eval.py"
DEFAULT_DATASET = PROJECT_ROOT / "benchmark" / "test.parquet"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs" / "agent_runs"

# Let docker_pool resolve whether this file runs as a script or is imported.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import docker_pool  # noqa: E402  (needs the path insert above)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedRun:
    """Represents a single seed evaluation target."""

    seed_dir: Path
    instance_id: str
    model_patch: str


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def load_specs(dataset_path: Path) -> dict[str, dict[str, Any]]:
    """Load instance specs from parquet, keyed by instance_id."""
    import pyarrow.parquet as pq

    table = pq.read_table(dataset_path)
    rows = table.to_pylist()
    return {row["instance_id"]: row for row in rows}


def _extract_patch_from_preds(seed_dir: Path) -> tuple[str, str] | None:
    """Try to read instance_id and patch from preds.json."""
    preds_path = seed_dir / "preds.json"
    if not preds_path.exists():
        return None
    try:
        data = json.loads(preds_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or len(data) != 1:
        return None
    instance_id, payload = next(iter(data.items()))
    if not isinstance(payload, dict):
        return None
    patch = payload.get("model_patch") or payload.get("patch") or ""
    if not patch.strip():
        return None
    return instance_id, patch


def _extract_patch_from_traj(traj_path: Path) -> tuple[str, str] | None:
    """Try to read instance_id and patch from a .traj.json file."""
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    instance_id = data.get("instance_id") or ""
    if not instance_id:
        # A mini-swe-agent v2 traj has no top level instance_id,
        # fall back to the file name: foo__bar-123.traj.json -> foo__bar-123
        name = traj_path.name
        if name.endswith(".traj.json"):
            instance_id = name[: -len(".traj.json")]
    info = data.get("info") or {}
    exit_status = info.get("exit_status", "")
    submission = info.get("submission") or ""
    if not instance_id or exit_status not in ("Submitted", "LimitsExceeded"):
        return None
    return instance_id, submission


def discover_runs(runs_dir: Path, *, skip_existing: bool) -> list[SeedRun]:
    """Find all seed directories containing *.traj.json under runs_dir."""
    seen_dirs: set[Path] = set()
    runs: list[SeedRun] = []

    for traj_path in sorted(runs_dir.rglob("*.traj.json")):
        seed_dir = traj_path.parent
        # The traj file may be nested one level deeper (seed_dir/instance_id/file.traj.json)
        # Walk up to find the actual seed dir (contains seed.txt or preds.json)
        if (seed_dir / "preds.json").exists() or (seed_dir / "seed.txt").exists():
            pass
        elif (seed_dir.parent / "preds.json").exists() or (
            seed_dir.parent / "seed.txt"
        ).exists():
            seed_dir = seed_dir.parent
        else:
            # Use the traj file's parent as seed dir
            pass

        if seed_dir in seen_dirs:
            continue
        seen_dirs.add(seed_dir)

        if skip_existing and (seed_dir / "eval_report.json").exists():
            continue

        # Extract patch: prefer preds.json, fall back to traj.json
        result = _extract_patch_from_preds(seed_dir)
        if result is None:
            result = _extract_patch_from_traj(traj_path)
        if result is None:
            log.debug("Skipping %s: no valid patch found", seed_dir)
            continue

        instance_id, patch = result
        runs.append(
            SeedRun(seed_dir=seed_dir, instance_id=instance_id, model_patch=patch)
        )

    return runs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_seed(
    run: SeedRun,
    specs_by_id: dict[str, dict[str, Any]],
    eval_script: Path,
    timeout: int,
) -> dict[str, Any]:
    """Run SWE-rebench eval for a single seed directory."""
    spec = specs_by_id.get(run.instance_id)
    if spec is None:
        return {
            "seed_dir": str(run.seed_dir),
            "instance_id": run.instance_id,
            "status": "error",
            "resolved": False,
            "error": "instance spec not found in dataset",
        }

    # Prepare eval input
    eval_input = [dict(spec, patch=run.model_patch)]
    eval_input_path = run.seed_dir / "eval_input.json"
    eval_report_path = run.seed_dir / "eval_report.json"
    run.seed_dir.mkdir(parents=True, exist_ok=True)
    eval_input_path.write_text(
        json.dumps(eval_input, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # Call eval.py
    cmd = [
        sys.executable,
        str(eval_script),
        "--json",
        str(eval_input_path),
        "--max-workers",
        "1",
        "--report-json",
        str(eval_report_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=run.seed_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "seed_dir": str(run.seed_dir),
            "instance_id": run.instance_id,
            "status": "timeout",
            "resolved": False,
            "error": f"eval timed out after {timeout}s",
        }

    # Save stdout/stderr
    (run.seed_dir / "eval_stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (run.seed_dir / "eval_stderr.log").write_text(result.stderr or "", encoding="utf-8")

    # Parse report
    report: dict[str, Any] = {}
    if eval_report_path.exists():
        try:
            report = json.loads(eval_report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "seed_dir": str(run.seed_dir),
                "instance_id": run.instance_id,
                "status": "error",
                "resolved": False,
                "error": f"failed to parse eval report: {exc}",
            }

    # Extract result from report items
    items = report.get("items", [])
    item = items[0] if isinstance(items, list) and items else {}
    error = item.get("error") or ""
    passed_match = bool(item.get("passed_match", False))
    resolved = result.returncode == 0 and passed_match and not error

    return {
        "seed_dir": str(run.seed_dir),
        "instance_id": run.instance_id,
        "status": "ok" if result.returncode == 0 else "failed",
        "resolved": resolved,
        "passed_match": passed_match,
        "returncode": result.returncode,
        "error": error,
    }


def evaluate_all(
    *,
    runs: list[SeedRun],
    specs_by_id: dict[str, dict[str, Any]],
    eval_script: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Group the seed runs by image and evaluate them through docker_pool.

    Stage 2 may have evicted these images already, so scoring generally has to
    pull them again. Grouping by image keeps that to one pull per image instead
    of one per seed, and lets the same disk cap apply here.
    """

    def image_of(run: SeedRun) -> str | None:
        spec = specs_by_id.get(run.instance_id) or {}
        image = spec.get("image_name")
        return str(image) if image else None

    def eval_one(run: SeedRun) -> dict[str, Any]:
        result = evaluate_seed(run, specs_by_id, eval_script, args.timeout)
        duration = _load_run_duration(run.seed_dir)
        if duration is not None:
            result["duration_s"] = duration
        return result

    def on_pull_failure(run: SeedRun, image: str) -> dict[str, Any]:
        return {
            "seed_dir": str(run.seed_dir),
            "instance_id": run.instance_id,
            "status": "error",
            "resolved": False,
            "error": f"failed to pull image {image}",
        }

    def progress(result: dict[str, Any], done: int, total: int) -> None:
        icon = "+" if result.get("resolved") else "x"
        duration = result.get("duration_s")
        suffix = (
            f" {duration:.1f}s" if isinstance(duration, (int, float)) else ""
        )
        log.info(
            "[%d/%d] %s %s (%s)%s",
            done,
            total,
            icon,
            result.get("instance_id", "?"),
            result.get("status", "?"),
            suffix,
        )

    groups = docker_pool.group_by_image(runs, image_of)
    return docker_pool.run_grouped(
        groups,
        eval_one,
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


def write_summary_csv(runs_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write evaluation summary as CSV."""
    csv_path = runs_dir / "eval_summary.csv"
    fields = [
        "seed_dir",
        "instance_id",
        "status",
        "resolved",
        "passed_match",
        "duration_s",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _load_run_duration(seed_dir: Path) -> float | None:
    """Load wall-clock duration written by run_agent.py's result.json."""
    p = seed_dir / "result.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text(encoding="utf-8")).get("duration_s"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"runs directory to evaluate (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--pull-workers",
        type=int,
        default=docker_pool.DEFAULT_PULL_WORKERS,
        help=(
            "how many docker pulls run at once, ahead of evaluation "
            f"(default {docker_pool.DEFAULT_PULL_WORKERS})"
        ),
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=docker_pool.DEFAULT_IMAGE_WORKERS,
        help=(
            "how many distinct images are being evaluated at once; concurrent "
            "containers peak at image-workers * seed-workers "
            f"(default {docker_pool.DEFAULT_IMAGE_WORKERS})"
        ),
    )
    parser.add_argument(
        "--seed-workers",
        type=int,
        default=docker_pool.DEFAULT_SEED_WORKERS,
        help=(
            "how many seeds of one instance are evaluated at once; they share "
            f"the image but get separate containers (default "
            f"{docker_pool.DEFAULT_SEED_WORKERS})"
        ),
    )
    parser.add_argument(
        "--image-pool-size",
        type=int,
        default=docker_pool.DEFAULT_POOL_SIZE,
        help=(
            "cap on images resident on disk; an image is removed once all its "
            "seeds are scored, freeing the slot for the next pull. Set 0 to "
            f"keep every image (default {docker_pool.DEFAULT_POOL_SIZE})"
        ),
    )
    parser.add_argument(
        "--pull-timeout",
        type=int,
        default=docker_pool.DEFAULT_PULL_TIMEOUT,
        help=(
            "seconds allowed for one docker pull "
            f"(default {docker_pool.DEFAULT_PULL_TIMEOUT})"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip directories that already have eval_report.json",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="per-evaluation timeout in seconds (default 600)",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=None,
        help="path to the SWE-rebench eval script (auto-detected by default)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="path to the dataset parquet (auto-detected by default)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve eval script path
    eval_script = (args.eval_script or DEFAULT_EVAL_SCRIPT).resolve()
    if not eval_script.exists():
        log.error("Eval script not found: %s", eval_script)
        return 1

    # Resolve dataset path
    dataset_path = (args.dataset or DEFAULT_DATASET).resolve()
    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        return 1

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        log.error("Runs directory does not exist: %s", runs_dir)
        return 1

    # Load specs
    log.info("Loading dataset specs from %s ...", dataset_path.name)
    specs_by_id = load_specs(dataset_path)
    log.info("Loaded %d instance specs", len(specs_by_id))

    # Discover runs
    runs = discover_runs(runs_dir, skip_existing=args.skip_existing)
    log.info("Found %d seed directories to evaluate", len(runs))
    if not runs:
        log.info("Nothing to evaluate.")
        return 0

    # Evaluation
    rows = evaluate_all(
        runs=runs, specs_by_id=specs_by_id, eval_script=eval_script, args=args
    )

    # Sort and summarize
    rows.sort(key=lambda r: (r.get("instance_id", ""), r.get("seed_dir", "")))

    total = len(rows)
    resolved = sum(1 for r in rows if r.get("resolved"))
    failed = sum(1 for r in rows if r.get("status") == "failed")
    errors = sum(1 for r in rows if r.get("status") == "error")
    timeouts = sum(1 for r in rows if r.get("status") == "timeout")

    log.info("=" * 50)
    log.info(
        "SUMMARY: total=%d resolved=%d failed=%d errors=%d timeouts=%d",
        total,
        resolved,
        failed,
        errors,
        timeouts,
    )
    if total > 0:
        log.info("Pass rate: %.1f%%", 100.0 * resolved / total)
    # Wall-clock stats (only counts rows where run_agent.py wrote result.json).
    durations = [
        r["duration_s"] for r in rows if isinstance(r.get("duration_s"), (int, float))
    ]
    if durations:
        total_s = sum(durations)
        avg_s = total_s / len(durations)
        log.info(
            "Wall-clock: n=%d total=%.0fs (%.1f min) avg=%.1fs min=%.1fs max=%.1fs",
            len(durations),
            total_s,
            total_s / 60.0,
            avg_s,
            min(durations),
            max(durations),
        )
    log.info("=" * 50)

    # Write CSV
    csv_path = write_summary_csv(runs_dir, rows)
    log.info("Summary CSV: %s", csv_path)

    return 0 if (failed == 0 and errors == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())

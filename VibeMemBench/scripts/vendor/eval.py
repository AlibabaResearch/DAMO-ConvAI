#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Timing patterns that some test runners embed in test names.
# These are stripped from both actual (by log parsers) and expected names (below)
# so that timing differences between runs don't cause spurious mismatches.
_TIMING_NORMALIZE_RES = [
    # PHP/JUnit bracket style: " [1.34 ms]" or "[123 ms]"
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    # Botan inline style: " in 29.08 msec" or " in 1 sec"
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    # Parenthesised duration: " (1.234s)" or " (123ms)"
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _normalize_test_name(name: str) -> str:
    """Strip known timing suffixes/infix patterns from a test name."""
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()

# Vendored from SWE-rebench-V2. Upstream resolved `agent.log_parsers` through
# a `lib/` package on sys.path. Here the parsers sit next to this file, so the
# import is flat and the original sys.path juggling is unnecessary.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import log_parsers

LOGS_DIR = Path("logs")


def load_specs(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of tasks.")
    for spec in data:
        if not isinstance(spec, dict):
            raise ValueError("Each task must be an object.")
    return data


def parse_instance_ids(value: str) -> list[str]:
    ids = [x.strip() for x in value.split(",") if x.strip()]
    if not ids:
        raise ValueError("--instance-ids provided but no valid ids found.")
    return ids


def filter_specs_by_instance_ids(specs: list[dict], wanted_ids: list[str]) -> list[dict]:
    wanted = set(wanted_ids)
    return [spec for spec in specs if spec.get("instance_id") in wanted]


def get_parser(parser_name: str):
    parser = log_parsers.NAME_TO_PARSER.get(parser_name)
    if parser is None:
        parser = getattr(log_parsers, parser_name, None)
    if parser is None:
        raise ValueError(f"Unknown log parser: {parser_name}")
    return parser


def normalize_command_list(value: object, field_name: str, instance_id: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"Task {instance_id} missing {field_name}.")

    commands = [item for item in value if isinstance(item, str) and item.strip()]
    if not commands:
        raise ValueError(f"Task {instance_id} missing {field_name}.")
    return commands


def write_patch(temp_dir: Path, name: str, content: str) -> Path:
    path = temp_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def run_in_container(
    image: str,
    workdir: str,
    patch_dir: Path,
    patch_name: str,
    test_patch_name: str,
    test_cmds: list[str],
) -> tuple[int, str]:
    cmd_lines = [
        "set -e",
        "git reset --hard HEAD",
        f"git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/{patch_name}",
        f"git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/{test_patch_name}",
    ]
    cmd_lines.extend(test_cmds)
    script = "\n".join(cmd_lines)

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network", "host",
        "-e", "_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
        "-v",
        f"{patch_dir}:/patches:ro",
        "-w",
        workdir,
        image,
        "/bin/bash",
        "-c",
        script,
    ]

    result = subprocess.run(
        docker_cmd, check=False, capture_output=True, text=True
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output


def evaluate_instance(
    spec: dict,
    image: str,
    patch_override: dict | None,
) -> dict:
    instance_id = spec.get("instance_id")
    if not instance_id:
        raise ValueError("Task missing instance_id.")

    repo = spec.get("repo")
    if not repo or "/" not in repo:
        raise ValueError(f"Task {instance_id} missing repo.")

    install_config = spec.get("install_config", {})
    test_cmds = normalize_command_list(
        install_config.get("test_cmd", []),
        "install_config.test_cmd",
        instance_id,
    )

    parser_name = install_config.get("log_parser")
    if not parser_name:
        raise ValueError(f"Task {instance_id} missing install_config.log_parser.")
    parser = get_parser(parser_name)

    patch = spec.get("patch", "")
    if patch_override:
        patch = patch_override.get("patch", patch)
    test_patch = spec.get("test_patch", "")
    if not patch or not test_patch:
        raise ValueError(f"Task {instance_id} missing patch/test_patch.")

    workdir = f"/{repo.split('/')[1]}"

    with tempfile.TemporaryDirectory(prefix="eval_patches_") as tmp:
        patch_dir = Path(tmp)
        write_patch(patch_dir, "patch.diff", patch)
        write_patch(patch_dir, "test_patch.diff", test_patch)
        exit_code, output = run_in_container(
            image=image,
            workdir=workdir,
            patch_dir=patch_dir,
            patch_name="patch.diff",
            test_patch_name="test_patch.diff",
            test_cmds=test_cmds,
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{instance_id}_log.txt"
    log_path.write_text(output, encoding="utf-8")

    parsed = parser(output)
    # Normalize actual names (strip timing) in case the parser didn't already.
    parsed = {_normalize_test_name(k): v for k, v in parsed.items()}
    passed = sorted(k for k, v in parsed.items() if v == "PASSED")
    failed = sorted(k for k, v in parsed.items() if v == "FAILED")

    # Normalize expected names so timing differences between runs don't cause
    # spurious mismatches (some datasets were captured with timing in test names).
    expected_passed = sorted(
        _normalize_test_name(n)
        for n in spec.get("PASS_TO_PASS", []) + spec.get("FAIL_TO_PASS", [])
    )

    result = {
        "instance_id": instance_id,
        "exit_code": exit_code,
        "passed_match": passed == expected_passed,
        "pass_to_pass_expected": sorted(spec.get("PASS_TO_PASS", [])),
        "fail_to_pass_expected": sorted(spec.get("FAIL_TO_PASS", [])),
        "passed_expected": expected_passed,
        "passed_actual": passed,
        "failed_actual": failed,
        "log_length": len(output),
        "log_path": str(log_path),
    }
    return result


def resolve_task_image(
    spec: dict,
    image_registry: str,
    tag_prefix: str,
) -> str:
    instance_id = spec.get("instance_id")
    if not instance_id:
        raise ValueError("Task missing instance_id.")

    image = spec.get("image_name")
    if image is not None:
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"Task {instance_id} missing top-level image_name.")
        return image

    tag = f"{tag_prefix}{instance_id}"
    if image_registry:
        tag = f"{image_registry}/{tag}"
    return tag


def evaluate_task(
    spec: dict,
    image_registry: str,
    tag_prefix: str,
    golden_eval: bool,
    patch_overrides: dict[str, dict],
) -> dict:
    instance_id = spec.get("instance_id")
    if not instance_id:
        return {"instance_id": "", "error": "Task missing instance_id."}

    try:
        image = resolve_task_image(
            spec=spec,
            image_registry=image_registry,
            tag_prefix=tag_prefix,
        )
        patch_override = None if golden_eval else patch_overrides.get(instance_id)
        result = evaluate_instance(spec, image, patch_override)
        return {"instance_id": instance_id, "result": result}
    except Exception as exc:
        return {"instance_id": instance_id, "error": str(exc)}


def build_report_item(spec: dict, outcome: dict) -> dict:
    instance_id = spec.get("instance_id", "")
    # Normalize expected names to match what log parsers produce.
    fail_to_pass_expected = {_normalize_test_name(n) for n in spec.get("FAIL_TO_PASS", [])}
    pass_to_pass_expected = {_normalize_test_name(n) for n in spec.get("PASS_TO_PASS", [])}
    error = outcome.get("error")
    if error:
        return {
            "instance_id": instance_id,
            "from_fail_to_pass": [],
            "failed_from_pass_to_pass": sorted(pass_to_pass_expected),
            "error": error,
        }

    result = outcome["result"]
    passed_actual = set(result.get("passed_actual", []))
    from_fail_to_pass = sorted(passed_actual.intersection(fail_to_pass_expected))
    failed_from_pass_to_pass = sorted(pass_to_pass_expected.difference(passed_actual))
    return {
        "instance_id": instance_id,
        "from_fail_to_pass": from_fail_to_pass,
        "failed_from_pass_to_pass": failed_from_pass_to_pass,
        "passed_match": result.get("passed_match", False),
        "exit_code": result.get("exit_code"),
        "log_path": result.get("log_path"),
        "error": "",
    }


def render_progress_bar(completed: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = max(0, min(width, int((completed / total) * width)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate instances via docker.")
    parser.add_argument(
        "--json",
        required=True,
        help="Path to JSON file with tasks.",
    )
    parser.add_argument(
        "--patches",
        default="",
        help="Optional JSON with instance_id and patch overrides.",
    )
    parser.add_argument(
        "--image-registry",
        default="",
        help="Optional registry/repo prefix for instance images.",
    )
    parser.add_argument(
        "--tag-prefix",
        default="",
        help="Optional prefix for instance image tags.",
    )
    parser.add_argument(
        "--golden-eval",
        action="store_true",
        help="Ignore --patches and use patch/test_patch from each task.",
    )
    parser.add_argument(
        "--instance-ids",
        default="",
        help="Comma-separated instance_id list to run a subset of tasks.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of tasks to evaluate in parallel.",
    )
    parser.add_argument(
        "--report-json",
        default="eval_report.json",
        help="Path to write JSON report.",
    )
    args = parser.parse_args()

    if args.max_workers < 1:
        print("--max-workers must be >= 1.", file=sys.stderr)
        return 1

    try:
        json_path = Path(args.json)
        if not json_path.is_file():
            print(f"JSON file not found: {json_path}", file=sys.stderr)
            return 1
        specs = load_specs(json_path)
    except Exception as exc:
        print(f"Failed to load tasks: {exc}", file=sys.stderr)
        return 1

    if args.instance_ids:
        try:
            wanted_ids = parse_instance_ids(args.instance_ids)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        specs = filter_specs_by_instance_ids(specs, wanted_ids)
        if not specs:
            print("No tasks matched --instance-ids.", file=sys.stderr)
            return 1

    patch_overrides = {}
    if args.patches and not args.golden_eval:
        patches_path = Path(args.patches)
        if not patches_path.is_file():
            print(f"Patches JSON not found: {patches_path}", file=sys.stderr)
            return 1
        patches_data = json.loads(patches_path.read_text(encoding="utf-8"))
        if not isinstance(patches_data, list):
            print("Patches JSON root must be a list.", file=sys.stderr)
            return 1
        for item in patches_data:
            if not isinstance(item, dict):
                print("Each patches entry must be an object.", file=sys.stderr)
                return 1
            pid = item.get("instance_id")
            if not pid:
                print("Patches entry missing instance_id.", file=sys.stderr)
                return 1
            patch_overrides[pid] = item

    if args.patches and args.golden_eval:
        print("warning: --golden-eval enabled; ignoring --patches", file=sys.stderr)

    all_ok = True
    outcomes_by_id: dict[str, dict] = {}
    for spec in specs:
        instance_id = spec.get("instance_id")
        if not instance_id:
            print("Task missing instance_id.", file=sys.stderr)
            return 1

    total_tasks = len(specs)
    completed_tasks = 0
    ok_tasks = 0
    mismatch_tasks = 0
    error_tasks = 0

    def print_progress() -> None:
        running_tasks = max(total_tasks - completed_tasks, 0)
        bar = render_progress_bar(completed_tasks, total_tasks)
        line = (
            f"\r{bar} done {completed_tasks}/{total_tasks} | "
            f"ok {ok_tasks} | mismatch {mismatch_tasks} | error {error_tasks} | "
            f"running {running_tasks}"
        )
        print(line, end="", flush=True)

    print_progress()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_id = {
            executor.submit(
                evaluate_task,
                spec=spec,
                image_registry=args.image_registry,
                tag_prefix=args.tag_prefix,
                golden_eval=args.golden_eval,
                patch_overrides=patch_overrides,
            ): spec["instance_id"]
            for spec in specs
        }

        for future in as_completed(future_to_id):
            instance_id = future_to_id[future]
            try:
                outcome = future.result()
            except Exception as exc:
                outcome = {"instance_id": instance_id, "error": str(exc)}
            outcomes_by_id[instance_id] = outcome

            error = outcome.get("error")
            if error:
                all_ok = False
                error_tasks += 1
            else:
                result = outcome["result"]
                ok = result["passed_match"]
                if ok:
                    ok_tasks += 1
                else:
                    mismatch_tasks += 1
                    all_ok = False

            completed_tasks += 1
            print_progress()

    print()

    report_items = []
    for spec in specs:
        instance_id = spec["instance_id"]
        outcome = outcomes_by_id.get(instance_id, {"instance_id": instance_id, "error": "No outcome produced."})
        report_items.append(build_report_item(spec, outcome))

    report_path = Path(args.report_json)
    report_payload = {
        "max_workers": args.max_workers,
        "total": len(report_items),
        "all_ok": all_ok,
        "items": report_items,
    }
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON report written: {report_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

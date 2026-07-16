"""Validate one or more leaderboard submissions.

A submission is a directory ``leaderboard/<name>/`` containing ``results.json``
(the trimmed benchmark scores) and ``config.json`` (run metadata). This script
checks both files for the leaderboard contract and exits non-zero on any
failure, so it can gate pull requests in CI.

Usage:
    uv run python scripts/validate_submission.py leaderboard/locf
    uv run python scripts/validate_submission.py --all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Allowed enum values for config fields.
_MODEL_TYPES = {"statistical", "pretrained", "fine-tuned", "from-scratch"}
_EVAL_MODES = {"zero-shot", "fine-tuned", "trained-from-scratch"}

# Config fields that must be present and filled in (no empty / "TODO" values).
_REQUIRED_CONFIG_FIELDS = (
    "model_name",
    "model_type",
    "evaluation_mode",
    "organization",
    "code_url",
    "test_data_leakage",
    "benchmark_commit",
)


def _is_unfilled(value: object) -> bool:
    """True if a config value is missing or still a placeholder."""
    return value is None or (
        isinstance(value, str) and (not value.strip() or value.startswith("TODO"))
    )


def _check_score_block(block: dict, where: str, errors: list[str]) -> None:
    """Validate that a block has finite, non-negative nmae and nrmse."""
    for key in ("nmae", "nrmse"):
        if key not in block:
            errors.append(f"{where}: missing '{key}'")
            continue
        val = block[key]
        if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
            errors.append(f"{where}: '{key}' must be a finite number >= 0, got {val!r}")


def validate_results(results: dict, errors: list[str]) -> None:
    """Validate the necessary fields of a results.json dict.

    ``results.json`` is the full ``run_benchmark`` output, so it carries far more
    than the leaderboard needs (per-sample and per-signal-channel detail, skip
    counts). This checks only the fields the table and ranking depend on — the
    top-level, per-domain, and per-task NMAE/NRMSE — and tolerates any extra
    keys.
    """
    _check_score_block(results, "results.json", errors)
    per_domain = results.get("per_domain")
    if not isinstance(per_domain, list) or not per_domain:
        errors.append("results.json: 'per_domain' must be a non-empty array")
        return
    for i, block in enumerate(per_domain):
        if not isinstance(block, dict):
            errors.append(f"results.json: per_domain[{i}] must be an object")
            continue
        name = block.get("domain")
        if not isinstance(name, str) or not name:
            errors.append(f"results.json: per_domain[{i}] missing string 'domain'")
            name = f"#{i}"
        _check_score_block(block, f"per_domain['{name}']", errors)
        per_task = block.get("per_task", [])
        if not isinstance(per_task, list):
            errors.append(f"per_domain['{name}'].per_task must be an array")
            continue
        for tblock in per_task:
            if not isinstance(tblock, dict):
                errors.append(f"per_domain['{name}'].per_task item must be an object")
                continue
            task = tblock.get("task", "?")
            if "task" not in tblock:
                errors.append(f"per_domain['{name}'].per_task item missing 'task'")
            _check_score_block(
                tblock, f"per_domain['{name}'].per_task['{task}']", errors
            )


def validate_config(config: dict, errors: list[str]) -> None:
    """Validate required fields and enum values of a config.json dict."""
    for field in _REQUIRED_CONFIG_FIELDS:
        if field not in config or _is_unfilled(config[field]):
            errors.append(
                f"config.json: required field '{field}' is missing or unfilled"
            )
    mtype = config.get("model_type")
    if mtype is not None and not _is_unfilled(mtype) and mtype not in _MODEL_TYPES:
        errors.append(
            f"config.json: 'model_type' must be one of {sorted(_MODEL_TYPES)}, got {mtype!r}"
        )
    mode = config.get("evaluation_mode")
    if mode is not None and not _is_unfilled(mode) and mode not in _EVAL_MODES:
        errors.append(
            f"config.json: 'evaluation_mode' must be one of {sorted(_EVAL_MODES)}, got {mode!r}"
        )


def validate_entry(entry_dir: Path) -> list[str]:
    """Validate a single submission directory.

    Args:
        entry_dir: Path to ``leaderboard/<name>/``.

    Returns:
        A list of human-readable error strings; empty means the entry is valid.
    """
    errors: list[str] = []
    for filename, validator in (
        ("results.json", validate_results),
        ("config.json", validate_config),
    ):
        path = entry_dir / filename
        if not path.exists():
            errors.append(f"{entry_dir.name}: missing {filename}")
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{entry_dir.name}/{filename}: invalid JSON ({exc})")
            continue
        validator(data, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "entries",
        nargs="*",
        type=Path,
        help="Submission directories to validate (e.g. leaderboard/locf).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate every subdirectory of leaderboard/.",
    )
    parser.add_argument(
        "--leaderboard-root",
        type=Path,
        default=Path("leaderboard"),
        help="Root used with --all (default: leaderboard).",
    )
    args = parser.parse_args()

    entries = list(args.entries)
    if args.all:
        entries += sorted(p for p in args.leaderboard_root.glob("*") if p.is_dir())
    if not entries:
        parser.error("Provide submission directories or use --all.")

    total_errors = 0
    for entry in entries:
        errors = validate_entry(entry)
        if errors:
            total_errors += len(errors)
            print(f"FAIL {entry}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"OK   {entry}")

    if total_errors:
        print(f"\n{total_errors} validation error(s).")
        return 1
    print("\nAll submissions valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

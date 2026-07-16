"""Top-level entry point for full benchmark evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

from benchmark.evaluate import (
    VALID_TASKS,
    evaluate_benchmark,
    evaluate_domain,
    evaluate_split,
)
from benchmark.specs import BenchmarkSpec

logger = logging.getLogger(__name__)


def run_benchmark(
    predictions_root: Path,
    domain_specs: dict[str, BenchmarkSpec],
    split_dirs: dict[str, Path],
    tasks: list[str] | None = None,
) -> dict:
    """Evaluate a model against the full benchmark in one call.

    Discovers prediction files under predictions_root using the expected
    directory layout:

        predictions_root/
          <domain>/
            <task>/
              <sample_filename>.parquet

    where <domain> matches the keys of domain_specs and split_dirs,
    and <task> is one of the VALID_TASKS in evaluate.py.

    For each (domain, task) combination, the function checks whether the
    predictions directory exists and contains at least one .parquet file.
    Present tasks are evaluated via evaluate_split; absent tasks are skipped
    with a logged warning. This allows partial submissions: a model that
    only supports forecasting can submit predictions for that task alone
    and receive a meaningful partial score.

    Args:
        predictions_root: Root directory containing per-domain, per-task
            prediction subdirectories.
        domain_specs: Mapping from domain name (str) to BenchmarkSpec.
            The domain name is used to locate predictions_root/<domain>/
            and must match the keys of split_dirs exactly.
        split_dirs: Mapping from domain name to the Path of the test-split
            directory containing the ground-truth observation Parquet files.
            Keys must match domain_specs exactly.
        tasks: Optional list of task names to evaluate. Defaults to all
            four tasks: ["forecasting", "offline_imputation",
            "causal_imputation", "cross_modal"]. Unknown task names raise
            ValueError.

    Returns:
        Dict with the following structure:

            {
                "nmae": float,       # L5 benchmark NMAE
                "nrmse": float,      # L5 benchmark NRMSE
                "n_domains": int,
                "per_domain": [
                    {
                        "domain": str,       # domain name
                        "nmae": float,       # L4 domain NMAE
                        "nrmse": float,      # L4 domain NRMSE
                        "n_tasks": int,
                        "tasks_run": list[str],
                        "tasks_skipped": list[str],
                        "per_task": [        # one entry per task that was run
                            {
                                "task": str,
                                "domain": str,
                                "nmae": float,
                                "nrmse": float,
                                "n_samples": int,
                                "n_skipped": int,
                                "per_sample": list[dict],
                            },
                            ...
                        ],
                    },
                    ...
                ],
            }

    Raises:
        FileNotFoundError: If predictions_root does not exist, or if any
            split_dir does not exist.
        ValueError: If domain_specs and split_dirs have different keys.
        ValueError: If any entry in tasks is not a valid task name.
        ValueError: If no prediction files are found for any task in a
            domain (propagated from evaluate_domain).
        ValueError: If domain_specs is empty.
    """
    predictions_root = Path(predictions_root)
    if not predictions_root.exists():
        raise FileNotFoundError(
            f"run_benchmark: predictions_root does not exist: {predictions_root}"
        )

    if not domain_specs:
        raise ValueError("run_benchmark: domain_specs must not be empty.")

    spec_keys = set(domain_specs.keys())
    split_keys = set(split_dirs.keys())
    if spec_keys != split_keys:
        missing = spec_keys - split_keys
        extra = split_keys - spec_keys
        raise ValueError(
            f"run_benchmark: domain_specs and split_dirs keys do not match. "
            f"Missing from split_dirs: {sorted(missing)}. "
            f"Extra in split_dirs: {sorted(extra)}."
        )

    for domain_name, split_dir in split_dirs.items():
        split_dir = Path(split_dir)
        if not split_dir.exists():
            raise FileNotFoundError(
                f"run_benchmark: split_dir for domain {domain_name!r} does not exist: "
                f"{split_dir}"
            )

    if tasks is None:
        tasks = list(VALID_TASKS)
    else:
        invalid = [t for t in tasks if t not in VALID_TASKS]
        if invalid:
            raise ValueError(
                f"run_benchmark: unknown task name(s) {invalid}. "
                f"Valid tasks: {list(VALID_TASKS)}."
            )

    domain_results: list[dict] = []

    for domain_name in sorted(domain_specs.keys()):
        spec = domain_specs[domain_name]
        split_dir = Path(split_dirs[domain_name])
        task_results: list[dict] = []
        tasks_run: list[str] = []
        tasks_skipped: list[str] = []

        tasks_set = set(tasks)
        for task_name in VALID_TASKS:
            if task_name not in tasks_set:
                tasks_skipped.append(task_name)
                continue
            pred_dir = predictions_root / domain_name / task_name
            has_preds = pred_dir.exists() and any(
                p for p in pred_dir.glob("*.parquet") if "_grid" not in p.name
            )
            if has_preds:
                result = evaluate_split(split_dir, pred_dir, task_name, spec)
                task_results.append(result)
                tasks_run.append(task_name)
            else:
                logger.warning(
                    "run_benchmark: no predictions found for domain=%r task=%r at %s — skipping.",
                    domain_name,
                    task_name,
                    pred_dir,
                )
                tasks_skipped.append(task_name)

        domain_result = evaluate_domain(task_results)
        domain_result["domain"] = domain_name
        domain_result["tasks_run"] = tasks_run
        domain_result["tasks_skipped"] = tasks_skipped
        domain_results.append(domain_result)

    return evaluate_benchmark(domain_results)

"""Full evaluation pipeline orchestration.
This module provides three functions that compose the lower-level building blocks from ``benchmark.data``, ``benchmark.tasks``, and ``benchmark.metrics`` into complete evaluation runs:

- ``evaluate_sample``  — one sample, one task
- ``evaluate_split``   — all samples in a split directory, one task
- ``evaluate_domain``  — all tasks in a domain
- ``evaluate_benchmark`` — all domains in the benchmark
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.data import load_sample
from benchmark.metrics import aggregate_sample, compute_scores
from benchmark.specs import BenchmarkSpec
from benchmark.tasks import (
    partition_causal_imputation,
    partition_cross_modal,
    partition_forecasting,
    partition_offline_imputation,
)

logger = logging.getLogger(__name__)

VALID_TASKS: tuple[str, ...] = (
    "forecasting",
    "offline_imputation",
    "causal_imputation",
    "cross_modal",
)


def evaluate_sample(
    parquet_path: Path,
    predictions: pd.DataFrame,
    task_name: str,
    spec: BenchmarkSpec,
) -> dict:
    """Run the full evaluation pipeline for a single sample and task.

    Loads the observation Parquet file, partitions it into context and query
    sets according to ``task_name``, aligns ``predictions`` to the query set,
    computes per-observation NMAE and NMSE, and aggregates hierarchically.

    Args:
        parquet_path: Path to the observation Parquet file for this sample
        predictions: DataFrame with columns ``(t, s, j, x_pred)`` where ``x_pred`` is a numpy array of float32 predictions.  Rows should correspond to query-set observations.
        task_name: One of ``VALID_TASKS``.  Determines how the observation set is partitioned
        spec: Full benchmark specification.

    Returns:
        Dict containing the ``aggregate_sample`` result (keys ``nmae``,``nrmse``, ``n_signal_channels``, ``per_signal_channel``) extended
        with:

            {
                "sample_id": str,    # stem of parquet_path, e.g. "shot_30421"
                "task": str,         # task_name
                "domain": str,       # spec.domain.name
                "n_query_obs": int,  # total rows in Q
                "n_matched": int,    # rows with a matching prediction
            }

    Raises:
        ValueError: If ``task_name`` is not one of ``VALID_TASKS``, or if the corresponding spec field is ``None`` in ``spec``.
    """
    if task_name not in VALID_TASKS:
        raise ValueError(
            f"evaluate_sample: unknown task {task_name!r}. "
            f"Must be one of {VALID_TASKS}."
        )

    obs = load_sample(parquet_path)

    # Partition into context and query sets
    if task_name == "forecasting":
        if spec.forecasting is None:
            raise ValueError("evaluate_sample: spec.forecasting is None.")
        _, Q = partition_forecasting(obs, spec.forecasting)
    elif task_name == "offline_imputation":
        if spec.offline_imputation is None:
            raise ValueError("evaluate_sample: spec.offline_imputation is None.")
        _, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
    elif task_name == "causal_imputation":
        if spec.causal_imputation is None:
            raise ValueError("evaluate_sample: spec.causal_imputation is None.")
        _, Q = partition_causal_imputation(obs, spec.causal_imputation, spec.domain)
    else:  # cross_modal
        if spec.cross_modal is None:
            raise ValueError("evaluate_sample: spec.cross_modal is None.")
        _, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)

    scores = compute_scores(Q, predictions, spec.domain.train_stats)
    result = aggregate_sample(scores, spec.domain.registry)

    result["sample_id"] = parquet_path.stem
    result["task"] = task_name
    result["domain"] = spec.domain.name
    result["n_query_obs"] = len(Q)
    result["n_matched"] = int(scores["matched"].sum())

    return result


def evaluate_split(
    split_dir: Path,
    predictions_dir: Path,
    task_name: str,
    spec: BenchmarkSpec,
) -> dict:
    """Evaluate all samples in a split directory for one task. Implements level-3 (task-level) aggregation.
    Args:
        split_dir: Directory containing the split's Parquet files.
        predictions_dir: Directory containing prediction Parquet files, one
            per sample.  Each prediction file must have the same filename as
            its corresponding observation file.  Samples with no matching
            prediction file are skipped with a logged warning.
        task_name: One of ``VALID_TASKS``.
        spec: Full benchmark specification.

    Returns:
        Dict::

            {
                "task": str,
                "domain": str,
                "nmae": float,           # L^NMAE_tau
                "nrmse": float,          # L^NRMSE_tau
                "n_samples": int,        # samples successfully evaluated
                "n_skipped": int,        # samples skipped (missing predictions)
                "per_sample": list[dict] # evaluate_sample results
            }
    """
    split_dir = Path(split_dir)
    predictions_dir = Path(predictions_dir)

    parquet_files = sorted(
        p for p in split_dir.glob("*.parquet") if "_grid" not in p.name
    )

    per_sample: list[dict] = []
    n_skipped = 0

    for pq_path in parquet_files:
        pred_path = predictions_dir / pq_path.name
        if not pred_path.exists():
            logger.warning(
                "evaluate_split: no prediction file found for %s at %s — skipping.",
                pq_path.name,
                pred_path,
            )
            n_skipped += 1
            continue

        predictions = pd.read_parquet(pred_path)
        result = evaluate_sample(pq_path, predictions, task_name, spec)
        per_sample.append(result)

    n_samples = len(per_sample)

    if n_samples == 0:
        logger.warning(
            "evaluate_split: no samples evaluated for task %r in %s — returning NaN scores.",
            task_name,
            split_dir,
        )
        nmae = float("nan")
        nrmse = float("nan")
    else:
        # Use nanmean so that samples with entirely-NaN signal channels (e.g.
        # completely offline sensors) are excluded rather than poisoning the
        # task-level aggregate.
        nmae = float(np.nanmean([r["nmae"] for r in per_sample]))
        nrmse = float(np.nanmean([r["nrmse"] for r in per_sample]))

    return {
        "task": task_name,
        "domain": spec.domain.name,
        "nmae": nmae,
        "nrmse": nrmse,
        "n_samples": n_samples,
        "n_skipped": n_skipped,
        "per_sample": per_sample,
    }


def evaluate_domain(task_results: list[dict]) -> dict:
    """Aggregate task-level results to a domain-level score. Implements level-4 (domain-level) aggregation.

    Args:
        task_results: List of dicts as returned by ``evaluate_split``, one per benchmark task in the domain.

    Returns:
        Dict::

            {
                "nmae": float,          # L^NMAE_d
                "nrmse": float,         # L^NRMSE_d
                "n_tasks": int,         # |B_d|
                "per_task": list[dict]  # the input task_results
            }

    Raises:
        ValueError: If ``task_results`` is empty.
    """
    if not task_results:
        raise ValueError(
            "evaluate_domain: task_results must not be empty. "
            "An empty list means no tasks were evaluated — this indicates a "
            "pipeline or configuration error, not a model failure mode."
        )

    nmae = float(np.nanmean([r["nmae"] for r in task_results]))
    nrmse = float(np.nanmean([r["nrmse"] for r in task_results]))

    return {
        "nmae": nmae,
        "nrmse": nrmse,
        "n_tasks": len(task_results),
        "per_task": task_results,
    }


def evaluate_benchmark(domain_results: list[dict]) -> dict:
    """Aggregate domain-level results to the overall benchmark score. Implements level-5 (benchmark-level) aggregation.

    Args:
        domain_results: List of dicts as returned by ``evaluate_domain``, one
            per scientific domain in the benchmark.  Each dict must have
            keys ``"nmae"``, ``"nrmse"``, and ``"n_tasks"``.

    Returns:
        Dict::

            {
                "nmae": float,            # overall benchmark NMAE
                "nrmse": float,           # overall benchmark NRMSE
                "n_domains": int,         # |D|
                "per_domain": list[dict]  # the input domain_results
            }

    Raises:
        ValueError: If ``domain_results`` is empty.
        ValueError: If any result dict is missing ``"nmae"`` or ``"nrmse"``
            keys.
    """
    if not domain_results:
        raise ValueError("evaluate_benchmark: domain_results must not be empty.")

    for i, result in enumerate(domain_results):
        if "nmae" not in result:
            raise ValueError(
                f"evaluate_benchmark: result at index {i} is missing 'nmae' key."
            )
        if "nrmse" not in result:
            raise ValueError(
                f"evaluate_benchmark: result at index {i} is missing 'nrmse' key."
            )
        if "n_tasks" not in result:
            raise ValueError(
                f"evaluate_benchmark: result at index {i} is missing 'n_tasks' key."
            )

    nan_indices = [
        i
        for i, r in enumerate(domain_results)
        if not math.isfinite(r["nmae"]) or not math.isfinite(r["nrmse"])
    ]
    if nan_indices:
        logger.warning(
            "evaluate_benchmark: domain result(s) at index %s contain NaN/inf scores — "
            "benchmark aggregate will be NaN.",
            nan_indices,
        )

    nmae = float(np.nanmean([r["nmae"] for r in domain_results]))
    nrmse = float(np.nanmean([r["nrmse"] for r in domain_results]))

    return {
        "nmae": nmae,
        "nrmse": nrmse,
        "n_domains": len(domain_results),
        "per_domain": domain_results,
    }

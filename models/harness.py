"""Shared evaluation harness: run one model over a split for every task.

This single loop replaces the per-model ``runner.py`` files. It loads the
domain spec, runs the model's per-domain :meth:`~models.base.Model.setup`, then
for each sample partitions the observation set into context/query for **every**
requested task and writes a prediction file. Because a file is written for every
task, every model is scored on every task (rows the model cannot predict get the
training-mean penalty in the evaluator).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from benchmark.data import load_sample
from benchmark.evaluate import VALID_TASKS
from benchmark.specs import (
    BenchmarkSpec,
    load_fusion_spec,
    load_healthcare_spec,
    load_tcprimed_spec,
)
from benchmark.tasks import (
    partition_causal_imputation,
    partition_cross_modal,
    partition_forecasting,
    partition_offline_imputation,
)
from models._io import write_predictions
from models.base import Model

logger = logging.getLogger(__name__)

# Domain name -> spec loader. Matches the keys used everywhere else.
DOMAIN_LOADERS = {
    "fusion": load_fusion_spec,
    "healthcare": load_healthcare_spec,
    "tcprimed": load_tcprimed_spec,
}

_EMPTY_PRED = pd.DataFrame(columns=["t", "s", "j", "x_pred"])


def _partition(
    task: str, obs: pd.DataFrame, spec: BenchmarkSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition ``obs`` into (context, query) for ``task`` using ``spec``.

    The caller (:func:`run_model`) only invokes this for tasks whose spec is not
    ``None``; the asserts document and narrow that precondition.
    """
    if task == "forecasting":
        assert spec.forecasting is not None
        return partition_forecasting(obs, spec.forecasting)
    if task == "offline_imputation":
        assert spec.offline_imputation is not None
        return partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
    if task == "causal_imputation":
        assert spec.causal_imputation is not None
        return partition_causal_imputation(obs, spec.causal_imputation, spec.domain)
    if task == "cross_modal":
        assert spec.cross_modal is not None
        return partition_cross_modal(obs, spec.cross_modal, spec.domain)
    raise ValueError(f"_partition: unknown task {task!r}")


def run_model(
    model: Model,
    spec: BenchmarkSpec,
    split_dir: Path,
    output_root: Path,
    tasks: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
) -> Path:
    """Run ``model`` over a split, writing predictions for each task.

    Args:
        model: The model to run.
        spec: The domain's loaded benchmark specification (e.g. from
            ``load_fusion_spec``). Its ``spec.domain.name`` names the output
            subdirectory and ``spec.domain.data_root / "train"`` is passed to
            ``model.setup`` as the per-domain training directory.
        split_dir: Directory of observation Parquet files to evaluate
            (``_grid`` files are skipped).
        output_root: Predictions land at
            ``output_root/<domain>/<task>/<sample>.parquet``.
        tasks: Subset of ``VALID_TASKS`` to run; defaults to all four. Tasks
            whose spec is ``None`` for this domain are skipped.

    Returns:
        ``output_root`` (the predictions root, for passing to ``run_benchmark``).

    Raises:
        ValueError: If ``tasks`` contains an invalid task name.
    """
    selected = tuple(tasks) if tasks else VALID_TASKS
    invalid = [t for t in selected if t not in VALID_TASKS]
    if invalid:
        raise ValueError(f"run_model: unknown task(s) {invalid}. Valid: {VALID_TASKS}.")

    split_dir = Path(split_dir)
    output_root = Path(output_root)
    domain_name = spec.domain.name
    train_dir = Path(spec.domain.data_root) / "train"

    logger.info("Preparing %s for domain %r ...", model.name, domain_name)
    model.setup(spec, train_dir)

    parquet_files = sorted(
        p for p in split_dir.glob("*.parquet") if "_grid" not in p.name
    )
    if limit is not None:
        parquet_files = parquet_files[:limit]
    if not parquet_files:
        logger.warning("run_model: no parquet files found in %s.", split_dir)

    for parquet_path in parquet_files:
        logger.info("Processing %s", parquet_path.name)
        obs = load_sample(parquet_path)

        for task in selected:
            if getattr(spec, task) is None:
                logger.debug(
                    "Skipping task %r: spec is None for %s.", task, domain_name
                )
                continue

            context, query = _partition(task, obs, spec)
            out_path = output_root / domain_name / task / parquet_path.name

            if len(query) == 0:
                write_predictions(_EMPTY_PRED, out_path)
                continue

            pred = model.predict(context, query, spec)
            write_predictions(pred, out_path)

    return output_root

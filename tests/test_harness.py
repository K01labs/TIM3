"""Tests for the unified model harness (models.harness.run_model).

Synthetic data only — no network, no real datasets. Verifies the structural
guarantee that the harness writes a prediction file for *every* task, so every
model is scored on every task (rows a model omits get the training-mean penalty).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from benchmark.evaluate import VALID_TASKS
from benchmark.run import run_benchmark
from benchmark.specs import (
    BenchmarkSpec,
    CausalImputationSpec,
    CrossModalSpec,
    DomainSpec,
    ForecastingSpec,
    OfflineImputationSpec,
)
from models.base import Model

_OBS_SCHEMA = pa.schema(
    [
        ("t", pa.float32()),
        ("s", pa.int32()),
        ("j", pa.int32()),
        ("x", pa.list_(pa.float32())),
    ]
)


def _make_spec(tmp_path: Path) -> BenchmarkSpec:
    """A two-signal synthetic domain with all four task specs populated."""
    registry = {
        "0": {"name": "static", "category": "static"},
        "1": {"name": "vital_a", "category": "scalar_single"},
        "2": {"name": "vital_b", "category": "scalar_single"},
    }
    train_stats = pd.DataFrame(
        {
            "s": np.array([1, 2], dtype=np.int32),
            "j": np.array([0, 0], dtype=np.int32),
            "m": np.array([0, 0], dtype=np.int32),
            "mean": np.array([0.0, 0.0], dtype=np.float32),
            "std": np.array([1.0, 1.0], dtype=np.float32),
        }
    )
    domain = DomainSpec(
        name="fusion",  # any DOMAIN_LOADERS key; only used for the output subdir
        data_root=tmp_path,
        registry=registry,
        train_stats=train_stats,
    )
    return BenchmarkSpec(
        domain=domain,
        forecasting=ForecastingSpec(t_c=0.5, t_low=0.0, t_high=0.5),
        offline_imputation=OfflineImputationSpec(
            held_out_signals=["vital_b"],
            held_out_intervals={"vital_b": (0.3, 0.7)},
        ),
        causal_imputation=CausalImputationSpec(
            held_out_signals=["vital_b"],
            t_failure={"vital_b": 0.3},
            t_present=0.7,
        ),
        cross_modal=CrossModalSpec(
            source_signals=["vital_a"], target_signals=["vital_b"]
        ),
    )


def _write_sample(split_dir: Path, name: str = "sample_001.parquet") -> None:
    """One synthetic sample: a static row + two scalar signals over [0, 1)."""
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = [(-1.0, 0, 0, [0.0])]  # static
    t_grid = np.linspace(0.0, 0.9, 10, dtype=np.float32)
    for s in (1, 2):
        for t in t_grid:
            rows.append((float(t), s, 0, [float(np.sin(t + s))]))
    df = pd.DataFrame(rows, columns=["t", "s", "j", "x"])
    pq.write_table(
        pa.Table.from_pandas(df, schema=_OBS_SCHEMA, preserve_index=False),
        split_dir / name,
    )


class _EchoModel(Model):
    """Predicts every query row (copies its value): exercises the predict path."""

    name = "echo"

    def predict(self, context, query, spec):
        out = query[["t", "s", "j"]].copy()
        out["x_pred"] = [np.asarray(v, dtype=np.float32) for v in query["x"]]
        return out


class _EmptyModel(Model):
    """Predicts nothing: every row falls back to the training-mean penalty."""

    name = "empty"

    def predict(self, context, query, spec):
        return pd.DataFrame(columns=["t", "s", "j", "x_pred"])


def _run(model: Model, tmp_path: Path):
    from models.harness import run_model

    spec = _make_spec(tmp_path)
    split_dir = tmp_path / "test"
    _write_sample(split_dir)
    out_root = tmp_path / "predictions"
    run_model(model, spec, split_dir=split_dir, output_root=out_root)
    return spec, split_dir, out_root


def test_harness_writes_a_file_for_every_task(tmp_path: Path) -> None:
    _, _, out_root = _run(_EchoModel(), tmp_path)
    for task in VALID_TASKS:
        f = out_root / "fusion" / task / "sample_001.parquet"
        assert f.exists(), f"missing prediction file for task {task}"


def test_harness_scores_all_four_tasks(tmp_path: Path) -> None:
    spec, split_dir, out_root = _run(_EchoModel(), tmp_path)
    result = run_benchmark(
        predictions_root=out_root,
        domain_specs={"fusion": spec},
        split_dirs={"fusion": split_dir},
    )
    domain = result["per_domain"][0]
    assert domain["n_tasks"] == 4
    assert np.isfinite(result["nmae"]) and np.isfinite(result["nrmse"])


def test_empty_model_still_scored_on_every_task(tmp_path: Path) -> None:
    # A model that predicts nothing must still be scored on all four tasks
    # (mean penalty), not silently skipped.
    spec, split_dir, out_root = _run(_EmptyModel(), tmp_path)
    result = run_benchmark(
        predictions_root=out_root,
        domain_specs={"fusion": spec},
        split_dirs={"fusion": split_dir},
    )
    domain = result["per_domain"][0]
    assert domain["n_tasks"] == 4
    assert np.isfinite(result["nmae"])


def test_predict_output_schema(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    split_dir = tmp_path / "test"
    _write_sample(split_dir)
    from benchmark.data import load_sample
    from benchmark.tasks import partition_forecasting

    obs = load_sample(split_dir / "sample_001.parquet")
    assert spec.forecasting is not None
    context, query = partition_forecasting(obs, spec.forecasting)
    pred = _EchoModel().predict(context, query, spec)
    assert list(pred.columns) == ["t", "s", "j", "x_pred"]
    assert len(pred) == len(query)


def test_tasks_subset_only_runs_selected(tmp_path: Path) -> None:
    from models.harness import run_model

    spec = _make_spec(tmp_path)
    split_dir = tmp_path / "test"
    _write_sample(split_dir)
    out_root = tmp_path / "predictions"
    run_model(
        _EchoModel(),
        spec,
        split_dir=split_dir,
        output_root=out_root,
        tasks=["forecasting"],
    )
    assert (out_root / "fusion" / "forecasting" / "sample_001.parquet").exists()
    assert not (out_root / "fusion" / "cross_modal").exists()

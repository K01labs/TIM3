"""Golden regression test for the scoring pipeline.

Pins the exact NMAE/NRMSE the benchmark produces on a frozen, fully
deterministic synthetic domain scored with a deterministic model. If anything in
the scoring path changes the numbers — `benchmark/metrics.py` (per-observation
errors, hierarchical aggregation), `benchmark/tasks.py` (partitions), the
mean-penalty for unmatched rows, or `models/harness.py` — these assertions fail.

This is the behavioural half of "the benchmark code stays the same and runs
properly": the integrity (CODEOWNERS + the CI path-guard) stops unauthorised
edits; this test catches any edit that would shift the scores.

No randomness, no network, no real data — safe to run anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
from models.harness import run_model

_OBS_SCHEMA = pa.schema(
    [
        ("t", pa.float32()),
        ("s", pa.int32()),
        ("j", pa.int32()),
        ("x", pa.list_(pa.float32())),
    ]
)

_T_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _make_spec(tmp_path: Path) -> BenchmarkSpec:
    """Two scalar signals, unit training stats, all four task specs populated."""
    registry = {
        "0": {"name": "static", "category": "static"},
        "1": {"name": "sig_a", "category": "scalar_single"},
        "2": {"name": "sig_b", "category": "scalar_single"},
    }
    train_stats = pd.DataFrame(
        {
            "s": np.array([1, 2], dtype=np.int32),
            "j": np.array([0, 0], dtype=np.int32),
            "m": np.array([0, 0], dtype=np.int32),
            "mean": np.array([0.0, 0.0], dtype=np.float32),
            "std": np.array([2.0, 2.0], dtype=np.float32),
        }
    )
    domain = DomainSpec(
        name="fusion",
        data_root=tmp_path,
        registry=registry,
        train_stats=train_stats,
    )
    return BenchmarkSpec(
        domain=domain,
        forecasting=ForecastingSpec(t_c=0.5, t_low=0.0, t_high=0.5),
        offline_imputation=OfflineImputationSpec(
            held_out_signals=["sig_b"], held_out_intervals={"sig_b": (0.3, 0.7)}
        ),
        causal_imputation=CausalImputationSpec(
            held_out_signals=["sig_b"], t_failure={"sig_b": 0.3}, t_present=0.7
        ),
        cross_modal=CrossModalSpec(source_signals=["sig_a"], target_signals=["sig_b"]),
    )


def _write_sample(split_dir: Path, k: int) -> None:
    """Deterministic sample k: value = signal + 0.1*index + k (no randomness)."""
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = [(-1.0, 0, 0, [0.0])]  # static
    for s in (1, 2):
        for i, t in enumerate(_T_GRID):
            rows.append((float(t), s, 0, [float(s) + 0.1 * i + float(k)]))
    df = pd.DataFrame(rows, columns=["t", "s", "j", "x"])
    pq.write_table(
        pa.Table.from_pandas(df, schema=_OBS_SCHEMA, preserve_index=False),
        split_dir / f"sample_{k:03d}.parquet",
    )


class _ConstantModel(Model):
    """Predicts a fixed value for every query row (deterministic, full coverage)."""

    name = "constant"

    def predict(self, context, query, spec):
        out = query[["t", "s", "j"]].copy()
        out["x_pred"] = [np.array([1.0], dtype=np.float32) for _ in range(len(query))]
        return out


def test_golden_scores(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    split_dir = tmp_path / "test"
    for k in range(2):
        _write_sample(split_dir, k)
    out_root = tmp_path / "predictions"
    run_model(_ConstantModel(), spec, split_dir=split_dir, output_root=out_root)

    result = run_benchmark(
        predictions_root=out_root,
        domain_specs={"fusion": spec},
        split_dirs={"fusion": split_dir},
    )

    # Frozen reference values (compared at 6 dp: any real scoring change shifts
    # these by far more, while float-reordering noise stays below the rounding).
    def r6(x: float) -> float:
        return round(float(x), 6)

    assert r6(result["nmae"]) == GOLDEN_OVERALL["nmae"]
    assert r6(result["nrmse"]) == GOLDEN_OVERALL["nrmse"]
    per_task = {t["task"]: t for t in result["per_domain"][0]["per_task"]}
    for task, ref in GOLDEN_PER_TASK.items():
        assert r6(per_task[task]["nmae"]) == ref["nmae"], task
        assert r6(per_task[task]["nrmse"]) == ref["nrmse"], task


# Reference scores, pinned from a known-good run (see module docstring).
GOLDEN_OVERALL: dict[str, float] = {"nmae": 0.956250, "nrmse": 0.961315}
GOLDEN_PER_TASK: dict[str, dict[str, float]] = {
    "forecasting": {"nmae": 0.850000, "nrmse": 0.853699},
    "offline_imputation": {"nmae": 1.000000, "nrmse": 1.002662},
    "causal_imputation": {"nmae": 1.000000, "nrmse": 1.002662},
    "cross_modal": {"nmae": 0.975000, "nrmse": 0.986238},
}

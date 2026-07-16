"""Tests for benchmark.run.run_benchmark.

All tests use synthetic data only — no real Parquet files, no network access.
evaluate_split is mocked so tests do not depend on real observation data.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from benchmark.run import run_benchmark
from benchmark.specs import BenchmarkSpec, DomainSpec, ForecastingSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_split_result(task: str, domain: str) -> dict:
    return {
        "task": task,
        "domain": domain,
        "nmae": 0.5,
        "nrmse": 0.7,
        "n_samples": 3,
        "n_skipped": 0,
        "per_sample": [],
    }


def make_pred_dir(root: Path, domain: str, task: str) -> Path:
    d = root / domain / task
    d.mkdir(parents=True)
    (d / "sample_001.parquet").touch()
    return d


def make_split_dir(root: Path, domain: str) -> Path:
    d = root / "data" / domain
    d.mkdir(parents=True)
    return d


def make_spec(domain_name: str = "fusion") -> BenchmarkSpec:
    registry = {"1": {"name": "sig_1"}}
    train_stats = pd.DataFrame(
        {
            "s": np.array([1], dtype=np.int32),
            "j": np.array([0], dtype=np.int32),
            "m": np.array([0], dtype=np.int32),
            "mean": np.array([0.0], dtype=np.float32),
            "std": np.array([1.0], dtype=np.float32),
        }
    )
    domain = DomainSpec(
        name=domain_name,
        data_root=Path("/tmp"),
        registry=registry,
        train_stats=train_stats,
    )
    return BenchmarkSpec(
        domain=domain,
        forecasting=ForecastingSpec(t_c=0.2, t_low=0.0, t_high=0.1),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_benchmark_all_tasks_all_domains(tmp_path: Path) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()

    all_tasks = [
        "forecasting",
        "offline_imputation",
        "causal_imputation",
        "cross_modal",
    ]
    domains = ["fusion", "tcprimed"]

    for domain in domains:
        for task in all_tasks:
            make_pred_dir(pred_root, domain, task)

    split_dirs = {d: make_split_dir(tmp_path, d) for d in domains}
    domain_specs = {d: make_spec(d) for d in domains}

    def mock_evaluate_split(split_dir, pred_dir, task_name, spec):
        return _mock_split_result(task_name, spec.domain.name)

    with patch("benchmark.run.evaluate_split", side_effect=mock_evaluate_split):
        result = run_benchmark(pred_root, domain_specs, split_dirs)

    assert "nmae" in result
    assert "nrmse" in result
    assert "n_domains" in result
    assert result["n_domains"] == 2
    assert len(result["per_domain"]) == 2

    for domain_result in result["per_domain"]:
        assert "domain" in domain_result
        assert "tasks_run" in domain_result
        assert "tasks_skipped" in domain_result
        assert len(domain_result["tasks_run"]) == 4
        assert domain_result["tasks_skipped"] == []

    assert math.isfinite(result["nmae"])


def test_run_benchmark_partial_tasks(tmp_path: Path) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()

    make_pred_dir(pred_root, "fusion", "forecasting")
    split_dirs = {"fusion": make_split_dir(tmp_path, "fusion")}
    domain_specs = {"fusion": make_spec("fusion")}

    def mock_evaluate_split(split_dir, pred_dir, task_name, spec):
        return _mock_split_result(task_name, spec.domain.name)

    with patch("benchmark.run.evaluate_split", side_effect=mock_evaluate_split):
        result = run_benchmark(pred_root, domain_specs, split_dirs)

    domain = result["per_domain"][0]
    assert domain["tasks_run"] == ["forecasting"]
    assert len(domain["tasks_skipped"]) == 3
    assert math.isfinite(result["nmae"])


def test_run_benchmark_predictions_root_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_benchmark(
            tmp_path / "nonexistent",
            {"fusion": make_spec()},
            {"fusion": make_split_dir(tmp_path, "fusion")},
        )


def test_run_benchmark_split_dir_missing(tmp_path: Path) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()

    with pytest.raises(FileNotFoundError):
        run_benchmark(
            pred_root,
            {"fusion": make_spec()},
            {"fusion": tmp_path / "nonexistent_split"},
        )


def test_run_benchmark_mismatched_keys(tmp_path: Path) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()

    with pytest.raises(ValueError):
        run_benchmark(
            pred_root,
            {"fusion": make_spec("fusion")},
            {"tcprimed": make_split_dir(tmp_path, "tcprimed")},
        )


def test_run_benchmark_empty_domain_specs(tmp_path: Path) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()

    with pytest.raises(ValueError):
        run_benchmark(pred_root, {}, {})


def test_run_benchmark_invalid_task(tmp_path: Path) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    split_dirs = {"fusion": make_split_dir(tmp_path, "fusion")}

    with pytest.raises(ValueError):
        run_benchmark(
            pred_root,
            {"fusion": make_spec()},
            split_dirs,
            tasks=["forecasting", "not_a_task"],
        )


def test_run_benchmark_no_predictions_for_domain(tmp_path: Path) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    (pred_root / "fusion").mkdir()  # domain dir exists but no task subdirs
    split_dirs = {"fusion": make_split_dir(tmp_path, "fusion")}

    with pytest.raises(ValueError):
        run_benchmark(pred_root, {"fusion": make_spec()}, split_dirs)


def test_run_benchmark_domain_name_in_result(tmp_path: Path) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    make_pred_dir(pred_root, "fusion", "forecasting")
    split_dirs = {"fusion": make_split_dir(tmp_path, "fusion")}

    def mock_evaluate_split(split_dir, pred_dir, task_name, spec):
        return _mock_split_result(task_name, spec.domain.name)

    with patch("benchmark.run.evaluate_split", side_effect=mock_evaluate_split):
        result = run_benchmark(
            pred_root,
            {"fusion": make_spec("fusion")},
            split_dirs,
            tasks=["forecasting"],
        )

    assert result["per_domain"][0]["domain"] == "fusion"


def test_run_benchmark_tasks_filter(tmp_path: Path) -> None:
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    make_pred_dir(pred_root, "fusion", "forecasting")
    make_pred_dir(pred_root, "fusion", "offline_imputation")
    split_dirs = {"fusion": make_split_dir(tmp_path, "fusion")}

    def mock_evaluate_split(split_dir, pred_dir, task_name, spec):
        return _mock_split_result(task_name, spec.domain.name)

    with patch("benchmark.run.evaluate_split", side_effect=mock_evaluate_split):
        result = run_benchmark(
            pred_root,
            {"fusion": make_spec("fusion")},
            split_dirs,
            tasks=["forecasting"],
        )

    domain = result["per_domain"][0]
    assert domain["tasks_run"] == ["forecasting"]
    assert "offline_imputation" in domain["tasks_skipped"]

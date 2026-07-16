"""Validation tests for the MIMIC-IV healthcare dataloader.

Most tests use tiny synthetic CSVs under tests/fixtures/mimic_synthetic/;
the full-MIMIC integration test is skipped unless the real dataset is
available at ~/data/mimic-4/physionet.org/files/mimiciv/3.1/.

Run with:
    uv run pytest tests/test_healthcare_dataset.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# Import surface
# ---------------------------------------------------------------------------


def test_module_exports_constants() -> None:
    from data.dataloaders import healthcare_dataset as hd

    assert isinstance(hd.SIGNALS, dict)
    assert set(hd.SIGNALS.keys()) == {1, 2, 3, 4, 5, 6}
    assert hd.MIN_STAY_HOURS == 24
    assert hd.SPLIT_RATIOS == (0.70, 0.15, 0.15)
    assert hd.RANDOM_SEED == 42

    schema = hd.PARQUET_SCHEMA
    assert schema.field("t").type.equals(pa.float32())
    assert schema.field("s").type.equals(pa.int32())
    assert schema.field("j").type.equals(pa.int32())

    # Signal 4 = blood_pressure should have 3 channels (sbp, dbp, map)
    assert len(hd.SIGNALS[4]["channels"]) == 3
    # Signal 6 = gcs should have 3 channels (eye, verbal, motor)
    assert len(hd.SIGNALS[6]["channels"]) == 3
    # Signal 5 = temperature — both F and C itemids mapped
    temp_itemids = hd.SIGNALS[5]["channels"][0][1]
    assert 223761 in temp_itemids and 223762 in temp_itemids


def test_build_itemid_index() -> None:
    from data.dataloaders.healthcare_dataset import build_itemid_index

    idx = build_itemid_index()
    # Heart rate
    assert idx[220045] == (1, 0)
    # SpO2
    assert idx[220277] == (2, 0)
    # Blood pressure channels
    assert idx[220179] == (4, 0)  # sbp
    assert idx[220180] == (4, 1)  # dbp
    assert idx[220181] == (4, 2)  # map
    # Temperature (both F and C map to the same (s, j))
    assert idx[223761] == (5, 0)
    assert idx[223762] == (5, 0)
    # GCS
    assert idx[220739] == (6, 0)
    assert idx[223900] == (6, 1)
    assert idx[223901] == (6, 2)
    # Unknown itemid must not appear
    assert 999999 not in idx


# ---------------------------------------------------------------------------
# Synthetic fixture tests (Task 4)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "mimic_synthetic"


def test_fixture_files_exist() -> None:
    assert (FIX / "icustays.csv").exists()
    assert (FIX / "chartevents.csv").exists()


def test_fixture_icustays_structure() -> None:
    df = pd.read_csv(FIX / "icustays.csv")
    # Must have the columns we actually use.
    assert {"subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"}.issubset(
        df.columns
    )
    # At least 3 stays: one >24h, one <24h (to test filter), one edge case.
    assert len(df) >= 3


def test_load_icustays_filters_short_stays(tmp_path: Path) -> None:
    from data.dataloaders.healthcare_dataset import load_icustays

    icu_path = FIX / "icustays.csv"
    df = load_icustays(icu_path, min_hours=24)

    # Only 30001 (36h) and 30003 (30h) should pass.
    assert set(df["stay_id"]) == {30001, 30003}
    # Returned frame must carry the columns downstream needs.
    assert {"stay_id", "subject_id", "hadm_id", "intime"}.issubset(df.columns)
    # intime must be parsed as a datetime.
    assert pd.api.types.is_datetime64_any_dtype(df["intime"])


def test_load_icustays_missing_file_raises(tmp_path: Path) -> None:
    from data.dataloaders.healthcare_dataset import load_icustays

    with pytest.raises(FileNotFoundError):
        load_icustays(tmp_path / "nope.csv")


def test_normalize_temperature_converts_fahrenheit() -> None:
    from data.dataloaders.healthcare_dataset import normalize_temperature

    df = pd.DataFrame(
        {"itemid": [223761, 223762, 220045], "valuenum": [98.6, 37.0, 85.0]}
    )
    out = normalize_temperature(df)
    # 98.6 F -> 37.0 C
    assert np.isclose(
        out.loc[out["itemid"] == 223761, "valuenum"].iloc[0], 37.0, atol=0.05
    )
    # 223762 (already Celsius) unchanged
    assert np.isclose(out.loc[out["itemid"] == 223762, "valuenum"].iloc[0], 37.0)
    # Non-temperature rows unchanged
    assert out.loc[out["itemid"] == 220045, "valuenum"].iloc[0] == 85.0


def test_clamp_physio_drops_out_of_range() -> None:
    from data.dataloaders.healthcare_dataset import clamp_physio, build_itemid_index

    idx = build_itemid_index()
    df = pd.DataFrame(
        {
            "itemid": [220045, 220045, 220277, 220739, 220739, 223901],
            "valuenum": [85.0, 9999.0, 150.0, 3.0, 5.0, 6.0],
        }
    )
    out = clamp_physio(df, idx)
    # 9999 bpm dropped; 150% SpO2 dropped; gcs_eye=5 dropped (range 1-4); others kept.
    kept = list(zip(out["itemid"].tolist(), out["valuenum"].tolist()))
    assert (220045, 85.0) in kept
    assert (220045, 9999.0) not in kept
    assert (220277, 150.0) not in kept
    assert (220739, 3.0) in kept
    assert (220739, 5.0) not in kept
    assert (223901, 6.0) in kept


def test_load_chartevents_for_stays_filters_by_stay_and_itemid() -> None:
    from data.dataloaders.healthcare_dataset import (
        build_itemid_index,
        load_chartevents_for_stays,
    )

    idx = build_itemid_index()
    wanted_stays = {30001, 30003}  # drop 30002
    df = load_chartevents_for_stays(
        FIX / "chartevents.csv",
        stay_ids=wanted_stays,
        itemid_index=idx,
        chunksize=4,  # force multi-chunk path
    )
    # All kept rows belong to wanted stays.
    assert set(df["stay_id"].unique()) <= wanted_stays
    # Stay 30002's HR row must be dropped.
    assert not (df["stay_id"] == 30002).any()
    # charttime parsed as datetime.
    assert pd.api.types.is_datetime64_any_dtype(df["charttime"])
    # All itemids are in the known registry.
    assert set(df["itemid"].unique()).issubset(set(idx.keys()))


def test_build_observation_set_aligns_to_first_vital() -> None:
    from data.dataloaders.healthcare_dataset import (
        build_itemid_index,
        build_observation_set,
    )

    idx = build_itemid_index()
    # Three rows, 5 minutes apart. First is HR at 00:05.
    df = pd.DataFrame(
        {
            "stay_id": [30001, 30001, 30001],
            "charttime": pd.to_datetime(
                [
                    "2150-01-01 00:05:00",
                    "2150-01-01 00:10:00",
                    "2150-01-01 00:15:00",
                ]
            ),
            "itemid": [220045, 220277, 220179],
            "valuenum": [85.0, 98.0, 120.0],
        }
    )
    obs = build_observation_set(df, idx)
    # Schema check
    assert list(obs.columns) == ["t", "s", "j", "x"]
    # First observation has t=0
    assert float(obs["t"].iloc[0]) == 0.0
    # Second row at +5 min = 300 s
    assert float(obs["t"].iloc[1]) == pytest.approx(300.0, abs=1e-3)
    # Third row at +10 min = 600 s
    assert float(obs["t"].iloc[2]) == pytest.approx(600.0, abs=1e-3)
    # x is a length-1 list/array
    assert len(obs["x"].iloc[0]) == 1
    assert obs["x"].iloc[0][0] == pytest.approx(85.0)
    # dtypes
    assert obs["t"].dtype == np.float32
    assert obs["s"].dtype == np.int32
    assert obs["j"].dtype == np.int32


def test_build_observation_set_empty_returns_none() -> None:
    from data.dataloaders.healthcare_dataset import (
        build_itemid_index,
        build_observation_set,
    )

    idx = build_itemid_index()
    empty = pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"])
    assert build_observation_set(empty, idx) is None


def test_write_stay_parquet_roundtrip(tmp_path: Path) -> None:
    from data.dataloaders.healthcare_dataset import (
        build_itemid_index,
        build_observation_set,
        build_stay_grid,
        write_grid_parquet,
        write_stay_parquet,
    )

    idx = build_itemid_index()
    events = pd.DataFrame(
        {
            "stay_id": [30001, 30001, 30001],
            "charttime": pd.to_datetime(
                [
                    "2150-01-01 00:05:00",
                    "2150-01-01 00:10:00",
                    "2150-01-01 00:15:00",
                ]
            ),
            "itemid": [220045, 220277, 220045],
            "valuenum": [85.0, 98.0, 90.0],
        }
    )
    obs = build_observation_set(events, idx)
    assert obs is not None
    grid = build_stay_grid(obs)

    obs_path = tmp_path / "stay_30001.parquet"
    grid_path = tmp_path / "stay_30001_grid.parquet"
    write_stay_parquet(obs, obs_path)
    write_grid_parquet(grid, grid_path)

    # Roundtrip obs
    loaded = pd.read_parquet(obs_path)
    assert set(loaded.columns) == {"t", "s", "j", "x"}
    # Parquet file uses the canonical schema.
    pq_schema = pq.read_schema(obs_path)
    assert pq_schema.field("t").type.equals(pa.float32())
    assert pq_schema.field("x").type.equals(pa.list_(pa.float32()))

    # Grid: one row per (s, t) at which ANY channel of s was recorded.
    grid_loaded = pd.read_parquet(grid_path)
    assert set(grid_loaded.columns) == {"s", "t"}
    # s=1 (HR) appears twice in events -> grid should have 2 HR rows.
    assert (grid_loaded["s"] == 1).sum() == 2


def test_split_stays_ratios_and_determinism() -> None:
    from data.dataloaders.healthcare_dataset import split_stays

    stay_ids = list(range(100))
    a = split_stays(stay_ids, ratios=(0.7, 0.15, 0.15), seed=42)
    b = split_stays(stay_ids, ratios=(0.7, 0.15, 0.15), seed=42)

    assert a == b  # determinism
    assert set(a["train"] + a["val"] + a["test"]) == set(stay_ids)
    assert 65 <= len(a["train"]) <= 75
    assert 10 <= len(a["val"]) <= 20
    assert 10 <= len(a["test"]) <= 20
    # No overlap.
    assert set(a["train"]).isdisjoint(a["val"])
    assert set(a["train"]).isdisjoint(a["test"])
    assert set(a["val"]).isdisjoint(a["test"])


def test_build_signals_registry_shape() -> None:
    from data.dataloaders.healthcare_dataset import build_signals_registry

    reg = build_signals_registry()
    assert set(reg.keys()) == {"1", "2", "3", "4", "5", "6"}  # stringified IDs
    assert reg["1"]["name"] == "heart_rate"
    assert reg["4"]["n_channels"] == 3
    assert reg["6"]["n_channels"] == 3
    # Every entry carries the fields downstream tooling reads.
    for sid, entry in reg.items():
        assert {"name", "category", "unit", "n_channels", "value_dim"}.issubset(entry)
        assert entry["value_dim"] == 1  # MIMIC vitals are scalar


def test_compute_and_write_train_stats(tmp_path: Path) -> None:
    from data.dataloaders.healthcare_dataset import compute_and_write_train_stats

    # Build a minimal train/ dir with one synthetic parquet.
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (tmp_path / "splits.json").write_text(json.dumps({"train": ["stay_1"]}))
    (tmp_path / "signals.json").write_text(json.dumps({"1": {"name": "hr"}}))

    obs = pd.DataFrame(
        {
            "t": np.array([0.0, 60.0, 120.0], dtype=np.float32),
            "s": np.array([1, 1, 1], dtype=np.int32),
            "j": np.array([0, 0, 0], dtype=np.int32),
            "x": [
                np.array([80.0], dtype=np.float32),
                np.array([90.0], dtype=np.float32),
                np.array([100.0], dtype=np.float32),
            ],
        }
    )
    from data.dataloaders.healthcare_dataset import PARQUET_SCHEMA

    table = pa.Table.from_pandas(obs, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, train_dir / "stay_1.parquet")

    compute_and_write_train_stats(tmp_path)

    stats = pd.read_parquet(tmp_path / "train_stats.parquet")
    row = stats[(stats["s"] == 1) & (stats["j"] == 0) & (stats["m"] == 0)]
    assert len(row) == 1
    assert row["mean"].iloc[0] == pytest.approx(90.0, abs=1e-4)
    # std of {80,90,100} (population) = sqrt(200/3) ≈ 8.1650
    assert row["std"].iloc[0] == pytest.approx(np.sqrt(200.0 / 3.0), abs=1e-3)


def test_main_end_to_end_on_synthetic(tmp_path: Path, monkeypatch) -> None:
    """Run the full pipeline on the synthetic MIMIC fixtures."""
    from data.dataloaders import healthcare_dataset as hd

    # Point the script at the fixture dir rather than the real MIMIC base.
    out = tmp_path / "healthcare_data"
    monkeypatch.setattr(
        "sys.argv",
        [
            "healthcare_dataset.py",
            "--mimic_base",
            str(FIX),
            "--output_dir",
            str(out),
            "--min_stay_hours",
            "24",
        ],
    )
    hd.main()

    # Expected artefacts
    assert (out / "signals.json").exists()
    assert (out / "splits.json").exists()
    assert (out / "train_stats.parquet").exists()
    for split in ("train", "val", "test"):
        assert (out / split).is_dir()

    splits = json.loads((out / "splits.json").read_text())
    all_stays = splits["train"] + splits["val"] + splits["test"]
    # Only stays 30001 and 30003 passed the 24h filter; 30002 dropped.
    assert {"stay_30001", "stay_30003"} == set(all_stays)

    # Every listed stay has both an observation file and a grid file.
    for split, labels in splits.items():
        for label in labels:
            assert (out / split / f"{label}.parquet").exists()
            assert (out / split / f"{label}_grid.parquet").exists()

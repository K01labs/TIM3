"""Convert raw MIMIC-IV ICU data into benchmark observation-set Parquet.

Each ICU stay is converted to an observation-set DataFrame with columns:
    t (float32)        — seconds since stay start (T0 = first vital)
    s (int32)          — signal type ID (1..6)
    j (int32)          — channel index within that signal type
    x (list[float32])  — value array (always length 1 for MIMIC vitals)

Usage:
    uv run python data/dataloaders/healthcare_dataset.py --mimic_base PATH
        [--output_dir PATH] [--limit N] [--stats_only]

``--mimic_base`` is required and must point at your own credentialed MIMIC-IV
v3.1 download (the data cannot be redistributed). Every ICU stay that satisfies
MIN_STAY_HOURS is processed and written to data/healthcare_data/. See
data/dataloaders/HEALTHCARE.md for the full walkthrough.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Signal registry. Keys are integer signal IDs (1-indexed; s=0 reserved for
# static observations in the benchmark spec but unused for MIMIC v1).
# Channels are (name, [itemid, ...]) tuples; j is the 0-based channel index.
SIGNALS: dict[int, dict] = {
    1: {
        "name": "heart_rate",
        "category": "vital",
        "unit": "bpm",
        "channels": [("hr", [220045])],
        "range": (0.0, 300.0),
    },
    2: {
        "name": "spo2",
        "category": "vital",
        "unit": "%",
        "channels": [("spo2", [220277])],
        "range": (0.0, 100.0),
    },
    3: {
        "name": "respiratory_rate",
        "category": "vital",
        "unit": "breaths/min",
        "channels": [("rr", [220210])],
        "range": (0.0, 80.0),
    },
    4: {
        "name": "blood_pressure",
        "category": "vital",
        "unit": "mmHg",
        "channels": [
            ("sbp", [220179]),
            ("dbp", [220180]),
            ("map", [220181]),
        ],
        "range": (0.0, 300.0),
    },
    5: {
        "name": "temperature",
        "category": "vital",
        "unit": "C",
        # 223761 = Fahrenheit (converted to C in normalize_temperature);
        # 223762 = Celsius.
        "channels": [("temp", [223761, 223762])],
        "range": (25.0, 45.0),
    },
    6: {
        "name": "gcs",
        "category": "neuro",
        "unit": "score",
        "channels": [
            ("gcs_eye", [220739]),
            ("gcs_verbal", [223900]),
            ("gcs_motor", [223901]),
        ],
        # GCS ranges differ per channel; see clamp_physio.
        "range_per_channel": [(1.0, 4.0), (1.0, 5.0), (1.0, 6.0)],
    },
}

# Itemid values stored in Fahrenheit (must be converted to C before clamp).
FAHRENHEIT_ITEMIDS: frozenset[int] = frozenset({223761})

# Pipeline parameters.
MIN_STAY_HOURS: int = 24
SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
RANDOM_SEED: int = 42
CHARTEVENTS_CHUNKSIZE: int = 1_000_000  # rows per pandas chunk

# PyArrow schema for observation-set Parquet files.
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("t", pa.float32()),
        pa.field("s", pa.int32()),
        pa.field("j", pa.int32()),
        pa.field("x", pa.list_(pa.float32())),
    ]
)

# Per-shot/stay grid schema (mirrors fusion/climate).
GRID_SCHEMA = pa.schema(
    [
        pa.field("s", pa.int32()),
        pa.field("t", pa.float32()),
    ]
)

# ---------------------------------------------------------------------------
# Itemid -> (s, j) lookup
# ---------------------------------------------------------------------------


def build_itemid_index() -> dict[int, tuple[int, int]]:
    """Return a mapping from MIMIC itemid -> (signal_id, channel_index).

    Built once per run and reused for every chartevents chunk.
    """
    idx: dict[int, tuple[int, int]] = {}
    for s, info in SIGNALS.items():
        for j, (_name, itemids) in enumerate(info["channels"]):
            for itemid in itemids:
                if itemid in idx:
                    raise ValueError(f"Duplicate itemid {itemid} in SIGNALS registry")
                idx[itemid] = (s, j)
    return idx


# ---------------------------------------------------------------------------
# ICU stay manifest
# ---------------------------------------------------------------------------


def load_icustays(path: Path, min_hours: int = MIN_STAY_HOURS) -> pd.DataFrame:
    """Load the ICU stays manifest, filter by minimum stay length.

    Args:
        path: Path to ``icustays.csv`` or ``icustays.csv.gz`` (MIMIC-IV icu).
        min_hours: Stays shorter than this are dropped.

    Returns:
        DataFrame with at least ``stay_id, subject_id, hadm_id, intime`` columns.
        ``intime`` is parsed to datetime64.
    """
    if not path.exists():
        raise FileNotFoundError(f"icustays file not found: {path}")
    df = pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )
    df = df[df["los"] * 24.0 >= float(min_hours)].copy()
    df["stay_id"] = df["stay_id"].astype(np.int64)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Value normalization and physiological clamping
# ---------------------------------------------------------------------------


def normalize_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Fahrenheit temperature rows to Celsius in-place.

    Rows whose ``itemid`` is in FAHRENHEIT_ITEMIDS have ``valuenum`` rewritten
    as (valuenum - 32) * 5/9.  Other rows are untouched.
    """
    mask = df["itemid"].isin(FAHRENHEIT_ITEMIDS)
    if mask.any():
        df = df.copy()
        df.loc[mask, "valuenum"] = (df.loc[mask, "valuenum"] - 32.0) * (5.0 / 9.0)
    return df


def clamp_physio(
    df: pd.DataFrame, itemid_index: dict[int, tuple[int, int]]
) -> pd.DataFrame:
    """Drop rows whose ``valuenum`` is outside the per-signal physiological range.

    For signals with per-channel ranges (only GCS), the range for channel j
    is used; for all others the single ``range`` tuple from SIGNALS applies.
    Rows whose itemid is not in the registry are also dropped.
    """
    if df.empty:
        return df

    # Build aligned (lo, hi) arrays indexed by df row.
    pairs = df["itemid"].map(itemid_index)
    mask_known = pairs.notna()
    df = df.loc[mask_known].copy()
    pairs = pairs.loc[mask_known]

    lo = np.empty(len(df), dtype=np.float64)
    hi = np.empty(len(df), dtype=np.float64)
    for row_pos, (s, j) in enumerate(pairs):
        info = SIGNALS[s]
        if "range_per_channel" in info:
            lo_v, hi_v = info["range_per_channel"][j]
        else:
            lo_v, hi_v = info["range"]
        lo[row_pos] = lo_v
        hi[row_pos] = hi_v

    vals = df["valuenum"].to_numpy(dtype=np.float64)
    keep = np.isfinite(vals) & (vals >= lo) & (vals <= hi)
    return df.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# chartevents streaming reader
# ---------------------------------------------------------------------------


def load_chartevents_for_stays(
    path: Path,
    stay_ids: set[int],
    itemid_index: dict[int, tuple[int, int]],
    chunksize: int = CHARTEVENTS_CHUNKSIZE,
) -> pd.DataFrame:
    """Stream chartevents in chunks, keep only rows matching wanted stays/itemids.

    Args:
        path: Path to ``chartevents.csv`` or ``chartevents.csv.gz``.
        stay_ids: The set of stay_ids to retain.
        itemid_index: Itemid -> (s, j) map; rows with itemid not in the keys
            are discarded.
        chunksize: pandas read_csv chunksize (rows).

    Returns:
        DataFrame with columns ``stay_id, charttime, itemid, valuenum``
        (sorted by charttime within each stay).
    """
    if not path.exists():
        raise FileNotFoundError(f"chartevents file not found: {path}")

    wanted_itemids = set(itemid_index.keys())
    pieces: list[pd.DataFrame] = []

    reader = pd.read_csv(
        path,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        chunksize=chunksize,
        dtype={"stay_id": "Int64", "itemid": "Int64", "valuenum": "Float64"},
    )
    for chunk in reader:
        mask = chunk["stay_id"].isin(stay_ids) & chunk["itemid"].isin(wanted_itemids)
        if mask.any():
            pieces.append(chunk.loc[mask].copy())

    if not pieces:
        return pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"])

    df = pd.concat(pieces, ignore_index=True)
    df["stay_id"] = df["stay_id"].astype(np.int64)
    df["itemid"] = df["itemid"].astype(np.int64)
    df["valuenum"] = df["valuenum"].astype(np.float64)
    df = df.dropna(subset=["valuenum", "charttime"])
    df = df.sort_values(["stay_id", "charttime"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Observation-set construction
# ---------------------------------------------------------------------------


def build_observation_set(
    events: pd.DataFrame, itemid_index: dict[int, tuple[int, int]]
) -> pd.DataFrame | None:
    """Convert per-stay chartevent rows to a (t, s, j, x) observation-set DataFrame.

    T0 is the earliest charttime in ``events`` (first recorded vital for the
    stay). All timestamps are expressed in seconds since T0, so t >= 0 always.
    ``x`` is a length-1 float32 list (MIMIC vitals are scalar measurements).

    Returns None when ``events`` is empty after filtering.
    """
    if events.empty:
        return None

    s_j = events["itemid"].map(itemid_index)
    if s_j.isna().any():
        events = events.loc[s_j.notna()].copy()
        s_j = s_j.loc[s_j.notna()]
    if events.empty:
        return None

    t0 = events["charttime"].min()
    t_seconds = (events["charttime"] - t0).dt.total_seconds().astype(np.float32)
    s_vals = np.fromiter((sj[0] for sj in s_j), dtype=np.int32, count=len(events))
    j_vals = np.fromiter((sj[1] for sj in s_j), dtype=np.int32, count=len(events))
    x_vals = [np.asarray([v], dtype=np.float32) for v in events["valuenum"].to_numpy()]

    obs = pd.DataFrame(
        {
            "t": t_seconds.to_numpy(),
            "s": s_vals,
            "j": j_vals,
            "x": x_vals,
        }
    )
    return obs.sort_values(["t", "s", "j"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parquet writers + observation grid
# ---------------------------------------------------------------------------


def build_stay_grid(obs: pd.DataFrame) -> pd.DataFrame:
    """Build the (s, t) grid of times at which each signal type was recorded.

    The grid is the de-duplicated set of (signal, timestamp) pairs across all
    channels of that signal. Mirrors fusion/climate semantics.
    """
    grid = (
        obs[["s", "t"]].drop_duplicates().sort_values(["s", "t"]).reset_index(drop=True)
    )
    return grid


def write_stay_parquet(obs: pd.DataFrame, path: Path) -> None:
    """Write an observation-set DataFrame to Parquet using the canonical schema."""
    table = pa.Table.from_pandas(obs, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, path)


def write_grid_parquet(grid: pd.DataFrame, path: Path) -> None:
    """Write the observation grid to Parquet."""
    table = pa.Table.from_pandas(grid, schema=GRID_SCHEMA, preserve_index=False)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def split_stays(
    stay_ids: list[int],
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> dict[str, list[int]]:
    """Deterministically split stay IDs into train/val/test.

    The input is sorted before shuffling, so the split depends only on the set of
    stay IDs and the seed — never on the order they were passed in. This is what
    makes the published ``splits.json`` checksum reproducible.

    Args:
        stay_ids: Stay IDs to split (any order; sorted internally).
        ratios: (train, val, test) fractions; remaining fraction goes to test.
        seed: Random seed for shuffling.

    Returns:
        Dict with keys ``train``, ``val``, ``test`` and sorted integer lists.
    """
    rng = np.random.default_rng(seed)
    ids = sorted(stay_ids)
    perm = rng.permutation(len(ids))
    ordered = [ids[i] for i in perm]

    n = len(ordered)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train = sorted(ordered[:n_train])
    val = sorted(ordered[n_train : n_train + n_val])
    test = sorted(ordered[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Signal registry
# ---------------------------------------------------------------------------


def build_signals_registry() -> dict[str, dict]:
    """Return the signals.json registry as a {str(signal_id): metadata} mapping.

    Keys are the integer signal IDs stringified (matches fusion/climate convention).
    The ``category`` field uses the structural taxonomy shared with fusion/climate
    (``scalar_single`` for 1-channel vitals, ``scalar_multichannel`` for >1) so the
    benchmark scalar filter sees healthcare signals. The clinical category is
    preserved separately under ``clinical_category``.
    """
    registry: dict[str, dict] = {}
    for s, info in SIGNALS.items():
        n_channels = len(info["channels"])
        category = "scalar_single" if n_channels == 1 else "scalar_multichannel"
        registry[str(s)] = {
            "name": info["name"],
            "category": category,
            "clinical_category": info["category"],
            "unit": info["unit"],
            "n_channels": n_channels,
            "value_dim": 1,  # every MIMIC vital is a scalar measurement
            "channels": [name for name, _itemids in info["channels"]],
        }
    return registry


# ---------------------------------------------------------------------------
# Training statistics
# ---------------------------------------------------------------------------


def compute_and_write_train_stats(output_dir: Path) -> None:
    """Compute per-(s, j, m) mean and std from the training split.

    Reads every training Parquet one at a time and accumulates running sums.
    MIMIC vitals are all scalar (value_dim=1), so ``m`` is always 0.
    Writes ``<output_dir>/train_stats.parquet`` with schema
    (s int32, j int32, m int32, mean float32, std float32).
    """
    splits = json.loads((output_dir / "splits.json").read_text())
    train_files = [
        output_dir / "train" / f"{label}.parquet" for label in splits["train"]
    ]

    accum: dict[tuple[int, int], dict[str, float]] = {}

    for fpath in train_files:
        if not fpath.exists():
            logger.warning("Training file not found, skipping: %s", fpath)
            continue
        table = pq.read_table(fpath)
        s_col = table.column("s").to_pylist()
        j_col = table.column("j").to_pylist()
        x_col = table.column("x").to_pylist()

        for s_val, j_val, x_list in zip(s_col, j_col, x_col):
            key = (int(s_val), int(j_val))
            v = float(x_list[0])  # scalar signal: x is length-1
            if not np.isfinite(v):
                continue
            if key not in accum:
                accum[key] = {"n": 0.0, "sum": 0.0, "sum_sq": 0.0}
            accum[key]["n"] += 1.0
            accum[key]["sum"] += v
            accum[key]["sum_sq"] += v * v

    stats_schema = pa.schema(
        [
            pa.field("s", pa.int32()),
            pa.field("j", pa.int32()),
            pa.field("m", pa.int32()),
            pa.field("mean", pa.float32()),
            pa.field("std", pa.float32()),
        ]
    )

    rows: list[dict] = []
    for (s, j), st in sorted(accum.items()):
        n = st["n"]
        if n == 0.0:
            continue
        mean = st["sum"] / n
        variance = max(st["sum_sq"] / n - mean * mean, 0.0)
        std = max(float(np.sqrt(variance)), 1e-8)
        rows.append(
            {
                "s": np.int32(s),
                "j": np.int32(j),
                "m": np.int32(0),
                "mean": np.float32(mean),
                "std": np.float32(std),
            }
        )

    stats_df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(stats_df, schema=stats_schema, preserve_index=False)
    pq.write_table(table, output_dir / "train_stats.parquet")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: ingest MIMIC-IV ICU into benchmark observation-set Parquet."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw MIMIC-IV ICU CSVs to observation-set Parquet "
            "(benchmark-native format)."
        )
    )
    parser.add_argument(
        "--mimic_base",
        type=Path,
        required=True,
        help=(
            "Directory that contains ``icu/icustays.csv(.gz)`` and "
            "``icu/chartevents.csv(.gz)`` from your credentialed MIMIC-IV v3.1 "
            "download (typically ``physionet.org/files/mimiciv/3.1``). For the "
            "synthetic fixture pass the fixture directory directly (the script "
            "falls back to flat CSVs)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/healthcare_data"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--min_stay_hours",
        type=int,
        default=MIN_STAY_HOURS,
        help="Drop ICU stays shorter than this (hours).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, process at most N stays (smoke test).",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Skip ingestion; recompute train_stats.parquet from existing splits.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.stats_only:
        compute_and_write_train_stats(args.output_dir)
        return

    t0 = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Locate input CSVs. Real MIMIC layout is <base>/icu/<file>(.csv|.csv.gz);
    # the synthetic fixture uses flat <base>/<file>.csv.
    def _resolve(name: str) -> Path:
        for candidate in (
            args.mimic_base / "icu" / f"{name}.csv.gz",
            args.mimic_base / "icu" / f"{name}.csv",
            args.mimic_base / f"{name}.csv.gz",
            args.mimic_base / f"{name}.csv",
        ):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not locate {name}.csv[.gz] under {args.mimic_base}"
        )

    icustays_path = _resolve("icustays")
    chartevents_path = _resolve("chartevents")

    print(f"Loading ICU stays from {icustays_path} ...")
    stays = load_icustays(icustays_path, min_hours=args.min_stay_hours)
    if args.limit is not None:
        stays = stays.head(args.limit).copy()
    stay_ids = set(stays["stay_id"].astype(int).tolist())
    print(f"  {len(stay_ids)} stays after min_hours={args.min_stay_hours} filter.")

    itemid_index = build_itemid_index()

    print(f"Streaming chartevents from {chartevents_path} ...")
    events = load_chartevents_for_stays(
        chartevents_path, stay_ids=stay_ids, itemid_index=itemid_index
    )
    events = normalize_temperature(events)
    events = clamp_physio(events, itemid_index)
    print(f"  {len(events):,} events after normalisation + clamp.")

    # Staging dir for per-stay parquets.
    staging = args.output_dir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    successful_stays: list[int] = []
    total_obs_rows = 0
    total_grid_rows = 0

    for sid, group in tqdm(
        events.groupby("stay_id"), desc="Building obs sets", unit="stay"
    ):
        obs = build_observation_set(group, itemid_index)
        if obs is None or obs.empty:
            continue
        grid = build_stay_grid(obs)
        write_stay_parquet(obs, staging / f"stay_{int(sid)}.parquet")
        write_grid_parquet(grid, staging / f"stay_{int(sid)}_grid.parquet")
        successful_stays.append(int(sid))
        total_obs_rows += len(obs)
        total_grid_rows += len(grid)

    # Deterministic split.
    splits = split_stays(successful_stays)
    for split in ("train", "val", "test"):
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)

    splits_json: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split, ids in splits.items():
        for sid in ids:
            label = f"stay_{sid}"
            splits_json[split].append(label)
            for suffix in ("", "_grid"):
                src = staging / f"{label}{suffix}.parquet"
                if src.exists():
                    src.rename(args.output_dir / split / f"{label}{suffix}.parquet")

    shutil.rmtree(staging, ignore_errors=True)

    (args.output_dir / "signals.json").write_text(
        json.dumps(build_signals_registry(), indent=2)
    )
    (args.output_dir / "splits.json").write_text(json.dumps(splits_json, indent=2))
    compute_and_write_train_stats(args.output_dir)

    elapsed = time.perf_counter() - t0
    print("\n=== Processing complete ===")
    print(f"  Stays       : {len(successful_stays)}")
    print(
        f"  Split sizes : train={len(splits_json['train'])} "
        f"val={len(splits_json['val'])} test={len(splits_json['test'])}"
    )
    print(f"  Obs rows    : {total_obs_rows:,}")
    print(f"  Grid rows   : {total_grid_rows:,}")
    print(f"  Elapsed     : {elapsed:.1f}s")


if __name__ == "__main__":
    main()

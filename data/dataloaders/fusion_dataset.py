"""Fetch MAST shots from S3 Zarr and convert to benchmark Parquet format.
Each shot is converted to an observation-set DataFrame with columns:
    t (float32)  — timestamp in seconds
    s (int32)    — signal type ID
    j (int32)    — channel index within that signal type
    x (list[float32]) — value array, variable length

Usage:
    uv run python data/dataloaders/fusion_dataset.py [--output_dir PATH]
        [--start_shot N] [--end_shot N] [--reference_shot ID] [--step N] [--resume]

    Run without arguments to fetch shots in the range [11766, 30471) at a step of
    4, using shot 30421 as the reference for signal discovery, and write Parquet
    files plus signals.json / splits.json / train_stats.parquet to data/fusion_data.
"""

import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S3_URL_TEMPLATE = "https://s3.echo.stfc.ac.uk/mast/level2/shots/{shot_id}.zarr"

# Signals to fetch from the MAST database, organised by instrument group.
# Follows the Tokamak benchmark signal list as reference.
SIGNALS: dict[str, list[str]] = {
    "magnetics": [
        "flux_loop_flux",
        "b_field_pol_probe_ccbv_field",
        "b_field_pol_probe_obr_field",
        "b_field_pol_probe_obv_field",
        "b_field_tor_probe_saddle_voltage",
        "b_field_tor_probe_cc_field",
        "b_field_pol_probe_omv_voltage",
    ],
    "thomson_scattering": ["t_e", "n_e"],
    "interferometer": ["n_e_line"],
    "spectrometer_visible": ["filter_spectrometer_dalpha_voltage"],
    "soft_x_rays": ["horizontal_cam_lower", "horizontal_cam_upper"],
    "pf_active": ["coil_current", "solenoid_current", "coil_voltage"],
    "pulse_schedule": ["i_plasma", "n_e_line"],
    "summary": ["ip", "power_nbi"],
    "gas_injection": ["total_injected"],
    "equilibrium": [
        "elongation",
        "elongation_axis",
        "triangularity_upper",
        "triangularity_lower",
        "x_point_r",
        "x_point_z",
        "minor_radius",
        "magnetic_axis_r",
        "magnetic_axis_z",
        "q95",
        "beta_tor",
        "beta_pol",
        "beta_tor_normal",
        "bvac_rmag",
        "bphi_rmag",
        "lcfs_r",
        "lcfs_z",
        "psi",
    ],
}

# Dimension names that indicate a spatial axis (not a channel index).
SPATIAL_DIMS: frozenset[str] = frozenset(
    {"major_radius", "n_boundary_coords", "z", "r"}
)

# PyArrow schema for the observation-set Parquet files.
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("t", pa.float32()),
        pa.field("s", pa.int32()),
        pa.field("j", pa.int32()),
        pa.field("x", pa.list_(pa.float32())),
    ]
)

# PyArrow schema for the per-shot observation grid files.
GRID_SCHEMA = pa.schema(
    [
        pa.field("s", pa.int32()),
        pa.field("t", pa.float32()),
    ]
)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_shot(
    shot_id: int,
    signals: dict[str, list[str]],
) -> xr.Dataset | None:
    """Fetch a single shot from the MAST S3 Zarr store.

    Opens each instrument group via ``xr.open_zarr``, extracts the requested
    variables, renames every time-like dimension to ``time_{group}_{variable}``
    so that each signal has its own unique time base, and returns a single
    in-memory Dataset.

    Args:
        shot_id: The shot ID to fetch.
        signals: Mapping from instrument group name to list of variable names.

    Returns:
        An ``xr.Dataset`` with all successfully fetched variables loaded into
        memory, or ``None`` if no variables could be retrieved.
    """
    raw_shot_ds = xr.Dataset()
    url = S3_URL_TEMPLATE.format(shot_id=shot_id)

    for group, variables in signals.items():
        try:
            profiles = xr.open_zarr(url, group=group)
            for value in variables:
                if value not in profiles:
                    continue
                var_name = f"{group}_{value}"
                data_array = profiles[value]

                # Rename time-like and spatial dimensions to unique per-signal
                # names so that each signal has its own independent axes.
                # Without this, xarray aligns on shared dimension names
                # (e.g. major_radius) when merging groups with different
                # coordinate sizes, inflating data with NaN.
                rename_map: dict[str, str] = {}
                for dim in list(data_array.dims):
                    if dim == "time" or dim.startswith("time_"):
                        rename_map[dim] = f"time_{group}_{value}"
                    elif dim in SPATIAL_DIMS:
                        rename_map[dim] = f"{dim}_{group}_{value}"
                if rename_map:
                    data_array = data_array.rename(rename_map)

                raw_shot_ds = raw_shot_ds.assign({var_name: data_array})
        except Exception:
            logger.debug("Failed to fetch group %s for shot %d", group, shot_id)

    if len(raw_shot_ds.data_vars) == 0:
        return None

    # Materialise lazy Zarr arrays into memory so the S3 connection can close.
    raw_shot_ds.load()
    return raw_shot_ds


# ---------------------------------------------------------------------------
# Signal classification helpers
# ---------------------------------------------------------------------------


def get_time_dim(da: xr.DataArray) -> str | None:
    """Return the dimension whose name starts with 'time_', or None.

    Args:
        da: The DataArray to inspect.

    Returns:
        The name of the time dimension, or None if not found.
    """
    for dim in da.dims:
        if dim.startswith("time_"):
            return dim
    return None


def _is_spatial_dim(dim: str | object) -> bool:
    """Check if a dimension name is spatial (exact match or renamed variant).

    Spatial dims may have been renamed from e.g. ``major_radius`` to
    ``major_radius_equilibrium_psi`` during fetch to prevent xarray
    alignment across groups with different coordinate sizes.
    """
    dim_str = str(dim)
    return dim_str in SPATIAL_DIMS or any(
        dim_str.startswith(f"{s}_") for s in SPATIAL_DIMS
    )


def classify_variable(da: xr.DataArray, time_dim: str) -> tuple[str, list[str], int]:
    """Classify a DataArray into one of four signal categories.

    Categories are determined by inspecting the non-time dimensions:
      - scalar_single:       no extra dims          -> (j=0, x=[scalar])
      - scalar_multichannel: 1+ channel dims        -> (j=channel_idx, x=[scalar])
      - profile_1d:          1 spatial dim           -> (j=0, x=spatial_vector)
      - field_2d:            2+ spatial dims         -> (j=0, x=flattened_field)

    Args:
        da: The DataArray to classify.
        time_dim: Name of the time dimension.

    Returns:
        Tuple of (category, non_time_dims, value_dim) where value_dim is the
        length of the x vector produced per observation.
    """
    non_time = [d for d in da.dims if d != time_dim]
    spatial = [d for d in non_time if _is_spatial_dim(d)]

    if len(non_time) == 0:
        return "scalar_single", [], 1

    if len(spatial) >= 2:
        value_dim = math.prod(da.sizes[d] for d in non_time if _is_spatial_dim(d))
        return "field_2d", non_time, value_dim

    if len(spatial) == 1:
        value_dim = da.sizes[spatial[0]]
        return "profile_1d", non_time, value_dim

    # One or more channel dims, no spatial dims.
    # value_dim is 1 because each Parquet row stores x=[scalar]; j encodes the channel.
    return "scalar_multichannel", non_time, 1


# ---------------------------------------------------------------------------
# Signal registry
# ---------------------------------------------------------------------------


def build_signal_registry(
    ds: xr.Dataset,
) -> tuple[dict[str, dict], dict[str, int]]:
    """Build the signal registry by inspecting the reference shot Dataset.

    Iterates over all data variables in the Dataset, classifies each, and
    assigns signal IDs sequentially.

    Args:
        ds: The already-loaded reference shot Dataset.

    Returns:
        Tuple of (registry, var_to_id) where registry maps string signal ID to
        signal metadata dict, and var_to_id maps variable name to integer ID.
    """
    registry: dict[str, dict] = {}
    var_to_id: dict[str, int] = {}

    for sid, var_name in enumerate(ds.data_vars, start=1):
        da = ds[var_name]
        time_dim = get_time_dim(da)
        if time_dim is None:
            continue  # skip variables without a recognised time dimension

        category, non_time_dims, value_dim = classify_variable(da, time_dim)

        n_channels: int
        if category == "scalar_multichannel" and non_time_dims:
            n_channels = da.sizes[non_time_dims[0]]
        else:
            n_channels = 1

        unit: str = da.attrs.get("units", da.attrs.get("unit", ""))

        entry: dict = {
            "name": var_name,
            "category": category,
            "n_channels": n_channels,
            "value_dim": value_dim,
            "time_dim": time_dim,
            "unit": unit,
        }

        # Preserve spatial coordinate values for profile_1d / field_2d signals
        # so that downstream consumers can map array indices back to physical
        # positions (e.g. major_radius in metres).
        spatial = [d for d in non_time_dims if _is_spatial_dim(d)]
        if category == "field_2d" and len(spatial) >= 2:
            entry["shape"] = [da.sizes[d] for d in spatial]
            entry["coords"] = {
                str(d): ds[d].values.astype(float).tolist() for d in spatial
            }
        elif category == "profile_1d" and len(spatial) == 1:
            entry["coords"] = {
                str(spatial[0]): ds[spatial[0]].values.astype(float).tolist()
            }
            # Boundary contour coords (n_boundary_coords_*) have a variable
            # number of points per shot depending on the equilibrium reconstruction.
            # Flag these so compute_and_write_train_stats uses global stats.
            if "n_boundary_coords" in str(spatial[0]):
                entry["variable_length"] = True

        registry[str(sid)] = entry
        var_to_id[str(var_name)] = sid

    return registry, var_to_id


# ---------------------------------------------------------------------------
# Variable -> observations (vectorised)
# ---------------------------------------------------------------------------


def process_variable(
    ds: xr.Dataset,
    var_name: str,
    signal_id: int,
    time_dim: str,
    category: str,
) -> pd.DataFrame | None:
    """Convert one variable in a shot to a DataFrame of observations.

    Args:
        ds: The opened xarray Dataset for this shot.
        var_name: Name of the variable to process.
        signal_id: Integer signal ID (s column value).
        time_dim: Name of the variable's time dimension.
        category: One of scalar_single, scalar_multichannel, profile_1d, field_2d.

    Returns:
        DataFrame with columns t, s, j, x; or None if all values are NaN.
    """
    da = ds[var_name]
    times = ds[time_dim].values.astype(np.float64)
    values = da.values.astype(np.float64)

    if category == "scalar_single":
        # values shape: (T,)
        mask = ~np.isnan(values)
        n = int(mask.sum())
        if n == 0:
            return None
        t_vals = times[mask].astype(np.float32)
        x_vals = values[mask].astype(np.float32)
        return pd.DataFrame(
            {
                "t": t_vals,
                "s": np.full(n, signal_id, dtype=np.int32),
                "j": np.zeros(n, dtype=np.int32),
                "x": x_vals.reshape(-1, 1).tolist(),
            }
        )

    elif category == "scalar_multichannel":
        # values shape may be (C, T) or (T, C) — move time to axis -1
        time_axis = list(da.dims).index(time_dim)
        if time_axis != len(da.dims) - 1:
            values = np.moveaxis(values, time_axis, -1)
        # values shape: (C, T)
        if values.ndim == 1:
            values = values[np.newaxis, :]  # edge case: single channel
        c_idx, t_idx = np.where(~np.isnan(values))
        n = len(t_idx)
        if n == 0:
            return None
        t_vals = times[t_idx].astype(np.float32)
        x_vals = values[c_idx, t_idx].astype(np.float32)
        return pd.DataFrame(
            {
                "t": t_vals,
                "s": np.full(n, signal_id, dtype=np.int32),
                "j": c_idx.astype(np.int32),
                "x": x_vals.reshape(-1, 1).tolist(),
            }
        )

    elif category == "profile_1d":
        # values shape: (R, T) — move time to axis -1
        time_axis = list(da.dims).index(time_dim)
        if time_axis != len(da.dims) - 1:
            values = np.moveaxis(values, time_axis, -1)
        # values shape: (R, T)
        valid = ~np.all(np.isnan(values), axis=0)  # (T,)
        n = int(valid.sum())
        if n == 0:
            return None
        profiles = values[:, valid].T  # (n, R)
        t_vals = times[valid].astype(np.float32)
        return pd.DataFrame(
            {
                "t": t_vals,
                "s": np.full(n, signal_id, dtype=np.int32),
                "j": np.zeros(n, dtype=np.int32),
                "x": profiles.astype(np.float32).tolist(),
            }
        )

    elif category == "field_2d":
        # values shape: (*spatial, T) — move time to axis -1
        time_axis = list(da.dims).index(time_dim)
        if time_axis != len(da.dims) - 1:
            values = np.moveaxis(values, time_axis, -1)
        # values shape: (*spatial, T)
        T = values.shape[-1]
        flat = values.reshape(-1, T)  # (spatial_flat, T)
        valid = ~np.all(np.isnan(flat), axis=0)  # (T,)
        n = int(valid.sum())
        if n == 0:
            return None
        fields = flat[:, valid].T  # (n, spatial_flat)
        t_vals = times[valid].astype(np.float32)
        return pd.DataFrame(
            {
                "t": t_vals,
                "s": np.full(n, signal_id, dtype=np.int32),
                "j": np.zeros(n, dtype=np.int32),
                "x": fields.astype(np.float32).tolist(),
            }
        )

    return None


# ---------------------------------------------------------------------------
# Shot processing
# ---------------------------------------------------------------------------


def build_shot_grid(
    ds: xr.Dataset,
    registry: dict[str, dict],
    var_to_id: dict[str, int],
) -> pd.DataFrame | None:
    """Build the expected observation grid for a shot.

    For each signal present in the dataset, emits one row per timestep in the
    raw time coordinate — regardless of whether the value is NaN. Multichannel
    signals are NOT expanded per channel; ``n_channels`` from ``signals.json``
    lets models do that expansion themselves.

    Args:
        ds: The already-opened xarray Dataset for this shot.
        registry: Signal registry mapping string sid to metadata.
        var_to_id: Mapping from variable name to integer signal ID.

    Returns:
        DataFrame with columns s (int32) and t (float32), or None if no
        signals found.
    """
    parts: list[pd.DataFrame] = []

    for var_name, sid in var_to_id.items():
        if var_name not in ds.data_vars:
            continue
        time_dim = registry[str(sid)]["time_dim"]
        if time_dim not in ds.dims:
            continue
        times = ds[time_dim].values.astype(np.float32)
        n = len(times)
        parts.append(
            pd.DataFrame(
                {
                    "s": np.full(n, sid, dtype=np.int32),
                    "t": times,
                }
            )
        )

    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def process_shot(
    ds: xr.Dataset,
    registry: dict[str, dict],
    var_to_id: dict[str, int],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Convert a single shot's Dataset to observation and grid DataFrames.

    Timestamps are zero-shifted so that every shot starts at t=0: the minimum
    timestamp across all signals is subtracted from every time coordinate before
    any observations are written.

    Args:
        ds: The already-loaded shot Dataset.
        registry: Signal registry mapping string sid to metadata.
        var_to_id: Mapping from variable name to integer signal ID.

    Returns:
        Tuple of (obs_df, grid_df). Either may be None if no data is present.
    """
    # Compute t_offset = minimum timestamp across all present time dimensions.
    time_dim_mins: dict[str, float] = {}
    for var_name, sid in var_to_id.items():
        if var_name not in ds.data_vars:
            continue
        time_dim = registry[str(sid)]["time_dim"]
        if time_dim not in ds.dims or time_dim in time_dim_mins:
            continue
        times = ds[time_dim].values.astype(np.float64)
        if len(times) > 0 and not np.all(np.isnan(times)):
            time_dim_mins[time_dim] = float(np.nanmin(times))

    t_offset = min(time_dim_mins.values()) if time_dim_mins else 0.0

    # Shift all time coordinates so the shot starts at t=0.
    if t_offset != 0.0 and time_dim_mins:
        ds = ds.assign_coords(
            {td: ds[td].values.astype(np.float64) - t_offset for td in time_dim_mins}
        )

    dfs: list[pd.DataFrame] = []

    for var_name, sid in var_to_id.items():
        if var_name not in ds.data_vars:
            continue
        info = registry[str(sid)]
        time_dim = info["time_dim"]
        if time_dim not in ds.dims:
            continue
        df = process_variable(ds, var_name, sid, time_dim, info["category"])
        if df is not None:
            dfs.append(df)

    grid_df = build_shot_grid(ds, registry, var_to_id)

    obs_df = pd.concat(dfs, ignore_index=True) if dfs else None
    return obs_df, grid_df


# ---------------------------------------------------------------------------
# Parquet writers
# ---------------------------------------------------------------------------


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write an observation DataFrame to a Parquet file using the benchmark schema.

    Args:
        df: DataFrame with columns t, s, j, x.
        path: Output file path.
    """
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, path)


def write_grid_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write an observation grid DataFrame to a Parquet file.

    Args:
        df: DataFrame with columns s, t.
        path: Output file path.
    """
    table = pa.Table.from_pandas(df, schema=GRID_SCHEMA, preserve_index=False)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Training statistics
# ---------------------------------------------------------------------------


def compute_and_write_train_stats(output_dir: Path) -> None:
    """Compute per-(s, j, m) mean and std from the training split and write to Parquet.

    Reads every training Parquet file one at a time (no simultaneous loading) and
    accumulates running sum, sum-of-squares, and count per (signal, channel, x-index)
    triple using numpy arrays.  After all files are processed, mean and std are
    derived and written to ``output_dir/train_stats.parquet``.

    Signals marked ``variable_length: true`` in ``signals.json`` have a spatial
    dimension whose size varies across shots (e.g. the LCFS boundary contour, whose
    number of points depends on the equilibrium reconstruction).  Because the index
    m does not correspond to a fixed physical location across shots for such signals,
    per-index stats would mix measurements from different physical positions.
    Instead, all non-NaN values across every index and every shot are pooled into a
    single global mean and std, stored as ``m=0``.

    For signals where std = 0 (constant signal), std is set to 1e-8 to prevent
    division-by-zero when the stats are used for normalisation.

    Args:
        output_dir: Root output directory that contains ``splits.json``,
            ``signals.json``, and the ``train/`` subdirectory.
    """
    splits: dict[str, list[str]] = json.loads((output_dir / "splits.json").read_text())
    registry: dict[str, dict] = json.loads((output_dir / "signals.json").read_text())

    variable_length_sids: set[int] = {
        int(sid) for sid, info in registry.items() if info.get("variable_length", False)
    }

    train_files = [
        output_dir / "train" / f"{label}.parquet" for label in splits["train"]
    ]

    # Fixed-length accumulator: (s, j) -> {"n": ndarray, "sum": ndarray, "sum_sq": ndarray}
    accum: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    # Variable-length accumulator: (s, j) -> {"n": int, "sum": float, "sum_sq": float}
    vl_accum: dict[tuple[int, int], dict[str, float]] = {}

    for fpath in train_files:
        if not fpath.exists():
            logger.warning("Training file not found, skipping: %s", fpath)
            continue
        table = pq.read_table(fpath)
        s_col = table.column("s").to_pylist()
        j_col = table.column("j").to_pylist()
        x_col = table.column("x").to_pylist()

        for s_val, j_val, x_list in zip(s_col, j_col, x_col):
            s_val, j_val = int(s_val), int(j_val)
            x_arr = np.array(x_list, dtype=np.float64)
            key = (s_val, j_val)

            if s_val in variable_length_sids:
                # Pool all valid values globally — one mean/std for the whole signal.
                valid_vals = x_arr[~np.isnan(x_arr)]
                if key not in vl_accum:
                    vl_accum[key] = {"n": 0, "sum": 0.0, "sum_sq": 0.0}
                vl_accum[key]["n"] += len(valid_vals)
                vl_accum[key]["sum"] += float(valid_vals.sum())
                vl_accum[key]["sum_sq"] += float((valid_vals**2).sum())
                continue

            vdim = len(x_arr)
            if key not in accum:
                accum[key] = {
                    "n": np.zeros(vdim, dtype=np.int64),
                    "sum": np.zeros(vdim, dtype=np.float64),
                    "sum_sq": np.zeros(vdim, dtype=np.float64),
                }
            valid = ~np.isnan(x_arr)
            accum[key]["n"][valid] += 1
            accum[key]["sum"][valid] += x_arr[valid]
            accum[key]["sum_sq"][valid] += x_arr[valid] ** 2

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
    for (s, j), stats in sorted(accum.items()):
        for m in range(len(stats["n"])):
            n = int(stats["n"][m])
            if n == 0:
                continue
            mean = stats["sum"][m] / n
            variance = stats["sum_sq"][m] / n - mean**2
            std = max(float(np.sqrt(max(variance, 0.0))), 1e-8)
            rows.append(
                {
                    "s": np.int32(s),
                    "j": np.int32(j),
                    "m": np.int32(m),
                    "mean": np.float32(mean),
                    "std": np.float32(std),
                }
            )

    for (s, j), stats in sorted(vl_accum.items()):
        n = stats["n"]
        if n == 0:
            continue
        mean = stats["sum"] / n
        variance = stats["sum_sq"] / n - mean**2
        std = max(float(np.sqrt(max(variance, 0.0))), 1e-8)
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
    stats_path = output_dir / "train_stats.parquet"
    pq.write_table(table, stats_path)
    print(
        f"  train_stats.parquet: {len(rows)} (s, j, m) combinations written to {stats_path}"
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: fetch shots from S3, process to Parquet, organise splits."""
    parser = argparse.ArgumentParser(
        description="Fetch MAST shots from S3 Zarr and convert to benchmark Parquet."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/fusion_data"),
        help="Root output directory for Parquet files and metadata.",
    )
    parser.add_argument(
        "--start_shot",
        type=int,
        default=11766,
        help="First shot ID to fetch (inclusive).",
    )
    parser.add_argument(
        "--end_shot",
        type=int,
        default=30471,
        help="Last shot ID to fetch (exclusive).",
    )
    parser.add_argument(
        "--reference_shot",
        type=int,
        default=30421,
        help="Shot ID used to build the signal registry.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="Step size between shot IDs (e.g. 4 = take one shot, skip three).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip shots whose Parquet output already exists in staging.",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Skip fetching and processing; compute train_stats.parquet from existing split data.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.stats_only:
        compute_and_write_train_stats(args.output_dir)
        return

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase 1a: Fetch reference shot and build signal registry
    # ------------------------------------------------------------------
    print(f"Fetching reference shot {args.reference_shot} from S3 ...")
    ref_ds = fetch_shot(args.reference_shot, SIGNALS)
    if ref_ds is None:
        raise RuntimeError(
            f"Failed to fetch reference shot {args.reference_shot}. "
            "Cannot build signal registry."
        )

    registry, var_to_id = build_signal_registry(ref_ds)
    print(f"  {len(registry)} signals discovered.")
    del ref_ds

    # ------------------------------------------------------------------
    # Phase 1b: Fetch, process, and write each shot to staging
    # ------------------------------------------------------------------
    staging_dir = args.output_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    shot_ids = list(range(args.start_shot, args.end_shot, args.step))
    successful_shots: list[int] = []
    total_obs_rows = 0
    total_grid_rows = 0

    for shot_id in tqdm(shot_ids, desc="Fetching & processing", unit="shot"):
        shot_label = f"shot_{shot_id}"
        obs_path = staging_dir / f"{shot_label}.parquet"

        # Resume support: skip if already processed.
        if args.resume and obs_path.exists():
            successful_shots.append(shot_id)
            continue

        try:
            ds = fetch_shot(shot_id, SIGNALS)
        except Exception:
            logger.warning("Shot %d: fetch failed, skipping.", shot_id)
            continue

        if ds is None:
            continue

        try:
            obs_df, grid_df = process_shot(ds, registry, var_to_id)
        except Exception:
            logger.warning("Shot %d: processing failed, skipping.", shot_id)
            del ds
            continue

        del ds

        if obs_df is None:
            successful_shots.append(shot_id)
            continue

        write_parquet(obs_df, obs_path)
        total_obs_rows += len(obs_df)

        if grid_df is not None:
            grid_path = staging_dir / f"{shot_label}_grid.parquet"
            write_grid_parquet(grid_df, grid_path)
            total_grid_rows += len(grid_df)

        successful_shots.append(shot_id)

    # ------------------------------------------------------------------
    # Phase 2: Compute splits and organise into final directory layout
    # ------------------------------------------------------------------
    successful_shots.sort()
    n_total = len(successful_shots)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    split_assignment: dict[int, str] = {}
    for i, sid in enumerate(successful_shots):
        if i < n_train:
            split_assignment[sid] = "train"
        elif i < n_train + n_val:
            split_assignment[sid] = "val"
        else:
            split_assignment[sid] = "test"

    # Create split directories.
    for split in ("train", "val", "test"):
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)

    # Move files from staging into split directories.
    splits_json: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for shot_id in successful_shots:
        split = split_assignment[shot_id]
        shot_label = f"shot_{shot_id}"
        splits_json[split].append(shot_label)

        src_obs = staging_dir / f"{shot_label}.parquet"
        if src_obs.exists():
            src_obs.rename(args.output_dir / split / f"{shot_label}.parquet")

        src_grid = staging_dir / f"{shot_label}_grid.parquet"
        if src_grid.exists():
            src_grid.rename(args.output_dir / split / f"{shot_label}_grid.parquet")

    # Clean up staging directory.
    shutil.rmtree(staging_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Write metadata files
    # ------------------------------------------------------------------
    signals_path = args.output_dir / "signals.json"
    signals_path.write_text(json.dumps(registry, indent=2))

    splits_path = args.output_dir / "splits.json"
    splits_path.write_text(json.dumps(splits_json, indent=2))
    compute_and_write_train_stats(args.output_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    print("\n=== Processing complete ===")
    print(f"  Total shots : {n_total}  (from {len(shot_ids)} attempted)")
    print(
        f"  Split sizes : train={len(splits_json['train'])}  "
        f"val={len(splits_json['val'])}  "
        f"test={len(splits_json['test'])}"
    )
    print(f"  Obs rows    : {total_obs_rows:,}")
    print(f"  Grid rows   : {total_grid_rows:,}")
    print(f"  signals.json: {signals_path}")
    print(f"  splits.json : {splits_path}")
    print(f"  Elapsed     : {elapsed:.1f}s")


if __name__ == "__main__":
    main()

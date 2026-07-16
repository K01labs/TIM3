"""Fetch TC PRIMED storms and convert to benchmark Parquet format.

Each storm lifecycle is converted to an observation-set DataFrame with columns:
    t (float32)  — timestamp in seconds since storm start
    s (int32)    — signal type ID
    j (int32)    — channel index within that signal type
    x (list[float32]) — value array, variable length

Design notes:

- Overpass times are decoded carefully: TC PRIMED often stores them as
  datetime64 coordinates rather than plain floats.
- The overpass observation grid contains a row only for signals actually present
  in each overpass file, not every overpass signal at every overpass time — so
  the grid's missingness reflects true sensor availability.
- Only stable overpass products are used by default (GPROF precipitation, GEO IR,
  optional radar precipitation, and overpass storm metadata). Raw passive
  microwave TB_* channels are intentionally excluded because their channel sets
  and swath shapes differ by sensor.

Usage:
    uv run python data/dataloaders/climate_dataset.py [--output_dir PATH]
        [--basins AL,EP,WP] [--start_season N] [--end_season N] [--resume]

    Run without arguments to fetch every basin for seasons 1987-2023 — a very
    large, multi-day download. For a quick run, restrict the scope, e.g.:
        --basins AL --start_season 2019 --end_season 2020
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from tqdm import tqdm
import tcprimedapi
import netCDF4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("t", pa.float32()),
        pa.field("s", pa.int32()),
        pa.field("j", pa.int32()),
        pa.field("x", pa.list_(pa.float32())),
    ]
)

GRID_SCHEMA = pa.schema(
    [
        pa.field("s", pa.int32()),
        pa.field("t", pa.float32()),
    ]
)

# Environmental scalars at synoptic times.
ENV_STORM_SCALARS: list[dict] = [
    {
        "name": "track_intensity",
        "group": "storm_metadata",
        "var": "intensity",
        "unit": "kt",
    },
    {
        "name": "track_pressure",
        "group": "storm_metadata",
        "var": "central_min_pressure",
        "unit": "hPa",
    },
    {
        "name": "track_speed",
        "group": "storm_metadata",
        "var": "storm_speed",
        "unit": "m s-1",
    },
    {
        "name": "track_heading",
        "group": "storm_metadata",
        "var": "storm_heading",
        "unit": "degree",
    },
]

ENV_DIAGNOSTIC_SCALARS: list[dict] = [
    # Sea surface temperature. shape (64, 1) → scalar_single.
    {"name": "diag_sst", "group": "diagnostics", "var": "sst", "unit": "K"},
    # Empirical maximum potential intensity — thermodynamic ceiling. shape (64, 1) → scalar_single.
    {
        "name": "diag_mpi",
        "group": "diagnostics",
        "var": "potential_intensity_empirical",
        "unit": "kt",
    },
    # Layered wind shear magnitude. shape (64, 2, 4) → 8-channel scalar_multichannel.
    {
        "name": "diag_shear_magnitude",
        "group": "diagnostics",
        "var": "shear_magnitude",
        "unit": "m s-1",
    },
    # Layered wind shear direction. shape (64, 2, 4) → 8-channel scalar_multichannel.
    {
        "name": "diag_shear_direction",
        "group": "diagnostics",
        "var": "shear_direction",
        "unit": "degree",
    },
    # Precipitable water in 5 radial bins (200 km each). shape (64, 5) → 5-channel scalar_multichannel.
    {
        "name": "diag_precipitable_water",
        "group": "diagnostics",
        "var": "precipitable_water",
        "unit": "kg m-2",
    },
    # Warm-core temperature anomaly over 23 pressure levels. shape (64, 23, 1) → 23-channel scalar_multichannel.
    {
        "name": "diag_warm_core",
        "group": "diagnostics",
        "var": "temperature_anomaly",
        "unit": "K",
    },
    # Cyclone phase space B parameter. shape (64, 1) → scalar_single.
    {
        "name": "diag_phase_b",
        "group": "diagnostics",
        "var": "cyclone_phase_space_b_parameter",
        "unit": "m",
    },
    # 1D cloud model vertical velocity in 4 annular regions. shape (64, 4) → 4-channel scalar_multichannel.
    {
        "name": "diag_vertical_velocity",
        "group": "diagnostics",
        "var": "vertical_velocity_average",
        "unit": "m s-1",
    },
]

SHEAR_CHANNEL_LABELS: list[str] = [
    "shallow_0_300km",
    "shallow_0_500km",
    "shallow_0_800km",
    "shallow_200_800km",
    "deep_0_300km",
    "deep_0_500km",
    "deep_0_800km",
    "deep_200_800km",
]

# Stable overpass-level signals to use by default.
# We intentionally exclude raw passive-microwave TB_* channels in this
# simplified loader because the available channels and swath geometry differ
# by sensor/platform across TC PRIMED.
OVP_FIELD_SIGNALS: list[dict] = [
    # GPROF surface precipitation rate. Shape varies by sensor swath → variable_length.
    {
        "name": "gprof_surface_precip",
        "group": "GPROF/S1",
        "var": "surfacePrecipitation",
        "category": "field_2d",
        "unit": "mm hr-1",
    },
    # GPROF ice water path — deep convection signature. Shape varies by sensor swath → variable_length.
    {
        "name": "gprof_ice_water_path",
        "group": "GPROF/S1",
        "var": "iceWaterPath",
        "category": "field_2d",
        "unit": "kg m-2",
    },
    # IR brightness temperature at 10.3 um. Shape varies by IR sensor platform (480x640, 301x301, 560x560).
    {
        "name": "ir_brightness_temp",
        "group": "infrared",
        "var": "IRWIN",
        "category": "field_2d",
        "unit": "K",
    },
    # Radar near-surface precipitation. GPM/TRMM overpasses only — high missingness by design.
    {
        "name": "radar_precip_rate",
        "group": "radar_radiometer",
        "var": "nearSurfPrecipTotRate",
        "category": "field_2d",
        "unit": "mm hr-1",
    },
]

OVP_SCALAR_SIGNALS: list[dict] = [
    {
        "name": "overpass_intensity",
        "group": "overpass_storm_metadata",
        "var": "intensity",
        "unit": "knots",
    },
    {
        "name": "overpass_pressure",
        "group": "overpass_storm_metadata",
        "var": "central_min_pressure",
        "unit": "hPa",
    },
    {
        "name": "overpass_speed",
        "group": "overpass_storm_metadata",
        "var": "storm_speed",
        "unit": "m/s",
    },
    {
        "name": "overpass_distance_to_land",
        "group": "overpass_storm_metadata",
        "var": "distance_to_land",
        "unit": "km",
    },
]

_ATCF_RE = re.compile(r"([a-zA-Z]{2}\d{2}\d{4})")


# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------


def open_netcdf_group_from_bytes(data: bytes, group: str) -> xr.Dataset | None:
    """Open a specific NetCDF group from in-memory bytes."""
    fname = f"memory_{id(data)}_{group.replace('/', '_')}.nc"
    nc = None
    try:
        nc = netCDF4.Dataset(fname, memory=data)
        grp = nc
        for part in group.split("/"):
            if part not in grp.groups:
                return None
            grp = grp.groups[part]
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(grp), decode_timedelta=False)
        ds.load()
        return ds
    except Exception:
        return None
    finally:
        if nc is not None:
            try:
                nc.close()
            except Exception:
                pass


def _get_fill_values(da: xr.DataArray) -> set[float]:
    """Extract fill/missing values from NetCDF variable attributes."""
    fills: set[float] = set()
    for attr_name in ("_FillValue", "missing_value"):
        val = da.attrs.get(attr_name)
        if val is None:
            continue
        if hasattr(val, "__iter__") and not isinstance(val, str):
            fills.update(float(v) for v in val)
        else:
            fills.add(float(val))
    if not fills:
        fills = {-999.0, -9999.0, -9999.9, -999.99}
    return fills


def _atcf_sort_key(atcf_id: str) -> tuple[int, str, int]:
    basin = atcf_id[:2]
    number = int(atcf_id[2:4])
    season = int(atcf_id[4:8])
    return (season, basin, number)


def _find_time_coord(grp_ds: xr.Dataset, da: xr.DataArray) -> str | None:
    """Find the time coordinate name in a NetCDF group dataset."""
    for coord_name in ("time", "env_time"):
        if coord_name in grp_ds.coords:
            return coord_name
    for dim in da.dims:
        if "time" in str(dim).lower():
            return str(dim)
    return None


def _times_to_seconds(times_raw: np.ndarray) -> np.ndarray:
    """Convert a time array to float64 seconds since epoch."""
    times_raw = np.asarray(times_raw)
    if np.issubdtype(times_raw.dtype, np.datetime64):
        return (times_raw - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(
            1, "s"
        )
    return times_raw.astype(np.float64)


def _single_time_to_seconds(values: np.ndarray) -> float:
    """Convert a length-1 time array or scalar to seconds since epoch."""
    sec = _times_to_seconds(np.asarray(values).reshape(-1))
    return float(sec[0])


def _infer_env_signal_layout(da: xr.DataArray, time_dim: str) -> tuple[str, int, int]:
    """Infer whether an env variable is scalar_single or scalar_multichannel."""
    non_time_dims = [d for d in da.dims if d != time_dim]
    if not non_time_dims:
        return "scalar_single", 1, 1

    n_channels = int(np.prod([da.sizes[d] for d in non_time_dims]))
    if n_channels == 1:
        return "scalar_single", 1, 1

    # value_dim is 1: each Parquet row stores x=[scalar]; j encodes the channel.
    return "scalar_multichannel", n_channels, 1


# ---------------------------------------------------------------------------
# Signal registry
# ---------------------------------------------------------------------------


def build_signal_registry(env_data: bytes) -> tuple[dict[str, dict], dict[str, int]]:
    """Build the signal registry.

    Environmental signals are included if present in the reference storm.
    Overpass signals are registered from stable definitions rather than from a
    single overpass file because TC PRIMED overpass products are sensor-
    heterogeneous.
    """
    registry: dict[str, dict] = {}
    name_to_id: dict[str, int] = {}
    registry["0"] = {
        "name": "static_context",
        "category": "static",
        "n_channels": 3,
        "value_dim": 1,
        "time_dim": None,
        "unit": "",
        "source": "static",
        "group": "atcf_id",
        "channel_labels": ["basin_code", "season", "storm_number"],
    }
    name_to_id["static_context"] = 0
    sid = 1

    env_storm = open_netcdf_group_from_bytes(env_data, "storm_metadata")
    env_diag = open_netcdf_group_from_bytes(env_data, "diagnostics")

    for sig_def in ENV_STORM_SCALARS:
        var_name = sig_def["var"]
        if env_storm is None or var_name not in env_storm:
            continue
        da = env_storm[var_name]
        time_coord = _find_time_coord(env_storm, da)
        registry[str(sid)] = {
            "name": sig_def["name"],
            "category": "scalar_single",
            "n_channels": 1,
            "value_dim": 1,
            "time_dim": time_coord or "time",
            "unit": sig_def.get("unit", ""),
            "source": "env",
            "group": sig_def["group"],
            "var": var_name,
        }
        name_to_id[sig_def["name"]] = sid
        sid += 1

    for sig_def in ENV_DIAGNOSTIC_SCALARS:
        var_name = sig_def["var"]
        if env_diag is None or var_name not in env_diag:
            continue
        da = env_diag[var_name]
        time_coord = _find_time_coord(env_diag, da)
        if time_coord is None:
            continue

        category, n_channels, value_dim = _infer_env_signal_layout(da, time_coord)
        entry = {
            "name": sig_def["name"],
            "category": category,
            "n_channels": n_channels,
            "value_dim": value_dim,
            "time_dim": time_coord,
            "unit": sig_def.get("unit", ""),
            "source": "env",
            "group": sig_def["group"],
            "var": var_name,
        }
        if (
            sig_def["name"] in {"diag_shear_magnitude", "diag_shear_direction"}
            and n_channels == 8
        ):
            entry["channel_labels"] = SHEAR_CHANNEL_LABELS

        registry[str(sid)] = entry
        name_to_id[sig_def["name"]] = sid
        sid += 1

    for sig_def in OVP_SCALAR_SIGNALS:
        registry[str(sid)] = {
            "name": sig_def["name"],
            "category": "scalar_single",
            "n_channels": 1,
            "value_dim": 1,
            "time_dim": "time",
            "unit": sig_def.get("unit", ""),
            "source": "overpass",
            "group": sig_def["group"],
            "var": sig_def["var"],
        }
        name_to_id[sig_def["name"]] = sid
        sid += 1

    for sig_def in OVP_FIELD_SIGNALS:
        vdim = sig_def.get("value_dim", -1)
        entry: dict = {
            "name": sig_def["name"],
            "category": "field_2d",
            "n_channels": 1,
            "value_dim": vdim,
            "time_dim": "time",
            "unit": sig_def.get("unit", ""),
            "source": "overpass",
            "group": sig_def["group"],
            "var": sig_def["var"],
        }
        if vdim == -1:
            entry["variable_length"] = True
        registry[str(sid)] = entry
        name_to_id[sig_def["name"]] = sid
        sid += 1

    # Register the static context signal so signals.json is fully self-describing.
    # j=0 → basin_code, j=1 → season, j=2 → storm_number
    print(f"  Signal registry: {len(registry)} signals discovered")
    return registry, name_to_id


# ---------------------------------------------------------------------------
# Processing: environmental file -> observation rows
# ---------------------------------------------------------------------------


def process_env_file(
    env_data: bytes,
    registry: dict[str, dict],
    t_offset: float,
) -> pd.DataFrame | None:
    """Extract observations from a storm's environmental file."""
    dfs: list[pd.DataFrame] = []

    groups: dict[str, xr.Dataset | None] = {
        "storm_metadata": open_netcdf_group_from_bytes(env_data, "storm_metadata"),
        "diagnostics": open_netcdf_group_from_bytes(env_data, "diagnostics"),
    }

    for str_sid, info in registry.items():
        if info["source"] != "env":
            continue
        sid = int(str_sid)
        grp_ds = groups.get(info["group"])
        if grp_ds is None or info["var"] not in grp_ds:
            continue

        da = grp_ds[info["var"]]
        values = da.values.astype(np.float64)
        time_var = _find_time_coord(grp_ds, da)
        if time_var is None:
            continue

        times = _times_to_seconds(grp_ds[time_var].values) - t_offset
        fill_vals = _get_fill_values(da)

        if info["category"] == "scalar_single":
            values = values.ravel()
            mask = ~np.isnan(values)
            for fv in fill_vals:
                mask &= values != fv

            n = int(mask.sum())
            if n == 0:
                continue

            dfs.append(
                pd.DataFrame(
                    {
                        "t": times[mask].astype(np.float32),
                        "s": np.full(n, sid, dtype=np.int32),
                        "j": np.zeros(n, dtype=np.int32),
                        "x": values[mask].astype(np.float32).reshape(-1, 1).tolist(),
                    }
                )
            )
            continue

        if info["category"] == "scalar_multichannel":
            time_axis = list(da.dims).index(time_var)
            if time_axis != len(da.dims) - 1:
                values = np.moveaxis(values, time_axis, -1)
            values = values.reshape(-1, values.shape[-1])

            mask = ~np.isnan(values)
            for fv in fill_vals:
                mask &= values != fv

            c_idx, t_idx = np.where(mask)
            n = len(t_idx)
            if n == 0:
                continue

            x_vals = values[c_idx, t_idx].astype(np.float32)
            dfs.append(
                pd.DataFrame(
                    {
                        "t": times[t_idx].astype(np.float32),
                        "s": np.full(n, sid, dtype=np.int32),
                        "j": c_idx.astype(np.int32),
                        "x": x_vals.reshape(-1, 1).tolist(),
                    }
                )
            )

    return pd.concat(dfs, ignore_index=True) if dfs else None


# ---------------------------------------------------------------------------
# Processing: overpass file -> observation rows and presence map
# ---------------------------------------------------------------------------


def process_overpass_file(
    ovp_data: bytes,
    registry: dict[str, dict],
    t_offset: float,
) -> tuple[pd.DataFrame | None, float | None, set[int]]:
    """Extract observations from a single overpass file.

    Returns:
        (obs_df, overpass_time_rel, present_signal_ids)
    """
    ovp_meta = open_netcdf_group_from_bytes(ovp_data, "overpass_metadata")
    if ovp_meta is None or "time" not in ovp_meta:
        return None, None, set()

    ovp_time = np.float32(_single_time_to_seconds(ovp_meta["time"].values) - t_offset)
    dfs: list[pd.DataFrame] = []
    present_signal_ids: set[int] = set()

    for str_sid, info in registry.items():
        if info["source"] != "overpass":
            continue
        sid = int(str_sid)
        grp_ds = open_netcdf_group_from_bytes(ovp_data, info["group"])
        if grp_ds is None or info["var"] not in grp_ds:
            continue

        da = grp_ds[info["var"]]
        values = da.values.astype(np.float64)
        fill_vals = _get_fill_values(da)

        if info["category"] == "scalar_single":
            val = float(np.asarray(values).reshape(-1)[0])
            if np.isnan(val) or any(val == fv for fv in fill_vals):
                continue
            dfs.append(
                pd.DataFrame(
                    {
                        "t": np.array([ovp_time], dtype=np.float32),
                        "s": np.array([sid], dtype=np.int32),
                        "j": np.array([0], dtype=np.int32),
                        "x": [[np.float32(val)]],
                    }
                )
            )
            present_signal_ids.add(sid)
            continue

        if info["category"] == "field_2d":
            flat = values.ravel().astype(np.float32)
            for fv in fill_vals:
                flat[flat == np.float32(fv)] = np.nan
            if np.all(np.isnan(flat)):
                continue
            dfs.append(
                pd.DataFrame(
                    {
                        "t": np.array([ovp_time], dtype=np.float32),
                        "s": np.array([sid], dtype=np.int32),
                        "j": np.array([0], dtype=np.int32),
                        "x": [flat.tolist()],
                    }
                )
            )
            present_signal_ids.add(sid)

    obs_df = pd.concat(dfs, ignore_index=True) if dfs else None
    return obs_df, float(ovp_time), present_signal_ids


# ---------------------------------------------------------------------------
# Grid building
# ---------------------------------------------------------------------------


def build_storm_grid(
    env_data: bytes,
    overpass_signal_times: dict[int, list[float]],
    registry: dict[str, dict],
    t_offset: float,
) -> pd.DataFrame | None:
    """Build the expected observation grid for a storm.

    Environmental signals contribute one row per synoptic timestep. Overpass
    signals contribute rows only for times where that signal was actually
    present in an overpass file.
    """
    parts: list[pd.DataFrame] = []

    groups: dict[str, xr.Dataset | None] = {
        "storm_metadata": open_netcdf_group_from_bytes(env_data, "storm_metadata"),
        "diagnostics": open_netcdf_group_from_bytes(env_data, "diagnostics"),
    }

    for str_sid, info in registry.items():
        sid = int(str_sid)

        if info["source"] == "env":
            grp_ds = groups.get(info["group"])
            if grp_ds is None or info["var"] not in grp_ds:
                continue
            da = grp_ds[info["var"]]
            time_var = _find_time_coord(grp_ds, da)
            if time_var is None:
                continue
            times = _times_to_seconds(grp_ds[time_var].values) - t_offset
            parts.append(
                pd.DataFrame(
                    {
                        "s": np.full(len(times), sid, dtype=np.int32),
                        "t": times.astype(np.float32),
                    }
                )
            )
            continue

        if info["source"] == "overpass":
            sig_times = overpass_signal_times.get(sid)
            if not sig_times:
                continue
            parts.append(
                pd.DataFrame(
                    {
                        "s": np.full(len(sig_times), sid, dtype=np.int32),
                        "t": np.array(sig_times, dtype=np.float32),
                    }
                )
            )

    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Static context
# ---------------------------------------------------------------------------


def build_static_context(atcf_id: str, registry: dict[str, dict]) -> pd.DataFrame:
    """Extract static storm metadata as S_k observations.

    Uses the signal ID registered for 'static_context' in the registry so that
    the parquet output is fully self-describing without needing s=-1 as a magic
    value undocumented in signals.json.
    """
    static_sid = int(
        next(k for k, v in registry.items() if v["name"] == "static_context")
    )

    basin_map = {"AL": 0, "EP": 1, "CP": 2, "WP": 3, "IO": 4, "SH": 5, "SL": 6}
    basin = atcf_id[:2].upper()
    basin_code = basin_map.get(basin, -1)
    season = int(atcf_id[4:8])
    number = int(atcf_id[2:4])

    rows = [
        {
            "t": np.float32(-1.0),
            "s": np.int32(static_sid),
            "j": np.int32(0),
            "x": [np.float32(basin_code)],
        },
        {
            "t": np.float32(-1.0),
            "s": np.int32(static_sid),
            "j": np.int32(1),
            "x": [np.float32(season)],
        },
        {
            "t": np.float32(-1.0),
            "s": np.int32(static_sid),
            "j": np.int32(2),
            "x": [np.float32(number)],
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parquet writers
# ---------------------------------------------------------------------------


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, path)


def write_grid_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, schema=GRID_SCHEMA, preserve_index=False)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Storm-level processing
# ---------------------------------------------------------------------------


def get_storm_start_time(env_data: bytes) -> float:
    storm_meta = open_netcdf_group_from_bytes(env_data, "storm_metadata")
    if storm_meta is None:
        raise ValueError("No storm_metadata group in environmental file")
    if "time" not in storm_meta:
        raise ValueError("No 'time' coordinate in storm_metadata")
    return float(_times_to_seconds(storm_meta["time"].values)[0])


def process_storm(
    client: tcprimedapi.Client,
    atcf_id: str,
    registry: dict[str, dict],
    max_overpasses: int = 200,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Process an entire storm lifecycle into observation-set DataFrames."""
    client.query({"atcf_id": [atcf_id], "file_type": ["env"]})
    env_data = None
    for _key, data in client.inmemory():
        env_data = data
        break
    if env_data is None:
        logger.warning("Storm %s: failed to read env file.", atcf_id)
        return None, None

    t_offset = get_storm_start_time(env_data)
    all_obs: list[pd.DataFrame] = []

    env_obs = process_env_file(env_data, registry, t_offset)
    if env_obs is not None:
        all_obs.append(env_obs)

    # NOTE: tcprimedapi does not support fetching individual overpass files by key.
    # client.inmemory() streams ALL matching files regardless of max_overpasses.
    # For a storm with 296 overpasses this downloads ~3.8 GB to process N files.
    # There is no workaround at the API level without querying by sensor or time range.
    client.query({"atcf_id": [atcf_id]})
    ovp_keys = sorted(
        [k for k in client.object_keys if "_env_" not in k and k.endswith(".nc")]
    )
    if len(ovp_keys) > max_overpasses:
        indices = np.linspace(0, len(ovp_keys) - 1, max_overpasses, dtype=int)
        ovp_keys = [ovp_keys[i] for i in indices]
    ovp_key_set = set(ovp_keys)

    overpass_signal_times: dict[int, list[float]] = {}

    for key, ovp_data in client.inmemory():
        if key not in ovp_key_set:
            continue
        try:
            ovp_obs, ovp_time, present_signal_ids = process_overpass_file(
                ovp_data, registry, t_offset
            )
            if ovp_obs is not None:
                all_obs.append(ovp_obs)
            if ovp_time is not None:
                for sid in present_signal_ids:
                    overpass_signal_times.setdefault(sid, []).append(ovp_time)
        except Exception as exc:
            logger.warning(
                "Storm %s overpass %s: processing failed (%s)", atcf_id, key, exc
            )

    all_obs.append(build_static_context(atcf_id, registry))
    grid_df = build_storm_grid(env_data, overpass_signal_times, registry, t_offset)
    obs_df = pd.concat(all_obs, ignore_index=True) if all_obs else None
    return obs_df, grid_df


# ---------------------------------------------------------------------------
# Storm discovery
# ---------------------------------------------------------------------------


def discover_storms(basins: list[str], start_season: int, end_season: int) -> list[str]:
    """Discover available storm ATCF IDs for the given basins and seasons."""
    client = tcprimedapi.Client()
    storms: set[str] = set()

    for basin in basins:
        for season in range(start_season, end_season + 1):
            try:
                client.query(
                    {
                        "season": [str(season)],
                        "basin": [basin.upper()],
                        "file_type": ["env"],
                    }
                )
                for key in client.object_keys:
                    filename = key.split("/")[-1]
                    match = _ATCF_RE.search(filename)
                    if match:
                        storms.add(match.group(1).lower())
            except Exception:
                continue

    return sorted(storms, key=_atcf_sort_key)


# ---------------------------------------------------------------------------
# API shape verification
# ---------------------------------------------------------------------------


def _verify_inmemory_api(client: tcprimedapi.Client, atcf_id: str) -> None:
    """Verify that client.inmemory() yields (key, bytes) tuples."""
    client.query({"atcf_id": [atcf_id], "file_type": ["env"]})
    for item in client.inmemory():
        if not isinstance(item, tuple) or len(item) != 2:
            raise RuntimeError(
                f"tcprimedapi.Client.inmemory() yielded {type(item).__name__}, expected a (key, bytes) tuple."
            )
        key, data = item
        if not isinstance(key, str):
            raise RuntimeError(
                f"tcprimedapi.Client.inmemory() key is {type(key).__name__}, expected str."
            )
        if not isinstance(data, (bytes, bytearray)):
            raise RuntimeError(
                f"tcprimedapi.Client.inmemory() data is {type(data).__name__}, expected bytes."
            )
        break


# ---------------------------------------------------------------------------
# Training statistics
# ---------------------------------------------------------------------------


def compute_and_write_train_stats(output_dir: Path) -> None:
    """Compute per-(s, j, m) mean and std from the training split and write to Parquet.

    Reads every training Parquet file one at a time and accumulates running sum,
    sum-of-squares, and count per (signal, channel, x-index) triple.

    Signals marked ``variable_length: true`` in ``signals.json`` have a swath
    geometry that varies by sensor (e.g. GPROF precipitation fields from GMI vs
    SSMI have different numbers of pixels).  Per-index stats would mix values from
    different physical positions across sensors, so instead all non-NaN values
    across every index and every storm are pooled into a single global mean and
    std, stored as ``m=0``.  This gives valid normalisation statistics despite the
    variable swath size.

    For signals where std = 0 (constant signal), std is set to 1e-8 to avoid
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
    """Entry point: fetch TC PRIMED storms, process to Parquet, organise splits."""
    parser = argparse.ArgumentParser(
        description="Fetch TC PRIMED storms from AWS S3 and convert to benchmark Parquet format."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/tcprimed_data"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--basins",
        type=str,
        default="AL,SL,EP,CP,WP,IO,SH",
        help="Comma-separated basin codes (e.g., AL,EP,WP).",
    )
    parser.add_argument(
        "--start_season", type=int, default=1987, help="First season (inclusive)."
    )
    parser.add_argument(
        "--end_season", type=int, default=2023, help="Last season (inclusive)."
    )
    parser.add_argument(
        "--reference_storm",
        type=str,
        default="al092019",
        help="ATCF ID of storm used to build the environmental signal registry.",
    )
    parser.add_argument(
        "--max_overpasses_per_storm",
        type=int,
        default=900,
        help="Max overpass files to process per storm.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip storms whose Parquet output already exists.",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Skip fetching and processing; compute train_stats.parquet from existing split data.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.stats_only:
        compute_and_write_train_stats(args.output_dir)
        return

    t0 = time.perf_counter()
    basins = [b.strip().upper() for b in args.basins.split(",")]
    client = tcprimedapi.Client()

    print("Verifying tcprimedapi API shape ...")
    _verify_inmemory_api(client, args.reference_storm)
    print("  API verified: inmemory() yields (key, bytes) tuples.")

    print(f"Fetching reference storm {args.reference_storm} ...")
    client.query({"atcf_id": [args.reference_storm], "file_type": ["env"]})
    ref_env_data = None
    for _key, data in client.inmemory():
        ref_env_data = data
        break
    if ref_env_data is None:
        raise RuntimeError(
            f"Failed to fetch env file for reference storm {args.reference_storm}"
        )

    registry, name_to_id = build_signal_registry(ref_env_data)
    del ref_env_data, name_to_id

    print(
        f"Discovering storms for basins={basins}, seasons={args.start_season}-{args.end_season} ..."
    )
    storm_ids = discover_storms(basins, args.start_season, args.end_season)
    print(f"  Found {len(storm_ids)} storms.")

    staging_dir = args.output_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    successful_storms: list[str] = []
    total_obs_rows = 0
    total_grid_rows = 0

    for atcf_id in tqdm(storm_ids, desc="Fetching & processing", unit="storm"):
        storm_label = f"storm_{atcf_id}"
        obs_path = staging_dir / f"{storm_label}.parquet"

        if args.resume and obs_path.exists():
            successful_storms.append(atcf_id)
            continue

        try:
            obs_df, grid_df = process_storm(
                client,
                atcf_id,
                registry,
                max_overpasses=args.max_overpasses_per_storm,
            )
        except Exception as exc:
            logger.warning("Storm %s: processing failed (%s), skipping.", atcf_id, exc)
            continue

        if obs_df is None:
            continue

        write_parquet(obs_df, obs_path)
        total_obs_rows += len(obs_df)

        if grid_df is not None:
            grid_path = staging_dir / f"{storm_label}_grid.parquet"
            write_grid_parquet(grid_df, grid_path)
            total_grid_rows += len(grid_df)

        successful_storms.append(atcf_id)

    successful_storms.sort(key=_atcf_sort_key)
    n_total = len(successful_storms)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    split_assignment: dict[str, str] = {}
    for i, sid in enumerate(successful_storms):
        if i < n_train:
            split_assignment[sid] = "train"
        elif i < n_train + n_val:
            split_assignment[sid] = "val"
        else:
            split_assignment[sid] = "test"

    for split in ("train", "val", "test"):
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)

    splits_json: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for atcf_id in successful_storms:
        split = split_assignment[atcf_id]
        storm_label = f"storm_{atcf_id}"
        splits_json[split].append(storm_label)

        src_obs = staging_dir / f"{storm_label}.parquet"
        if src_obs.exists():
            src_obs.rename(args.output_dir / split / f"{storm_label}.parquet")

        src_grid = staging_dir / f"{storm_label}_grid.parquet"
        if src_grid.exists():
            src_grid.rename(args.output_dir / split / f"{storm_label}_grid.parquet")

    shutil.rmtree(staging_dir, ignore_errors=True)

    signals_path = args.output_dir / "signals.json"
    signals_path.write_text(json.dumps(registry, indent=2))

    splits_path = args.output_dir / "splits.json"
    splits_path.write_text(json.dumps(splits_json, indent=2))

    compute_and_write_train_stats(args.output_dir)

    elapsed = time.perf_counter() - t0
    print("\n=== Processing complete ===")
    print(f"  Total storms : {n_total}  (from {len(storm_ids)} discovered)")
    print(
        f"  Split sizes  : train={len(splits_json['train'])}  val={len(splits_json['val'])}  test={len(splits_json['test'])}"
    )
    print(f"  Obs rows     : {total_obs_rows:,}")
    print(f"  Grid rows    : {total_grid_rows:,}")
    print(f"  signals.json : {signals_path}")
    print(f"  splits.json  : {splits_path}")
    print(f"  Elapsed      : {elapsed:.1f}s")


if __name__ == "__main__":
    main()

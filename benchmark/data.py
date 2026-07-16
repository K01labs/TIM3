"""Low-level data loading and training-statistics lookup.

This module provides two functions:

- ``load_sample`` — reads one Parquet observation file into a DataFrame.
- ``get_sigma`` — looks up the per-dimension training standard deviation
  used to normalise errors.

Parquet schema (canonical):
    t  : float32  — timestamp in seconds (t >= 0 for temporal, t = -1 for static)
    s  : int32    — signal ID (s = 0 for static observations)
    j  : int32    — channel index (c_i in the paper; NOT renamed here)
    x  : object   — numpy ndarray of float32, shape (d_{s},)
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


_REQUIRED_COLUMNS: frozenset[str] = frozenset({"t", "s", "j", "x"})


def load_sample(parquet_path: Path) -> pd.DataFrame:
    """Load a single observation-set Parquet file.

    Returns the DataFrame exactly as stored, with no filtering, imputation,
    or row reordering.  Static observations (``s = 0``, ``t = -1``) are
    included.

    Args:
        parquet_path: Path to a ``.parquet`` file produced by the data
            pipeline (e.g. ``data/fusion_data/test/shot_30451.parquet``).

    Returns:
        DataFrame with columns:

        * ``t``  (float32) — timestamp in seconds
        * ``s``  (int32)   — signal ID
        * ``j``  (int32)   — channel index
        * ``x``  (object)  — numpy ndarray of float32 values

    Raises:
        ValueError: If the file is missing any of the required columns
            ``{t, s, j, x}``.
    """
    df = pd.read_parquet(parquet_path)
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"load_sample: Parquet file {parquet_path} is missing required "
            f"columns: {sorted(missing)}.  Found: {sorted(df.columns)}"
        )
    return df


def get_sigma(train_stats: pd.DataFrame, s: int, j: int, m: int) -> float:
    """Look up the training standard deviation sigma_{s, c, m}.

    Returns the standard deviation of the ``m``-th component of signal ``s``,
    channel ``j`` (== ``c_i`` in paper notation), estimated on the training
    data.

    A safe fallback of ``1e-8`` is returned when:
    - the ``(s, j, m)`` combination is not present in ``train_stats``, or
    - the stored standard deviation is zero or non-finite.

    This prevents division-by-zero in metric computation while still
    producing a meaningful (large) normalised error for constant-valued
    channels.

    Args:
        train_stats: DataFrame with columns
            ``(s: int32, j: int32, m: int32, mean: float32, std: float32)``.
        s: Signal ID (integer).
        j: Channel index (integer; ``c_i`` in the paper).
        m: Dimension index within the observation vector (0-indexed).

    Returns:
        Standard deviation as a Python float.  Falls back to ``1e-8`` if
        the combination is not found or the stored value is degenerate.
    """
    mask = (train_stats["s"] == s) & (train_stats["j"] == j) & (train_stats["m"] == m)
    rows = train_stats.loc[mask, "std"]
    if rows.empty:
        return 1e-8
    sigma = float(rows.iloc[0])
    if not math.isfinite(sigma) or sigma <= 0.0:
        return 1e-8
    return sigma

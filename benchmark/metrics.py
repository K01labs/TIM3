"""Metric computation and hierarchical aggregation.

This module implements the normalised error metrics, applied hierarchically from per-observation up to per-sample level.  Higher levels
(task, domain, benchmark) are handled in ``evaluate.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from benchmark.data import get_sigma

logger = logging.getLogger(__name__)


def nmae_obs(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Compute the per-observation normalised mean absolute error.

    Only finite dimensions of ``x_true`` contribute to the score.  Dimensions
    where ``x_true`` is NaN or infinite are silently excluded.  If no finite
    dimensions remain, ``nan`` is returned.

    Args:
        x_true: Ground-truth observation vector, shape ``(d,)``.
        x_pred: Predicted observation vector, shape ``(d,)``.
        sigma: Per-dimension training standard deviations, shape ``(d,)``. All entries must be strictly positive.

    Returns:
        Scalar NMAE value as a Python float, or ``nan`` if all dimensions are
        non-finite.
    """
    residuals = np.abs(x_true - x_pred) / sigma
    finite_mask = np.isfinite(x_true)
    d_finite = int(finite_mask.sum())
    if d_finite == 0:
        return float("nan")
    return float(residuals[finite_mask].sum() / d_finite)


def nmse_obs(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Compute the per-observation normalised mean squared error.

    Only finite dimensions of ``x_true`` contribute to the score.  Dimensions
    where ``x_true`` is NaN or infinite are silently excluded.  If no finite
    dimensions remain, ``nan`` is returned.

    Args:
        x_true: Ground-truth observation vector, shape ``(d,)``.
        x_pred: Predicted observation vector, shape ``(d,)``.
        sigma: Per-dimension training standard deviations, shape ``(d,)``. All entries must be strictly positive.

    Returns:
        Scalar NMSE value as a Python float, or ``nan`` if all dimensions are
        non-finite.
    """
    finite_mask = np.isfinite(x_true)
    d_finite = int(finite_mask.sum())
    if d_finite == 0:
        return float("nan")
    sq_residuals = ((x_true - x_pred) / sigma) ** 2
    return float(sq_residuals[finite_mask].sum() / d_finite)


# ---------------------------------------------------------------------------
# Prediction alignment
# ---------------------------------------------------------------------------


def align_predictions(
    Q: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Align a predictions DataFrame to the query set on canonicalised ``(t, s, j)``.

    Matching is performed on exact ``(s, j)`` together with ``t`` cast to
    float32 on both sides.  This preserves the benchmark contract that one
    returned prediction row is required for one query observation row: matching
    on the full float32 value (rather than a coarser rounded key) keeps distinct
    high-frequency fusion timestamps from collapsing to the same key.

    Query rows whose canonicalised ``(s, j, t)`` key is absent from
    ``predictions`` receive ``x_pred=None`` (``matched=False``), and the caller
    (``compute_scores``) fills them with the training-mean penalty.  Duplicate
    prediction rows for the same canonicalised key are warned about and
    deduplicated by keeping the first occurrence.

    Args:
        Q: Query set DataFrame with columns ``(t, s, j, x)`` where ``x``
            is a numpy array of float32.
        predictions: Predictions DataFrame with columns
            ``(t, s, j, x_pred)`` where ``x_pred`` is a numpy array of
            float32 with the same length as the corresponding ``x``.

    Returns:
        DataFrame with columns ``(t, s, j, x_true, x_pred, matched)``
        where ``matched`` is a boolean indicating whether a prediction was
        found for that observation.  Unmatched rows have ``x_pred=None``.
    """
    Q_keyed = Q.rename(columns={"x": "x_true"}).copy()
    Q_keyed["_orig_idx"] = np.arange(len(Q_keyed))
    Q_keyed["_t_key"] = Q_keyed["t"].to_numpy(dtype=np.float32)

    if predictions.empty:
        Q_keyed["x_pred"] = None
        Q_keyed["matched"] = False
        return Q_keyed[["t", "s", "j", "x_true", "x_pred", "matched"]].reset_index(
            drop=True
        )

    pred_keyed = predictions[["s", "j", "t", "x_pred"]].copy()
    pred_keyed["_t_key"] = pred_keyed["t"].to_numpy(dtype=np.float32)

    pred_keyed_deduped = pred_keyed.drop_duplicates(
        subset=["s", "j", "_t_key"], keep="first"
    )
    n_dups = len(pred_keyed) - len(pred_keyed_deduped)
    if n_dups > 0:
        logger.warning(
            "align_predictions: dropped %d duplicate prediction row(s) with the same "
            "canonicalised (s, j, t) key — keeping first occurrence.",
            n_dups,
        )

    merged = Q_keyed.merge(
        pred_keyed_deduped[["s", "j", "_t_key", "x_pred"]],
        on=["s", "j", "_t_key"],
        how="left",
    )

    merged["matched"] = merged["x_pred"].notna()

    # Restore original query order before returning.
    merged = merged.sort_values("_orig_idx").reset_index(drop=True)

    return merged[["t", "s", "j", "x_true", "x_pred", "matched"]].reset_index(drop=True)


def _build_sigma_cache(
    train_stats: pd.DataFrame,
    s_j_pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    """Build a per (s, j) sigma array cache from training statistics.

    Precomputes sigma vectors for each unique ``(s, j)`` pair that appears
    in the query set, avoiding repeated full-table scans during row iteration.
    Rows in ``train_stats`` are sorted by ``m`` within each ``(s, j)`` group,
    so the resulting arrays are dimension-ordered.

    Args:
        train_stats: Training statistics DataFrame with columns
            ``(s, j, m, mean, std)``.
        s_j_pairs: List of unique ``(s, j)`` pairs to build sigma vectors for.

    Returns:
        Dict mapping ``(s, j)`` to a numpy float64 array of sigmas, one entry
        per dimension.  Dimensions with zero or non-finite std are replaced
        by ``1e-8``.
    """
    cache: dict[tuple[int, int], np.ndarray] = {}
    for s, j in s_j_pairs:
        mask = (train_stats["s"] == s) & (train_stats["j"] == j)
        sub = train_stats.loc[mask].sort_values("m")
        if sub.empty:
            # Placeholder; actual length unknown until observation is seen.
            cache[(s, j)] = np.array([], dtype=np.float64)
        else:
            stds = sub["std"].to_numpy(dtype=np.float64)
            # Replace degenerate values with safe fallback.
            stds = np.where(np.isfinite(stds) & (stds > 0), stds, 1e-8)
            cache[(s, j)] = stds
    return cache


def _build_mean_cache(
    train_stats: pd.DataFrame,
    s_j_pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    """Build a per (s, j) mean array cache from training statistics.

    Precomputes mean vectors for each unique ``(s, j)`` pair that appears
    in the query set, avoiding repeated full-table scans during row iteration.
    Rows in ``train_stats`` are sorted by ``m`` within each ``(s, j)`` group,
    so the resulting arrays are dimension-ordered.

    Used by ``compute_scores`` to fill unmatched predictions with the training
    mean penalty: a model that fails to predict receives mean imputation as
    its prediction, yielding NMSE ≈ 1.0 in normalised space.

    Args:
        train_stats: Training statistics DataFrame with columns
            ``(s, j, m, mean, std)``.
        s_j_pairs: List of unique ``(s, j)`` pairs to build mean vectors for.

    Returns:
        Dict mapping ``(s, j)`` to a numpy float64 array of means, one entry
        per dimension.  Returns an empty array for ``(s, j)`` pairs not found
        in ``train_stats``.
    """
    cache: dict[tuple[int, int], np.ndarray] = {}
    for s, j in s_j_pairs:
        mask = (train_stats["s"] == s) & (train_stats["j"] == j)
        sub = train_stats.loc[mask].sort_values("m")
        if sub.empty:
            logger.warning(
                "_build_mean_cache: (s=%d, j=%d) not found in train_stats — "
                "unmatched predictions for this pair will use zeros as fallback.",
                s,
                j,
            )
            cache[(s, j)] = np.array([], dtype=np.float64)
        else:
            cache[(s, j)] = sub["mean"].to_numpy(dtype=np.float64)
    return cache


def compute_scores(
    Q: pd.DataFrame,
    predictions: pd.DataFrame,
    train_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-observation NMAE and NMSE scores for a query set.

    Steps:
    1. Align predictions to the query set via ``align_predictions``.
    2. Pre-build per-(s, j) sigma and mean caches to avoid repeated table
       scans.
    3. For each observation, retrieve sigma from the cache.
    4. Unmatched predictions (matched=False) receive ``x_pred = training mean``.
    5. Compute ``nmae_obs`` and ``nmse_obs`` per observation.

    Args:
        Q: Query set DataFrame with columns ``(t, s, j, x)``.
        predictions: Predictions DataFrame with columns
            ``(t, s, j, x_pred)``.
        train_stats: Training statistics DataFrame with columns
            ``(s, j, m, mean, std)``.

    Returns:
        DataFrame with columns ``(t, s, j, nmae, nmse, matched)``.
    """
    aligned = align_predictions(Q, predictions)

    # Pre-build sigma and mean caches for all (s, j) pairs in the query.
    unique_sj = list({(int(r.s), int(r.j)) for r in aligned[["s", "j"]].itertuples()})
    sigma_cache = _build_sigma_cache(train_stats, unique_sj)
    mean_cache = _build_mean_cache(train_stats, unique_sj)

    nmae_vals: list[float] = []
    nmse_vals: list[float] = []
    _warned_no_mean: set[tuple[int, int]] = set()

    for _, row in aligned.iterrows():
        x_true = np.asarray(row["x_true"], dtype=np.float64)
        d = len(x_true)
        s = int(row["s"])
        j = int(row["j"])

        if not row["matched"]:
            # Unmatched: fill x_pred with training mean (mean imputation penalty).
            cached_mean = mean_cache.get((s, j))
            if cached_mean is not None and len(cached_mean) == d:
                x_pred = cached_mean.copy()
            elif cached_mean is not None and len(cached_mean) == 1 and d > 1:
                x_pred = np.full(d, cached_mean[0], dtype=np.float64)
            else:
                # (s, j) absent from train_stats: no mean available, use zeros.
                if (s, j) not in _warned_no_mean:
                    logger.warning(
                        "compute_scores: no training mean for (s=%d, j=%d) — "
                        "using zeros as penalty for unmatched prediction.",
                        s,
                        j,
                    )
                    _warned_no_mean.add((s, j))
                x_pred = np.zeros(d, dtype=np.float64)
        else:
            x_pred = np.asarray(row["x_pred"], dtype=np.float64)
            if len(x_pred) != d:
                # Dimension mismatch: prediction length differs from ground truth.
                # Treat as unmatched and apply training mean penalty.
                logger.warning(
                    "compute_scores: x_pred length %d != x_true length %d for "
                    "(s=%d, j=%d, t=%.4f) — treating as unmatched.",
                    len(x_pred),
                    d,
                    s,
                    j,
                    float(row["t"]),
                )
                cached_mean = mean_cache.get((s, j))
                if cached_mean is not None and len(cached_mean) == d:
                    x_pred = cached_mean.copy()
                elif cached_mean is not None and len(cached_mean) == 1 and d > 1:
                    x_pred = np.full(d, cached_mean[0], dtype=np.float64)
                else:
                    x_pred = np.zeros(d, dtype=np.float64)

        cached = sigma_cache.get((s, j))
        if cached is not None and len(cached) == d:
            # Exact match: use cached sigma array as-is.
            sigma = cached
        elif cached is not None and len(cached) == 1 and d > 1:
            # Variable-length signal: train_stats stores a single global sigma
            # (m=0 only).  Broadcast to all d dimensions.
            sigma = np.full(d, cached[0], dtype=np.float64)
        elif cached is not None and len(cached) > 1 and d != len(cached):
            # Dimension mismatch (unexpected): fall back to per-dim lookup.
            sigma = np.array(
                [get_sigma(train_stats, s, j, m) for m in range(d)],
                dtype=np.float64,
            )
        else:
            # Not in train_stats at all: use fallback 1e-8 for all dims.
            sigma = np.full(d, 1e-8, dtype=np.float64)

        nmae_vals.append(nmae_obs(x_true, x_pred, sigma))
        nmse_vals.append(nmse_obs(x_true, x_pred, sigma))

    scores = aligned[["t", "s", "j", "matched"]].copy()
    scores["nmae"] = nmae_vals
    scores["nmse"] = nmse_vals

    return scores[["t", "s", "j", "nmae", "nmse", "matched"]].reset_index(drop=True)


def aggregate_sample(
    scores: pd.DataFrame,
    signals_registry: dict,
) -> dict:
    """Hierarchically aggregate per-observation scores for a single sample.

    Implements two levels of aggregation:

     - Level 1 — per (s, j) pair
     - Level 2 — sample level


    Args:
        scores: Per-observation scores DataFrame with columns
            ``(t, s, j, nmae, nmse, matched)``, as returned by
            ``compute_scores``.
        signals_registry: Raw ``signals.json`` dict (string-keyed by signal
            ID) used to map integer signal IDs to human-readable names for
            the ``per_signal_channel`` output keys.

    Returns:
        Dict with the following structure::

            {
                "nmae": float,              # L^NMAE_k
                "nrmse": float,             # L^NRMSE_k
                "n_signal_channels": int,   # |I_k|
                "per_signal_channel": {
                    "<signal_name>/<j>": {
                        "nmae": float,      # L^NMAE_{k,(s,j)}
                        "nrmse": float,     # L^NRMSE_{k,(s,j)}
                        "n_obs": int,
                        "signal_name": str,
                        "j": int,
                    },
                    ...
                },
            }
    """
    per_sc: dict[str, dict] = {}
    sc_nmae_vals: list[float] = []
    sc_nrmse_vals: list[float] = []

    for _, group in scores.groupby(["s", "j"]):
        s = int(group["s"].iloc[0])
        j = int(group["j"].iloc[0])

        sc_nmae = float(group["nmae"].mean())
        sc_nrmse = float(np.sqrt(group["nmse"].mean()))

        # Resolve human-readable signal name from the registry.
        # Registry keys are strings; values have a "name" field.
        sig_name = signals_registry.get(str(s), {}).get("name", f"signal_{s}")
        key = f"{sig_name}/{j}"

        per_sc[key] = {
            "nmae": sc_nmae,
            "nrmse": sc_nrmse,
            "n_obs": len(group),
            "signal_name": sig_name,
            "j": j,
        }

        sc_nmae_vals.append(sc_nmae)
        sc_nrmse_vals.append(sc_nrmse)

    n_sc = len(sc_nmae_vals)
    n_nan_sc = sum(1 for v in sc_nmae_vals if not np.isfinite(v))
    # Drop NaN signal channels from the sample-level aggregate so that a single
    # completely-offline sensor does not poison the whole sample score.
    sample_nmae = float(np.nanmean(sc_nmae_vals)) if sc_nmae_vals else float("nan")
    sample_nrmse = float(np.nanmean(sc_nrmse_vals)) if sc_nrmse_vals else float("nan")

    return {
        "nmae": sample_nmae,
        "nrmse": sample_nrmse,
        "n_signal_channels": n_sc,
        "n_nan_signal_channels": n_nan_sc,
        "per_signal_channel": per_sc,
    }

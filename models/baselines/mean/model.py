"""Mean baseline: predict the per-(s, j) training mean for every query point.

Context- and task-independent. Because errors are normalised by the per-signal
training standard deviation, this baseline scores NRMSE ~ 1.0 by construction
(and NMAE ~ 0.8 for Gaussian signals). It is the reference point any
context-aware model should beat.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from benchmark.specs import BenchmarkSpec
from models.base import Model

logger = logging.getLogger(__name__)


def predict_mean(
    query_df: pd.DataFrame,
    train_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Predict the training mean vector for every query observation.

    For each row in query_df the function looks up the per-(s, j) mean vector
    from train_stats.  Variable-length signals (where train_stats stores only
    m=0) are handled by broadcasting the scalar mean across all dimensions of
    the query observation's x vector.

    Args:
        query_df: Query set DataFrame with columns (t, s, j, x) where x is a
            numpy float32 array.  Timestamps are copied verbatim into the
            output
        train_stats: Training statistics DataFrame with columns
            (s: int32, j: int32, m: int32, mean: float32, std: float32).

    Returns:
        DataFrame with columns (t, s, j, x_pred) where x_pred is a numpy
        float32 array of the same length as the corresponding x in query_df.
        Timestamps t are copied exactly from query_df.
    """
    unique_sj: list[tuple[int, int]] = list(
        {(int(r.s), int(r.j)) for r in query_df[["s", "j"]].itertuples()}
    )

    mean_cache: dict[tuple[int, int], np.ndarray | None] = {}
    for s, j in unique_sj:
        mask = (train_stats["s"] == s) & (train_stats["j"] == j)
        sub = train_stats.loc[mask].sort_values("m")
        if sub.empty:
            logger.warning(
                "predict_mean: no training stats for (s=%d, j=%d) — predicting zeros.",
                s,
                j,
            )
            mean_cache[(s, j)] = None
        else:
            mean_cache[(s, j)] = sub["mean"].to_numpy(dtype=np.float32)

    t_vals: list[float] = []
    s_vals: list[int] = []
    j_vals: list[int] = []
    x_pred_vals: list[np.ndarray] = []

    for _, row in query_df.iterrows():
        s = int(row["s"])
        j = int(row["j"])
        d = len(row["x"])

        mean_vec = mean_cache.get((s, j))
        if mean_vec is None:
            x_pred = np.zeros(d, dtype=np.float32)
        elif len(mean_vec) == d:
            x_pred = mean_vec.copy()
        elif len(mean_vec) == 1:
            # Variable-length signal: broadcast the single stored mean.
            x_pred = np.full(d, mean_vec[0], dtype=np.float32)
        else:
            logger.warning(
                "predict_mean: dimension mismatch for (s=%d, j=%d): "
                "train_stats has %d dims but query x has %d — predicting zeros.",
                s,
                j,
                len(mean_vec),
                d,
            )
            x_pred = np.zeros(d, dtype=np.float32)

        t_vals.append(row["t"])
        s_vals.append(s)
        j_vals.append(j)
        x_pred_vals.append(x_pred)

    return pd.DataFrame({"t": t_vals, "s": s_vals, "j": j_vals, "x_pred": x_pred_vals})


class MeanModel(Model):
    """Training-mean baseline. Context-independent: predicts the per-(s, j)
    training mean for every query observation, on every task."""

    name = "mean"

    def predict(
        self, context: pd.DataFrame, query: pd.DataFrame, spec: BenchmarkSpec
    ) -> pd.DataFrame:
        return predict_mean(query, spec.domain.train_stats)

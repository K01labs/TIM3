"""Last-observation carry-forward (LOCF) baseline.

For each query point, finds the most recent context observation with the same
(s, j) and a strictly earlier timestamp.  Uses np.searchsorted for O(log N_C)
per-query lookup after a one-time sort of context observations per (s, j) group.
Operates directly on the native irregular timestamps — no gridding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.specs import BenchmarkSpec
from models.base import Model


def predict_last_observation(
    context_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict each query value by carrying forward the last context observation.

    For each query row (s, j, t_q), finds the context observation with the same
    (s, j) and the largest t_c < t_q (strict predecessor).  If no strict
    predecessor exists but a forward observation does (t_c > t_q), uses it as a
    fallback — this only activates for offline imputation with t_a = 0 where the
    entire signal is masked.  If neither exists the query row is omitted and the
    evaluation harness applies the training-mean penalty.

    Complexity is O(U_Q * log N_C) after an O(N_C log N_C) pre-sort, where U_Q
    is the number of unique (s, j) pairs in the query and N_C is the context size.

    Note on cross-modal tasks: CrossModalSpec enforces source_signals ∩
    target_signals = ∅.  The context contains only source-signal observations
    while the query contains only target-signal observations.  Therefore no (s, j)
    pair in the query ever appears in the context, and this function returns an
    empty DataFrame for cross-modal tasks by construction — not a bug.

    Args:
        context_df: Context set DataFrame with columns (t, s, j, x) where x is a
            numpy float32 array.
        query_df: Query set DataFrame with columns (t, s, j, x) where x is a
            numpy float32 array.  The x values are ground truth (unused here).

    Returns:
        DataFrame with columns (t, s, j, x_pred) for query rows that had at least
        one predecessor (or forward fallback) in the context.  Query rows with no
        matching context observation are omitted.
    """
    # Pre-group context by (s, j); sort each group by t once.
    context_lookup: dict[tuple[int, int], tuple[np.ndarray, list[np.ndarray]]] = {}
    ctx_pairs: list[tuple[int, int]] = list(
        {(int(r.s), int(r.j)) for r in context_df[["s", "j"]].itertuples()}
    )
    for s, j in ctx_pairs:
        sub = context_df[(context_df["s"] == s) & (context_df["j"] == j)]
        sub_sorted = sub.sort_values("t")
        t_arr = sub_sorted["t"].to_numpy(dtype=np.float64)
        x_list = list(sub_sorted["x"])
        context_lookup[(s, j)] = (t_arr, x_list)

    t_vals: list[float] = []
    s_vals: list[int] = []
    j_vals: list[int] = []
    x_pred_vals: list[np.ndarray] = []

    query_pairs: list[tuple[int, int]] = list(
        {(int(r.s), int(r.j)) for r in query_df[["s", "j"]].itertuples()}
    )

    for s, j in query_pairs:
        if (s, j) not in context_lookup:
            # No context for this (s, j); omit — harness applies training-mean penalty.
            # This is the expected path for cross-modal tasks (see docstring).
            continue

        t_arr, x_list = context_lookup[(s, j)]
        q_sub = query_df[(query_df["s"] == s) & (query_df["j"] == j)]
        q_t = q_sub["t"].to_numpy(dtype=np.float64)

        # Vectorised backward search: idx = last context index with t_c < t_q.
        idxs = np.searchsorted(t_arr, q_t, side="left") - 1

        for i, (_, row) in enumerate(q_sub.iterrows()):
            idx = idxs[i]
            if idx >= 0:
                # Strict predecessor found.
                x_pred = np.asarray(x_list[idx], dtype=np.float32)
            else:
                # No strict predecessor — try forward fallback (offline imputation
                # with t_a = 0 edge case).  Do NOT apply for causal tasks; the
                # partition already ensures causal context has no future observations,
                # so this branch simply won't fire for causal tasks.
                fwd_idx = int(np.searchsorted(t_arr, q_t[i], side="right"))
                if fwd_idx < len(t_arr):
                    x_pred = np.asarray(x_list[fwd_idx], dtype=np.float32)
                else:
                    # No context at all for this query timestamp; omit.
                    continue

            # Variable-length signals (e.g. TCPrimed swath imagery) may change
            # dimension between context and query timesteps.  A wrong-sized
            # carry-forward is meaningless — omit so the evaluator applies the
            # training-mean penalty instead.
            x_true_dim = len(np.asarray(row["x"]))
            if len(x_pred) != x_true_dim:
                continue

            t_vals.append(float(row["t"]))
            s_vals.append(s)
            j_vals.append(j)
            x_pred_vals.append(x_pred)

    return pd.DataFrame({"t": t_vals, "s": s_vals, "j": j_vals, "x_pred": x_pred_vals})


class LOCFModel(Model):
    """Last-observation-carry-forward baseline. Carries the most recent context
    value of each (s, j) forward to the query timestamps (no gridding)."""

    name = "locf"

    def predict(
        self, context: pd.DataFrame, query: pd.DataFrame, spec: BenchmarkSpec
    ) -> pd.DataFrame:
        return predict_last_observation(context, query)

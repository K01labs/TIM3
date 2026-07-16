"""Context/query partition functions for each benchmark task."""

from __future__ import annotations

import math

import pandas as pd

from benchmark.specs import (
    CausalImputationSpec,
    CrossModalSpec,
    DomainSpec,
    ForecastingSpec,
    OfflineImputationSpec,
)


def _split_static(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate static from temporal observations.

    Args:
        obs: Full observation DataFrame with columns ``(t, s, j, x)``.

    Returns:
        A tuple ``(static_rows, temporal_rows)`` where ``static_rows``
        contains all rows with ``s == 0`` and ``temporal_rows`` all others.
    """
    is_static = obs["s"] == 0
    return obs[is_static], obs[~is_static]


def _assert_partition(
    temporal: pd.DataFrame,
    context_temporal: pd.DataFrame,
    query: pd.DataFrame,
    *,
    allow_dropped: bool = False,
) -> None:
    """Assert that context and query form a valid partition of the input.

    Checks:
    1. No row appears in both context_temporal and query.
    2. Unless ``allow_dropped`` is True, every temporal row appears in
       exactly one of context_temporal or query.

    Args:
        temporal: The full set of temporal rows before partitioning.
        context_temporal: The temporal portion of the context set
            (excluding static rows, which are always in C).
        query: The query set.
        allow_dropped: If ``True``, skip the completeness check.  Use this
            for tasks that deliberately drop observations (e.g. forecasting
            drops observations outside the horizon band).
    """
    ctx_idx = set(context_temporal.index)
    q_idx = set(query.index)
    overlap = ctx_idx & q_idx
    assert not overlap, (
        f"_assert_partition: {len(overlap)} rows appear in both C and Q. "
        f"Sample indices: {list(overlap)[:5]}"
    )
    if not allow_dropped:
        all_idx = set(temporal.index)
        covered = ctx_idx | q_idx
        missing = all_idx - covered
        assert not missing, (
            f"_assert_partition: {len(missing)} temporal rows are in neither "
            f"C nor Q. Sample indices: {list(missing)[:5]}"
        )


def partition_forecasting(
    obs: pd.DataFrame,
    spec: ForecastingSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition observations for the forecasting task.

    Args:
        obs: Full observation DataFrame with columns ``(t, s, j, x)``.
        spec: Forecasting task parameters.

    Returns:
        Tuple ``(C, Q)`` where both DataFrames have the same schema as
        ``obs``.  Static rows are prepended to C.
    """
    static, temporal = _split_static(obs)

    t = temporal["t"]
    context_mask = t < spec.t_c
    horizon_offset = t - spec.t_c
    query_mask = (
        (t >= spec.t_c)
        & (horizon_offset >= spec.t_low)
        & (horizon_offset <= spec.t_high)
    )

    context_temporal = temporal[context_mask]
    query = temporal[query_mask]

    _assert_partition(temporal, context_temporal, query, allow_dropped=True)

    context = pd.concat([static, context_temporal], ignore_index=True)
    return context, query


def partition_offline_imputation(
    obs: pd.DataFrame,
    spec: OfflineImputationSpec,
    domain: DomainSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition observations for the offline imputation task.
    If ``spec.held_out_intervals`` is ``None``, the entire time range of each
    held-out signal is masked (equivalent to ``t_a = -inf``, ``t_b = +inf``).

    Args:
        obs: Full observation DataFrame with columns ``(t, s, j, x)``.
        spec: Offline imputation task parameters.
        domain: Domain spec used to resolve signal names to integer IDs.

    Returns:
        Tuple ``(C, Q)`` where both DataFrames have the same schema as
        ``obs``.  Static rows are prepended to C.
    """
    static, temporal = _split_static(obs)

    held_out_ids = set(domain.signal_ids(spec.held_out_signals))

    is_heldout = temporal["s"].isin(held_out_ids)

    if spec.held_out_intervals is None:
        # Entire duration of held-out signals goes to Q.
        is_query = is_heldout
    else:
        # Resolve interval dict by signal name -> signal ID.
        interval_by_id: dict[int, tuple[float, float]] = {
            domain.signal_id(name): interval
            for name, interval in spec.held_out_intervals.items()
        }

        # Build per-row t_a and t_b arrays using vectorised map.
        # Rows whose signal is not held out get t_a = +inf, t_b = -inf so the
        # interval condition is always False.
        t_a_map = {sid: iv[0] for sid, iv in interval_by_id.items()}
        t_b_map = {sid: iv[1] for sid, iv in interval_by_id.items()}

        row_t_a = temporal["s"].map(t_a_map).fillna(math.inf)
        row_t_b = temporal["s"].map(t_b_map).fillna(-math.inf)

        t = temporal["t"]
        in_interval = (t >= row_t_a) & (t <= row_t_b)
        is_query = is_heldout & in_interval

    context_temporal = temporal[~is_query]
    query = temporal[is_query]

    _assert_partition(temporal, context_temporal, query)

    context = pd.concat([static, context_temporal], ignore_index=True)
    return context, query


def partition_causal_imputation(
    obs: pd.DataFrame,
    spec: CausalImputationSpec,
    domain: DomainSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition observations for the causal imputation task.
    Observations with ``t_i > t_p`` are discarded entirely from both C and Q,
    enforcing the causal (real-time) constraint.

    Args:
        obs: Full observation DataFrame with columns ``(t, s, j, x)``.
        spec: Causal imputation task parameters.
        domain: Domain spec used to resolve signal names to integer IDs.

    Returns:
        Tuple ``(C, Q)`` where both DataFrames have the same schema as
        ``obs``.  Static rows are prepended to C.
    """
    static, temporal = _split_static(obs)

    # Enforce causal constraint: discard everything after t_present.
    causal_temporal = temporal[temporal["t"] <= spec.t_present]

    held_out_ids = set(domain.signal_ids(spec.held_out_signals))

    # Resolve t_failure from signal names to signal IDs.
    t_failure_by_id: dict[int, float] = {
        domain.signal_id(name): t_f for name, t_f in spec.t_failure.items()
    }

    # Vectorised per-row failure time lookup.
    # Non-held-out signals get t_f = +inf so the failure condition is always False.
    row_t_fail = causal_temporal["s"].map(t_failure_by_id).fillna(math.inf)

    is_heldout = causal_temporal["s"].isin(held_out_ids)
    is_query = is_heldout & (causal_temporal["t"] >= row_t_fail)

    context_temporal = causal_temporal[~is_query]
    query = causal_temporal[is_query]

    # Sanity: causal_temporal ∪ dropped_future = temporal, but C and Q only
    # cover causal_temporal, so we check partition within causal_temporal.
    _assert_partition(causal_temporal, context_temporal, query)

    # Safety check: no future observations leaked into context.
    assert context_temporal["t"].le(spec.t_present).all(), (
        "partition_causal_imputation: context contains observations with "
        f"t > t_present ({spec.t_present})"
    )

    context = pd.concat([static, context_temporal], ignore_index=True)
    return context, query


def partition_cross_modal(
    obs: pd.DataFrame,
    spec: CrossModalSpec,
    domain: DomainSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition observations for the cross-modal prediction task.

    Observations belonging to neither source nor target signals are dropped
    from both C and Q.  Disjointness of source and target is already enforced
    at ``CrossModalSpec`` construction time.

    This is an offline task: the model may use all source-signal observations
    to predict all target-signal observations (no temporal causal constraint).

    Args:
        obs: Full observation DataFrame with columns ``(t, s, j, x)``.
        spec: Cross-modal prediction task parameters.
        domain: Domain spec used to resolve signal names to integer IDs.

    Returns:
        Tuple ``(C, Q)`` where both DataFrames have the same schema as
        ``obs``.  Static rows are prepended to C.
    """
    static, temporal = _split_static(obs)

    src_ids = set(domain.signal_ids(spec.source_signals))
    tgt_ids = set(domain.signal_ids(spec.target_signals))

    is_source = temporal["s"].isin(src_ids)
    is_target = temporal["s"].isin(tgt_ids)

    context_temporal = temporal[is_source]
    query = temporal[is_target]

    # Source and target are disjoint by CrossModalSpec invariant.
    _assert_partition(temporal, context_temporal, query, allow_dropped=True)

    context = pd.concat([static, context_temporal], ignore_index=True)
    return context, query

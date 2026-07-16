"""Tests for the benchmark evaluation suite.

All tests use synthetic in-memory data only — no network access, no real
data files.

Synthetic data conventions:
- Signals s=1..n_signals are temporal, scalar (d=1) unless noted.
- Signals s=4,5 have two channels (j=0 and j=1).
- Static rows have s=0, t=-1.0, j=0..2, x=[float].
- All train_stats have std=1.0 so sigma cancels out.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from benchmark.data import get_sigma
from benchmark.evaluate import evaluate_benchmark, evaluate_domain
from benchmark.metrics import (
    aggregate_sample,
    align_predictions,
    compute_scores,
    nmae_obs,
    nmse_obs,
)
from benchmark.specs import (
    CausalImputationSpec,
    CrossModalSpec,
    DomainSpec,
    ForecastingSpec,
    OfflineImputationSpec,
)
from benchmark.tasks import (
    partition_causal_imputation,
    partition_cross_modal,
    partition_forecasting,
    partition_offline_imputation,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_obs(n_signals: int = 5, n_timesteps: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic observation DataFrame.

    Structure:
    - 3 static rows: s=0, t=-1.0, j=0/1/2, x=[float]
    - n_signals * n_timesteps temporal rows:
        - s=1..3: scalar, j=0 only (d=1)
        - s=4..5: scalar, j=0 and j=1 (multichannel; 2*n_timesteps rows each)
      Timestamps are linspace(0, 1, n_timesteps).

    Args:
        n_signals: Number of distinct temporal signal IDs (1-indexed).
        n_timesteps: Number of timesteps per signal-channel pair.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns (t: float32, s: int32, j: int32, x: object).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # Static rows.
    for jj in range(3):
        rows.append(
            {
                "t": np.float32(-1.0),
                "s": np.int32(0),
                "j": np.int32(jj),
                "x": np.array([rng.standard_normal()], dtype=np.float32),
            }
        )

    t_vals = np.linspace(0.0, 1.0, n_timesteps, dtype=np.float32)

    for sig in range(1, n_signals + 1):
        # Signals 4 and 5 are multichannel (j=0 and j=1).
        channels = [0, 1] if sig >= 4 else [0]
        for jj in channels:
            for t in t_vals:
                rows.append(
                    {
                        "t": t,
                        "s": np.int32(sig),
                        "j": np.int32(jj),
                        "x": np.array([rng.standard_normal()], dtype=np.float32),
                    }
                )

    return pd.DataFrame(rows)


def make_train_stats(
    signals: list[int],
    j_max: dict[int, int] | None = None,
    d: int = 1,
) -> pd.DataFrame:
    """Create synthetic train_stats with std=1.0 for all (s, j, m) entries.

    Args:
        signals: List of signal IDs to include.
        j_max: Mapping from signal ID to maximum channel index (inclusive).
            Defaults to {s: 0} for all signals (single channel).
        d: Number of dimensions per signal (same for all signals).

    Returns:
        DataFrame with columns (s, j, m, mean, std).
    """
    if j_max is None:
        j_max = {s: 0 for s in signals}

    rows = []
    for s in signals:
        for jj in range(j_max.get(s, 0) + 1):
            for m in range(d):
                rows.append(
                    {
                        "s": np.int32(s),
                        "j": np.int32(jj),
                        "m": np.int32(m),
                        "mean": np.float32(0.0),
                        "std": np.float32(1.0),
                    }
                )
    return pd.DataFrame(rows)


def make_domain(
    signal_names: dict[int, str] | None = None,
    n_signals: int = 5,
) -> DomainSpec:
    """Create a minimal DomainSpec backed by synthetic data.

    Args:
        signal_names: Mapping from integer signal ID to name.  Defaults to
            ``{1: "sig_1", ..., n_signals: "sig_<n>"}``.
        n_signals: Used when signal_names is None.

    Returns:
        DomainSpec with in-memory registry and train_stats.
    """
    if signal_names is None:
        signal_names = {i: f"sig_{i}" for i in range(1, n_signals + 1)}

    registry = {str(sid): {"name": name} for sid, name in signal_names.items()}
    j_max = {4: 1, 5: 1}
    train_stats = make_train_stats(
        list(signal_names.keys()),
        j_max={k: v for k, v in j_max.items() if k in signal_names},
    )

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        return DomainSpec(
            name="test_domain",
            data_root=Path(tmp),
            registry=registry,
            train_stats=train_stats,
        )


# ---------------------------------------------------------------------------
# Specs tests
# ---------------------------------------------------------------------------


def test_domain_spec_signal_id() -> None:
    domain = make_domain()
    assert domain.signal_id("sig_1") == 1
    assert domain.signal_id("sig_5") == 5


def test_domain_spec_signal_id_missing() -> None:
    domain = make_domain()
    with pytest.raises(KeyError):
        domain.signal_id("does_not_exist")


def test_domain_spec_signal_ids() -> None:
    domain = make_domain()
    assert domain.signal_ids(["sig_1", "sig_3"]) == [1, 3]


def test_cross_modal_spec_disjoint_raises() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        CrossModalSpec(
            source_signals=["sig_1", "sig_2"],
            target_signals=["sig_2", "sig_3"],
        )


def test_cross_modal_spec_valid() -> None:
    spec = CrossModalSpec(source_signals=["sig_1"], target_signals=["sig_2"])
    assert spec.source_signals == ["sig_1"]
    assert spec.target_signals == ["sig_2"]


# ---------------------------------------------------------------------------
# Helper: partition all four tasks on a shared observation set
# ---------------------------------------------------------------------------


def _make_partition_fixtures() -> tuple[pd.DataFrame, DomainSpec]:
    """Return (obs, domain) ready for partition tests."""
    obs = make_obs(n_signals=5, n_timesteps=50)
    domain = make_domain()
    return obs, domain


# ---------------------------------------------------------------------------
# test_static_always_in_context
# ---------------------------------------------------------------------------


def test_static_always_in_context_forecasting() -> None:
    obs, _ = _make_partition_fixtures()
    spec = ForecastingSpec(t_c=0.5, t_low=0.0, t_high=0.5)
    C, Q = partition_forecasting(obs, spec)
    assert (C["s"] == 0).sum() == 3, "All 3 static rows must be in C"
    assert (Q["s"] == 0).sum() == 0, "No static rows may be in Q"


def test_static_always_in_context_offline_imputation() -> None:
    obs, domain = _make_partition_fixtures()
    spec = OfflineImputationSpec(held_out_signals=["sig_1", "sig_2"])
    C, Q = partition_offline_imputation(obs, spec, domain)
    assert (C["s"] == 0).sum() == 3
    assert (Q["s"] == 0).sum() == 0


def test_static_always_in_context_causal_imputation() -> None:
    obs, domain = _make_partition_fixtures()
    spec = CausalImputationSpec(
        held_out_signals=["sig_1"],
        t_failure={"sig_1": 0.4},
        t_present=0.8,
    )
    C, Q = partition_causal_imputation(obs, spec, domain)
    assert (C["s"] == 0).sum() == 3
    assert (Q["s"] == 0).sum() == 0


def test_static_always_in_context_cross_modal() -> None:
    obs, domain = _make_partition_fixtures()
    spec = CrossModalSpec(
        source_signals=["sig_1", "sig_2", "sig_3"],
        target_signals=["sig_4", "sig_5"],
    )
    C, Q = partition_cross_modal(obs, spec, domain)
    assert (C["s"] == 0).sum() == 3
    assert (Q["s"] == 0).sum() == 0


# ---------------------------------------------------------------------------
# test_partition_disjoint
# ---------------------------------------------------------------------------


def _temporal_overlap(C: pd.DataFrame, Q: pd.DataFrame) -> int:
    """Return number of temporal rows appearing in both C and Q by value."""
    # Compare by (t, s, j) tuples to be robust against reset_index in concat.
    c_keys = set(
        zip(C.loc[C["s"] != 0, "t"], C.loc[C["s"] != 0, "s"], C.loc[C["s"] != 0, "j"])
    )
    q_keys = set(zip(Q["t"], Q["s"], Q["j"]))
    return len(c_keys & q_keys)


def test_partition_disjoint_forecasting() -> None:
    obs, _ = _make_partition_fixtures()
    spec = ForecastingSpec(t_c=0.5, t_low=0.0, t_high=0.3)
    C, Q = partition_forecasting(obs, spec)
    assert _temporal_overlap(C, Q) == 0


def test_partition_disjoint_offline_imputation() -> None:
    obs, domain = _make_partition_fixtures()
    spec = OfflineImputationSpec(held_out_signals=["sig_1"])
    C, Q = partition_offline_imputation(obs, spec, domain)
    assert _temporal_overlap(C, Q) == 0


def test_partition_disjoint_causal_imputation() -> None:
    obs, domain = _make_partition_fixtures()
    spec = CausalImputationSpec(
        held_out_signals=["sig_1"],
        t_failure={"sig_1": 0.4},
        t_present=0.8,
    )
    C, Q = partition_causal_imputation(obs, spec, domain)
    assert _temporal_overlap(C, Q) == 0


def test_partition_disjoint_cross_modal() -> None:
    obs, domain = _make_partition_fixtures()
    spec = CrossModalSpec(
        source_signals=["sig_1", "sig_2"],
        target_signals=["sig_3", "sig_4"],
    )
    C, Q = partition_cross_modal(obs, spec, domain)
    assert _temporal_overlap(C, Q) == 0


# ---------------------------------------------------------------------------
# test_partition_union_is_complete
# ---------------------------------------------------------------------------


def test_partition_union_complete_offline_imputation() -> None:
    obs, domain = _make_partition_fixtures()
    spec = OfflineImputationSpec(held_out_signals=["sig_2"])
    C, Q = partition_offline_imputation(obs, spec, domain)

    temporal = obs[obs["s"] != 0]
    c_temporal = C[C["s"] != 0]
    covered = len(c_temporal) + len(Q)
    assert covered == len(temporal), (
        f"Offline imputation: {covered} covered, {len(temporal)} expected"
    )


def test_partition_union_complete_causal_imputation() -> None:
    obs, domain = _make_partition_fixtures()
    t_present = 0.8
    spec = CausalImputationSpec(
        held_out_signals=["sig_1"],
        t_failure={"sig_1": 0.4},
        t_present=t_present,
    )
    C, Q = partition_causal_imputation(obs, spec, domain)

    temporal = obs[obs["s"] != 0]
    # Only observations with t <= t_present are in scope.
    causal_temporal = temporal[temporal["t"] <= t_present]
    c_temporal = C[C["s"] != 0]
    covered = len(c_temporal) + len(Q)
    assert covered == len(causal_temporal), (
        f"Causal imputation: {covered} covered within t<=t_present, "
        f"{len(causal_temporal)} expected"
    )


def test_partition_forecasting_drops_outside_horizon() -> None:
    """Observations outside the horizon band must not appear in C or Q."""
    obs, _ = _make_partition_fixtures()
    # Tight horizon: only t in [0.55, 0.65] relative to t_c=0.5
    spec = ForecastingSpec(t_c=0.5, t_low=0.05, t_high=0.15)
    C, Q = partition_forecasting(obs, spec)

    temporal = obs[obs["s"] != 0]
    in_scope = temporal[temporal["t"] < 0.5]  # context
    horizon_obs = temporal[
        (temporal["t"] >= 0.5)
        & (temporal["t"] - 0.5 >= 0.05)
        & (temporal["t"] - 0.5 <= 0.15)
    ]
    assert len(C[C["s"] != 0]) == len(in_scope)
    assert len(Q) == len(horizon_obs)


# ---------------------------------------------------------------------------
# Per-observation metric tests
# ---------------------------------------------------------------------------


def test_nmae_obs_hand_computed() -> None:
    x_true = np.array([2.0])
    x_pred = np.array([0.0])
    sigma = np.array([1.0])
    assert nmae_obs(x_true, x_pred, sigma) == pytest.approx(2.0)


def test_nmse_obs_hand_computed() -> None:
    x_true = np.array([2.0])
    x_pred = np.array([0.0])
    sigma = np.array([1.0])
    assert nmse_obs(x_true, x_pred, sigma) == pytest.approx(4.0)


def test_nmae_obs_multidim() -> None:
    # d=3: |errors| = [1, 2, 3], sigma=[1,1,1] -> nmae = (1+2+3)/3 = 2.0
    x_true = np.array([1.0, 2.0, 3.0])
    x_pred = np.array([0.0, 0.0, 0.0])
    sigma = np.array([1.0, 1.0, 1.0])
    assert nmae_obs(x_true, x_pred, sigma) == pytest.approx(2.0)


def test_nmse_obs_multidim() -> None:
    # d=2: squared errors / sigma^2 = [1, 4], nmse = (1+4)/2 = 2.5
    x_true = np.array([1.0, 2.0])
    x_pred = np.array([0.0, 0.0])
    sigma = np.array([1.0, 1.0])
    assert nmse_obs(x_true, x_pred, sigma) == pytest.approx(2.5)


def test_nrmse_is_sqrt_of_mean_nmse() -> None:
    """NRMSE must be sqrt(mean(NMSE)), not mean(sqrt(NMSE))."""
    scores = pd.DataFrame(
        {
            "t": [0.1, 0.2],
            "s": [1, 1],
            "j": [0, 0],
            "nmae": [1.0, 1.0],
            "nmse": [1.0, 9.0],  # mean = 5.0, sqrt(mean) = sqrt(5) ≈ 2.236
            "matched": [True, True],
        }
    )
    registry = {"1": {"name": "sig_1"}}
    result = aggregate_sample(scores, registry)

    # NRMSE at signal-channel level: sqrt(mean([1.0, 9.0])) = sqrt(5)
    sc = result["per_signal_channel"]["sig_1/0"]
    assert sc["nrmse"] == pytest.approx(math.sqrt(5.0))

    # Confirm it's NOT mean(sqrt): mean([1, 3]) = 2.0 != sqrt(5)
    assert abs(sc["nrmse"] - 2.0) > 0.1


# ---------------------------------------------------------------------------
# align_predictions tests
# ---------------------------------------------------------------------------


def test_align_predictions_missing() -> None:
    """Query has 5 obs; 3 exact matches and 2 rows in an absent (s, j) group."""
    Q = pd.DataFrame(
        {
            "t": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            "s": np.array([1, 1, 1, 2, 2], dtype=np.int32),
            "j": np.array([0, 0, 0, 0, 0], dtype=np.int32),
            "x": [np.array([float(i)], dtype=np.float32) for i in range(5)],
        }
    )
    preds = pd.DataFrame(
        {
            "t": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "s": np.array([1, 1, 1], dtype=np.int32),
            "j": np.array([0, 0, 0], dtype=np.int32),
            "x_pred": [np.array([float(i)], dtype=np.float32) for i in range(3)],
        }
    )
    aligned = align_predictions(Q, preds)
    assert len(aligned) == 5
    assert aligned["matched"].sum() == 3
    assert (~aligned["matched"]).sum() == 2
    # Unmatched rows have x_pred=None (caller fills with training mean).
    for _, row in aligned[~aligned["matched"]].iterrows():
        assert pd.isna(row["x_pred"])


def test_align_predictions_float32_canonicalization() -> None:
    """Copied timestamps match after canonicalising both sides to float32."""
    Q = pd.DataFrame(
        {
            "t": np.array([0.10000, 0.20000], dtype=np.float32),
            "s": np.array([1, 1], dtype=np.int32),
            "j": np.array([0, 0], dtype=np.int32),
            "x": [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)],
        }
    )
    # The query carries float32 values; a model may return plain float64 values
    # copied from those coordinates.
    preds = pd.DataFrame(
        {
            "t": np.array([0.1, 0.2], dtype=np.float64),
            "s": np.array([1, 1], dtype=np.int32),
            "j": np.array([0, 0], dtype=np.int32),
            "x_pred": [
                np.array([1.0], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
            ],
        }
    )
    aligned = align_predictions(Q, preds)
    assert aligned["matched"].all(), (
        f"Expected all matched; got {aligned['matched'].tolist()}"
    )


def test_align_predictions_all_missing() -> None:
    """When no predictions are provided, all rows are unmatched with x_pred=None."""
    Q = pd.DataFrame(
        {
            "t": np.array([0.1, 0.2], dtype=np.float32),
            "s": np.array([1, 1], dtype=np.int32),
            "j": np.array([0, 0], dtype=np.int32),
            "x": [np.array([3.0], dtype=np.float32), np.array([4.0], dtype=np.float32)],
        }
    )
    preds = pd.DataFrame(columns=["t", "s", "j", "x_pred"])
    aligned = align_predictions(Q, preds)
    assert len(aligned) == 2
    assert not aligned["matched"].any()


def test_align_predictions_missing_timestamp_stays_unmatched() -> None:
    """Rows without an exact canonicalised timestamp match remain unmatched."""
    Q = pd.DataFrame(
        {
            "t": np.array([0.1, 0.5], dtype=np.float32),
            "s": np.array([1, 1], dtype=np.int32),
            "j": np.array([0, 0], dtype=np.int32),
            "x": [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)],
        }
    )
    preds = pd.DataFrame(
        {
            "t": np.array([0.1], dtype=np.float32),
            "s": np.array([1], dtype=np.int32),
            "j": np.array([0], dtype=np.int32),
            "x_pred": [np.array([1.0], dtype=np.float32)],
        }
    )
    aligned = align_predictions(Q, preds)
    assert len(aligned) == 2
    assert aligned["matched"].tolist() == [True, False], (
        f"Expected only the exact timestamp matched; got {aligned['matched'].tolist()}"
    )


def test_align_predictions_high_frequency_rows_do_not_collapse() -> None:
    """Distinct dense fusion timestamps must survive alignment one-to-one."""
    Q = pd.DataFrame(
        {
            "t": np.array([0.150000, 0.150002, 0.150004], dtype=np.float32),
            "s": np.array([1, 1, 1], dtype=np.int32),
            "j": np.array([0, 0, 0], dtype=np.int32),
            "x": [np.array([1.0], dtype=np.float32) for _ in range(3)],
        }
    )
    preds = pd.DataFrame(
        {
            "t": np.array([0.150000, 0.150002, 0.150004], dtype=np.float32),
            "s": np.array([1, 1, 1], dtype=np.int32),
            "j": np.array([0, 0, 0], dtype=np.int32),
            "x_pred": [
                np.array([10.0], dtype=np.float32),
                np.array([20.0], dtype=np.float32),
                np.array([30.0], dtype=np.float32),
            ],
        }
    )
    aligned = align_predictions(Q, preds)
    assert aligned["matched"].all()
    got = [float(np.asarray(v)[0]) for v in aligned["x_pred"]]
    assert got == [10.0, 20.0, 30.0]


def test_align_predictions_duplicate_predictions_warn_and_keep_first(caplog) -> None:
    """Duplicate prediction keys log a warning and keep the first row."""
    Q = pd.DataFrame(
        {
            "t": np.array([0.1], dtype=np.float32),
            "s": np.array([1], dtype=np.int32),
            "j": np.array([0], dtype=np.int32),
            "x": [np.array([1.0], dtype=np.float32)],
        }
    )
    preds = pd.DataFrame(
        {
            "t": np.array([0.1, 0.1], dtype=np.float32),
            "s": np.array([1, 1], dtype=np.int32),
            "j": np.array([0, 0], dtype=np.int32),
            "x_pred": [
                np.array([10.0], dtype=np.float32),
                np.array([20.0], dtype=np.float32),
            ],
        }
    )
    with caplog.at_level(logging.WARNING):
        aligned = align_predictions(Q, preds)
    assert "duplicate prediction row(s)" in caplog.text
    assert aligned["matched"].tolist() == [True]
    assert float(np.asarray(aligned.iloc[0]["x_pred"])[0]) == 10.0


def test_align_predictions_preserves_query_order() -> None:
    """Alignment should return rows in the original query order."""
    Q = pd.DataFrame(
        {
            "t": np.array([0.2, 0.1, 0.3], dtype=np.float32),
            "s": np.array([2, 1, 1], dtype=np.int32),
            "j": np.array([0, 0, 0], dtype=np.int32),
            "x": [np.array([1.0], dtype=np.float32) for _ in range(3)],
        }
    )
    preds = pd.DataFrame(
        {
            "t": np.array([0.1, 0.2], dtype=np.float32),
            "s": np.array([1, 2], dtype=np.int32),
            "j": np.array([0, 0], dtype=np.int32),
            "x_pred": [
                np.array([10.0], dtype=np.float32),
                np.array([20.0], dtype=np.float32),
            ],
        }
    )
    aligned = align_predictions(Q, preds)
    assert aligned["t"].tolist() == pytest.approx([0.2, 0.1, 0.3])
    assert aligned["matched"].tolist() == [True, True, False]


# ---------------------------------------------------------------------------
# aggregate_sample tests
# ---------------------------------------------------------------------------


def test_aggregate_sample_structure() -> None:
    scores = pd.DataFrame(
        {
            "t": [0.1, 0.2, 0.3],
            "s": [1, 1, 2],
            "j": [0, 0, 0],
            "nmae": [1.0, 2.0, 3.0],
            "nmse": [1.0, 4.0, 9.0],
            "matched": [True, True, True],
        }
    )
    registry = {"1": {"name": "sig_1"}, "2": {"name": "sig_2"}}
    result = aggregate_sample(scores, registry)

    assert "nmae" in result and isinstance(result["nmae"], float)
    assert "nrmse" in result and isinstance(result["nrmse"], float)
    assert "n_signal_channels" in result and isinstance(
        result["n_signal_channels"], int
    )
    assert "per_signal_channel" in result and len(result["per_signal_channel"]) > 0

    sc = result["per_signal_channel"]
    assert "sig_1/0" in sc
    assert "sig_2/0" in sc
    assert sc["sig_1/0"]["n_obs"] == 2
    assert sc["sig_2/0"]["n_obs"] == 1


def test_aggregate_sample_values() -> None:
    """Verify level-1 and level-2 aggregation numerics exactly."""
    scores = pd.DataFrame(
        {
            "t": [0.1, 0.2, 0.3, 0.4],
            "s": [1, 1, 2, 2],
            "j": [0, 0, 0, 0],
            "nmae": [1.0, 3.0, 2.0, 4.0],
            "nmse": [1.0, 9.0, 4.0, 16.0],
            "matched": [True, True, True, True],
        }
    )
    registry = {"1": {"name": "sig_a"}, "2": {"name": "sig_b"}}
    result = aggregate_sample(scores, registry)

    # Level 1:
    # sig_a/0: nmae = (1+3)/2 = 2.0, nrmse = sqrt((1+9)/2) = sqrt(5)
    # sig_b/0: nmae = (2+4)/2 = 3.0, nrmse = sqrt((4+16)/2) = sqrt(10)
    assert result["per_signal_channel"]["sig_a/0"]["nmae"] == pytest.approx(2.0)
    assert result["per_signal_channel"]["sig_a/0"]["nrmse"] == pytest.approx(
        math.sqrt(5)
    )
    assert result["per_signal_channel"]["sig_b/0"]["nmae"] == pytest.approx(3.0)
    assert result["per_signal_channel"]["sig_b/0"]["nrmse"] == pytest.approx(
        math.sqrt(10)
    )

    # Level 2:
    assert result["nmae"] == pytest.approx(2.5)
    assert result["nrmse"] == pytest.approx((math.sqrt(5) + math.sqrt(10)) / 2)
    assert result["n_signal_channels"] == 2


# ---------------------------------------------------------------------------
# test_causal_imputation_no_future_leak
# ---------------------------------------------------------------------------


def test_causal_imputation_no_future_leak() -> None:
    """Context must not contain any rows with t > t_present."""
    obs, domain = _make_partition_fixtures()
    t_present = 0.6
    spec = CausalImputationSpec(
        held_out_signals=["sig_2"],
        t_failure={"sig_2": 0.3},
        t_present=t_present,
    )
    C, Q = partition_causal_imputation(obs, spec, domain)

    temporal_C = C[C["s"] != 0]
    assert temporal_C["t"].le(t_present).all(), (
        f"Context contains t > {t_present}: "
        f"{temporal_C.loc[temporal_C['t'] > t_present, 't'].tolist()}"
    )


def test_causal_imputation_query_within_t_present() -> None:
    """All query observations must have t <= t_present."""
    obs, domain = _make_partition_fixtures()
    spec = CausalImputationSpec(
        held_out_signals=["sig_1"],
        t_failure={"sig_1": 0.3},
        t_present=0.7,
    )
    _, Q = partition_causal_imputation(obs, spec, domain)
    if len(Q) > 0:
        assert Q["t"].le(0.7).all()


# ---------------------------------------------------------------------------
# get_sigma tests
# ---------------------------------------------------------------------------


def test_get_sigma_found() -> None:
    stats = make_train_stats([1, 2], j_max={1: 0, 2: 1})
    sigma = get_sigma(stats, s=1, j=0, m=0)
    assert sigma == pytest.approx(1.0)


def test_get_sigma_missing_returns_fallback() -> None:
    stats = make_train_stats([1])
    sigma = get_sigma(stats, s=999, j=0, m=0)
    assert sigma == pytest.approx(1e-8)


def test_get_sigma_zero_std_returns_fallback() -> None:
    stats = pd.DataFrame(
        {
            "s": np.array([1], dtype=np.int32),
            "j": np.array([0], dtype=np.int32),
            "m": np.array([0], dtype=np.int32),
            "mean": np.array([0.0], dtype=np.float32),
            "std": np.array([0.0], dtype=np.float32),
        }
    )
    sigma = get_sigma(stats, s=1, j=0, m=0)
    assert sigma == pytest.approx(1e-8)


# ---------------------------------------------------------------------------
# compute_scores integration test
# ---------------------------------------------------------------------------


def test_compute_scores_perfect_predictions() -> None:
    """Perfect predictions should yield nmae=0 and nmse=0."""
    x = np.array([1.5], dtype=np.float32)
    Q = pd.DataFrame(
        {
            "t": np.array([0.1], dtype=np.float32),
            "s": np.array([1], dtype=np.int32),
            "j": np.array([0], dtype=np.int32),
            "x": [x],
        }
    )
    preds = pd.DataFrame(
        {
            "t": np.array([0.1], dtype=np.float32),
            "s": np.array([1], dtype=np.int32),
            "j": np.array([0], dtype=np.int32),
            "x_pred": [x.copy()],
        }
    )
    train_stats = make_train_stats([1])
    scores = compute_scores(Q, preds, train_stats)
    assert scores["nmae"].iloc[0] == pytest.approx(0.0)
    assert scores["nmse"].iloc[0] == pytest.approx(0.0)


def test_compute_scores_unmatched_penalty() -> None:
    """Unmatched observations receive a training-mean penalty.

    With ``make_train_stats`` defaulting to ``mean=0.0``, the training mean
    coincides with the zero vector, so the asserted NMAE/NMSE values are the
    same either way.  The dedicated ``test_mean_imputation_*`` tests below
    use non-zero means to unambiguously verify mean-imputation semantics.
    """
    x = np.array([2.0], dtype=np.float32)
    Q = pd.DataFrame(
        {
            "t": np.array([0.1], dtype=np.float32),
            "s": np.array([1], dtype=np.int32),
            "j": np.array([0], dtype=np.int32),
            "x": [x],
        }
    )
    # Empty predictions — no match.
    preds = pd.DataFrame(columns=["t", "s", "j", "x_pred"])
    train_stats = make_train_stats([1])  # std=1.0, mean=0.0
    scores = compute_scores(Q, preds, train_stats)
    # mean=0.0, so x_pred = training mean = 0.0
    # nmae = |2.0 - 0.0| / 1.0 = 2.0
    # nmse = (2.0 / 1.0)^2 = 4.0
    assert scores["nmae"].iloc[0] == pytest.approx(2.0)
    assert scores["nmse"].iloc[0] == pytest.approx(4.0)
    assert not scores["matched"].iloc[0]


# ---------------------------------------------------------------------------
# Helpers for mean-imputation and evaluate_benchmark tests
# ---------------------------------------------------------------------------


def make_train_stats_sc(
    signal_channels: list[tuple[int, int, int, float, float]],
) -> pd.DataFrame:
    """Build a train_stats DataFrame from explicit (s, j, m, mean, std) tuples.

    Args:
        signal_channels: List of ``(s, j, m, mean, std)`` tuples.

    Returns:
        DataFrame with columns ``(s, j, m, mean, std)``.
    """
    rows = []
    for s, j, m, mean, std in signal_channels:
        rows.append(
            {
                "s": np.int32(s),
                "j": np.int32(j),
                "m": np.int32(m),
                "mean": np.float32(mean),
                "std": np.float32(std),
            }
        )
    return pd.DataFrame(rows)


def make_obs_df(rows: list[dict]) -> pd.DataFrame:
    """Build a query-set DataFrame from a list of row dicts.

    Args:
        rows: List of dicts with keys ``t``, ``s``, ``j``, ``x``
            (``x`` is a list of floats).

    Returns:
        DataFrame with columns ``(t: float32, s: int32, j: int32, x: object)``.
    """
    result_rows = []
    for row in rows:
        result_rows.append(
            {
                "t": np.float32(row["t"]),
                "s": np.int32(row["s"]),
                "j": np.int32(row["j"]),
                "x": np.array(row["x"], dtype=np.float32),
            }
        )
    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Mean-imputation penalty tests
# ---------------------------------------------------------------------------


def test_mean_imputation_penalty_correct() -> None:
    """Missing prediction uses training mean; hand-computed: nmae=1.0, nmse=1.0."""
    # s=1, j=0, m=0: mean=5.0, std=2.0
    train_stats = make_train_stats_sc([(1, 0, 0, 5.0, 2.0)])
    Q = make_obs_df([{"t": 0.1, "s": 1, "j": 0, "x": [3.0]}])
    preds = pd.DataFrame(columns=["t", "s", "j", "x_pred"])

    scores = compute_scores(Q, preds, train_stats)

    # x_pred = mean = 5.0
    # nmae = |3.0 - 5.0| / 2.0 = 1.0
    assert scores["nmae"].iloc[0] == pytest.approx(1.0)
    # nmse = ((3.0 - 5.0) / 2.0)^2 = 1.0
    assert scores["nmse"].iloc[0] == pytest.approx(1.0)
    assert not scores["matched"].iloc[0]


def test_mean_imputation_zero_error_when_true_equals_mean() -> None:
    """When x_true == training mean and prediction is missing, error = 0."""
    train_stats = make_train_stats_sc([(1, 0, 0, 5.0, 2.0)])
    Q = make_obs_df([{"t": 0.1, "s": 1, "j": 0, "x": [5.0]}])
    preds = pd.DataFrame(columns=["t", "s", "j", "x_pred"])

    scores = compute_scores(Q, preds, train_stats)

    # x_pred = mean = 5.0 = x_true → error = 0
    assert scores["nmae"].iloc[0] == pytest.approx(0.0)
    assert scores["nmse"].iloc[0] == pytest.approx(0.0)


def test_matched_prediction_ignores_training_mean() -> None:
    """Matched prediction uses x_pred, not training mean."""
    # mean=5.0 must NOT be used; x_pred=7.0 is the actual prediction.
    train_stats = make_train_stats_sc([(1, 0, 0, 5.0, 1.0)])
    Q = make_obs_df([{"t": 0.1, "s": 1, "j": 0, "x": [3.0]}])
    preds = pd.DataFrame(
        [
            {
                "t": np.float32(0.1),
                "s": np.int32(1),
                "j": np.int32(0),
                "x_pred": np.array([7.0], dtype=np.float32),
            }
        ]
    )

    scores = compute_scores(Q, preds, train_stats)

    # Uses prediction 7.0, not mean 5.0: nmae = |3.0 - 7.0| / 1.0 = 4.0
    assert scores["nmae"].iloc[0] == pytest.approx(4.0)
    assert scores["matched"].iloc[0]


# ---------------------------------------------------------------------------
# evaluate_benchmark tests
# ---------------------------------------------------------------------------


def test_evaluate_benchmark_basic() -> None:
    """Two domains are averaged equally regardless of n_tasks."""
    result1 = {"nmae": 0.5, "nrmse": 0.7, "n_tasks": 2, "per_task": []}
    result2 = {"nmae": 0.3, "nrmse": 0.5, "n_tasks": 1, "per_task": []}
    result = evaluate_benchmark([result1, result2])
    assert result["nmae"] == pytest.approx(0.4)
    assert result["nrmse"] == pytest.approx(0.6)
    assert result["n_domains"] == 2
    assert result["per_domain"] == [result1, result2]


def test_evaluate_benchmark_single_domain() -> None:
    """Single domain passes through unchanged."""
    domain = {"nmae": 0.42, "nrmse": 0.88, "n_tasks": 3, "per_task": []}
    result = evaluate_benchmark([domain])
    assert result["nmae"] == pytest.approx(0.42)
    assert result["nrmse"] == pytest.approx(0.88)
    assert result["n_domains"] == 1


def test_evaluate_benchmark_empty_raises() -> None:
    """Empty domain list must raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_benchmark([])


def test_evaluate_domain_empty_raises() -> None:
    """Empty task list must raise ValueError, not silently return NaN."""
    with pytest.raises(ValueError):
        evaluate_domain([])

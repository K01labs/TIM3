"""Integration tests for the benchmark suite using real data files.

These tests exercise the full pipeline against both the fusion (MAST) and
TCPrimed (tropical cyclone) datasets.  They require data files to be present
and do NOT mock any I/O.

Coverage:
    - load_fusion_spec / load_tcprimed_spec factory functions
    - load_sample on real Parquet files (schema, dtypes, content)
    - Static-row handling: fusion has 0, TCPrimed has 3 per storm
    - All four partition functions on both domains
    - Partition correctness: disjointness, union completeness, no-future-leak
    - get_sigma against real train_stats (including large IR signal)
    - compute_scores with perfect / zero predictions
    - sigma broadcast for variable-length GPROF signals (each obs is different size)
    - sigma per-dimension for fixed-large IR signal (307200 dims)
    - aggregate_sample structure and NRMSE = sqrt(mean(NMSE)) verified numerically
    - evaluate_sample end-to-end on both domains
    - Cross-domain sanity: both produce finite NMAE=0 on perfect predictions
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.data import get_sigma, load_sample
from benchmark.evaluate import evaluate_sample
from benchmark.metrics import aggregate_sample, compute_scores
from benchmark.specs import BenchmarkSpec, load_fusion_spec, load_tcprimed_spec
from benchmark.tasks import (
    _split_static,
    partition_causal_imputation,
    partition_cross_modal,
    partition_forecasting,
    partition_offline_imputation,
)

# ---------------------------------------------------------------------------
# Paths and skip markers
# ---------------------------------------------------------------------------

FUSION_ROOT = Path("data/fusion_data")
TCP_ROOT = Path("data/tcprimed_data")

FUSION_SAMPLE = FUSION_ROOT / "test" / "shot_30451.parquet"
FUSION_SAMPLE_2 = FUSION_ROOT / "test" / "shot_30454.parquet"
TCP_SAMPLE = TCP_ROOT / "test" / "storm_al182019.parquet"
TCP_SAMPLE_2 = TCP_ROOT / "test" / "storm_al192019.parquet"

fusion_data = pytest.mark.skipif(
    not FUSION_SAMPLE.exists(),
    reason="Fusion test data not present",
)
tcp_data = pytest.mark.skipif(
    not TCP_SAMPLE.exists(),
    reason="TCPrimed test data not present",
)
both_data = pytest.mark.skipif(
    not (FUSION_SAMPLE.exists() and TCP_SAMPLE.exists()),
    reason="Both fusion and TCPrimed data required",
)


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def assert_static_invariant(C: pd.DataFrame, Q: pd.DataFrame, n_static: int) -> None:
    """All n_static static rows (s=0) must be in C and none in Q."""
    actual_c = int((C["s"] == 0).sum())
    actual_q = int((Q["s"] == 0).sum())
    assert actual_c == n_static, f"Expected {n_static} static rows in C, got {actual_c}"
    assert actual_q == 0, f"Expected 0 static rows in Q, got {actual_q}"


def assert_disjoint(C: pd.DataFrame, Q: pd.DataFrame) -> None:
    """No temporal row may appear in both C and Q (matched on t, s, j)."""
    c_keys = set(
        zip(
            C.loc[C["s"] != 0, "t"].round(6),
            C.loc[C["s"] != 0, "s"],
            C.loc[C["s"] != 0, "j"],
        )
    )
    q_keys = set(zip(Q["t"].round(6), Q["s"], Q["j"]))
    overlap = c_keys & q_keys
    assert not overlap, f"C and Q share {len(overlap)} temporal rows"


def assert_union_complete(
    obs: pd.DataFrame,
    C: pd.DataFrame,
    Q: pd.DataFrame,
    t_present: float | None = None,
) -> None:
    """C_temporal ∪ Q covers all in-scope temporal obs (accounting for causal cutoff)."""
    temporal = obs[obs["s"] != 0]
    if t_present is not None:
        temporal = temporal[temporal["t"] <= t_present]
    c_temporal = C[C["s"] != 0]
    assert len(c_temporal) + len(Q) == len(temporal), (
        f"Union incomplete: {len(c_temporal)} + {len(Q)} != {len(temporal)}"
    )


# ===========================================================================
# FUSION — loading
# ===========================================================================


@fusion_data
class TestFusionLoad:
    def test_schema(self) -> None:
        df = load_sample(FUSION_SAMPLE)
        assert set(df.columns) == {"t", "s", "j", "x"}
        assert df["t"].dtype == np.float32
        assert df["s"].dtype == np.int32
        assert df["j"].dtype == np.int32

    def test_row_count_reasonable(self) -> None:
        df = load_sample(FUSION_SAMPLE)
        assert len(df) > 100_000

    def test_no_static_rows(self) -> None:
        """Real fusion shots carry no static rows (no patient metadata)."""
        df = load_sample(FUSION_SAMPLE)
        assert (df["s"] == 0).sum() == 0

    def test_all_t_nonneg(self) -> None:
        df = load_sample(FUSION_SAMPLE)
        assert (df["t"] >= 0).all()

    def test_signal_ids_in_expected_range(self) -> None:
        df = load_sample(FUSION_SAMPLE)
        assert set(df["s"].unique()).issubset(set(range(1, 40)))

    def test_x_arrays_are_numpy(self) -> None:
        df = load_sample(FUSION_SAMPLE)
        for x in df["x"].iloc[:10]:
            assert isinstance(x, np.ndarray)

    def test_second_sample_loads(self) -> None:
        """Sanity check that multiple shots load consistently."""
        df = load_sample(FUSION_SAMPLE_2)
        assert set(df.columns) == {"t", "s", "j", "x"}
        assert len(df) > 0


# ===========================================================================
# FUSION — specs
# ===========================================================================


@fusion_data
class TestFusionSpec:
    @pytest.fixture(scope="class")
    def spec(self):
        return load_fusion_spec(FUSION_ROOT)

    def test_all_task_specs_set(self, spec) -> None:
        assert spec.domain.name == "fusion"
        assert spec.forecasting is not None
        assert spec.offline_imputation is not None
        assert spec.causal_imputation is not None
        assert spec.cross_modal is not None

    def test_signal_lookup_first_and_last(self, spec) -> None:
        assert spec.domain.signal_id("magnetics_flux_loop_flux") == 1
        assert spec.domain.signal_id("equilibrium_psi") == 39

    def test_cross_modal_disjoint(self, spec) -> None:
        src = set(spec.cross_modal.source_signals)
        tgt = set(spec.cross_modal.target_signals)
        assert src & tgt == set()

    def test_all_equilibrium_targets(self, spec) -> None:
        for name in spec.cross_modal.target_signals:
            assert "equilibrium" in name

    def test_train_stats_has_required_columns(self, spec) -> None:
        assert set(spec.domain.train_stats.columns) >= {"s", "j", "m", "mean", "std"}


# ===========================================================================
# FUSION — partition functions
# ===========================================================================


@fusion_data
class TestFusionPartitions:
    @pytest.fixture(scope="class")
    def obs(self):
        return load_sample(FUSION_SAMPLE)

    @pytest.fixture(scope="class")
    def spec(self):
        return load_fusion_spec(FUSION_ROOT)

    # -- Forecasting --

    def test_forecasting_context_before_tc(self, obs, spec) -> None:
        C, Q = partition_forecasting(obs, spec.forecasting)
        t_c = spec.forecasting.t_c
        assert (C.loc[C["s"] != 0, "t"] < t_c).all()

    def test_forecasting_query_in_horizon(self, obs, spec) -> None:
        C, Q = partition_forecasting(obs, spec.forecasting)
        t_c, t_low, t_high = (
            spec.forecasting.t_c,
            spec.forecasting.t_low,
            spec.forecasting.t_high,
        )
        if len(Q) > 0:
            offset = Q["t"] - t_c
            assert (offset >= t_low - 1e-5).all()
            assert (offset <= t_high + 1e-5).all()

    def test_forecasting_no_static_rows(self, obs, spec) -> None:
        C, Q = partition_forecasting(obs, spec.forecasting)
        assert_static_invariant(C, Q, n_static=0)

    def test_forecasting_disjoint(self, obs, spec) -> None:
        C, Q = partition_forecasting(obs, spec.forecasting)
        assert_disjoint(C, Q)

    # -- Offline imputation --

    def test_offline_no_static_rows(self, obs, spec) -> None:
        C, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
        assert_static_invariant(C, Q, n_static=0)

    def test_offline_disjoint(self, obs, spec) -> None:
        C, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
        assert_disjoint(C, Q)

    def test_offline_union_complete(self, obs, spec) -> None:
        C, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
        assert_union_complete(obs, C, Q)

    def test_offline_query_signals_heldout(self, obs, spec) -> None:
        C, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
        held = set(spec.domain.signal_ids(spec.offline_imputation.held_out_signals))
        assert set(Q["s"].unique()).issubset(held)

    def test_offline_context_has_no_heldout_in_interval(self, obs, spec) -> None:
        """Held-out observations inside their masked intervals must not be in C."""
        C, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
        held = set(spec.domain.signal_ids(spec.offline_imputation.held_out_signals))
        intervals = {
            spec.domain.signal_id(name): iv
            for name, iv in spec.offline_imputation.held_out_intervals.items()
        }
        c_temporal = C[C["s"] != 0]
        for sid in held:
            t_a, t_b = intervals[sid]
            in_interval = c_temporal[
                (c_temporal["s"] == sid)
                & (c_temporal["t"] >= t_a)
                & (c_temporal["t"] <= t_b)
            ]
            assert len(in_interval) == 0, (
                f"s={sid} has {len(in_interval)} rows inside held-out interval in C"
            )

    # -- Causal imputation --

    def test_causal_no_future_in_context(self, obs, spec) -> None:
        C, Q = partition_causal_imputation(obs, spec.causal_imputation, spec.domain)
        t_p = spec.causal_imputation.t_present
        assert (C.loc[C["s"] != 0, "t"] <= t_p + 1e-5).all()

    def test_causal_union_complete(self, obs, spec) -> None:
        C, Q = partition_causal_imputation(obs, spec.causal_imputation, spec.domain)
        assert_union_complete(obs, C, Q, t_present=spec.causal_imputation.t_present)

    def test_causal_disjoint(self, obs, spec) -> None:
        C, Q = partition_causal_imputation(obs, spec.causal_imputation, spec.domain)
        assert_disjoint(C, Q)

    # -- Cross-modal --

    def test_cross_modal_context_source_only(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        src = set(spec.domain.signal_ids(spec.cross_modal.source_signals))
        c_sigs = set(C.loc[C["s"] != 0, "s"].unique())
        assert c_sigs.issubset(src)

    def test_cross_modal_query_target_only(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        tgt = set(spec.domain.signal_ids(spec.cross_modal.target_signals))
        assert set(Q["s"].unique()).issubset(tgt)

    def test_cross_modal_disjoint(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        assert_disjoint(C, Q)

    def test_cross_modal_no_static_rows(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        assert_static_invariant(C, Q, n_static=0)


# ===========================================================================
# FUSION — metrics
# ===========================================================================


@fusion_data
class TestFusionMetrics:
    @pytest.fixture(scope="class")
    def spec(self):
        return load_fusion_spec(FUSION_ROOT)

    @pytest.fixture(scope="class")
    def obs(self):
        return load_sample(FUSION_SAMPLE)

    @pytest.fixture(scope="class")
    def forecasting_query_small(self, obs, spec):
        _, Q = partition_forecasting(obs, spec.forecasting)
        return Q.iloc[:100].copy()

    def test_get_sigma_known_signal_positive_finite(self, spec) -> None:
        sigma = get_sigma(spec.domain.train_stats, s=1, j=0, m=0)
        assert sigma > 0 and math.isfinite(sigma)

    def test_get_sigma_all_train_stats_positive_finite(self, spec) -> None:
        stats = spec.domain.train_stats
        for _, row in stats.iterrows():
            sigma = get_sigma(stats, int(row["s"]), int(row["j"]), int(row["m"]))
            assert sigma > 0 and math.isfinite(sigma), (
                f"sigma={sigma} at s={row['s']}, j={row['j']}, m={row['m']}"
            )

    def test_perfect_predictions_give_zero_error(
        self, forecasting_query_small, spec
    ) -> None:
        Q = forecasting_query_small
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [x.copy() for x in Q["x"].values],
            }
        )
        scores = compute_scores(Q, preds, spec.domain.train_stats)
        assert (scores["nmae"] < 1e-10).all(), "Perfect predictions must yield NMAE=0"
        assert (scores["nmse"] < 1e-10).all(), "Perfect predictions must yield NMSE=0"
        assert scores["matched"].all()

    def test_zero_predictions_give_finite_positive_error(
        self, forecasting_query_small, spec
    ) -> None:
        Q = forecasting_query_small
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [np.zeros(len(x), dtype=np.float32) for x in Q["x"].values],
            }
        )
        scores = compute_scores(Q, preds, spec.domain.train_stats)
        assert np.isfinite(scores["nmae"]).all()
        assert np.isfinite(scores["nmse"]).all()
        assert (scores["nmae"] >= 0).all()
        assert (scores["nmse"] >= 0).all()

    def test_aggregate_sample_keys_and_types(
        self, forecasting_query_small, spec
    ) -> None:
        Q = forecasting_query_small
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [x.copy() for x in Q["x"].values],
            }
        )
        scores = compute_scores(Q, preds, spec.domain.train_stats)
        result = aggregate_sample(scores, spec.domain.registry)

        assert isinstance(result["nmae"], float)
        assert isinstance(result["nrmse"], float)
        assert isinstance(result["n_signal_channels"], int)
        assert result["n_signal_channels"] > 0
        assert len(result["per_signal_channel"]) == result["n_signal_channels"]
        assert result["nrmse"] >= 0
        assert result["nmae"] >= 0

    def test_nrmse_equals_sqrt_mean_nmse(self, forecasting_query_small, spec) -> None:
        """NRMSE must equal sqrt(mean(NMSE)) at signal-channel level — verified numerically."""
        Q = forecasting_query_small
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [np.zeros(len(x), dtype=np.float32) for x in Q["x"].values],
            }
        )
        scores = compute_scores(Q, preds, spec.domain.train_stats)
        result = aggregate_sample(scores, spec.domain.registry)

        for sc_key, sc_data in result["per_signal_channel"].items():
            sig_name = sc_data["signal_name"]
            j = sc_data["j"]
            sid = spec.domain.signal_id(sig_name)
            group = scores[(scores["s"] == sid) & (scores["j"] == j)]
            expected = math.sqrt(float(group["nmse"].mean()))
            assert abs(sc_data["nrmse"] - expected) < 1e-10, (
                f"NRMSE mismatch for {sc_key}: got {sc_data['nrmse']}, expected {expected}"
            )


# ===========================================================================
# TCPRIMED — loading
# ===========================================================================


@tcp_data
class TestTCPrimedLoad:
    def test_schema(self) -> None:
        df = load_sample(TCP_SAMPLE)
        assert set(df.columns) == {"t", "s", "j", "x"}

    def test_three_static_rows(self) -> None:
        """TCPrimed has exactly 3 static context rows per storm."""
        df = load_sample(TCP_SAMPLE)
        assert (df["s"] == 0).sum() == 3

    def test_static_rows_t_is_minus_one(self) -> None:
        df = load_sample(TCP_SAMPLE)
        static = df[df["s"] == 0]
        assert (static["t"] == -1.0).all()

    def test_static_rows_three_channels(self) -> None:
        """Channels j=0 (basin), j=1 (season), j=2 (storm number)."""
        df = load_sample(TCP_SAMPLE)
        static = df[df["s"] == 0]
        assert set(static["j"].unique()) == {0, 1, 2}

    def test_temporal_t_range_seconds(self) -> None:
        """Temporal observations span at least 1 hour in seconds."""
        df = load_sample(TCP_SAMPLE)
        temporal = df[df["s"] != 0]
        assert temporal["t"].max() > 3600

    def test_ir_signal_fixed_307200_dims(self) -> None:
        df = load_sample(TCP_SAMPLE)
        ir = df[df["s"] == 19]
        if len(ir) > 0:
            assert all(len(x) == 307_200 for x in ir["x"])

    def test_gprof_signals_are_variable_length(self) -> None:
        """GPROF (s=17, 18) overpass swaths vary in pixel count — val_dim=-1 in registry."""
        df = load_sample(TCP_SAMPLE)
        for s_id in [17, 18]:
            sub = df[df["s"] == s_id]
            if len(sub) > 1:
                lengths = {len(x) for x in sub["x"]}
                # Variable-length signals should have more than one unique length.
                assert len(lengths) > 1, (
                    f"s={s_id}: expected variable lengths but all are the same: {lengths}"
                )

    def test_second_sample_loads(self) -> None:
        df = load_sample(TCP_SAMPLE_2)
        assert set(df.columns) == {"t", "s", "j", "x"}
        assert len(df) > 0


# ===========================================================================
# TCPRIMED — specs
# ===========================================================================


@tcp_data
class TestTCPrimedSpec:
    @pytest.fixture(scope="class")
    def spec(self):
        return load_tcprimed_spec(TCP_ROOT)

    def test_all_task_specs_set(self, spec) -> None:
        assert spec.domain.name == "tcprimed"
        for attr in (
            "forecasting",
            "offline_imputation",
            "causal_imputation",
            "cross_modal",
        ):
            assert getattr(spec, attr) is not None

    def test_forecasting_t_c_is_in_seconds(self, spec) -> None:
        assert spec.forecasting.t_c >= 3600

    def test_cross_modal_source_is_environmental(self, spec) -> None:
        for name in spec.cross_modal.source_signals:
            assert name.startswith("track_") or name.startswith("diag_")

    def test_cross_modal_target_is_overpass_scalar(self, spec) -> None:
        for name in spec.cross_modal.target_signals:
            assert name.startswith("overpass_")

    def test_cross_modal_disjoint(self, spec) -> None:
        src = set(spec.cross_modal.source_signals)
        tgt = set(spec.cross_modal.target_signals)
        assert src & tgt == set()

    def test_registry_has_static_signal_entry(self, spec) -> None:
        assert "0" in spec.domain.registry
        assert spec.domain.registry["0"]["name"] == "static_context"

    def test_train_stats_ir_is_single_broadcast_sigma(self, spec) -> None:
        # ir_brightness_temp (s=19) is a variable-length field, so train_stats
        # stores a single global sigma at m=0 that the evaluator broadcasts
        # across all dimensions — not one row per (480x640) pixel.
        ir_stats = spec.domain.train_stats[spec.domain.train_stats["s"] == 19]
        assert len(ir_stats) == 1
        assert int(ir_stats.iloc[0]["m"]) == 0


# ===========================================================================
# TCPRIMED — partition functions
# ===========================================================================


@tcp_data
class TestTCPrimedPartitions:
    @pytest.fixture(scope="class")
    def obs(self):
        return load_sample(TCP_SAMPLE)

    @pytest.fixture(scope="class")
    def spec(self):
        return load_tcprimed_spec(TCP_ROOT)

    def test_split_static_returns_three(self, obs) -> None:
        static, temporal = _split_static(obs)
        assert len(static) == 3
        assert (static["s"] == 0).all()
        assert (temporal["s"] != 0).all()
        assert len(static) + len(temporal) == len(obs)

    # -- Forecasting --

    def test_forecasting_static_invariant(self, obs, spec) -> None:
        C, Q = partition_forecasting(obs, spec.forecasting)
        assert_static_invariant(C, Q, n_static=3)

    def test_forecasting_context_count(self, obs, spec) -> None:
        """Context row count must equal observations with t < t_c."""
        t_c = spec.forecasting.t_c
        C, Q = partition_forecasting(obs, spec.forecasting)
        expected = len(obs[(obs["s"] != 0) & (obs["t"] < t_c)])
        assert len(C[C["s"] != 0]) == expected

    def test_forecasting_query_in_horizon(self, obs, spec) -> None:
        t_c = spec.forecasting.t_c
        C, Q = partition_forecasting(obs, spec.forecasting)
        if len(Q) > 0:
            offset = Q["t"] - t_c
            assert (offset >= spec.forecasting.t_low - 1).all()
            assert (offset <= spec.forecasting.t_high + 1).all()

    def test_forecasting_disjoint(self, obs, spec) -> None:
        C, Q = partition_forecasting(obs, spec.forecasting)
        assert_disjoint(C, Q)

    # -- Offline imputation --

    def test_offline_static_invariant(self, obs, spec) -> None:
        C, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
        assert_static_invariant(C, Q, n_static=3)

    def test_offline_disjoint(self, obs, spec) -> None:
        C, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
        assert_disjoint(C, Q)

    def test_offline_union_complete(self, obs, spec) -> None:
        C, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
        assert_union_complete(obs, C, Q)

    def test_offline_query_signals_are_heldout(self, obs, spec) -> None:
        C, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
        held = set(spec.domain.signal_ids(spec.offline_imputation.held_out_signals))
        if len(Q) > 0:
            assert set(Q["s"].unique()).issubset(held)

    def test_offline_query_in_interval(self, obs, spec) -> None:
        """Query obs must fall within each signal's held-out interval."""
        C, Q = partition_offline_imputation(obs, spec.offline_imputation, spec.domain)
        if len(Q) == 0:
            pytest.skip("No query observations for this storm's default interval")
        for name, (t_a, t_b) in spec.offline_imputation.held_out_intervals.items():
            sid = spec.domain.signal_id(name)
            q_sig = Q[Q["s"] == sid]
            if len(q_sig) > 0:
                assert (q_sig["t"] >= t_a - 1).all()
                assert (q_sig["t"] <= t_b + 1).all()

    # -- Causal imputation --

    def test_causal_static_invariant(self, obs, spec) -> None:
        C, Q = partition_causal_imputation(obs, spec.causal_imputation, spec.domain)
        assert_static_invariant(C, Q, n_static=3)

    def test_causal_no_future_in_context(self, obs, spec) -> None:
        t_p = spec.causal_imputation.t_present
        C, Q = partition_causal_imputation(obs, spec.causal_imputation, spec.domain)
        c_t = C.loc[C["s"] != 0, "t"]
        assert (c_t <= t_p + 1).all(), f"max context t = {c_t.max()}, t_present = {t_p}"

    def test_causal_disjoint(self, obs, spec) -> None:
        C, Q = partition_causal_imputation(obs, spec.causal_imputation, spec.domain)
        assert_disjoint(C, Q)

    def test_causal_union_complete(self, obs, spec) -> None:
        C, Q = partition_causal_imputation(obs, spec.causal_imputation, spec.domain)
        assert_union_complete(obs, C, Q, t_present=spec.causal_imputation.t_present)

    def test_causal_query_geq_t_failure(self, obs, spec) -> None:
        """Query observations must have t >= t_failure for their signal."""
        C, Q = partition_causal_imputation(obs, spec.causal_imputation, spec.domain)
        for name, t_f in spec.causal_imputation.t_failure.items():
            sid = spec.domain.signal_id(name)
            q_sig = Q[Q["s"] == sid]
            if len(q_sig) > 0:
                assert (q_sig["t"] >= t_f - 1).all()

    # -- Cross-modal --

    def test_cross_modal_static_invariant(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        assert_static_invariant(C, Q, n_static=3)

    def test_cross_modal_context_env_signals_only(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        src = set(spec.domain.signal_ids(spec.cross_modal.source_signals))
        c_sigs = set(C.loc[C["s"] != 0, "s"].unique())
        assert c_sigs.issubset(src)

    def test_cross_modal_query_overpass_scalar_only(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        tgt = set(spec.domain.signal_ids(spec.cross_modal.target_signals))
        assert set(Q["s"].unique()).issubset(tgt)

    def test_cross_modal_disjoint(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        assert_disjoint(C, Q)


# ===========================================================================
# TCPRIMED — metrics
# ===========================================================================


@tcp_data
class TestTCPrimedMetrics:
    @pytest.fixture(scope="class")
    def spec(self):
        return load_tcprimed_spec(TCP_ROOT)

    @pytest.fixture(scope="class")
    def obs(self):
        return load_sample(TCP_SAMPLE)

    def test_get_sigma_track_intensity(self, spec) -> None:
        sigma = get_sigma(spec.domain.train_stats, s=1, j=0, m=0)
        assert sigma > 0 and math.isfinite(sigma)

    def test_get_sigma_ir_spot_check(self, spec) -> None:
        """IR has 307200 entries — verify first, middle, and last."""
        stats = spec.domain.train_stats
        for m in [0, 153_600, 307_199]:
            sigma = get_sigma(stats, s=19, j=0, m=m)
            assert sigma > 0 and math.isfinite(sigma), f"IR m={m}: sigma={sigma}"

    def test_get_sigma_gprof_only_m0_in_stats(self, spec) -> None:
        """GPROF has only m=0 in train_stats; m=1 should fall back to 1e-8."""
        stats = spec.domain.train_stats
        sigma_0 = get_sigma(stats, s=17, j=0, m=0)
        assert sigma_0 > 0 and math.isfinite(sigma_0)
        # m=1 not in stats -> fallback
        sigma_1 = get_sigma(stats, s=17, j=0, m=1)
        assert sigma_1 == pytest.approx(1e-8)

    def test_perfect_predictions_overpass_scalars(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [x.copy() for x in Q["x"].values],
            }
        )
        scores = compute_scores(Q, preds, spec.domain.train_stats)
        assert (scores["nmae"] < 1e-10).all(), "Perfect predictions must yield NMAE=0"
        assert (scores["nmse"] < 1e-10).all()
        assert scores["matched"].all()

    def test_sigma_broadcast_gprof_perfect_predictions(self, obs, spec) -> None:
        """Variable-length GPROF signals must use global sigma broadcast, giving nmae=0."""
        Q_gprof = obs[obs["s"] == 17].copy()
        if len(Q_gprof) == 0:
            pytest.skip("No GPROF observations in this storm")
        preds = pd.DataFrame(
            {
                "t": Q_gprof["t"].values,
                "s": Q_gprof["s"].values,
                "j": Q_gprof["j"].values,
                "x_pred": [x.copy() for x in Q_gprof["x"].values],
            }
        )
        scores = compute_scores(Q_gprof, preds, spec.domain.train_stats)
        assert (scores["nmae"] < 1e-10).all(), (
            "GPROF perfect predictions must give NMAE=0 with sigma broadcast"
        )

    def test_sigma_broadcast_gprof_different_obs_sizes(self, obs, spec) -> None:
        """Sigma must broadcast correctly even when two observations have different dims."""
        Q_gprof = obs[obs["s"] == 17].copy()
        if len(Q_gprof) < 2:
            pytest.skip("Need at least 2 GPROF observations")
        # Take two obs with different x lengths.
        lengths = [(i, len(x)) for i, x in enumerate(Q_gprof["x"].values)]
        unique_lens = {ln for _, ln in lengths}
        if len(unique_lens) < 2:
            pytest.skip("All GPROF obs have same length — can't test size variability")
        # Both should give zero error with perfect predictions.
        subset = Q_gprof.iloc[:5].copy()
        preds = pd.DataFrame(
            {
                "t": subset["t"].values,
                "s": subset["s"].values,
                "j": subset["j"].values,
                "x_pred": [x.copy() for x in subset["x"].values],
            }
        )
        scores = compute_scores(subset, preds, spec.domain.train_stats)
        assert (scores["nmae"] < 1e-10).all()

    def test_ir_perfect_predictions_zero_error(self, obs, spec) -> None:
        """IR signal (307200 dims) must give nmae=0 with perfect predictions."""
        Q_ir = obs[obs["s"] == 19].copy()
        if len(Q_ir) == 0:
            pytest.skip("No IR observations in this storm")
        preds = pd.DataFrame(
            {
                "t": Q_ir["t"].values,
                "s": Q_ir["s"].values,
                "j": Q_ir["j"].values,
                "x_pred": [x.copy() for x in Q_ir["x"].values],
            }
        )
        import time

        t0 = time.time()
        scores = compute_scores(Q_ir, preds, spec.domain.train_stats)
        elapsed = time.time() - t0
        assert elapsed < 30, f"IR compute_scores too slow: {elapsed:.1f}s"
        assert (scores["nmae"] < 1e-10).all()
        assert (scores["nmse"] < 1e-10).all()

    def test_ir_zero_predictions_finite_nonzero_error(self, obs, spec) -> None:
        """Zero predictions against IR data must produce finite positive NMAE."""
        Q_ir = obs[obs["s"] == 19].iloc[:3].copy()
        if len(Q_ir) == 0:
            pytest.skip("No IR observations")
        preds = pd.DataFrame(
            {
                "t": Q_ir["t"].values,
                "s": Q_ir["s"].values,
                "j": Q_ir["j"].values,
                "x_pred": [
                    np.zeros(len(x), dtype=np.float32) for x in Q_ir["x"].values
                ],
            }
        )
        scores = compute_scores(Q_ir, preds, spec.domain.train_stats)
        assert np.isfinite(scores["nmae"]).all()
        assert np.isfinite(scores["nmse"]).all()
        assert (scores["nmae"] > 0).any()

    def test_aggregate_cross_modal_perfect_zero_nmae(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [x.copy() for x in Q["x"].values],
            }
        )
        scores = compute_scores(Q, preds, spec.domain.train_stats)
        result = aggregate_sample(scores, spec.domain.registry)
        assert result["nmae"] == pytest.approx(0.0, abs=1e-8)
        assert result["nrmse"] == pytest.approx(0.0, abs=1e-8)
        assert result["n_signal_channels"] == 4

    def test_aggregate_per_signal_keys_are_overpass(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [x.copy() for x in Q["x"].values],
            }
        )
        scores = compute_scores(Q, preds, spec.domain.train_stats)
        result = aggregate_sample(scores, spec.domain.registry)
        for key in result["per_signal_channel"]:
            assert key.startswith("overpass_"), f"Unexpected key: {key}"

    def test_nrmse_equals_sqrt_mean_nmse(self, obs, spec) -> None:
        C, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [np.zeros(len(x), dtype=np.float32) for x in Q["x"].values],
            }
        )
        scores = compute_scores(Q, preds, spec.domain.train_stats)
        result = aggregate_sample(scores, spec.domain.registry)
        for sc_key, sc_data in result["per_signal_channel"].items():
            sid = spec.domain.signal_id(sc_data["signal_name"])
            group = scores[(scores["s"] == sid) & (scores["j"] == sc_data["j"])]
            expected = math.sqrt(float(group["nmse"].mean()))
            assert abs(sc_data["nrmse"] - expected) < 1e-10, (
                f"NRMSE mismatch for {sc_key}"
            )


# ===========================================================================
# TCPRIMED — evaluate_sample end-to-end
# ===========================================================================


@tcp_data
class TestTCPrimedEvaluateSample:
    @pytest.fixture(scope="class")
    def spec(self):
        return load_tcprimed_spec(TCP_ROOT)

    @pytest.fixture(scope="class")
    def obs(self):
        return load_sample(TCP_SAMPLE)

    def test_cross_modal_end_to_end(self, obs, spec) -> None:
        _, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [x.copy() for x in Q["x"].values],
            }
        )
        result = evaluate_sample(TCP_SAMPLE, preds, "cross_modal", spec)
        assert result["sample_id"] == "storm_al182019"
        assert result["task"] == "cross_modal"
        assert result["domain"] == "tcprimed"
        assert isinstance(result["nmae"], float)
        assert math.isfinite(result["nmae"])
        assert result["n_query_obs"] > 0
        assert result["n_matched"] == result["n_query_obs"]
        assert result["nmae"] == pytest.approx(0.0, abs=1e-8)

    def test_forecasting_end_to_end(self, obs, spec) -> None:
        _, Q = partition_forecasting(obs, spec.forecasting)
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [x.copy() for x in Q["x"].values],
            }
        )
        result = evaluate_sample(TCP_SAMPLE, preds, "forecasting", spec)
        assert result["task"] == "forecasting"
        assert result["n_matched"] == result["n_query_obs"]
        assert result["nmae"] == pytest.approx(0.0, abs=1e-8)

    def test_invalid_task_raises(self, spec) -> None:
        with pytest.raises(ValueError, match="unknown task"):
            evaluate_sample(TCP_SAMPLE, pd.DataFrame(), "bad_task", spec)

    def test_missing_spec_raises(self, spec) -> None:
        spec_no_forecast = BenchmarkSpec(domain=spec.domain, forecasting=None)
        with pytest.raises(ValueError, match="spec.forecasting is None"):
            evaluate_sample(TCP_SAMPLE, pd.DataFrame(), "forecasting", spec_no_forecast)

    def test_second_sample_runs(self, spec) -> None:
        """evaluate_sample must work across all test storms."""
        obs2 = load_sample(TCP_SAMPLE_2)
        _, Q = partition_cross_modal(obs2, spec.cross_modal, spec.domain)
        preds = pd.DataFrame(
            {
                "t": Q["t"].values,
                "s": Q["s"].values,
                "j": Q["j"].values,
                "x_pred": [x.copy() for x in Q["x"].values],
            }
        )
        result = evaluate_sample(TCP_SAMPLE_2, preds, "cross_modal", spec)
        assert math.isfinite(result["nmae"])
        assert result["nmae"] == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# CROSS-DOMAIN CONSISTENCY
# ===========================================================================


@both_data
class TestCrossDomainConsistency:
    def test_both_domains_perfect_predictions_zero_nmae(self) -> None:
        """Both domains must report nmae≈0 under perfect predictions."""
        for data_root, pq_path, spec_fn in [
            (FUSION_ROOT, FUSION_SAMPLE, load_fusion_spec),
            (TCP_ROOT, TCP_SAMPLE, load_tcprimed_spec),
        ]:
            spec = spec_fn(data_root)
            obs = load_sample(pq_path)
            _, Q = partition_forecasting(obs, spec.forecasting)
            Q_small = Q.iloc[:50].copy()
            preds = pd.DataFrame(
                {
                    "t": Q_small["t"].values,
                    "s": Q_small["s"].values,
                    "j": Q_small["j"].values,
                    "x_pred": [x.copy() for x in Q_small["x"].values],
                }
            )
            scores = compute_scores(Q_small, preds, spec.domain.train_stats)
            result = aggregate_sample(scores, spec.domain.registry)
            assert math.isfinite(result["nmae"]), f"{spec.domain.name}: nmae not finite"
            assert result["nmae"] == pytest.approx(0.0, abs=1e-8), (
                f"{spec.domain.name}: nmae={result['nmae']} != 0 under perfect preds"
            )

    def test_both_domains_all_partition_functions_run(self) -> None:
        """All four partition functions must execute without error on both domains."""
        for data_root, pq_path, spec_fn in [
            (FUSION_ROOT, FUSION_SAMPLE, load_fusion_spec),
            (TCP_ROOT, TCP_SAMPLE, load_tcprimed_spec),
        ]:
            spec = spec_fn(data_root)
            obs = load_sample(pq_path)
            for fn, task_spec, extra in [
                (partition_forecasting, spec.forecasting, {}),
                (
                    partition_offline_imputation,
                    spec.offline_imputation,
                    {"domain": spec.domain},
                ),
                (
                    partition_causal_imputation,
                    spec.causal_imputation,
                    {"domain": spec.domain},
                ),
                (partition_cross_modal, spec.cross_modal, {"domain": spec.domain}),
            ]:
                C, Q = fn(obs, task_spec, **extra)
                assert len(C) + len(Q) > 0, (
                    f"{spec.domain.name} / {fn.__name__}: empty C+Q"
                )

    def test_both_domains_evaluate_sample_runs(self) -> None:
        """evaluate_sample must produce finite results on both domains."""
        for data_root, pq_path, spec_fn in [
            (FUSION_ROOT, FUSION_SAMPLE, load_fusion_spec),
            (TCP_ROOT, TCP_SAMPLE, load_tcprimed_spec),
        ]:
            spec = spec_fn(data_root)
            obs = load_sample(pq_path)
            _, Q = partition_cross_modal(obs, spec.cross_modal, spec.domain)
            preds = pd.DataFrame(
                {
                    "t": Q["t"].values,
                    "s": Q["s"].values,
                    "j": Q["j"].values,
                    "x_pred": [x.copy() for x in Q["x"].values],
                }
            )
            result = evaluate_sample(pq_path, preds, "cross_modal", spec)
            assert math.isfinite(result["nmae"]), (
                f"{spec.domain.name}: nmae not finite: {result['nmae']}"
            )

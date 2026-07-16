"""Benchmark specification dataclasses and factory functions.

This module defines the specification objects that parameterise each benchmark
task and domain.  A ``BenchmarkSpec`` collects a ``DomainSpec`` (data paths,
signal registry, training statistics) together with one optional spec per task
type.

Notation follows the paper:
    - s_i  : integer signal ID (1-indexed)
    - j    : channel index within a signal (== c_i in the paper)
    - t    : continuous timestamp in seconds
    - x    : observed value vector of shape (d_{s_i},)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class DomainSpec:
    """Describes one scientific domain (e.g. fusion, climate, healthcare).

    Attributes:
        name: Human-readable domain identifier (e.g. ``"fusion"``).
        data_root: Path to the domain's data directory (e.g.
            ``data/fusion_data/``).
        registry: Raw contents of ``signals.json`` as parsed by
            ``json.load``.  Keys are string-encoded integer signal IDs;
            values are dicts with at least a ``"name"`` field.
        train_stats: DataFrame loaded from ``train_stats.parquet`` with
            columns ``(s: int32, j: int32, m: int32, mean: float32,
            std: float32)``.

    Example:
        >>> spec = DomainSpec(name="fusion", data_root=Path("data/fusion_data"),
        ...                   registry=registry, train_stats=stats)
        >>> spec.signal_id("magnetics_flux_loop_flux")
        1
    """

    name: str
    data_root: Path
    registry: dict
    train_stats: pd.DataFrame

    # Internal reverse-lookup map built in __post_init__.
    _name_to_id: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the name -> integer ID reverse map from the registry."""
        self._name_to_id = {v["name"]: int(k) for k, v in self.registry.items()}

    def signal_id(self, name: str) -> int:
        """Look up the integer signal ID for a given signal name.

        Args:
            name: The signal name as it appears in ``signals.json``
                (e.g. ``"magnetics_flux_loop_flux"``).

        Returns:
            Integer signal ID (1-indexed).

        Raises:
            KeyError: If ``name`` is not present in the registry.
        """
        if name not in self._name_to_id:
            raise KeyError(
                f"Signal name {name!r} not found in registry. "
                f"Available names: {sorted(self._name_to_id)}"
            )
        return self._name_to_id[name]

    def signal_ids(self, names: list[str]) -> list[int]:
        """Look up multiple signal IDs by name.

        Args:
            names: List of signal names.

        Returns:
            List of integer signal IDs in the same order as ``names``.

        Raises:
            KeyError: If any name is not present in the registry.
        """
        return [self.signal_id(n) for n in names]


@dataclass
class ForecastingSpec:
    """Parameters for the forecasting task.

    The context set contains all observations strictly before ``t_c``.
    The query set is the horizon band relative to ``t_c``:

    Observations with ``t_i >= t_c`` but outside the horizon band are dropped
    from both context and query.

    Attributes:
        t_c: Cut-off time in seconds.  Observations at ``t < t_c`` are
            context; observations at ``t >= t_c`` are candidate queries.
        t_low: Horizon band lower bound (seconds, relative to ``t_c``).
            Must satisfy ``0 <= t_low <= t_high``.
        t_high: Horizon band upper bound (seconds, relative to ``t_c``).
    """

    t_c: float
    t_low: float
    t_high: float


@dataclass
class OfflineImputationSpec:
    """Parameters for the offline imputation task.

    A set of signals H is held out, optionally within per-signal time
    intervals.  The model sees all other observations (and the held-out
    signals outside their masked windows) and must reconstruct the missing
    portions.

    Attributes:
        held_out_signals: Signal names (not IDs) to hold out.
        held_out_intervals: Per-signal time intervals ``{name: (t_a, t_b)}``
            within which observations are masked.  If ``None``, the entire
            time range of each held-out signal is masked.
    """

    held_out_signals: list[str]
    held_out_intervals: dict[str, tuple[float, float]] | None = None


@dataclass
class CausalImputationSpec:
    """Parameters for the causal imputation task.

    Similar to offline imputation, but restricted to observations up to a
    present time ``t_p``, simulating real-time deployment.  Each held-out
    signal has a failure time ``t_f^{(s)}``: observations for that signal
    after failure are removed from context and become the query.

    Attributes:
        held_out_signals: Signal names to hold out.
        t_failure: Mapping from signal name to failure time ``t_f^{(s)}``
            in seconds.  Must include an entry for every signal in
            ``held_out_signals``.
        t_present: Present time ``t_p`` in seconds.  Only observations with
            ``t_i <= t_p`` are considered; everything beyond is discarded.
    """

    held_out_signals: list[str]
    t_failure: dict[str, float]
    t_present: float


@dataclass
class CrossModalSpec:
    """Parameters for the cross-modal prediction task.

    All observations from source signals form the context; observations from
    target signals are the query.  This is an offline task (the model may use
    all source-signal observations to predict all target-signal observations).

    Attributes:
        source_signals: Signal names in the context set (S_src).
        target_signals: Signal names to predict (S_tgt).

    Raises:
        ValueError: At construction time if ``source_signals`` and
            ``target_signals`` share any signal names.
    """

    source_signals: list[str]
    target_signals: list[str]

    def __post_init__(self) -> None:
        """Validate that source and target signal sets are disjoint."""
        overlap = set(self.source_signals) & set(self.target_signals)
        if overlap:
            raise ValueError(
                f"CrossModalSpec: source_signals and target_signals must be "
                f"disjoint, but the following names appear in both: "
                f"{sorted(overlap)}"
            )


@dataclass
class BenchmarkSpec:
    """Top-level benchmark specification for a single domain.

    Collects a ``DomainSpec`` together with optional per-task specifications.
    A ``None`` task spec means that task is not evaluated.

    Attributes:
        domain: Domain-level metadata, signal registry, and training stats.
        forecasting: Specification for the forecasting task, or ``None``.
        offline_imputation: Specification for the offline imputation task,
            or ``None``.
        causal_imputation: Specification for the causal imputation task,
            or ``None``.
        cross_modal: Specification for the cross-modal prediction task,
            or ``None``.
    """

    domain: DomainSpec
    forecasting: ForecastingSpec | None = None
    offline_imputation: OfflineImputationSpec | None = None
    causal_imputation: CausalImputationSpec | None = None
    cross_modal: CrossModalSpec | None = None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def load_fusion_spec(
    data_root: Path,
    t_c: float = 0.15,
    t_high: float = 0.05,
) -> BenchmarkSpec:
    """Load a ``BenchmarkSpec`` for the fusion domain with sensible defaults.

    Reads ``signals.json`` and ``train_stats.parquet`` from ``data_root``
    automatically, then constructs task specs using typical MAST shot
    parameters.

    Args:
        data_root: Path to the domain data directory.
        t_c: Forecasting cut-off time in seconds (default 0.15 s).
        t_high: Forecasting horizon upper bound in seconds relative to
            ``t_c`` (default 0.05 s = 50 ms). The default query window
            (0.15-0.20 s) lies within the typical MAST shot duration (median
            ~0.54 s); shots that end before a query timestamp simply contribute
            no query observations there.

    Default task specs:

    * **Forecasting**: ``t_c = 0.15 s``, horizon ``[0.0, t_high]``.
    * **Offline imputation**: hold out all equilibrium signals (names
      containing ``"equilibrium"``) for the fixed window ``[0.15, 0.35] s``,
      a reasonable default for MAST shots of ~0.5 s total duration.
    * **Causal imputation**: same equilibrium signals, failure at ``0.25 s``,
      present at ``0.4 s``.  Shots shorter than ``t_present`` will produce an
      empty query set; this is acceptable as a default configuration.
    * **Cross-modal**: source = all non-equilibrium signals;
      target = all equilibrium signals.

    Args:
        data_root: Path to the domain data directory, which must contain
            ``signals.json`` and ``train_stats.parquet``.

    Returns:
        A fully initialised ``BenchmarkSpec`` for the fusion domain.

    Raises:
        FileNotFoundError: If ``signals.json`` or ``train_stats.parquet``
            are not found under ``data_root``.
        ValueError: If any default signal name is not present in the
            registry (indicates a stale default).
    """
    data_root = Path(data_root)

    registry_path = data_root / "signals.json"
    stats_path = data_root / "train_stats.parquet"

    if not registry_path.exists():
        raise FileNotFoundError(f"signals.json not found at {registry_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"train_stats.parquet not found at {stats_path}")

    with registry_path.open() as fh:
        registry = json.load(fh)

    train_stats = pd.read_parquet(stats_path)

    domain = DomainSpec(
        name="fusion",
        data_root=data_root,
        registry=registry,
        train_stats=train_stats,
    )

    # Partition signals into equilibrium (target) and non-equilibrium (source).
    all_names = [v["name"] for v in registry.values()]
    equilibrium_signals = [n for n in all_names if "equilibrium" in n]
    source_signals = [n for n in all_names if "equilibrium" not in n]

    # Validate that all derived names are present in the registry.
    unrecognised = [
        n for n in equilibrium_signals + source_signals if n not in domain._name_to_id
    ]
    if unrecognised:
        raise ValueError(
            f"load_fusion_spec: the following signal names were not found in "
            f"the registry: {unrecognised}"
        )

    forecasting = ForecastingSpec(t_c=t_c, t_low=0.0, t_high=t_high)

    offline_imputation = OfflineImputationSpec(
        held_out_signals=equilibrium_signals,
        held_out_intervals={name: (0.15, 0.35) for name in equilibrium_signals},
    )

    causal_imputation = CausalImputationSpec(
        held_out_signals=equilibrium_signals,
        t_failure={name: 0.25 for name in equilibrium_signals},
        t_present=0.4,
    )

    cross_modal = CrossModalSpec(
        source_signals=source_signals,
        target_signals=equilibrium_signals,
    )

    return BenchmarkSpec(
        domain=domain,
        forecasting=forecasting,
        offline_imputation=offline_imputation,
        causal_imputation=causal_imputation,
        cross_modal=cross_modal,
    )


def load_healthcare_spec(data_root: Path) -> BenchmarkSpec:
    """Load a ``BenchmarkSpec`` for the MIMIC-IV ICU healthcare domain.

    Reads ``signals.json`` and ``train_stats.parquet`` from ``data_root`` and
    constructs task specs sized to the dataloader's ``MIN_STAY_HOURS = 24``
    filter. Timestamps are seconds since each stay's first vital recording
    (``T0 = first chartevent``), so all stays satisfy ``t >= 0`` and have
    duration ``>= 86400 s``.

    Default task specs:

    * **Forecasting**: ``t_c = 43200 s`` (12 h), horizon ``[0, 21600] s``
      (next 6 h). Picked so that the entire 24-h floor on stay length leaves
      both context (12 h) and queries (6 h) populated for every stay.
    * **Offline imputation**: hold out ``blood_pressure`` over
      ``[21600, 43200] s`` (hours 6-12). BP is the most clinically common
      missing-data target and has multi-channel structure (sbp/dbp/map).
    * **Causal imputation**: same held-out signal, failure at ``21600 s``
      (6 h), present at ``43200 s`` (12 h) — simulates a BP-cuff dropout.
    * **Cross-modal**: source = HR/SpO2/RR/BP/Temp (vitals);
      target = ``gcs`` (neuro). Tests whether continuous vitals predict
      intermittent neuro assessments.

    Args:
        data_root: Path to the domain data directory, which must contain
            ``signals.json`` and ``train_stats.parquet``.

    Returns:
        A fully initialised ``BenchmarkSpec`` for the healthcare domain.

    Raises:
        FileNotFoundError: If ``signals.json`` or ``train_stats.parquet``
            are not found under ``data_root``.
        ValueError: If any default signal name is not present in the
            registry (indicates a stale default).
    """
    data_root = Path(data_root)

    registry_path = data_root / "signals.json"
    stats_path = data_root / "train_stats.parquet"

    if not registry_path.exists():
        raise FileNotFoundError(f"signals.json not found at {registry_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"train_stats.parquet not found at {stats_path}")

    with registry_path.open() as fh:
        registry = json.load(fh)

    train_stats = pd.read_parquet(stats_path)

    domain = DomainSpec(
        name="healthcare",
        data_root=data_root,
        registry=registry,
        train_stats=train_stats,
    )

    all_names = [v["name"] for v in registry.values()]
    held_out = ["blood_pressure"]
    source_signals = [n for n in all_names if n != "gcs"]
    target_signals = ["gcs"]

    unrecognised = [n for n in held_out + all_names if n not in domain._name_to_id]
    if unrecognised:
        raise ValueError(
            f"load_healthcare_spec: the following signal names were not found "
            f"in the registry: {unrecognised}"
        )

    _6h = 21_600.0
    _12h = 43_200.0

    forecasting = ForecastingSpec(t_c=_12h, t_low=0.0, t_high=_6h)

    offline_imputation = OfflineImputationSpec(
        held_out_signals=held_out,
        held_out_intervals={name: (_6h, _12h) for name in held_out},
    )

    causal_imputation = CausalImputationSpec(
        held_out_signals=held_out,
        t_failure={name: _6h for name in held_out},
        t_present=_12h,
    )

    cross_modal = CrossModalSpec(
        source_signals=source_signals,
        target_signals=target_signals,
    )

    return BenchmarkSpec(
        domain=domain,
        forecasting=forecasting,
        offline_imputation=offline_imputation,
        causal_imputation=causal_imputation,
        cross_modal=cross_modal,
    )


def load_tcprimed_spec(data_root: Path) -> BenchmarkSpec:
    """Load a ``BenchmarkSpec`` for the TCPrimed tropical-cyclone domain.

    Reads ``signals.json`` and ``train_stats.parquet`` from ``data_root``
    automatically, then constructs task specs using typical TC-PRIMED storm
    parameters.  Timestamps are in **seconds** measured from the first
    best-track time of each storm.

    Signal groups (IDs taken from the registry):

    * **Environmental** (``track_*``, ``diag_*``): regularly sampled at 6-h
      synoptic times.  IDs are those whose names start with ``"track_"`` or
      ``"diag_"``.
    * **Overpass scalars** (``overpass_*``): irregularly sampled at satellite
      overpass times.  IDs are those whose names start with ``"overpass_"``.
    * **Overpass fields** (``gprof_*``, ``ir_*``, ``radar_*``): high-dimensional
      satellite imagery, also at overpass times.

    Default task specs:

    * **Forecasting**: ``t_c = 86400 s`` (24 h), horizon ``[0, 86400] s``
      (next 24 h).
    * **Offline imputation**: hold out overpass-scalar signals for the window
      ``[86400, 259200] s`` (days 1–3 of the storm).
    * **Causal imputation**: same overpass-scalar signals, failure at
      ``86400 s``, present at ``259200 s``.
    * **Cross-modal**: source = all environmental signals; target = all
      overpass-scalar signals.  Field signals are excluded from the target
      to keep the default evaluation tractable.

    Note:
        Static rows (``s = 0``, ``t = -1.0``) carry storm metadata
        (basin code, season, storm number) and are always included in the
        context for all tasks.

    Args:
        data_root: Path to the domain data directory, which must contain
            ``signals.json`` and ``train_stats.parquet``.

    Returns:
        A fully initialised ``BenchmarkSpec`` for the TCPrimed domain.

    Raises:
        FileNotFoundError: If ``signals.json`` or ``train_stats.parquet``
            are not found under ``data_root``.
        ValueError: If any default signal name is not present in the
            registry (indicates a stale default).
    """
    data_root = Path(data_root)

    registry_path = data_root / "signals.json"
    stats_path = data_root / "train_stats.parquet"

    if not registry_path.exists():
        raise FileNotFoundError(f"signals.json not found at {registry_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"train_stats.parquet not found at {stats_path}")

    with registry_path.open() as fh:
        registry = json.load(fh)

    train_stats = pd.read_parquet(stats_path)

    domain = DomainSpec(
        name="tcprimed",
        data_root=data_root,
        registry=registry,
        train_stats=train_stats,
    )

    # Classify signals by prefix; exclude s=0 (static) from all temporal groups.
    all_entries = [(int(k), v["name"]) for k, v in registry.items() if int(k) != 0]
    env_signals = [
        name
        for _, name in all_entries
        if name.startswith("track_") or name.startswith("diag_")
    ]
    overpass_scalar_signals = [
        name for _, name in all_entries if name.startswith("overpass_")
    ]

    # Validate all names against the registry.
    all_used = env_signals + overpass_scalar_signals
    unrecognised = [n for n in all_used if n not in domain._name_to_id]
    if unrecognised:
        raise ValueError(
            f"load_tcprimed_spec: the following signal names were not found "
            f"in the registry: {unrecognised}"
        )

    # Default time parameters (seconds).
    _24h = 86_400.0
    _3d = 259_200.0

    forecasting = ForecastingSpec(t_c=_24h, t_low=0.0, t_high=_24h)

    offline_imputation = OfflineImputationSpec(
        held_out_signals=overpass_scalar_signals,
        held_out_intervals={name: (_24h, _3d) for name in overpass_scalar_signals},
    )

    causal_imputation = CausalImputationSpec(
        held_out_signals=overpass_scalar_signals,
        t_failure={name: _24h for name in overpass_scalar_signals},
        t_present=_3d,
    )

    cross_modal = CrossModalSpec(
        source_signals=env_signals,
        target_signals=overpass_scalar_signals,
    )

    return BenchmarkSpec(
        domain=domain,
        forecasting=forecasting,
        offline_imputation=offline_imputation,
        causal_imputation=causal_imputation,
        cross_modal=cross_modal,
    )

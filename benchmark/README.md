# Benchmark

This directory implements the evaluation harness for TIM3, a cross-domain
scientific time-series benchmark. TIM3 measures how well a foundation model
generalises across heterogeneous scientific domains (nuclear fusion, tropical
cyclones, and healthcare) without task-specific finetuning. It does so through
four tasks (forecasting, offline imputation, causal imputation, cross-modal
prediction) evaluated under a unified, scale-invariant metric hierarchy.

## Contents

- [Observation set format](#observation-set-format)
- [Task definitions](#task-definitions)
  - [Forecasting](#forecasting)
  - [Offline imputation](#offline-imputation)
  - [Causal imputation](#causal-imputation)
  - [Cross-modal prediction](#cross-modal-prediction)
- [Metrics](#metrics)
- [Submitting predictions](#submitting-predictions)
- [Running the benchmark](#running-the-benchmark)
- [Signal registry](#signal-registry)

## Observation set format

Each experimental sample `k` (a plasma shot, ICU stay, storm) is represented as
a set of observations:

$$\mathcal{O}_k = \{(t_i,\ s_i,\ c_i,\ \mathbf{x}_i)\}_{i=1}^{N_k}$$

| Field | Type | Description |
|---|---|---|
| `t_i` | `float32` | Continuous timestamp in seconds. `t_i = 0` marks the start of the experiment; `t_i >= 0` for all temporal observations. |
| `s_i` | `int32` | Signal type identifier in `{1, ..., S}` (for example `magnetics_flux_loop_flux`, `heart_rate`). |
| `c_i` | `int32` | Channel index in `{1, ..., C_s}` within the signal type (for example which magnetic probe). |
| `x_i` | `object` (numpy `float32` array) | Observed value in `R^{d_s}`. For scalar signals `d_s = 1`; for spatially resolved profiles `d_s` equals the number of spatial grid points. |

The cardinality `N_k` varies across samples and encodes both experiment duration
and instrumentation completeness. No padding, imputation, or regular-grid
resampling is applied.

> **Code note.** The column `j` in every DataFrame is the channel index `c_i`
> from the paper. The codebase does not rename it: use `j` in code, `c_i` when
> citing the paper.

### Static observations

Static data (patient demographics, storm metadata, and similar) is represented
as a special observation subset:

$$\mathcal{S}_k = \{(t_{\text{static}},\ s_{\text{static}},\ c_i,\ \mathbf{x}_i)\}_{i=1}^{N^{\text{static}}_k}$$

with `t_static = -1` and `s_static = 0` for all static rows. Models should treat
`s = 0` as a distinct token type, not a temporal observation.

The complete information for sample `k` is:

$$\tilde{\mathcal{O}}_k = \mathcal{O}_k \cup \mathcal{S}_k$$

### Parquet schema

Each sample is stored as a single Parquet file. The canonical column schema is:

| Column | dtype | Notes |
|---|---|---|
| `t` | `float32` | Seconds from experiment start; `-1.0` for static rows |
| `s` | `int32` | Signal ID; `0` for static rows |
| `j` | `int32` | Channel index (`c_i`) |
| `x` | `object` | numpy `float32` array of shape `(d_s,)` |

Files whose names contain `_grid` are coordinate grids and are excluded from
evaluation automatically.

## Task definitions

For every task the data are split into a context set `C_k` and a query set
`Q_k`, with `C_k` and `Q_k` disjoint and together covering the whole sample.
Static observations are always placed in `C_k`. A model then predicts a value
for every `(t_i, s_i, c_i)` in `Q_k` from `C_k`.

### Forecasting

Given observations up to a cut-off time `t_c`, predict observations in a future
horizon band `h = [t_low, t_high]`:

$$\mathcal{C}_k = \mathcal{S}_k \cup \{ (t_i,s_i,c_i,\mathbf{x}_i)\in \mathcal{O}_k : t_i < t_c \}$$

$$\mathcal{Q}_{k,h} = \{ (t_i,s_i,c_i,\mathbf{x}_i)\in \mathcal{O}_k : t_i \geq t_c \land t_i - t_c \in h\}$$

Observations at or after `t_c` that fall outside the horizon band are dropped
from both sets.

**Spec parameters** (`ForecastingSpec`): `t_c`, `t_low`, `t_high`.

### Offline imputation

Given a set of held-out signals `H` and per-signal masking intervals
`[t_a^(s), t_b^(s)]` for each `s` in `H`:

$$\mathcal{C}_k = \mathcal{S}_k \cup \{ (t_i,s_i,c_i,\mathbf{x}_i)\in \mathcal{O}_k : s_i \notin \mathcal{H} \lor t_i \notin [t^{(s_i)}_a, t^{(s_i)}_b]\}$$

$$\mathcal{Q}_k = \{ (t_i,s_i,c_i,\mathbf{x}_i)\in \mathcal{O}_k : s_i \in \mathcal{H} \land t_i \in [t^{(s_i)}_a, t^{(s_i)}_b]\}$$

The model may use all observations from non-held-out signals, and observations
from held-out signals outside their masked windows.

**Spec parameters** (`OfflineImputationSpec`): `held_out_signals`, `held_out_intervals` (omit to mask the entire duration of each signal).

### Causal imputation

Like offline imputation, but restricted to observations up to a present time
`t_p`, simulating real-time deployment. Each held-out signal `s` in `H` has a
failure time `t_f^(s) < t_p` after which it stops reporting:

$$\mathcal{C}_k = \mathcal{S}_k \cup \{ (t_i,s_i,c_i,\mathbf{x}_i)\in \mathcal{O}_k : t_i \leq t_p \land (s_i \notin \mathcal{H} \lor t_i < t^{(s_i)}_f)\}$$

$$\mathcal{Q}_k = \{ (t_i,s_i,c_i,\mathbf{x}_i)\in \mathcal{O}_k : t_i \leq t_p \land s_i \in \mathcal{H} \land t_i \geq t^{(s_i)}_f\}$$

All observations after `t_p` are discarded from both sets.

**Spec parameters** (`CausalImputationSpec`): `held_out_signals`, `t_failure` (dict from signal name to `t_f^(s)`), `t_present` (`t_p`).

### Cross-modal prediction

Given disjoint source and target signal sets `S_src` and `S_tgt`, predict all
target-signal observations from all source-signal observations (offline, with no
causal constraint):

$$\mathcal{C}_k = \mathcal{S}_k \cup \{ (t_i,s_i,c_i,\mathbf{x}_i)\in \mathcal{O}_k : s_i \in \mathcal{S}_{\text{src}}\}$$

$$\mathcal{Q}_k = \{ (t_i,s_i,c_i,\mathbf{x}_i)\in \mathcal{O}_k : s_i \in \mathcal{S}_{\text{tgt}}\}$$

Observations belonging to neither source nor target are dropped. In the fusion
domain, for example, this task corresponds to predicting equilibrium
reconstructions (EFIT) from magnetic probe data.

**Spec parameters** (`CrossModalSpec`): `source_signals`, `target_signals` (must be disjoint).

## Metrics

### Per-observation errors

Let `(t_i, s_i, c_i, x_i)` be a query observation, `x_hat_i` the model
prediction, and `sigma_{s,c,m}` the training-data standard deviation of the
`m`-th component of signal `s`, channel `c`.

**Normalised mean absolute error (NMAE):**

$$l_i^{\text{NMAE}} = \frac{1}{d_{s_i}} \sum_{m=1}^{d_{s_i}} \frac{|x_{i,m} - \hat{x}_{i,m}|}{\sigma_{s_i, c_i, m}}$$

**Normalised mean squared error (NMSE):**

$$l_i^{\text{NMSE}} = \frac{1}{d_{s_i}} \sum_{m=1}^{d_{s_i}} \left(\frac{x_{i,m} - \hat{x}_{i,m}}{\sigma_{s_i, c_i, m}}\right)^2$$

The score aggregates these per-observation errors through five levels, so that
dense signals, high-dimensional modalities, and long samples cannot dominate.

### Level 1: per signal-channel pair

Average the per-observation errors within each `(signal, channel)` pair. NRMSE
is the square root of the mean NMSE.

$$\mathcal{L}^{\text{NMAE}}_{k,(s,c)} = \frac{1}{|\mathcal{Q}_{k,(s,c)}|} \sum_{i \in \mathcal{Q}_{k,(s,c)}} l_i^{\text{NMAE}}$$

$$\mathcal{L}^{\text{NRMSE}}_{k,(s,c)} = \sqrt{\frac{1}{|\mathcal{Q}_{k,(s,c)}|} \sum_{i \in \mathcal{Q}_{k,(s,c)}} l_i^{\text{NMSE}}}$$

### Level 2: per sample

Average over the `(signal, channel)` pairs present in the query. Each pair
contributes equally, regardless of how many observations it holds.

$$\mathcal{L}^{\text{NMAE}}_{k} = \frac{1}{|I_k|} \sum_{(s,c) \in I_k} \mathcal{L}^{\text{NMAE}}_{k,(s,c)}$$

### Level 3: per task

Average over the evaluation samples for the task. Each sample contributes
equally.

$$\mathcal{L}^{\text{NMAE}}_{\tau} = \frac{1}{|T_\tau|} \sum_{k \in T_\tau} \mathcal{L}^{\text{NMAE}}_{k}$$

### Level 4: per domain

Average over the four tasks. Each task contributes equally.

$$\mathcal{L}^{\text{NMAE}}_{d} = \frac{1}{|B_d|} \sum_{\tau \in B_d} \mathcal{L}^{\text{NMAE}}_{\tau}$$

### Level 5: overall benchmark

Average over the scientific domains. Each domain contributes equally.

$$\mathcal{L}^{\text{NMAE}} = \frac{1}{|D|} \sum_{d \in D} \mathcal{L}^{\text{NMAE}}_{d}$$

NRMSE aggregates the same way at every level (Level 2 onward averages the
per-pair NRMSE values).

### Training statistics

Normalisation uses per-dimension standard deviations estimated on the training
split only, stored in `train_stats.parquet`:

| Column | dtype | Description |
|---|---|---|
| `s` | `int32` | Signal ID |
| `j` | `int32` | Channel index (`c_i`) |
| `m` | `int32` | Dimension index within `x` (0-indexed) |
| `mean` | `float32` | Training mean `mu_{s,c,m}` |
| `std` | `float32` | Training standard deviation `sigma_{s,c,m}` |

Zero or non-finite standard deviations are replaced by `1e-8` to avoid division
by zero. For variable-length signals (profiles whose spatial resolution varies
between samples), a single global `sigma` (stored at `m = 0`) is broadcast
across all dimensions.

### Missing-prediction penalty

If a model does not return a prediction for a query observation, the evaluator
substitutes the training mean vector `mu_{s,c}`. In normalised space this gives
`NMSE` near `1.0`, matching the mean-imputation baseline. For `(s, c)` pairs
absent from `train_stats`, zeros are used as a fallback and a warning is logged.

## Submitting predictions

This section defines the prediction-file contract used by the evaluator. To turn
a directory of predictions into a scored **leaderboard entry**
(`leaderboard/<name>/{results.json, config.json}`), see
[`scripts/make_submission.py`](../scripts/make_submission.py) and the submission
guide in [CONTRIBUTING.md](../CONTRIBUTING.md#reporting-results-leaderboard).

### Prediction DataFrame format

Predictions for each sample must be provided as a Parquet file with the
following schema:

| Column | dtype | Description |
|---|---|---|
| `t` | `float32` | Timestamp; **must be copied from the query set** |
| `s` | `int32` | Signal ID |
| `j` | `int32` | Channel index |
| `x_pred` | `object` | numpy `float32` array of shape `(d_s,)` |

Write one prediction file per sample, with the **same filename** as the
corresponding observation Parquet file.

### Timestamp requirement

Prediction timestamps must be copied directly from the query-set Parquet files.
The evaluator matches predictions to query observations on exact `(s, j, t)`
keys after canonicalising both timestamp columns to `float32`. Matching on the
full `float32` value, rather than a coarser rounded key, keeps distinct
high-frequency timestamps distinct, so there is exactly one prediction row per
query row. Copy prediction timestamps from the query set rather than recomputing
them.

### Missing predictions

Any query observation with no matching prediction row receives the training-mean
penalty described above. Duplicate predictions for the same canonicalised
`(s, j, t)` key are deduplicated by keeping the first occurrence and logging a
warning.

### Directory layout

```
predictions/
  <domain>/
    <task>/
      <sample_filename>.parquet   # same name as the observation file
      ...
```

## Running the benchmark

### Prerequisites

```bash
uv sync
```

### `run_benchmark`: full evaluation in one call

The `BenchmarkSpec` bundles the domain metadata (signal registry, training
statistics, data root) with per-task parameters:

```python
from pathlib import Path
from benchmark.specs import load_fusion_spec, load_tcprimed_spec

fusion_spec    = load_fusion_spec(Path("data/fusion_data"))
tcprimed_spec  = load_tcprimed_spec(Path("data/tcprimed_data"))
```

Default task parameters for each domain are documented in `specs.py`.

```python
from pathlib import Path
from benchmark.run import run_benchmark
from benchmark.specs import load_fusion_spec, load_tcprimed_spec

result = run_benchmark(
    predictions_root=Path("predictions/my_model"),
    domain_specs={
        "fusion":   load_fusion_spec(Path("data/fusion_data")),
        "tcprimed": load_tcprimed_spec(Path("data/tcprimed_data")),
    },
    split_dirs={
        "fusion":   Path("data/fusion_data/test"),
        "tcprimed": Path("data/tcprimed_data/test"),
    },
)
print(f"Benchmark NMAE:  {result['nmae']:.4f}")
print(f"Benchmark NRMSE: {result['nrmse']:.4f}")
```

Predictions must be organised under `predictions_root` using the expected
directory layout:

```
predictions/my_model/
  <domain>/
    <task>/
      <sample_filename>.parquet   # same name as the observation file
      ...
```

Tasks for which no prediction directory is found are skipped with a warning. A
partial submission (for example forecasting only) still produces a meaningful
score for the tasks that were evaluated.

To inspect per-domain and per-task results:

```python
for domain in result["per_domain"]:
    print(f"\n{domain['domain']}: NMAE={domain['nmae']:.4f}")
    print(f"  Tasks run:     {domain['tasks_run']}")
    print(f"  Tasks skipped: {domain['tasks_skipped']}")
    for task in domain["per_task"]:
        print(f"  {task['task']}: NMAE={task['nmae']:.4f}  "
              f"({task['n_samples']} samples)")
```

**Returns:** dict with keys `nmae`, `nrmse`, `n_domains`, `per_domain`.

### Lower-level API

Use these functions to build custom evaluation pipelines or to evaluate
individual samples interactively.

#### `evaluate_sample`: one sample, one task

```python
from benchmark.evaluate import evaluate_sample
import pandas as pd

predictions = pd.read_parquet("predictions/fusion/forecasting/shot_30451.parquet")
result = evaluate_sample(
    parquet_path=Path("data/fusion_data/test/shot_30451.parquet"),
    predictions=predictions,
    task_name="forecasting",
    spec=fusion_spec,
)
# result["nmae"], result["nrmse"], result["per_signal_channel"], ...
```

**Returns:** dict with keys `nmae`, `nrmse`, `n_signal_channels`,
`per_signal_channel`, `sample_id`, `task`, `domain`, `n_query_obs`,
`n_matched`.

#### `evaluate_split`: all samples in a split, one task (level 3)

```python
from benchmark.evaluate import evaluate_split

task_result = evaluate_split(
    split_dir=Path("data/fusion_data/test"),
    predictions_dir=Path("predictions/fusion/forecasting"),
    task_name="forecasting",
    spec=fusion_spec,
)
# task_result["nmae"], task_result["nrmse"], task_result["n_samples"], ...
```

Samples with no matching prediction file are skipped with a warning.

**Returns:** dict with keys `task`, `domain`, `nmae`, `nrmse`, `n_samples`,
`n_skipped`, `per_sample`.

#### `evaluate_domain`: all tasks for a domain (level 4)

```python
from benchmark.evaluate import evaluate_domain

task_results = [
    evaluate_split(split_dir, predictions_dir / "forecasting",      "forecasting",      fusion_spec),
    evaluate_split(split_dir, predictions_dir / "offline_imputation","offline_imputation",fusion_spec),
    evaluate_split(split_dir, predictions_dir / "causal_imputation", "causal_imputation",fusion_spec),
    evaluate_split(split_dir, predictions_dir / "cross_modal",       "cross_modal",      fusion_spec),
]

domain_result = evaluate_domain(task_results)
# domain_result["nmae"], domain_result["nrmse"], domain_result["n_tasks"]
```

**Returns:** dict with keys `nmae`, `nrmse`, `n_tasks`, `per_task`.

#### `evaluate_benchmark`: all domains (level 5)

```python
from benchmark.evaluate import evaluate_benchmark

benchmark_result = evaluate_benchmark([fusion_domain_result, tcprimed_domain_result])
# benchmark_result["nmae"], benchmark_result["nrmse"], benchmark_result["n_domains"]
```

**Returns:** dict with keys `nmae`, `nrmse`, `n_domains`, `per_domain`.

## Signal registry

Each domain directory contains a `signals.json` file that maps integer signal
IDs (as string keys) to signal metadata. The evaluator uses this file to resolve
signal names to integer IDs and to produce human-readable keys in the
`per_signal_channel` output.

### Schema

```json
{
  "<signal_id>": {
    "name":       "snake_case signal name (used as the registry key)",
    "category":   "one of: scalar_single, scalar_multichannel, profile_1d, static",
    "n_channels": "number of channels C_s for this signal",
    "value_dim":  "dimensionality d_s of each observation vector x_i",
    "time_dim":   "name of the time dimension in the source file, or null",
    "unit":       "physical unit (empty string if dimensionless)"
  }
}
```

Domain-specific optional fields (for example `source`, `group`,
`channel_labels`, `var`) may also be present but are not used by the evaluator.

### Reserved IDs

| ID | Meaning |
|---|---|
| `"0"` | Static observations (`s = 0`, `t = -1`). Always in context, never in query. |
| `"1"` and up | Temporal signal types, 1-indexed. |

### Categories

| Category | `d_s` | Notes |
|---|---|---|
| `scalar_single` | 1 | Single channel, scalar value |
| `scalar_multichannel` | 1 | Multiple channels, each a scalar |
| `profile_1d` | > 1 | Spatially resolved 1-D profile; `d_s` is the number of grid points |
| `static` | 1 | Static metadata; always `s = 0` |

### Example (fusion domain)

```json
{
  "1": {
    "name": "magnetics_flux_loop_flux",
    "category": "scalar_multichannel",
    "n_channels": 15,
    "value_dim": 1,
    "time_dim": "time_magnetics_flux_loop_flux",
    "unit": "Wb"
  }
}
```

This entry defines signal `s = 1` with 15 channels (`j` in `{1, ..., 15}`), each
recording a scalar flux value in webers. In the `per_signal_channel` output,
entries for this signal appear as `magnetics_flux_loop_flux/<j>`.

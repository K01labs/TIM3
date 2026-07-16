# Dataloaders

A central challenge in benchmarking across heterogeneous scientific domains is defining a unified data representation that preserves the native structure of each domain's measurements. This includes: irregular sampling, per-signal missingness, mixed dimensionality, and static metadata. Furthermore, we want to shift away from the traditional assumptions such as regular temporal grids or uniform signal dimensionality.
Below, we introduce a set-based observation formulation that accommodates all three benchmark domains under a single framework.

## Observation sets

We represent each experimental sample (a plasma shot, an ICU admission) as an unordered set of observations:

$$\mathcal{O}_k = \{(t_i,\ s_i,\ j_i,\ \mathbf{x}_i)\}_{i=1}^{N_k}$$

Here `k` indexes the sample, `t_i` is the continuous timestamp of the `i`-th observation, `s_i` in `{1, ..., S}` identifies the signal type (for example "flux loop", "electron density profile", "heart rate"), `j_i` in `{1, ..., J_s}` identifies the channel within that signal type (for example which magnetic probe), and `x_i` in `R^{d_s}` is the observed value, whose dimensionality `d_s` is set by the signal type. For scalar signals such as plasma current, `d_s = 1`. For spatially resolved measurements such as radial electron density profiles, `d_s` equals the number of spatial grid points. For two-dimensional field measurements such as the poloidal magnetic flux map, `x_i` is the flattened field with `d_s = n_z * n_r`.

The cardinality `N_k` varies across samples and encodes both the duration of the experiment and the completeness of its instrumentation. A sample with extensive diagnostic coverage has a larger `N_k` than one with offline instruments. No explicit imputation, padding, or regular-grid resampling is required.

## Handling of core data challenges

The set-based formulation addresses the three defining properties of heterogeneous scientific time series:

**Missingness.** A signal that was not observed at a given time contributes no observation to `O_k`. Per-channel missingness (for example a single faulty probe among an array of sensors) is captured by the absence of observations with the corresponding `(s_i, j_i)` pair.

**Multi-frequency sampling.** Signals sampled at different rates (for example magnetics at ~500 kHz and Thomson scattering at ~100 Hz in fusion plasmas) coexist naturally as observations with different temporal densities. High-frequency signals contribute more observations per unit time. The continuous timestamps `t_i` record the actual measurement times without requiring alignment to a shared grid.

**Multi-modal and mixed-dimensionality data.** The signal type `s_i` and its associated dimensionality `d_s` accommodate scalar measurements, one-dimensional spatial profiles, and two-dimensional fields within a single set. Each signal's metadata (spatial coordinates, physical units, expected sampling frequency) is stored in an auxiliary signal registry rather than in the observation tuples themselves, keeping the per-observation schema uniform.

## Static context

Sample-level metadata that does not vary in time, such as machine configuration parameters and patient demographics, is represented as additional observations with a single timestamp `t_static` and a reserved signal type:

$$ \mathcal{S}_k = \{(t_{\text{static}},\ s_{\text{static}},\ j,\ c_k^{(j)})\}_{j=1}^{D_c}$$

Here `c_k^(j)` is the `j`-th static feature in sample `k` (encoded numerically) and `D_c` is the total number of static features. This lets models condition on static context without a separate input pathway. The full set for sample `k` is then:

$$ \tilde{\mathcal{O}}_k = \mathcal{O}_k \;\cup\ \mathcal{S}_k$$

## Fusion (MAST Tokamak)

### What the data represents

Each sample is a single plasma **shot** from the MAST (Mega Ampere Spherical Tokamak). A shot is one plasma discharge, typically lasting a few seconds, during which a large array of diagnostics record the state of the plasma simultaneously. 

Shots are heterogeneous: different diagnostics sample at very different rates, instruments occasionally fail mid-shot, and the plasma duration itself varies. This makes the dataset a natural fit for the observation-set format; no resampling or padding is applied.

### Directory layout

After running `uv run python data/dataloaders/fusion_dataset.py`, the processed data lives under `data/fusion_data/`. This single script fetches shots directly from the MAST S3 Zarr store and converts them to benchmark Parquet format in one step (no intermediate files are needed):

```
data/fusion_data/
├── signals.json              # signal registry (metadata for all 39 signals)
├── splits.json               # train/val/test assignment for every shot
├── train_stats.parquet      # per-(s, j, m) mean/std computed on train split
├── train/
│   ├── shot_11766.parquet    # observations for shot 11766
│   ├── shot_11766_grid.parquet  # expected observation grid for shot 11766
│   ├── shot_11774.parquet
│   ├── shot_11774_grid.parquet
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Shots are split **by temporal ordering** (not randomly) to prevent data leakage across the sequence of experiments:

| Split | Shots | Fraction |
|-------|-------|----------|
| train | 2022  | 70 %     |
| val   | 433   | 15 %     |
| test  | 434   | 15 %     |

---

### Observation files (`shot_XXXXX.parquet`)

Each shot's Parquet file is a table of observations following the format defined above. Every row is one measurement:

| Column | Type | Description |
|--------|------|-------------|
| `t` | float32 | Timestamp in seconds |
| `s` | int32 | Signal ID (maps to `signals.json`) |
| `j` | int32 | Channel index within that signal |
| `x` | list\[float32\] | Observed value; length 1 for scalars, 120 for Thomson scattering profiles, 4225 for the poloidal flux field |

Only non-NaN observations are stored. A typical shot contains several million rows across all signals.

```python
import pandas as pd

df = pd.read_parquet("data/fusion_data/train/shot_11766.parquet")
print(df.shape)       # e.g. (4124637, 4)
print(df.dtypes)
# t      float32
# s        int32
# j        int32
# x       object   ← list of float32

# All observations for plasma current (summary_ip, signal 19)
ip = df[df["s"] == 19]
print(ip["t"].values)   # timestamps in seconds
print(ip["x"].tolist()) # [[value], [value], ...]
```

---

### Signal registry (`signals.json`)

The registry maps each integer signal ID to its metadata. Load it once and use it to interpret the `s` column in any Parquet file.

```python
import json

with open("data/fusion_data/signals.json") as f:
    signals = json.load(f)

# signals["19"] →
# {
#   "name":       "summary_ip",
#   "category":   "scalar_single",
#   "n_channels": 1,
#   "value_dim":  1,
#   "time_dim":   "time_summary_ip",
#   "unit":       "A"
# }
```

The 39 signals fall into four categories:

| Category | Count | `x` length | Example signals |
|----------|-------|------------|-----------------|
| `scalar_single` | 20 | 1 | `summary_ip` (A), `equilibrium_q95`, `interferometer_n_e_line` |
| `scalar_multichannel` | 14 | 1 per row; `j` selects channel | `magnetics_flux_loop_flux` (15 ch, Wb), `soft_x_rays_horizontal_cam_lower` (18 ch) |
| `profile_1d` | 4 | spatial grid size | `thomson_scattering_t_e` (120 pts, eV), `equilibrium_lcfs_r` (139 pts, m) |
| `field_2d` | 1 | 4225 (flattened 2-D flux map) | `equilibrium_psi` (Wb/rad) |

**Sampling rates** vary by several orders of magnitude. High-speed magnetic probes (Mirnov coils) sample at ~500 kHz, producing hundreds of thousands of timesteps per shot. Equilibrium reconstructions and Thomson scattering run at ~50–100 Hz, producing tens to hundreds of timesteps. All coexist naturally as rows with different temporal densities.

A name-to-ID lookup is convenient for working with specific signals:

```python
name_to_id = {info["name"]: int(sid) for sid, info in signals.items()}
# name_to_id["summary_ip"] → 19
# name_to_id["equilibrium_psi"] → 39
```

---

### Observation grid files (`shot_XXXXX_grid.parquet`)

Each shot also has a companion grid file that records **every timestamp that was attempted** by each diagnostic, including timesteps where the instrument returned NaN. The observation Parquet only contains valid readings; the grid is the missingness oracle.

| Column | Type | Description |
|--------|------|-------------|
| `s` | int32 | Signal ID |
| `t` | float32 | Attempted timestamp |

For multichannel signals, the grid stores one row per `(signal, timestep)`, not per `(signal, channel, timestep)`, because all channels share the same time axis. A model that needs per-channel missingness can expand using `n_channels` from `signals.json`.

**Recovering missing observations:**

```python
grid = pd.read_parquet("data/fusion_data/train/shot_11766_grid.parquet")
obs  = pd.read_parquet("data/fusion_data/train/shot_11766.parquet")

# Full expected schedule for Thomson scattering electron temperature (sid=8)
expected = grid[grid["s"] == 8]["t"].values

# Observed timestamps
observed = obs[obs["s"] == 8]["t"].values

# Missing timestamps
missing = set(expected) - set(observed)
print(f"{len(missing)} / {len(expected)} Thomson T_e timesteps missing")
```

For a learnable missingness mask, join on `(s, t)`:

```python
full = grid[grid["s"] == 8][["t"]].copy()
full["observed"] = full["t"].isin(obs[obs["s"] == 8]["t"].values)
# True  → valid observation present
# False → measurement attempted but NaN in raw data
```

---

### Loading a shot for model training

A minimal PyTorch-style dataset that returns all observations for a single shot:

```python
import json
from pathlib import Path
import pandas as pd
import numpy as np

class FusionShotDataset:
    def __init__(self, split_dir: Path, signals_path: Path):
        self.files = sorted(
            p for p in split_dir.glob("shot_*.parquet")
            if "_grid" not in p.name
        )
        with open(signals_path) as f:
            self.signals = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_parquet(self.files[idx])
        return {
            "t": df["t"].to_numpy(dtype=np.float32),
            "s": df["s"].to_numpy(dtype=np.int32),
            "j": df["j"].to_numpy(dtype=np.int32),
            "x": df["x"].tolist(),   # list of variable-length arrays
        }

dataset = FusionShotDataset(
    split_dir=Path("data/fusion_data/train"),
    signals_path=Path("data/fusion_data/signals.json"),
)
shot = dataset[0]
print(f"Shot has {len(shot['t']):,} observations across all signals")
```

> **Note on `x`:** because `value_dim` differs across signals (1 for scalars, 120 for Thomson profiles, 4225 for the flux field), `x` is a ragged array. Models that require dense tensors should either process one signal category at a time, or pad to the maximum `value_dim` and mask accordingly.

## Climate (TC PRIMED)

### What the data represents

Each sample is a single **tropical cyclone storm lifecycle** from the TC PRIMED (Tropical Cyclone PRecipitation, Infrared, Microwave, and Environmental Dataset). A storm typically spans days to weeks, during which two types of data are collected:

- **Environmental data**: best-track storm metadata (position, intensity, pressure) and SHIPS-style diagnostics (SST, wind shear, humidity), recorded at 6-hourly synoptic times.
- **Satellite overpass data**: low-Earth orbit satellite observations (passive microwave brightness temperatures, GPROF precipitation retrievals, geostationary infrared imagery, radar-radiometer products), recorded aperiodically as satellites pass over the storm.

Storms are heterogeneous: the number and timing of overpasses varies by storm, different sensors produce different spatial resolutions, and some instruments are only available on certain satellite platforms (e.g., radar only on TRMM/GPM). This makes the dataset a natural fit for the observation-set format; no resampling or padding is applied.

Data is streamed from the NOAA Open Data Dissemination (NODD) AWS S3 bucket via the `tcprimedapi` Python package.

> **Download scale.** Run with no arguments, the loader fetches **every basin for
> seasons 1987–2023** and streams all overpass files per storm, a very large,
> multi-day download. For a quick run, restrict the scope, e.g.
> `--basins AL --start_season 2019 --end_season 2020`. The tracked
> `data/tcprimed_data/{signals.json, splits.json, train_stats.parquet}` pin the
> exact registry, split, and stats used in the paper.

### Directory layout

After running `uv run python data/dataloaders/climate_dataset.py`, the processed data lives under `data/tcprimed_data/`. This single script fetches storms directly from the TC PRIMED S3 bucket and converts them to benchmark Parquet format in one step:

```
data/tcprimed_data/
├── signals.json                  # signal registry (metadata for all signals)
├── splits.json                   # train/val/test assignment for every storm
├── train_stats.parquet          # per-(s, j, m) mean/std computed on train split
├── train/
│   ├── storm_al021987.parquet    # observations for one storm lifecycle
│   ├── storm_al021987_grid.parquet  # expected observation grid
│   ├── storm_al031987.parquet
│   ├── storm_al031987_grid.parquet
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Storms are split **by temporal ordering** (sorted by season, then basin and number) to prevent data leakage:

| Split | Storms | Fraction |
|-------|--------|----------|
| train | 2485   | 70 %     |
| val   | 532    | 15 %     |
| test  | 534    | 15 %     |

---

### Observation files (`storm_ATCFID.parquet`)

Each storm's Parquet file is a table of observations following the observation-set format. Every row is one measurement:

| Column | Type | Description |
|--------|------|-------------|
| `t` | float32 | Timestamp in seconds since storm start (t=0 is first best-track time) |
| `s` | int32 | Signal ID (maps to `signals.json`); signal 0 is `static_context`, emitted at `t = -1` |
| `j` | int32 | Channel index within that signal |
| `x` | list\[float32\] | Observed value; length 1 for scalars, variable for 2D fields |

Only non-NaN observations are stored (for 2D fields, the entire field is stored if it contains at least one valid pixel; individual invalid pixels remain as NaN within `x`).

```python
import pandas as pd

df = pd.read_parquet("data/tcprimed_data/train/storm_al021987.parquet")
print(df.shape)       # e.g. (5823, 4)
print(df.dtypes)
# t      float32
# s        int32
# j        int32
# x       object   <- list of float32

# All observations for track intensity (signal 1)
intensity = df[df["s"] == 1]
print(intensity["t"].values)   # timestamps in seconds since storm start
print(intensity["x"].tolist()) # [[value], [value], ...]
```

---

### Signal registry (`signals.json`)

The registry maps each integer signal ID to its metadata. Load it once and use it to interpret the `s` column in any Parquet file.

```python
import json

with open("data/tcprimed_data/signals.json") as f:
    signals = json.load(f)

# signals["1"] ->
# {
#   "name":       "track_intensity",
#   "category":   "scalar_single",
#   "n_channels": 1,
#   "value_dim":  1,
#   "time_dim":   "time",
#   "unit":       "kt",
#   "source":     "env",
#   "group":      "storm_metadata",
#   "var":        "intensity"
# }
```

The 21 signals fall into four categories:

| Category | Count | `x` length | Example signals |
|----------|-------|------------|-----------------|
| `static` | 1 | 1 per feature | `static_context` (basin, season, storm number) |
| `scalar_single` | 11 | 1 | `track_intensity` (kt), `track_pressure` (hPa), `diag_sst` (K), `overpass_intensity` |
| `scalar_multichannel` | 5 | 1 per row; `j` selects channel | `diag_shear_magnitude`, `diag_precipitable_water`, `diag_warm_core` |
| `field_2d` | 4 | spatial grid size | `gprof_surface_precip` (mm/hr), `ir_brightness_temp` (K), `radar_precip_rate` (mm/hr) |

Raw passive-microwave brightness-temperature (`TB_*`) channels are intentionally
excluded; their channel sets and swath shapes differ by sensor.

**Sampling rates** vary significantly. Environmental scalars are recorded at synoptic times (every 6 hours). Satellite overpasses are aperiodic, occurring whenever an LEO satellite passes over the storm (typically a few per day, but varies). 2D field sizes depend on the sensor: infrared grids, GPROF precipitation swaths, and radar footprints all have different spatial extents.

A name-to-ID lookup is convenient for working with specific signals:

```python
name_to_id = {info["name"]: int(sid) for sid, info in signals.items()}
# name_to_id["track_intensity"]  -> 1
# name_to_id["ir_brightness_temp"] -> 19
```

---

### Static context

Each storm includes static context encoded as observations with `t = -1.0` and `s = 0`:

| `j` | Value | Description |
|-----|-------|-------------|
| 0 | basin code | AL=0, EP=1, CP=2, WP=3, IO=4, SH=5, SL=6 |
| 1 | season | 4-digit year (e.g., 2019) |
| 2 | storm number | 2-digit number within basin+season (e.g., 9) |

```python
static = df[df["s"] == 0]   # static_context, emitted at t = -1
print(static[["j", "x"]].to_string())
# j       x
# 0  [0.0]    <- basin AL
# 1  [2019.0] <- season
# 2  [9.0]    <- storm number 09
```

---

### Observation grid files (`storm_ATCFID_grid.parquet`)

Each storm also has a companion grid file that records **every timestamp that was attempted** by each signal, including timesteps where the instrument returned NaN. The observation Parquet only contains valid readings; the grid is the missingness oracle.

| Column | Type | Description |
|--------|------|-------------|
| `s` | int32 | Signal ID |
| `t` | float32 | Attempted timestamp |

For environmental signals, the grid contains one row per synoptic timestep per signal. For overpass signals, the grid contains a row only for the signals actually present in each overpass file, so its missingness reflects true sensor availability, not every possible sensor at every overpass time.

```python
grid = pd.read_parquet("data/tcprimed_data/train/storm_al021987_grid.parquet")
obs  = pd.read_parquet("data/tcprimed_data/train/storm_al021987.parquet")

# Full expected schedule for track intensity (sid=1)
expected = grid[grid["s"] == 1]["t"].values

# Observed timestamps
observed = obs[obs["s"] == 1]["t"].values

# Missing timestamps
missing = set(expected) - set(observed)
print(f"{len(missing)} / {len(expected)} track intensity timesteps missing")
```

---

### Loading a storm for model training

A minimal PyTorch-style dataset that returns all observations for a single storm:

```python
import json
from pathlib import Path
import pandas as pd
import numpy as np

class ClimateStormDataset:
    def __init__(self, split_dir: Path, signals_path: Path):
        self.files = sorted(
            p for p in split_dir.glob("storm_*.parquet")
            if "_grid" not in p.name
        )
        with open(signals_path) as f:
            self.signals = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_parquet(self.files[idx])
        return {
            "t": df["t"].to_numpy(dtype=np.float32),
            "s": df["s"].to_numpy(dtype=np.int32),
            "j": df["j"].to_numpy(dtype=np.int32),
            "x": df["x"].tolist(),   # list of variable-length arrays
        }

dataset = ClimateStormDataset(
    split_dir=Path("data/tcprimed_data/train"),
    signals_path=Path("data/tcprimed_data/signals.json"),
)
storm = dataset[0]
print(f"Storm has {len(storm['t']):,} observations across all signals")
```

> **Note on `x`:** because `value_dim` differs across signals (1 for environmental scalars, variable for 2D satellite fields), `x` is a ragged array. 2D field values may contain NaN for invalid pixels. Models that require dense tensors should either process one signal category at a time, or pad to the maximum `value_dim` and mask accordingly.

## Healthcare (MIMIC-IV ICU)

> **MIMIC-IV is restricted data and is not shipped with this repo.** You need
> your own credentialed PhysioNet access and must regenerate it locally. See
> **[HEALTHCARE.md](HEALTHCARE.md)** for the full step-by-step guide (download,
> regenerate, verify against the published split checksum). The summary below
> describes the format the loader produces.

### What the data represents

Each sample is a single **ICU stay** from MIMIC-IV v3.1. Only stays with a minimum duration of 24 hours are included. During a stay, bedside monitors and nursing assessments record up to 10 physiological channels across 6 signal types (heart rate, SpO2, respiratory rate, blood pressure, temperature, and the Glasgow Coma Scale).

ICU stays are heterogeneous: observation frequency varies by signal and by clinical workflow, instruments are occasionally offline, and stay durations span from 24 hours to several weeks. This makes the dataset a natural fit for the observation-set format; no resampling or padding is applied.

### Directory layout

After running `uv run python data/dataloaders/healthcare_dataset.py`, the processed data lives under `data/healthcare_data/`. Use `--mimic_base` to point at a non-default MIMIC-IV location; use `--limit N` for a quick smoke run:

```
data/healthcare_data/
├── signals.json              # 6 signal types, 10 channels
├── splits.json               # train/val/test assignment per stay
├── train_stats.parquet       # per-(s, j, m) mean/std computed on train split
├── train/
│   ├── stay_30000001.parquet
│   ├── stay_30000001_grid.parquet
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Stays are split **by deterministic random assignment** (seed=42, 70/15/15) rather than by temporal ordering, because ICU stays in MIMIC have no meaningful global chronological ordering that would create data leakage:

| Split | Fraction |
|-------|----------|
| train | 70 %     |
| val   | 15 %     |
| test  | 15 %     |

> **Split granularity.** The split is by ICU **stay**, not by subject, so a
> patient with multiple ICU stays may appear in more than one split. This
> matches the paper and the published split checksum; see
> [HEALTHCARE.md](HEALTHCARE.md).

---

### Observation files (`stay_XXXXXXXX.parquet`)

Each stay's Parquet file is a table of observations following the observation-set format. Every row is one measurement:

| Column | Type | Description |
|--------|------|-------------|
| `t` | float32 | Seconds since T0 (first recorded vital charttime of the stay); always ≥ 0 |
| `s` | int32 | Signal ID (maps to `signals.json`) |
| `j` | int32 | Channel index within that signal |
| `x` | list\[float32\] | Observed value; length 1 for all healthcare channels |

Only non-NaN observations are stored (after physio-range clamping; out-of-range values are dropped). A typical stay contains a few thousand rows across all signals.

```python
import pandas as pd

df = pd.read_parquet("data/healthcare_data/train/stay_30000001.parquet")
print(df.shape)       # e.g. (3412, 4)
print(df.dtypes)
# t      float32
# s        int32
# j        int32
# x       object   <- list of float32

# All heart rate observations (signal 1)
hr = df[df["s"] == 1]
print(hr["t"].values)    # seconds since T0
print(hr["x"].tolist())  # [[value], [value], ...]
```

---

### Signal registry (`signals.json`)

The registry maps each integer signal ID to its metadata. Load it once and use it to interpret the `s` column in any Parquet file.

```python
import json

with open("data/healthcare_data/signals.json") as f:
    signals = json.load(f)

# signals["1"] →
# {
#   "name":       "heart_rate",
#   "n_channels": 1,
#   "channels":   ["hr"],
#   "unit":       "bpm",
#   "physio_range": [0, 300]
# }
```

The 6 signal types covering 10 channels:

| Signal ID | Name | Channels | Channel names | Unit | Physio range |
|-----------|------|----------|---------------|------|--------------|
| 1 | heart_rate | 1 | hr | bpm | 0–300 |
| 2 | spo2 | 1 | spo2 | % | 0–100 |
| 3 | respiratory_rate | 1 | rr | breaths/min | 0–80 |
| 4 | blood_pressure | 3 | sbp, dbp, map | mmHg | 0–300 |
| 5 | temperature | 1 | temp | °C | 25–45 |
| 6 | gcs | 3 | gcs_eye, gcs_verbal, gcs_motor | score | 1–4 / 1–5 / 1–6 |

**T0 alignment:** T0 is the first recorded vital sign charttime of the stay. All `t` values in the output Parquets are seconds since T0, so `t ≥ 0` always holds.

**Unit conversions:** MIMIC itemid `223761` (Temperature Fahrenheit) is converted to Celsius before the physio-range clamp is applied. All other channels are stored in their native MIMIC units.

A name-to-ID lookup is convenient for working with specific signals:

```python
name_to_id = {info["name"]: int(sid) for sid, info in signals.items()}
# name_to_id["heart_rate"] → 1
# name_to_id["blood_pressure"] → 4
```

---

### Observation grid files (`stay_XXXXXXXX_grid.parquet`)

Each stay also has a companion grid file that records **every timestamp that was attempted** by each signal, including timesteps where the chart event was present but the value was out of physio range (and therefore dropped). The observation Parquet only contains valid readings; the grid is the missingness oracle.

| Column | Type | Description |
|--------|------|-------------|
| `s` | int32 | Signal ID |
| `t` | float32 | Attempted timestamp (seconds since T0) |

For multichannel signals (blood pressure, GCS), the grid stores one row per `(signal, timestep)`, not per `(signal, channel, timestep)`, because all channels within a signal share the same charting time.

```python
grid = pd.read_parquet("data/healthcare_data/train/stay_30000001_grid.parquet")
obs  = pd.read_parquet("data/healthcare_data/train/stay_30000001.parquet")

# Full expected schedule for SpO2 (sid=2)
expected = grid[grid["s"] == 2]["t"].values

# Observed timestamps
observed = obs[obs["s"] == 2]["t"].values

# Missing timestamps
missing = set(expected) - set(observed)
print(f"{len(missing)} / {len(expected)} SpO2 timesteps missing")
```

---

### Static context

**No static observations in v1.** Demographics (age, sex, race, admission type) are deferred; no `s=0, t=-1` rows are emitted. The `tilde_O_k = O_k` for healthcare stays in this release.

---

### Loading a stay for model training

A minimal PyTorch-style dataset that returns all observations for a single ICU stay:

```python
import json
from pathlib import Path
import pandas as pd
import numpy as np

class HealthcareStayDataset:
    def __init__(self, split_dir: Path, signals_path: Path):
        self.files = sorted(
            p for p in split_dir.glob("stay_*.parquet")
            if "_grid" not in p.name
        )
        with open(signals_path) as f:
            self.signals = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_parquet(self.files[idx])
        return {
            "t": df["t"].to_numpy(dtype=np.float32),
            "s": df["s"].to_numpy(dtype=np.int32),
            "j": df["j"].to_numpy(dtype=np.int32),
            "x": df["x"].tolist(),   # list of length-1 arrays
        }

dataset = HealthcareStayDataset(
    split_dir=Path("data/healthcare_data/train"),
    signals_path=Path("data/healthcare_data/signals.json"),
)
stay = dataset[0]
print(f"Stay has {len(stay['t']):,} observations across all signals")
```

> **Note on `x`:** all healthcare channels are scalar (`value_dim = 1`), so every element of `x` is a length-1 list. This makes healthcare the simplest domain for dense-tensor construction: stack `x` directly to get a `(N, 1)` array with no padding required.


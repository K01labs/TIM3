# TIM3

[![ci](https://github.com/K01labs/TIM3/actions/workflows/ci.yml/badge.svg)](https://github.com/K01labs/TIM3/actions/workflows/ci.yml)

TIM3 is a cross-domain benchmark for evaluating time-series foundation models on **native scientific data**: irregular timestamps, mixed sampling rates, per-signal missingness, and multimodal channels, with no forced regridding.

Most time-series benchmarks align all observations to a uniform grid before any model sees the data. In scientific domains this preprocessing step discards temporal structure and silently changes model rankings. This benchmark stores data as **observation sets** at native resolution, and treats resampling as a modelling choice rather than a hidden design constant.

Companion code for the paper [*Towards Benchmarking Time Series Foundation Models on Native Scientific Data*](https://k01.is/research/towards-benchmarking-time-series-foundation-models-on-native-scientific-data) (ICML 2026 FMSD workshop; [OpenReview](https://openreview.net/forum?id=NtU2gLl4hN)).

## The benchmark at a glance

| | Fusion | Healthcare | Climate |
|---|---|---|---|
| Source | MAST tokamak (FAIR-MAST) | MIMIC-IV ICU | TC PRIMED tropical cyclones |
| Sample | plasma shot | ICU stay | storm |
| Samples (train/val/test) | 2,022 / 433 / 434 | 52,380 / 11,224 / 11,225 | 2,485 / 532 / 534 |
| Channels | 193 | 10 | 63 |
| Sampling rates | 200 Hz – 500 kHz | ~1 h – ~4 h, irregular | 6 h regular + ~45 min irregular overpasses |
| Access | public (S3) | credentialed ([PhysioNet](https://physionet.org/content/mimiciv/)) | public (NOAA) |

**Four tasks** (forecasting, offline imputation, causal imputation, cross-modal prediction), each defined as a context/query split over the observation set, scored with a five-level hierarchical NMAE/NRMSE metric so that no signal, sample, task, or domain dominates the aggregate.

Full format, task, metric, and submission specifications: [benchmark/README.md](benchmark/README.md).

## Installation

This project is managed with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/K01labs/TIM3.git
cd TIM3
uv sync
```

`uv sync` installs the project (editable) into the local environment, including
the `tim3` command used below. The install is lightweight: the shipped
models are pure NumPy/pandas baselines, so no deep-learning stack is required.

## Getting the data

Each domain has a single dataloader script that fetches the raw data and writes benchmark-format Parquet files plus a signal registry (`signals.json`), split assignments (`splits.json`), and training statistics (`train_stats.parquet`):

```bash
# Fusion: fetches MAST shots from the public STFC S3 Zarr store
uv run python data/dataloaders/fusion_dataset.py

# Climate: fetches TC PRIMED storms via tcprimedapi
uv run python data/dataloaders/climate_dataset.py

# Healthcare: requires your own credentialed copy of MIMIC-IV (it cannot be
# redistributed). Full instructions: data/dataloaders/HEALTHCARE.md
uv run python data/dataloaders/healthcare_dataset.py --mimic_base <path-to-mimic-iv>
```

For fusion and climate, the tracked `data/<domain>_data/{signals.json, splits.json, train_stats.parquet}` pin the exact registry, split, and training statistics used in the paper. Healthcare is regenerated locally under the [PhysioNet license](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/). Its split is byte-reproducible and verifiable against a published checksum (see [data/dataloaders/HEALTHCARE.md](data/dataloaders/HEALTHCARE.md)).

## Quickstart: run a baseline and score it

New here? To integrate your own model, including a foundation model, see the
**[model integration guide](models/README.md)**, which walks through the
interface and a complete worked adapter.

One CLI generates a model's predictions and scores them. Run it on a single
domain (all four tasks):

```bash
uv run tim3 run --model locf \
    --domain fusion \
    --data_root data/fusion_data \
    --split_dir data/fusion_data/test \
    --output_root predictions/locf
```

…or run the **whole benchmark** (all three domains and four tasks) in one command
by repeating `--domain`/`--data_root`/`--split_dir` in matching order. This prints
the overall cross-domain NMAE/NRMSE plus a per-domain, per-task breakdown:

```bash
uv run tim3 run --model mean \
    --domain    fusion                tcprimed                healthcare \
    --data_root data/fusion_data      data/tcprimed_data      data/healthcare_data \
    --split_dir data/fusion_data/test data/tcprimed_data/test data/healthcare_data/test \
    --output_root predictions/mean
```

The two built-in models are `mean` and `locf`; swap either into `--model`. Both share the one-method [model interface](models/base.py), so adding your own is the same pattern (see [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-model)). The harness writes a prediction file for every task, so every model is scored on all four tasks; rows a model can't predict (e.g. cross-modal for a forecaster) fall back to the training-mean penalty.

To score **your own** model instead of a built-in one, write its predictions in the schema described in [benchmark/README.md](benchmark/README.md#submitting-predictions), then score them. Unlike `tim3 run` (which generates *and* scores), these only score predictions already on disk:

```bash
uv run tim3 evaluate --predictions_root predictions/my_model
```

or, programmatically, call `run_benchmark` directly:

```python
from pathlib import Path
from benchmark.run import run_benchmark
from benchmark.specs import load_fusion_spec, load_healthcare_spec, load_tcprimed_spec

result = run_benchmark(
    predictions_root=Path("predictions/my_model"),
    domain_specs={
        "fusion": load_fusion_spec(Path("data/fusion_data")),
        "healthcare": load_healthcare_spec(Path("data/healthcare_data")),
        "tcprimed": load_tcprimed_spec(Path("data/tcprimed_data")),
    },
    split_dirs={
        "fusion": Path("data/fusion_data/test"),
        "healthcare": Path("data/healthcare_data/test"),
        "tcprimed": Path("data/tcprimed_data/test"),
    },
)
print(result["nmae"], result["nrmse"])
```

Partial submissions (e.g. forecasting only) are supported; missing predictions receive a training-mean penalty.

## Baseline scores (native resolution, NMAE)

| Model | Fusion | Climate | Healthcare | Overall |
|---|---|---|---|---|
| Mean | 0.493 | 0.667 | 0.783 | 0.647 |
| LOCF | 0.481 | 0.485 | 0.637 | 0.534 |

Domain scores are the mean over the four tasks; overall is the mean over domains. See the paper appendix for foundation-model forecasting results.

The full ranked table is in [LEADERBOARD.md](LEADERBOARD.md), regenerated from the entries under `leaderboard/`. To submit your own model, see [CONTRIBUTING.md](CONTRIBUTING.md#reporting-results-leaderboard).

## Repository structure

```
├── benchmark/          # Core: data loading, task partitions, metrics, evaluation
├── data/dataloaders/   # Per-domain dataset builders (fusion, healthcare, climate)
├── data/<domain>_data/ # Tracked signal registries and split assignments
├── models/             # Model interface + integration guide (README.md), harness, registry, CLI
│   └── baselines/       # Mean and LOCF reference implementations
├── scripts/            # make_submission / validate_submission / build_leaderboard
├── leaderboard/        # One folder per submission; LEADERBOARD.md is generated from it
└── tests/              # pytest suite
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the quality gate, how to add a
model adapter, and how to report results.

## Citation and license

Code is released under the [Apache-2.0 license](LICENSE). Datasets carry the
terms of their sources (MAST, TC PRIMED: open; MIMIC-IV: PhysioNet
credentialed access, never redistributed by this project).

If you use the benchmark, please cite the paper (see also [CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{odonnell2026towards,
  title={Towards Benchmarking Time Series Foundation Models on Native Scientific Data},
  author={Lewis O'Donnell and Arinbj{\"o}rn Kolbeinsson and Benedikt Kolbeinsson and Marc Peter Deisenroth},
  booktitle={2nd ICML Workshop on Foundation Models for Structured Data},
  year={2026},
  url={https://openreview.net/forum?id=NtU2gLl4hN}
}
```

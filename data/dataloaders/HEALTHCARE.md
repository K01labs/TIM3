# Healthcare domain (MIMIC-IV): regenerating the data

Unlike the fusion and climate domains, the healthcare data is **not shipped with
this repository** and cannot be. MIMIC-IV is restricted under the
[PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/),
whose Clause 3 states: *"The LICENSEE will not share access to PhysioNet
restricted data with anyone else."* Derived artifacts (the processed Parquet,
the `splits.json` of stay IDs, and `train_stats.parquet`) must be treated as
restricted too.

So this repository ships the **loader** ([`healthcare_dataset.py`](healthcare_dataset.py))
and this guide; you bring your own credentialed copy of MIMIC-IV and regenerate
the benchmark data locally. The split is deterministic, so you can verify you
reproduced the exact split used in the paper against the published checksum.

## 1. Get credentialed access to MIMIC-IV

1. Create a [PhysioNet](https://physionet.org/) account and complete the
   credentialing process (CITI "Data or Specimens Only Research" training +
   sign the Credentialed Health Data Use Agreement).
2. Accept the data use agreement on the
   [MIMIC-IV v3.1 project page](https://physionet.org/content/mimiciv/3.1/).
3. Download the **`icu` module** (the loader needs only `icu/icustays.csv.gz`
   and `icu/chartevents.csv.gz`):

   ```bash
   wget -r -N -c -np --user <your_physionet_user> --ask-password \
     https://physionet.org/files/mimiciv/3.1/icu/
   ```

   This produces `physionet.org/files/mimiciv/3.1/icu/...`.

## 2. Regenerate the benchmark data

```bash
uv run python data/dataloaders/healthcare_dataset.py \
    --mimic_base /path/to/physionet.org/files/mimiciv/3.1 \
    --output_dir data/healthcare_data
```

The loader keeps ICU stays ≥ 24 h, extracts the 10 channels (heart rate,
SpO₂, respiratory rate, blood pressure [sbp/dbp/map], temperature, and GCS
[eye/verbal/motor]), and writes `data/healthcare_data/` with `train/ val/ test/`
observation Parquet, `signals.json`, `splits.json`, and `train_stats.parquet`.

Expected output (MIMIC-IV v3.1): **52,380 / 11,224 / 11,225** stays
(74,829 total).

> **Runtime & memory.** `chartevents` is the large MIMIC table (tens of GB
> uncompressed, ~300M+ rows). The loader streams it, but a full run still takes
> on the order of an hour and needs several GB of free RAM and disk. Use
> `--limit N` for a quick trial on the first N stays before committing to a full
> run.

> Smoke test without credentials, using the synthetic fixture:
> ```bash
> uv run python data/dataloaders/healthcare_dataset.py \
>     --mimic_base tests/fixtures/mimic_synthetic \
>     --output_dir /tmp/hc_smoke --min_stay_hours 0
> ```

## 3. Verify you reproduced the paper's split

```bash
shasum -a 256 -c data/dataloaders/healthcare_splits.sha256
```

`splits.json` is byte-reproducible: stay IDs are sorted and a fixed seed
(`RANDOM_SEED = 42`) drives the 70/15/15 permutation, so the same loader on the
same MIMIC-IV version yields an identical file. A mismatch means a different
MIMIC-IV version, a modified loader, or non-default parameters.

> **The split is by ICU stay, not by subject.** A patient with multiple ICU
> stays may appear in more than one split. This is a deliberate design choice
> that matches the paper and the published checksum; if you need
> subject-disjoint splits for your own work, group by `subject_id` before
> splitting (this will not match the checksum).

`train_stats.parquet` is regenerated alongside it; its values will match but the
exact bytes can vary with the `pyarrow` version, so it is intentionally **not**
checksummed.

## 4. Run the benchmark

Once regenerated, the healthcare domain works exactly like the others:

```python
from pathlib import Path
from benchmark.specs import load_healthcare_spec
spec = load_healthcare_spec(Path("data/healthcare_data"))
```

See the [root README](../../README.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md)
for running baselines and producing a leaderboard submission.

## What you must not share

Per the license, do **not** commit or publish the regenerated
`data/healthcare_data/` (Parquet, `splits.json`, or `train_stats.parquet`),
it is gitignored for this reason. Only the loader code, this guide, and the
split checksum are public. If you build a derived dataset or model and wish to
share it, share it on PhysioNet under the same credentialed agreement, not on a
public repository.

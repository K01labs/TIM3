# Contributing

## What CI checks (you don't run anything by hand)

Every pull request is checked automatically by GitHub Actions
([`.github/workflows/`](.github/workflows)):

- **lint**: `ruff format --check` and `ruff check` (hard gates); `ty check`
  (advisory for now).
- **test**: `uv run pytest` on Python 3.12 and 3.14. This includes
  `tests/test_golden.py`, a **regression test that pins the exact benchmark
  scores**, so any change to the scoring code is caught.
- **guard**: outside contributors **cannot modify the benchmark/scoring/data
  core**. The core is owned via [`.github/CODEOWNERS`](.github/CODEOWNERS)
  (maintainer review required) and a CI job rejects a PR that touches protected
  paths. A model submission may only **add `models/<name>/`**, **append one line
  to `models/registry.py`**, and **add a `leaderboard/<name>/` entry**, nothing
  in `benchmark/`, `data/dataloaders/`, or the shared harness/metric code.
- **leaderboard**: submission entries are validated and `LEADERBOARD.md` is
  checked up to date.

## Development setup

The project is [`uv`](https://docs.astral.sh/uv/)-managed, Python >= 3.12.

```bash
uv sync
```

You don't have to run the gate locally (CI does), but to check before pushing:

```bash
uv run ruff format . && uv run ruff check . && uv run pytest
```

`uv run pytest` is offline by default; the two network tests (live S3 fetch,
model download) are opt-in with `uv run pytest -m network`.

## Running a model

One CLI runs any registered model over one or more domains and scores it. The
harness writes a prediction file for every task, so the model is scored on all
four tasks (rows it cannot predict get the training-mean penalty).

```bash
# one model, one domain, all four tasks
uv run tim3 run --model locf \
    --domain fusion --data_root data/fusion_data \
    --split_dir data/fusion_data/test --output_root predictions/locf

# a single task; cap samples for a quick check
uv run tim3 run --model locf --tasks forecasting --limit 20 \
    --domain fusion --data_root data/fusion_data \
    --split_dir data/fusion_data/test --output_root predictions/locf

# several domains at once (true cross-domain score); repeat each flag in order
uv run tim3 run --model mean \
    --domain fusion tcprimed healthcare \
    --data_root data/fusion_data data/tcprimed_data data/healthcare_data \
    --split_dir data/fusion_data/test data/tcprimed_data/test data/healthcare_data/test \
    --output_root predictions/mean
```

Registered models: `mean` and `locf` (the two reference baselines). Any flags a
model defines via `add_cli_args` appear in `--help` once `--model` is set. For
lower-level, single-(domain, task) or single-sample scoring, call
`benchmark.evaluate.evaluate_split` / `evaluate_sample` directly.

## Adding a model

A model is a **task-agnostic predictor**: implement one `predict` method and the
shared harness (`models/run.py`) handles per-task partitioning, scoring, and the
multi-domain CLI. You do **not** write a runner. For the full walkthrough,
including a complete worked **foundation-model** adapter and the design choices
an FM must make on this benchmark, see [models/README.md](models/README.md).
The short version:

1. Subclass `models.base.Model` (typically in `models/<name>/model.py`):

   ```python
   from models.base import Model

   class MyModel(Model):
       name = "mymodel"

       def predict(self, context, query, spec):
           # context, query: DataFrames with columns (t, s, j, x)
           # return (t, s, j, x_pred) for the query rows you can predict;
           # omit the rest; they get the training-mean penalty automatically.
           ...
   ```

   Optionally override `setup(self, spec, train_dir)` for per-domain
   preparation (loading weights, fitting normalisation, etc.), and
   `add_cli_args` / `from_args` for model-specific flags. `LOCFModel` and
   `MeanModel` in `models/baselines/` are minimal end-to-end examples.

2. Register it in `models/registry.py`: one line, `MyModel.name: MyModel`.

3. Add a `models/<name>/README.md` describing the model and its design
   decisions / failure modes.

Run it with the unified CLI (see "Running a model" above):

```bash
uv run tim3 run --model mymodel --domain fusion \
    --data_root data/fusion_data --split_dir data/fusion_data/test \
    --output_root predictions/mymodel
```

Because the harness writes a prediction file for every task, your model is
scored on all four tasks automatically, even tasks it cannot do (those fall
back to the training-mean penalty).

Rules:

- Predictions must copy `(t, s, j)` exactly from the query set
  (see `benchmark/README.md`, "Submitting predictions"). Use
  `models._io.write_predictions` if you write Parquet directly; the harness
  already does this for you.
- Models must not see query values, or any observation excluded from the
  context set by the task definition.
- All preprocessing (gridding, interpolation, normalisation) belongs in the
  model and must be documented; that trade-off is what the benchmark measures.

## Reporting results (leaderboard)

Results are reported by pull request. A submission is a folder
`leaderboard/<model_name>/` with two files:

1. `results.json`: the **full** benchmark scores: exactly what
   `benchmark.run.run_benchmark` returns (overall, per-domain, per-task, and
   per-sample / per-signal-channel detail and skip counts). The leaderboard
   table and the validator read only the summary fields; the rest is kept for
   diagnostics.
2. `config.json`: run metadata, including a required leakage disclosure.

Generate both from a directory of predictions with the helper script, which
runs the full evaluation and writes the entry:

```bash
uv run python scripts/make_submission.py \
    --predictions-root predictions/my_model \
    --name my_model
# writes leaderboard/my_model/results.json and a config.json stub
```

Then fill in the `config.json` stub, validate, and regenerate the table:

```bash
uv run python scripts/validate_submission.py leaderboard/my_model
uv run python scripts/build_leaderboard.py        # updates LEADERBOARD.md
```

### `results.json` schema

The file is the full `run_benchmark` output. Only the fields the leaderboard
reads are shown below; `per_domain` is an array, and each `per_task` entry also
carries `n_samples`, `n_skipped`, and a `per_sample` list (retained for
diagnostics, ignored by the table and validator):

```json
{
  "nmae": 0.534,
  "nrmse": 0.617,
  "per_domain": [
    {
      "domain": "fusion",
      "nmae": 0.48,
      "nrmse": 0.55,
      "per_task": [
        {"task": "forecasting", "nmae": 0.26, "nrmse": 0.31}
      ]
    }
  ]
}
```

> For very large test splits the `per_sample` detail can make `results.json`
> sizeable; that is expected.

### `config.json` fields

| Field | Required | Notes |
|---|---|---|
| `model_name` | yes | Display name |
| `model_type` | yes | `statistical`, `pretrained`, `fine-tuned`, or `from-scratch` |
| `evaluation_mode` | yes | `zero-shot`, `fine-tuned`, or `trained-from-scratch` |
| `organization` | yes | Submitter |
| `code_url` | yes | Link to the adapter / repo / PR used |
| `test_data_leakage` | yes | **Disclosure is mandatory.** State any overlap between the model's training corpus and the benchmark test data (MAST, MIMIC-IV, TC PRIMED), or `none known` |
| `benchmark_commit` | yes | Git SHA of the benchmark code used |
| `checkpoint`, `preprocessing`, `hardware`, `runtime_total_seconds` | no | Recommended for reproducibility |

`scripts/validate_submission.py` enforces these on every PR (also run in CI).

### Comparability and current status

Leaderboard numbers are only comparable across models evaluated on the **same
test split**. The repo tracks the exact registries, splits, and training
statistics used in the paper, but the bulk observation data is not yet hosted
publicly and the scoring hardening is still in progress, so the table is
**seeded with the paper's baselines and is not yet open to external
submissions**. Watch the README / release notes for when external submissions
open.

## Reporting bugs

Open an issue with the failing command, the full traceback, and (for data
issues) the domain, sample ID, and dataloader version/commit.

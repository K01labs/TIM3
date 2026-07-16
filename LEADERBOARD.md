# Leaderboard

Each score cell is **NMAE / NRMSE** (lower is better). Overall NMAE is the mean over the three domains. Regenerate with `uv run python scripts/build_leaderboard.py`.

> Numbers are comparable only across submissions evaluated on the same hosted test split. Until the data is hosted and the open scoring issues are resolved, the table is seeded with the paper's baselines and is not yet open to external submissions (see CONTRIBUTING.md).

## Overall and per-domain

| Model | Type | Mode | Overall | Fusion | Healthcare | Climate | Leakage | Code |
|---|---|---|---|--- | --- | ---|---|---|
| LOCF | statistical | zero-shot | **0.534** / 0.617 | 0.481 / 0.553 | 0.637 / 0.720 | 0.485 / 0.579 | none; deterministic baseline, uses no training data | [link](https://openreview.net/forum?id=NtU2gLl4hN) |
| Mean | statistical | zero-shot | **0.647** / 0.713 | 0.493 / 0.556 | 0.783 / 0.857 | 0.667 / 0.726 | none; predicts the training-set mean, uses no test data | [link](https://openreview.net/forum?id=NtU2gLl4hN) |

## Per-task (NMAE / NRMSE)

### Fusion

| Model | Forecasting | Offline imp. | Causal imp. | Cross-modal |
|---|--- | --- | --- | ---|
| LOCF | 0.262 / 0.305 | 0.736 / 0.824 | 0.383 / 0.424 | 0.545 / 0.659 |
| Mean | 0.474 / 0.505 | 0.468 / 0.545 | 0.483 / 0.517 | 0.545 / 0.659 |

### Healthcare

| Model | Forecasting | Offline imp. | Causal imp. | Cross-modal |
|---|--- | --- | --- | ---|
| LOCF | 0.448 / 0.516 | 0.646 / 0.749 | 0.646 / 0.749 | 0.807 / 0.867 |
| Mean | 0.753 / 0.807 | 0.786 / 0.877 | 0.786 / 0.877 | 0.807 / 0.867 |

### Climate

| Model | Forecasting | Offline imp. | Causal imp. | Cross-modal |
|---|--- | --- | --- | ---|
| LOCF | 0.430 / 0.500 | 0.396 / 0.496 | 0.396 / 0.496 | 0.719 / 0.823 |
| Mean | 0.688 / 0.732 | 0.630 / 0.675 | 0.630 / 0.675 | 0.719 / 0.823 |

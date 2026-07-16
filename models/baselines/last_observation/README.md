# Last-observation carry-forward (LOCF) baseline

Predicts each query observation by carrying forward the most recent context
observation that shares the same `(s, j)` pair and has a strictly earlier
timestamp. It works directly on the native irregular timestamps, with no gridding,
interpolation, or alignment, so it is equally valid for a 1 Hz vital sign and a
500 kHz plasma probe in the same sample.

## How it works

For each unique `(s, j)` in the query set:

1. Sort that pair's context observations by `t` once.
2. Use `np.searchsorted` to find the strict predecessor `t_c < t_q` in
   O(log N_C).
3. Carry the associated `x` vector forward as the prediction.

If no strict predecessor exists (e.g. the masked window starts at `t = 0` in
offline imputation), it falls back to the nearest **forward** context
observation. For forecasting and causal imputation the partition already
guarantees a backward predecessor, so the fallback never fires there.

## Expected behaviour per task

LOCF carries the last *earlier* observation forward, so it does best when the
gap to the query is short and the signal is slow-moving, and worst when it has
to hold a stale value across a long masked interval.

| Task | Behaviour |
|------|-----------|
| `forecasting` | Strongest task: the query directly follows the context, so the carried value is recent. |
| `offline_imputation` | Uses only the left edge of the masked window, so long interior gaps carry a stale value; often *worse* than forecasting. |
| `causal_imputation` | Between the two: a short causal gap behaves like forecasting. |
| `cross_modal` | Equal to the Mean baseline (see below). |

Actual per-task scores are on the [leaderboard](../../../LEADERBOARD.md); e.g.
on fusion, offline imputation (0.736 NMAE) is much harder for LOCF than
forecasting (0.262).

### Cross-modal falls back to the training mean

`CrossModalSpec` enforces `source_signals ∩ target_signals = ∅`, so no `(s, j)`
in the query ever appears in the context. `predict_last_observation` therefore
returns an empty DataFrame and the evaluator applies the training-mean penalty
to every query row, so LOCF's cross-modal score equals the Mean baseline's
(NMAE ≈ 0.55–0.81 across domains, not 1.0). A model that captures cross-modal
structure should beat this bound.

## Layout

- [`model.py`](model.py): `predict_last_observation(context_df, query_df)`.
- `LOCFModel` (in `model.py`): the `Model` adapter the shared runner uses.

## Run

```bash
uv run tim3 run --model locf \
  --domain fusion \
  --data_root data/fusion_data \
  --split_dir data/fusion_data/test \
  --output_root predictions/last_observation
```

Pass each flag once per domain (in matching order) to score several domains
together. Predictions land in
`predictions/last_observation/<domain>/<task>/<sample>.parquet` and a score
summary is printed; the `cross_modal/` files have the correct schema but zero
rows. To produce a leaderboard entry from the predictions, use
`scripts/make_submission.py` (see [CONTRIBUTING.md](../../../CONTRIBUTING.md)).

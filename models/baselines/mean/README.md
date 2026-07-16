# Mean baseline

Predicts the per-`(s, j)` training mean vector for every query observation,
ignoring the context set and the task entirely.

## Why the mean is the reference point

Errors are normalised by the per-dimension training standard deviation σ. With
`x_pred = μ` (the training mean), the squared error averages to the variance:

```
NRMSE = sqrt( mean_m (x_true_m - μ_m)^2 / σ_m^2 ) ≈ 1.0   (by construction)
```

so **NRMSE ≈ 1.0** for a signal drawn from the training distribution. The
absolute error is smaller: `E|x_true - μ| = sqrt(2/π)·σ ≈ 0.8·σ` for Gaussian
data (and at most σ in general), so **NMAE < 1.0**, around 0.8 for well-behaved
signals, and lower where the distribution is heavy-tailed or the training mean
is a good predictor. On the seeded [leaderboard](../../../LEADERBOARD.md) the
Mean baseline scores **0.647 / 0.713** (NMAE / NRMSE) overall.

The mean is the reference any context-using model should beat: it uses no
context and no test data, so anything that does should score lower.

## Layout

- [`model.py`](model.py): `predict_mean(query_df, train_stats)`, the pure
  predictor.
- `MeanModel` (in `model.py`): the `Model` adapter the shared runner uses.

## Run

```bash
# One domain
uv run tim3 run --model mean \
  --domain fusion \
  --data_root data/fusion_data \
  --split_dir data/fusion_data/test \
  --output_root predictions/mean

# Several domains at once (true cross-domain score); repeat each flag in order
uv run tim3 run --model mean \
  --domain fusion tcprimed healthcare \
  --data_root data/fusion_data data/tcprimed_data data/healthcare_data \
  --split_dir data/fusion_data/test data/tcprimed_data/test data/healthcare_data/test \
  --output_root predictions/mean
```

Predictions are written to `predictions/mean/<domain>/<task>/<sample>.parquet`
and a score summary is printed. To produce a leaderboard entry from the
predictions, use `scripts/make_submission.py` (see
[CONTRIBUTING.md](../../../CONTRIBUTING.md)).

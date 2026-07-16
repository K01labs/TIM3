## Summary

<!-- What does this PR do? -->

## Leaderboard submission?

If you are adding a model to the leaderboard, confirm:

- [ ] Added `leaderboard/<model_name>/results.json` and `config.json`
- [ ] Generated `results.json` with `scripts/make_submission.py` on the official hosted test split (see [CONTRIBUTING.md](../CONTRIBUTING.md))
- [ ] Filled in every `config.json` field, especially `test_data_leakage`
- [ ] Regenerated the table: `uv run python scripts/build_leaderboard.py`

<!--
CI runs lint + tests automatically (you don't need to run them here).
Contributions may add models/<name>/, append one line to models/registry.py, and
add leaderboard/<name>/. Changes to benchmark/, data/dataloaders/, or the shared
harness/metric code are reserved for maintainers and will be rejected by the
core-protection check.
-->


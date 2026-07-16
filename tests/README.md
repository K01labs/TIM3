# Tests

Pytest suite for the benchmark evaluator and the dataset builders. The whole
suite is **offline**: no network access or data download is required.

```bash
uv run pytest
```

This is exactly what CI runs.

| Module | Covers | Needs |
|---|---|---|
| `test_benchmark.py` | metrics, alignment, task partitions | nothing (synthetic) |
| `test_run.py` | `run_benchmark` orchestration | nothing (synthetic) |
| `test_benchmark_integration.py` | end-to-end evaluation | tracked `data/fusion_data/` + `data/tcprimed_data/` samples (skips if absent); run from the repo root |
| `test_harness.py` | unified model harness, per-task partitions | nothing (synthetic) |
| `test_golden.py` | **pins exact benchmark scores** (regression guard) | nothing (synthetic) |
| `test_healthcare_dataset.py` | MIMIC-IV builder | `tests/fixtures/mimic_synthetic/` |

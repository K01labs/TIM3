# Models: integrating your own model

Every model implements **one method**, `predict(context, query, spec)`, by
subclassing [`models.base.Model`](base.py). The shared harness
([`harness.py`](harness.py)) partitions each sample per task and scores the
result, so a single `predict` automatically covers all four benchmark tasks. You never write a training loop or a task switch.

This guide covers the interface, a minimal example, and a **complete worked
foundation-model adapter** so you can wrap a pretrained time-series model.

The benchmark core is deliberately dependency-light (NumPy/pandas only): it
ships no deep-learning stack. A model that needs `torch` (or any framework)
brings its own dependencies; see [Dependencies](#dependencies) below.

## The interface

```python
class Model(ABC):
    name: str = "model"                       # registry key + default output folder

    def predict(self, context, query, spec) -> pd.DataFrame: ...   # required
    def setup(self, spec, train_dir) -> None: ...                  # optional
    @classmethod
    def add_cli_args(cls, parser) -> None: ...                     # optional
    @classmethod
    def from_args(cls, args) -> "Model": ...                       # optional
```

- **`predict(context, query, spec)`**: the only required method.
  - `context`, `query`: DataFrames with columns `(t, s, j, x)`. `t` is the
    float32 timestamp (seconds), `s` the signal id, `j` the channel, and `x` a
    NumPy float32 array (length 1 for scalars, longer for profiles/fields).
    Static rows have `s = 0`. **Never** read `query["x"]`; it is the ground
    truth.
  - `spec`: the domain's [`BenchmarkSpec`](../benchmark/specs.py). Useful
    fields: `spec.domain.registry` (signal metadata) and
    `spec.domain.train_stats` (per-`(s, j, m)` mean/std).
  - **Returns** a DataFrame with columns `(t, s, j, x_pred)` for the query rows
    you can predict (`t, s, j` copied from `query`; `x_pred` a float32 array).
    Rows you omit (or an empty frame) receive the training-mean penalty, so
    **partial coverage is valid**. A pure forecaster can ignore the cross-modal
    task entirely and still get a meaningful score.
- **`setup(spec, train_dir)`**: called once per domain before any `predict`.
  Load weights, estimate per-domain statistics, or fine-tune on `train_dir` here.
- **`add_cli_args` / `from_args`**: expose model-specific `--flags` (e.g. a
  checkpoint id) so `tim3 run --model <name> --your-flag ...` works.

## Minimal example

A model that predicts each query point from the mean of that signal's context
values, the whole thing:

```python
import numpy as np
from models.base import Model


class ContextMeanModel(Model):
    name = "context_mean"

    def predict(self, context, query, spec):
        temporal = context[context.s != 0]  # drop static rows
        means = {
            (int(s), int(j)): np.mean([np.asarray(v)[0] for v in g.x])
            for (s, j), g in temporal.groupby(["s", "j"])
        }
        out = query[["t", "s", "j"]].copy()
        out["x_pred"] = [
            np.array([means.get((int(s), int(j)), 0.0)], dtype=np.float32)
            for s, j in zip(query.s, query.j)
        ]
        return out
```

The two shipped baselines, [`mean`](baselines/mean/model.py) and
[`locf`](baselines/last_observation/model.py), are complete, tested
implementations in the same spirit; read them first.

## Integrating a foundation model

Pretrained time-series models (Chronos, MOIRAI, MOMENT, TimesFM, …) need a bit
more glue: load the checkpoint once, run it **per scalar `(s, j)` pair**, and map
its fixed-length forecast back onto the query timestamps. The adapter below wraps
[Chronos](https://github.com/amazon-science/chronos-forecasting) as a zero-shot
forecaster. It is a faithful template; swap the three model-specific lines for
your own framework.

```python
# models/mychronos/model.py
import argparse

import numpy as np
import pandas as pd

from benchmark.specs import BenchmarkSpec
from models.base import Model

# Categories the benchmark treats as scalar time series (what an FM forecasts).
_SCALAR = {"scalar_single", "scalar_multichannel"}


class ChronosModel(Model):
    """Zero-shot Chronos forecaster: predicts each scalar (s, j) query causally."""

    name = "chronos"

    def __init__(self, model_id: str = "amazon/chronos-bolt-small"):
        self.model_id = model_id
        self.pipeline = None  # loaded lazily in setup()

    # --- optional CLI wiring: `--chronos_model_id ...` -----------------------
    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--chronos_model_id", default="amazon/chronos-bolt-small")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ChronosModel":
        return cls(model_id=args.chronos_model_id)

    # --- load the checkpoint once, before any predict() ----------------------
    def setup(self, spec: BenchmarkSpec, train_dir) -> None:
        # Import heavy deps here, not at module top, so importing the registry
        # never requires torch for users who only run the baselines.
        import torch
        from chronos import BaseChronosPipeline

        if self.pipeline is None:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
            )

    # --- the one required method ---------------------------------------------
    def predict(
        self, context: pd.DataFrame, query: pd.DataFrame, spec: BenchmarkSpec
    ) -> pd.DataFrame:
        import torch

        registry = spec.domain.registry
        rows = []
        # Only scalar signals; a pretrained forecaster has no notion of a
        # profile/field, and cross-modal query signals have no context here,
        # both are left out and fall back to the training-mean penalty.
        for (s, j), q in query.groupby(["s", "j"]):
            if registry.get(str(int(s)), {}).get("category") not in _SCALAR:
                continue
            ctx = context[(context.s == s) & (context.j == j)].sort_values("t")
            if ctx.empty:
                continue

            history = np.array([np.asarray(v)[0] for v in ctx.x], dtype=np.float32)
            q_t = q["t"].to_numpy(dtype=np.float64)

            # Forecast a fixed number of steps, then map onto the query times.
            # Chronos sees only values (no timestamps): step k is one context
            # sample-interval ahead. Convert query times to step indices.
            dt = float(np.median(np.diff(ctx["t"].to_numpy()))) or 1.0
            horizon = int(np.ceil((q_t.max() - float(ctx["t"].iloc[-1])) / dt))
            horizon = max(1, min(horizon, 64))  # Chronos native limit is 64

            _, mean = self.pipeline.predict_quantiles(
                context=torch.tensor(history),
                prediction_length=horizon,
                quantile_levels=[0.5],
            )
            forecast = mean[0].numpy()  # shape (horizon,)

            # Query timestamp -> forecast step; np.interp clamps beyond horizon.
            step = (q_t - float(ctx["t"].iloc[-1])) / dt
            x_pred = np.interp(step, np.arange(1, horizon + 1), forecast)

            for t, xp in zip(q_t, x_pred):
                rows.append((np.float32(t), int(s), int(j), np.array([xp], np.float32)))

        return pd.DataFrame(rows, columns=["t", "s", "j", "x_pred"])
```

Then register and run it:

```python
# models/registry.py: add one line
from models.mychronos.model import ChronosModel

MODEL_REGISTRY = {
    MeanModel.name: MeanModel,
    LOCFModel.name: LOCFModel,
    ChronosModel.name: ChronosModel,   # <-- your model
}
```

```bash
uv run tim3 run --model chronos \
    --domain fusion --data_root data/fusion_data \
    --split_dir data/fusion_data/test --output_root predictions/chronos
```

### Design decisions a foundation model must make

These are the choices the benchmark is built to expose; they are where models
differ, and worth documenting for your submission:

- **Irregular → uniform.** Most pretrained models assume evenly-spaced input.
  The example treats step `k` as one median sample-interval ahead; a stronger
  adapter resamples the irregular context onto a uniform grid first (and the
  grid step becomes a modelling choice that changes the score).
- **Fixed horizon.** Models emit a fixed number of steps; the benchmark's query
  window may need more or fewer. Above we clamp with `np.interp`; alternatives
  are autoregressive rollout or a longer-horizon model.
- **No timestamps.** A value-only model (like Chronos) cannot tell 2 µs from
  6.4 s between samples, and its horizon is measured in steps, not seconds. Account
  for the sampling rate when mapping steps to query times.
- **Univariate vs cross-modal.** A single-series forecaster has no context for
  the cross-modal task (disjoint source/target signals); returning nothing there
  is fine, and those rows get the mean penalty.

### Dependencies

The benchmark installs without any deep-learning framework. Install your model's
dependencies into the same environment yourself, e.g.:

```bash
uv pip install torch chronos-forecasting
```

Keep heavy imports **inside** `setup()` / `predict()` (as above) so that
importing `models.registry` never requires them for users running only the
baselines.

## Submitting results

Once your model runs, turn its predictions into a leaderboard entry and open a
PR. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the submission format, the
required leakage disclosure, and the review conventions.

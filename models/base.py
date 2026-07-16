"""Unified model interface for the benchmark.

A :class:`Model` is a **task-agnostic predictor**: given a context set and a
query set for one sample, it returns predictions for the query rows it can
handle. The harness (:mod:`models.harness`) owns per-task partitioning and
scoring, so a model never sees the task name and automatically supports every
benchmark task — returning an empty frame for rows it cannot predict (those
receive the training-mean penalty in the evaluator).

To add a new model: subclass :class:`Model`, implement :meth:`predict`, and
register it in ``models.registry.MODEL_REGISTRY``. Optionally override
:meth:`setup` (per-domain preparation, e.g. loading weights or fine-tuning) and
:meth:`add_cli_args` / :meth:`from_args` (model-specific CLI flags).
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from benchmark.specs import BenchmarkSpec


class Model(ABC):
    """Base class for benchmark models.

    Attributes:
        name: Short identifier used as the registry key and default output
            folder name.
    """

    name: str = "model"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register model-specific CLI flags on ``parser``.

        Default: no extra flags. Override to add e.g. a checkpoint id.
        """

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Model":
        """Construct a model instance from parsed CLI args.

        Default: call the no-argument constructor. Override when the model
        needs configuration from :meth:`add_cli_args`.
        """
        return cls()

    def setup(self, spec: BenchmarkSpec, train_dir: Path) -> None:
        """Prepare the model for one domain, called once before :meth:`predict`.

        Use this for work that depends on the domain: loading weights,
        estimating per-domain statistics, or fine-tuning on ``train_dir``.
        Default: no-op (suitable for stateless models like Mean and LOCF).

        Args:
            spec: The domain's benchmark specification (registry, training
                statistics, task specs).
            train_dir: Path to the domain's ``train/`` split directory.
        """

    @abstractmethod
    def predict(
        self,
        context: pd.DataFrame,
        query: pd.DataFrame,
        spec: BenchmarkSpec,
    ) -> pd.DataFrame:
        """Predict query observations from the context.

        The model is task-agnostic: the harness has already partitioned the
        sample into ``context`` and ``query`` for whichever task is running.

        Args:
            context: Context set with columns ``(t, s, j, x)``. Static rows
                (``s = 0``) are included; ``x`` is a numpy float32 array.
            query: Query set with columns ``(t, s, j, x)``. ``x`` is the ground
                truth and must not be used to form predictions.
            spec: The domain's :class:`~benchmark.specs.BenchmarkSpec`, giving
                access to ``spec.domain.train_stats`` and ``spec.domain.registry``.

        Returns:
            DataFrame with columns ``(t, s, j, x_pred)`` for the query rows the
            model can predict, where ``x_pred`` is a numpy float32 array. Rows
            that are omitted (or an empty frame) receive the training-mean
            penalty in the evaluator, so partial coverage is valid.
        """

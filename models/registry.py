"""Registry of available benchmark models, keyed by name.

To add a model: implement a :class:`models.base.Model` subclass (one
``predict`` method; optional ``setup`` / ``add_cli_args`` / ``from_args``) and
add one line here. The unified runner (``models/run.py``) looks models up by
name from this mapping.

This first release ships the two statistical baselines (Mean and LOCF). They
depend only on NumPy/pandas, so the benchmark installs and runs without any
deep-learning stack. Foundation-model adapters (Chronos, MOIRAI, MOMENT) are
maintained separately and can be dropped in as ``models/<name>/`` packages.
"""

from __future__ import annotations

from models.base import Model
from models.baselines.last_observation import LOCFModel
from models.baselines.mean import MeanModel

MODEL_REGISTRY: dict[str, type[Model]] = {
    MeanModel.name: MeanModel,
    LOCFModel.name: LOCFModel,
}

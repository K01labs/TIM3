"""Last-observation carry-forward (LOCF) baseline adapter."""

from models.baselines.last_observation.model import (
    LOCFModel,
    predict_last_observation,
)

__all__ = ["LOCFModel", "predict_last_observation"]

"""Abstract model port — all model adapters must implement this interface."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from src.domain.entities.experiment_config import ExperimentConfig


class ModelPort(ABC):
    """Port defining the contract for all ML model adapters."""

    @abstractmethod
    def build(self, hyperparameters: dict) -> None:
        """Build/initialize the model with given hyperparameters."""
        ...

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        """Train the model on the provided data."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate predictions for the input data."""
        ...

    @abstractmethod
    def get_search_space(self, trial: Any) -> dict:
        """
        Define the Optuna hyperparameter search space.

        Args:
            trial: An Optuna trial object.

        Returns:
            Dictionary of hyperparameters sampled from the search space.
        """
        ...

    @abstractmethod
    def get_model(self) -> Any:
        """Return the underlying model instance."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name identifier."""
        ...

    def get_scoring_metric(
        self,
        config: ExperimentConfig,
    ) -> str:
        """Return the scoring metric for cross-validation."""
        from src.domain.entities.experiment_config import TaskType

        scoring_map = {
            TaskType.REGRESSION: "neg_root_mean_squared_error",
            TaskType.BINARY_CLASSIFICATION: "roc_auc",
            TaskType.MULTICLASS_CLASSIFICATION: "accuracy",
        }
        return scoring_map[config.task_type]

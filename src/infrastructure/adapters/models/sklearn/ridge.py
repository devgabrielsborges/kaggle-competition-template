"""Ridge Regression adapter."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.domain.ports.model_port import ModelPort


class RidgeAdapter(ModelPort):
    """Adapter for sklearn Ridge Regression."""

    def __init__(self):
        self._model: Ridge | None = None

    @property
    def name(self) -> str:
        return "RidgeRegression"

    def build(self, hyperparameters: dict) -> None:
        self._model = Ridge(**hyperparameters)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_search_space(self, trial: Any) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["auto", "svd", "cholesky", "lsqr", "saga"]
            ),
            "max_iter": trial.suggest_int("max_iter", 100, 5000),
        }

    def get_model(self) -> Ridge:
        return self._model

    def get_default_trials(self) -> int:
        """Ridge has moderate search space: alpha (continuous) × 5 solvers × max_iter."""
        return 50

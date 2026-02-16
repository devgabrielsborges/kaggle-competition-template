"""Random Forest adapter."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.domain.ports.model_port import ModelPort


class RandomForestAdapter(ModelPort):
    """Adapter for sklearn Random Forest."""

    def __init__(self):
        self._model: RandomForestRegressor | None = None

    @property
    def name(self) -> str:
        return "RandomForest"

    def build(self, hyperparameters: dict) -> None:
        self._model = RandomForestRegressor(**hyperparameters)

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
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "random_state": 42,
        }

    def get_model(self) -> RandomForestRegressor:
        return self._model

    def get_default_trials(self) -> int:
        """Random Forest has large search space with 5 hyperparameters."""
        return 100

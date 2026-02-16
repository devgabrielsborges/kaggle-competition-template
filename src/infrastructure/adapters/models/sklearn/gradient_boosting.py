"""Gradient Boosting adapter."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.domain.ports.model_port import ModelPort


class GradientBoostingAdapter(ModelPort):
    """Adapter for sklearn Gradient Boosting."""

    def __init__(self):
        self._model: GradientBoostingRegressor | None = None

    @property
    def name(self) -> str:
        return "GradientBoosting"

    def build(self, hyperparameters: dict) -> None:
        self._model = GradientBoostingRegressor(**hyperparameters)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        self._model.fit(X, y)

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        return self._model.predict(X)

    def get_search_space(self, trial: Any) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_state": 42,
        }

    def get_model(self) -> GradientBoostingRegressor:
        return self._model

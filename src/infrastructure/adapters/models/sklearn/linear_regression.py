"""Linear Regression adapter."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.domain.ports.model_port import ModelPort


class LinearRegressionAdapter(ModelPort):
    """Adapter for sklearn LinearRegression."""

    def __init__(self):
        self._model: LinearRegression | None = None

    @property
    def name(self) -> str:
        return "LinearRegression"

    def build(self, hyperparameters: dict) -> None:
        self._model = LinearRegression(**hyperparameters)

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
            "fit_intercept": trial.suggest_categorical(
                "fit_intercept",
                [True, False],
            ),
            "positive": trial.suggest_categorical(
                "positive",
                [True, False],
            ),
        }

    def get_model(self) -> LinearRegression:
        return self._model

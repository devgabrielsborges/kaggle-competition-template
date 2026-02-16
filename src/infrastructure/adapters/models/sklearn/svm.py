"""SVM adapter."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from src.domain.ports.model_port import ModelPort


class SVMAdapter(ModelPort):
    """Adapter for sklearn Support Vector Machine."""

    def __init__(self):
        self._model: SVR | None = None

    @property
    def name(self) -> str:
        return "SVM"

    def build(self, hyperparameters: dict) -> None:
        self._model = SVR(**hyperparameters)

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
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": trial.suggest_categorical(
                "kernel",
                ["linear", "rbf", "poly"],
            ),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
        }

    def get_model(self) -> SVR:
        return self._model

    def get_default_trials(self) -> int:
        """SVM has medium complexity: C × 3 kernels × 2 gamma × epsilon."""
        return 50

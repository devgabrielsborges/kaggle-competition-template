"""XGBoost adapter."""

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from src.domain.entities.experiment_config import TaskType
from src.domain.ports.model_port import ModelPort


class XGBoostAdapter(ModelPort):
    """Adapter for XGBoost (supports both regression and classification)."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION):
        self._model = None
        self._task_type = task_type

    @property
    def name(self) -> str:
        return "XGBoost"

    def _get_model_class(self):
        if self._task_type == TaskType.REGRESSION:
            return XGBRegressor
        return XGBClassifier

    def build(self, hyperparameters: dict) -> None:
        model_cls = self._get_model_class()
        self._model = model_cls(**hyperparameters)

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
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0,
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42,
        }

    def get_model(self):
        return self._model

    def get_default_trials(self) -> int:
        """XGBoost has 9 hyperparameters with very large search space."""
        return 200

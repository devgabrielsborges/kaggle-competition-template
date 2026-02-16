"""Dataset entity representing train/test data splits."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """Core dataset entity for ML pipelines."""

    X_train: pd.DataFrame | np.ndarray
    X_test: pd.DataFrame | np.ndarray
    y_train: pd.Series | np.ndarray
    y_test: pd.Series | np.ndarray

    @property
    def train_shape(self) -> tuple:
        return self.X_train.shape

    @property
    def test_shape(self) -> tuple:
        return self.X_test.shape

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def feature_names(self) -> list[str] | None:
        if isinstance(self.X_train, pd.DataFrame):
            return list(self.X_train.columns)
        return None

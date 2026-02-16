"""Abstract preprocessor port."""

from abc import ABC, abstractmethod

import pandas as pd

from src.domain.entities.dataset import Dataset


class PreprocessorPort(ABC):
    """Port defining the contract for data preprocessing."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the preprocessor on training data."""
        ...

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted preprocessor."""
        ...

    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data in one step."""
        ...

    @abstractmethod
    def get_train_test_split(
        self,
        data: pd.DataFrame,
        target_col: str,
        test_size: float,
        random_state: int,
    ) -> Dataset:
        """Split data into train/test and return a Dataset entity."""
        ...

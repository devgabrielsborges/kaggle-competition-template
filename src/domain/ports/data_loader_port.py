"""Abstract data loader port."""

from abc import ABC, abstractmethod

import pandas as pd


class DataLoaderPort(ABC):
    """Port defining the contract for loading data."""

    @abstractmethod
    def load_train(self) -> pd.DataFrame:
        """Load the raw training data."""
        ...

    @abstractmethod
    def load_test(self) -> pd.DataFrame:
        """Load the raw test/submission data."""
        ...

    @abstractmethod
    def load_processed(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load preprocessed splits."""
        ...

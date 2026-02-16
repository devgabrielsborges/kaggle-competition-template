"""CSV data loader adapter."""

from pathlib import Path

import pandas as pd

from src.domain.ports.data_loader_port import DataLoaderPort


class CsvDataLoader(DataLoaderPort):
    """Loads data from CSV files (standard Kaggle format)."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        train_file: str = "train.csv",
        test_file: str = "test.csv",
    ):
        self._raw_dir = Path(raw_dir)
        self._processed_dir = Path(processed_dir)
        self._train_file = train_file
        self._test_file = test_file

    def load_train(self) -> pd.DataFrame:
        """Load raw training CSV."""
        path = self._raw_dir / self._train_file
        print(f"Loading training data from {path}...")
        df = pd.read_csv(path)
        print(f"✓ Training data loaded: {df.shape}")
        return df

    def load_test(self) -> pd.DataFrame:
        """Load raw test CSV (for submission)."""
        path = self._raw_dir / self._test_file
        print(f"Loading test data from {path}...")
        df = pd.read_csv(path)
        print(f"✓ Test data loaded: {df.shape}")
        return df

    def load_processed(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load preprocessed parquet splits."""
        path = self._processed_dir
        X_train = pd.read_parquet(path / "X_train.parquet")
        y_train = pd.read_parquet(path / "y_train.parquet").squeeze()
        X_test = pd.read_parquet(path / "X_test.parquet")
        y_test = pd.read_parquet(path / "y_test.parquet").squeeze()

        print("✓ Processed data loaded:")
        print(f"  - Training: {X_train.shape}")
        print(f"  - Test: {X_test.shape}")

        return X_train, X_test, y_train, y_test

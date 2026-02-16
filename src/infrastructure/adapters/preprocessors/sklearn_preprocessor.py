"""Sklearn-based preprocessor adapter.

Customize this file for each competition's specific feature engineering needs.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.domain.entities.dataset import Dataset
from src.domain.ports.preprocessor_port import PreprocessorPort


class SklearnPreprocessor(PreprocessorPort):
    """
    Adapter for sklearn-based preprocessing.

    Customize the transform logic for your specific competition data.
    """

    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the preprocessor on the data.

        Override with your competition-specific feature engineering:
        - Ordinal encoding
        - One-hot encoding
        - Feature creation
        - Scaling
        """
        numeric_cols = data.select_dtypes(include=["number"]).columns
        self._scaler.fit(data[numeric_cols])
        self._fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted preprocessor.

        Override with your competition-specific transformations.
        """
        data = data.copy()

        # TODO: Add competition-specific transformations here:
        # - Encode categorical variables
        # - Handle missing values
        # - Feature engineering

        numeric_cols = data.select_dtypes(include=["number"]).columns
        data[numeric_cols] = self._scaler.transform(data[numeric_cols])
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_train_test_split(
        self,
        data: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        id_col: str | None = "id",
    ) -> Dataset:
        """
        Preprocess, split, and scale data into a Dataset entity.

        Args:
            data: Raw dataframe.
            target_col: Name of the target column.
            test_size: Fraction of data for testing.
            random_state: Random seed.
            id_col: ID column to exclude from features (None to skip).

        Returns:
            Dataset entity with train/test splits.
        """
        data = data.copy()

        # TODO: Add encoding / feature engineering before splitting

        drop_cols = [target_col]
        if id_col and id_col in data.columns:
            drop_cols.append(id_col)

        X = data.drop(columns=drop_cols)
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale numeric features
        numeric_cols = X_train.select_dtypes(include=["number"]).columns
        self._scaler.fit(X_train[numeric_cols])

        X_train[numeric_cols] = self._scaler.transform(X_train[numeric_cols])
        X_test[numeric_cols] = self._scaler.transform(X_test[numeric_cols])

        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

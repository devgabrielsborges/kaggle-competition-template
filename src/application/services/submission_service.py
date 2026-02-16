"""Submission service — generates Kaggle submission files."""

import pandas as pd

from src.domain.ports.model_port import ModelPort
from src.domain.ports.preprocessor_port import PreprocessorPort


class SubmissionService:
    """Generates Kaggle competition submission files."""

    def __init__(
        self,
        model_adapter: ModelPort,
        preprocessor: PreprocessorPort,
        id_col: str = "id",
        target_col: str = "target",
    ):
        self.model_adapter = model_adapter
        self.preprocessor = preprocessor
        self.id_col = id_col
        self.target_col = target_col

    def generate(
        self,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        output_path: str = "submission.csv",
    ) -> pd.DataFrame:
        """
        Generate a submission CSV file.

        Args:
            test_df: Raw test dataframe (from competition).
            train_df: Raw train dataframe (to fit preprocessor).
            output_path: Path to save the submission file.

        Returns:
            Submission dataframe.
        """
        print("\n📝 Generating submission file...")

        test_ids = test_df[self.id_col].copy()

        # Fit preprocessor on training data, transform test data
        self.preprocessor.fit(train_df)
        X_test = self.preprocessor.transform(test_df)

        # Drop id column if present
        if self.id_col in X_test.columns:
            X_test = X_test.drop(columns=[self.id_col])

        # Generate predictions
        predictions = self.model_adapter.predict(X_test)

        # Build submission
        submission = pd.DataFrame({self.id_col: test_ids, self.target_col: predictions})
        submission.to_csv(output_path, index=False)

        print(f"✓ Submission saved to {output_path}")
        print(f"  - Shape: {submission.shape}")
        print(f"  - Sample predictions: {predictions[:5]}")

        return submission

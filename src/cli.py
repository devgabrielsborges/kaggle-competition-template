"""
CLI entry point — trains a model with Optuna + MLflow.

Usage:
    uv run python -m src.cli --model xgboost --trials 100
    uv run python -m src.cli --model ridge --trials 200
    uv run python -m src.cli --model pytorch --trials 50
"""

import argparse

from src.application.services.training_service import TrainingService
from src.domain.entities.dataset import Dataset
from src.domain.entities.experiment_config import ExperimentConfig, TaskType
from src.infrastructure.adapters.data_loaders.csv_loader import CsvDataLoader
from src.infrastructure.adapters.trackers.mlflow_tracker import MLflowTracker
from src.infrastructure.config.settings import Settings

# Registry of available model adapters
MODEL_REGISTRY: dict[str, callable] = {}


def _register_models():
    """Lazily register available models."""

    # sklearn models (always available)
    from src.infrastructure.adapters.models.sklearn import (
        GradientBoostingAdapter, LinearRegressionAdapter, RandomForestAdapter,
        RidgeAdapter, SVMAdapter)

    MODEL_REGISTRY["linear"] = lambda _task: LinearRegressionAdapter()
    MODEL_REGISTRY["ridge"] = lambda _task: RidgeAdapter()
    MODEL_REGISTRY["random_forest"] = lambda _task: RandomForestAdapter()
    MODEL_REGISTRY["gradient_boosting"] = lambda _task: GradientBoostingAdapter()
    MODEL_REGISTRY["svm"] = lambda _task: SVMAdapter()

    # XGBoost
    try:
        from src.infrastructure.adapters.models.xgboost import XGBoostAdapter

        MODEL_REGISTRY["xgboost"] = lambda task: XGBoostAdapter(task_type=task)
    except ImportError:
        pass

    # PyTorch
    try:
        from src.infrastructure.adapters.models.pytorch import PyTorchAdapter

        MODEL_REGISTRY["pytorch"] = lambda task: PyTorchAdapter(task_type=task)
    except ImportError:
        pass

    # TensorFlow
    try:
        from src.infrastructure.adapters.models.tensorflow import \
            TensorFlowAdapter

        MODEL_REGISTRY["tensorflow"] = lambda task: TensorFlowAdapter(
            task_type=task,
        )
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaggle competition model trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=("Model to train (e.g., linear, ridge, xgboost, pytorch)"),
    )
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=[
            "regression",
            "binary_classification",
            "multiclass_classification",
        ],
        help="Task type (default: regression)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna trials (default: auto-set based on model complexity)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--generate-submission",
        action="store_true",
        help="Generate submission file after training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    _register_models()

    if args.model not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        print(f"✗ Unknown model: '{args.model}'")
        print(f"  Available models: {available}")
        return

    settings = Settings()
    task_type = TaskType(args.task)

    # Create model adapter
    model_adapter = MODEL_REGISTRY[args.model](task_type)

    # Determine number of trials - use model-specific default if not specified
    if args.trials is not None:
        n_trials = args.trials
    else:
        n_trials = model_adapter.get_default_trials()

    # Load data
    loader = CsvDataLoader(
        raw_dir=settings.raw_data_dir,
        processed_dir=settings.processed_data_dir,
    )
    X_train, X_test, y_train, y_test = loader.load_processed()
    dataset = Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # Configure experiment
    config = ExperimentConfig(
        experiment_name=(f"{settings.competition_name} - {model_adapter.name}"),
        model_name=model_adapter.name,
        task_type=task_type,
        n_trials=n_trials,
        cv_folds=args.cv_folds,
        random_state=settings.random_state,
    )

    # Train
    tracker = MLflowTracker(settings)
    service = TrainingService(model_adapter, tracker, config)

    print("=" * 60)
    print(f"  {settings.competition_name.upper()} - {model_adapter.name}")
    print(f"  Trials: {n_trials} (model-optimized)")
    print("=" * 60)

    # Load raw data if submission is requested
    test_df = None
    train_df = None
    if args.generate_submission:
        import os

        import pandas as pd

        train_path = os.path.join(settings.raw_data_dir, "train.csv")
        test_path = os.path.join(settings.raw_data_dir, "test.csv")

        if os.path.exists(test_path) and os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print("✓ Loaded raw data for submission generation")
        else:
            print(
                f"⚠️  Test data not found at {test_path}, "
                "submission will not be generated"
            )
            args.generate_submission = False

    # Run training (with optional submission generation)
    model, study, metrics = service.run(
        dataset=dataset,
        generate_submission=args.generate_submission,
        test_df=test_df,
        train_df=train_df,
        settings=settings,
    )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()

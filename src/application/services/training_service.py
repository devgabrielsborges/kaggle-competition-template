"""Training service — orchestrates the full training pipeline."""

import optuna

from src.application.services.evaluation_service import EvaluationService
from src.application.services.optimization_service import OptimizationService
from src.domain.entities.dataset import Dataset
from src.domain.entities.experiment_config import ExperimentConfig
from src.domain.entities.prediction import Prediction
from src.domain.ports.experiment_tracker_port import ExperimentTrackerPort
from src.domain.ports.model_port import ModelPort


class TrainingService:
    """Orchestrates training with optimization and tracking."""

    def __init__(
        self,
        model_adapter: ModelPort,
        tracker: ExperimentTrackerPort,
        config: ExperimentConfig,
    ):
        self.model_adapter = model_adapter
        self.tracker = tracker
        self.config = config

    def run(
        self,
        dataset: Dataset,
        generate_submission: bool = False,
        test_df=None,
        train_df=None,
        settings=None,
    ) -> tuple[ModelPort, optuna.Study, dict]:
        """
        Execute the full training pipeline:
        1. Optimize hyperparameters with Optuna
        2. Train the best model
        3. Evaluate and log results
        4. Optionally generate submission file

        Args:
            dataset: The Dataset entity with train/test splits.
            generate_submission: Whether to generate a submission file.
            test_df: Raw test dataframe for submission generation.
            train_df: Raw training dataframe for submission generation.
            settings: Settings object with configuration.

        Returns:
            Tuple of (trained model adapter, study, metrics dict).
        """
        self.tracker.setup()
        self.tracker.create_experiment(self.config.experiment_name)

        with self.tracker.start_run(
            run_name=f"{self.model_adapter.name}_Optuna",
        ):
            # Optimize
            optimizer = OptimizationService(self.model_adapter, self.config)
            study = optimizer.optimize(dataset)

            # Train best model
            best_params = study.best_params
            self.model_adapter.build(best_params)
            self.model_adapter.fit(dataset.X_train, dataset.y_train)

            # Evaluate
            y_pred = self.model_adapter.predict(dataset.X_test)
            metrics = EvaluationService.evaluate(
                dataset.y_test, y_pred, self.config.task_type
            )

            prediction = Prediction(
                values=y_pred,
                model_name=self.model_adapter.name,
                metrics=metrics,
            )

            self._log_results(study, prediction, dataset)

            print("\n📊 Test Set Performance:")
            for name, value in metrics.items():
                print(f"  - {name}: {value:.4f}")

            # Generate submission if requested
            if generate_submission and test_df is not None and train_df is not None:
                self._generate_and_log_submission(test_df, train_df, settings)

            return self.model_adapter, study, metrics

    def _generate_and_log_submission(self, test_df, train_df, settings) -> None:
        """Generate submission file and log it as an artifact."""
        from src.application.services.submission_service import \
            SubmissionService
        from src.infrastructure.adapters.preprocessors.sklearn_preprocessor import \
            SklearnPreprocessor

        print("\n" + "=" * 60)
        print("  GENERATING SUBMISSION FILE")
        print("=" * 60)

        preprocessor = SklearnPreprocessor(
            target_col=settings.target_col,
            numeric_strategy="mean",
            categorical_strategy="most_frequent",
        )

        submission_service = SubmissionService(
            model_adapter=self.model_adapter,
            preprocessor=preprocessor,
            id_col=settings.id_col,
            target_col=settings.target_col,
        )

        submission_path = "submission.csv"
        submission_service.generate(
            test_df=test_df,
            train_df=train_df,
            output_path=submission_path,
        )

        # Log to MLflow (will be stored in MinIO)
        print("📦 Uploading submission to MinIO via MLflow...")
        self.tracker.log_artifact(submission_path)
        print("✓ Submission file logged to MLflow and stored in MinIO")

    def _log_results(
        self,
        study: optuna.Study,
        prediction: Prediction,
        dataset: Dataset,
    ) -> None:
        """Log all training results to the experiment tracker."""
        self.tracker.log_params(study.best_params)
        self.tracker.log_params(
            {
                "n_trials": self.config.n_trials,
                "cv_folds": self.config.cv_folds,
                "task_type": self.config.task_type.value,
                "best_trial_number": study.best_trial.number,
            }
        )
        self.tracker.log_metrics(prediction.metrics)
        self.tracker.log_metrics({"best_cv_score": study.best_value})

        # Save trials dataframe
        trials_df = study.trials_dataframe()
        trials_df.to_csv("optuna_trials.csv", index=False)
        self.tracker.log_artifact("optuna_trials.csv")

        # Log the model
        self.tracker.log_model(
            model=self.model_adapter.get_model(),
            artifact_path="model",
            input_example=(
                dataset.X_train.iloc[:5]
                if hasattr(dataset.X_train, "iloc")
                else dataset.X_train[:5]
            ),
        )

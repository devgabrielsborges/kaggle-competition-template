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
    ) -> tuple[ModelPort, optuna.Study, dict]:
        """
        Execute the full training pipeline:
        1. Optimize hyperparameters with Optuna
        2. Train the best model
        3. Evaluate and log results

        Args:
            dataset: The Dataset entity with train/test splits.

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

            return self.model_adapter, study, metrics

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

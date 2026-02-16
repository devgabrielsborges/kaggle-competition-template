"""Optimization service — runs Optuna hyperparameter search."""

import optuna
from sklearn.model_selection import cross_val_score

from src.domain.entities.dataset import Dataset
from src.domain.entities.experiment_config import ExperimentConfig
from src.domain.ports.model_port import ModelPort


class OptimizationService:
    """Orchestrates Optuna hyperparameter optimization."""

    def __init__(self, model_adapter: ModelPort, config: ExperimentConfig):
        self.model_adapter = model_adapter
        self.config = config

    def optimize(self, dataset: Dataset) -> optuna.Study:
        """
        Run Optuna optimization and return the study.

        Args:
            dataset: The Dataset entity with train/test splits.

        Returns:
            Completed Optuna study.
        """
        print(
            f"\n🔍 Starting Optuna optimization with {self.config.n_trials} trials..."
        )

        study = optuna.create_study(
            direction="maximize",
            study_name=(f"{self.model_adapter.name}_{self.config.experiment_name}"),
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )

        study.optimize(
            lambda trial: self._objective(trial, dataset),
            n_trials=self.config.n_trials,
            show_progress_bar=True,
        )

        print("\n✓ Optimization completed!")
        print(f"✓ Best trial: {study.best_trial.number}")
        print(f"✓ Best CV score: {study.best_value:.4f}")
        print("✓ Best parameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        return study

    def _objective(self, trial: optuna.Trial, dataset: Dataset) -> float:
        """Optuna objective function."""
        params = self.model_adapter.get_search_space(trial)
        self.model_adapter.build(params)

        scoring = self.model_adapter.get_scoring_metric(self.config)

        cv_scores = cross_val_score(
            self.model_adapter.get_model(),
            dataset.X_train,
            dataset.y_train,
            cv=self.config.cv_folds,
            scoring=scoring,
            n_jobs=-1,
        )

        return cv_scores.mean()

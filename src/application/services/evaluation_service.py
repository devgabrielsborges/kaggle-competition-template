"""Evaluation service — computes metrics based on task type."""

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, root_mean_squared_error)

from src.domain.entities.experiment_config import TaskType


class EvaluationService:
    """Evaluates model predictions and returns task-appropriate metrics."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: TaskType,
        y_prob: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Compute metrics based on the task type.

        Args:
            y_true: Ground truth values.
            y_pred: Model predictions.
            task_type: The type of ML task.
            y_prob: Predicted probabilities (for classification).

        Returns:
            Dictionary of metric name -> value.
        """
        if task_type == TaskType.REGRESSION:
            return EvaluationService._regression_metrics(y_true, y_pred)
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            return EvaluationService._binary_classification_metrics(
                y_true, y_pred, y_prob
            )
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return EvaluationService._multiclass_classification_metrics(
                y_true, y_pred, y_prob
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    @staticmethod
    def _regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    @staticmethod
    def _binary_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict[str, float]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["log_loss"] = log_loss(y_true, y_prob)
        return metrics

    @staticmethod
    def _multiclass_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict[str, float]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
        }
        if y_prob is not None:
            metrics["log_loss"] = log_loss(y_true, y_prob)
        return metrics

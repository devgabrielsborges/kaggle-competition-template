"""Experiment configuration entity."""

from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""

    experiment_name: str
    model_name: str
    task_type: TaskType
    n_trials: int = 100
    cv_folds: int = 5
    random_state: int = 42
    hyperparameters: dict = field(default_factory=dict)

"""Abstract experiment tracker port."""

from abc import ABC, abstractmethod
from typing import Any


class ExperimentTrackerPort(ABC):
    """Port defining the contract for experiment tracking."""

    @abstractmethod
    def setup(self) -> None:
        """Initialize and configure the tracker."""
        ...

    @abstractmethod
    def create_experiment(self, name: str) -> str:
        """Create or get an experiment. Returns experiment ID."""
        ...

    @abstractmethod
    def start_run(self, run_name: str) -> Any:
        """Start a tracking run. Returns a context manager."""
        ...

    @abstractmethod
    def log_params(self, params: dict) -> None:
        """Log a dictionary of parameters."""
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict) -> None:
        """Log a dictionary of metrics."""
        ...

    @abstractmethod
    def log_model(self, model: Any, artifact_path: str, **kwargs) -> None:
        """Log a trained model as an artifact."""
        ...

    @abstractmethod
    def log_artifact(self, file_path: str) -> None:
        """Log a file as an artifact."""
        ...

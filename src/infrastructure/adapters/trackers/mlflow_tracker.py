"""MLflow experiment tracker adapter."""

import os
from typing import Any

import mlflow
import mlflow.sklearn

from src.domain.ports.experiment_tracker_port import ExperimentTrackerPort
from src.infrastructure.config.settings import Settings


class MLflowTracker(ExperimentTrackerPort):
    """Adapter for MLflow experiment tracking with PostgreSQL + MinIO."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()

    def setup(self) -> None:
        """Configure MLflow tracking URI and S3/MinIO credentials."""
        tracking_uri = (
            f"postgresql://{self._settings.postgres_user}"
            f":{self._settings.postgres_password}"
            f"@{self._settings.postgres_host}"
            f":{self._settings.postgres_port}"
            f"/{self._settings.postgres_db}"
        )
        mlflow.set_tracking_uri(tracking_uri)

        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self._settings.s3_endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = self._settings.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = self._settings.aws_secret_access_key

        print("✓ MLflow configured:")
        print(f"  - Tracking URI: {tracking_uri}")
        print(f"  - S3 Endpoint: {self._settings.s3_endpoint_url}")
        print(f"  - Artifact Location: {self._settings.artifact_location}")

    def create_experiment(self, name: str) -> str:
        """Create experiment if it doesn't exist, then set it as active."""
        try:
            mlflow.create_experiment(
                name,
                artifact_location=self._settings.artifact_location,
            )
        except Exception:
            pass

        experiment = mlflow.set_experiment(name)
        return experiment.experiment_id

    def start_run(self, run_name: str) -> Any:
        """Start an MLflow run. Returns a context manager."""
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict) -> None:
        mlflow.log_metrics(metrics)

    def log_model(self, model: Any, artifact_path: str, **kwargs) -> None:
        mlflow.sklearn.log_model(model, artifact_path, **kwargs)

    def log_artifact(self, file_path: str) -> None:
        mlflow.log_artifact(file_path)

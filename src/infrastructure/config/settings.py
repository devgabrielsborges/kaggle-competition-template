"""Project settings — centralized configuration via environment variables."""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """
    All project settings with sensible defaults for the docker-compose setup.

    Override any setting via environment variables.
    """

    # PostgreSQL
    postgres_user: str = field(
        default_factory=lambda: os.getenv("MLFLOW_POSTGRES_USER", "mlflow")
    )
    postgres_password: str = field(
        default_factory=lambda: os.getenv("MLFLOW_POSTGRES_PASSWORD", "mlflow")
    )
    postgres_host: str = field(
        default_factory=lambda: os.getenv("MLFLOW_POSTGRES_HOST", "localhost")
    )
    postgres_port: str = field(
        default_factory=lambda: os.getenv("MLFLOW_POSTGRES_PORT", "5432")
    )
    postgres_db: str = field(
        default_factory=lambda: os.getenv("MLFLOW_POSTGRES_DB", "mlflow_db")
    )

    # MinIO / S3
    s3_endpoint_url: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
        )
    )
    aws_access_key_id: str = field(
        default_factory=lambda: os.getenv(
            "AWS_ACCESS_KEY_ID",
            "minioadmin",
        )
    )
    aws_secret_access_key: str = field(
        default_factory=lambda: os.getenv(
            "AWS_SECRET_ACCESS_KEY",
            "minioadmin",
        )
    )
    artifact_location: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_ARTIFACT_LOCATION", "s3://mlflow-artifacts/"
        )
    )

    # Data
    raw_data_dir: str = field(
        default_factory=lambda: os.getenv(
            "RAW_DATA_DIR",
            "data/raw",
        )
    )
    processed_data_dir: str = field(
        default_factory=lambda: os.getenv(
            "PROCESSED_DATA_DIR",
            "data/processed",
        )
    )

    # Competition-specific (customize per project)
    competition_name: str = field(
        default_factory=lambda: os.getenv(
            "COMPETITION_NAME",
            "kaggle-competition",
        )
    )
    target_col: str = field(
        default_factory=lambda: os.getenv(
            "TARGET_COL",
            "target",
        )
    )
    id_col: str = field(default_factory=lambda: os.getenv("ID_COL", "id"))
    test_size: float = 0.2
    random_state: int = 42

    @property
    def tracking_uri(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

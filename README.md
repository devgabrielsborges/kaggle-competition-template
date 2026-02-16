# Kaggle Competition Template

DDD-based ML project template for Kaggle competitions with experiment tracking and data versioning.

## Architecture

```
src/
├── domain/                    # Core business logic (framework-agnostic)
│   ├── entities/              # Data structures
│   │   ├── dataset.py         # Train/test data entity
│   │   ├── prediction.py      # Prediction result entity
│   │   └── experiment_config.py  # Experiment settings
│   └── ports/                 # Abstract interfaces (contracts)
│       ├── model_port.py      # Model interface
│       ├── preprocessor_port.py
│       ├── experiment_tracker_port.py
│       └── data_loader_port.py
├── application/               # Use cases / orchestration
│   └── services/
│       ├── training_service.py      # Full training pipeline
│       ├── optimization_service.py  # Optuna hyperparameter search
│       ├── evaluation_service.py    # Metrics computation
│       └── submission_service.py    # Kaggle submission generation
├── infrastructure/            # Concrete implementations
│   ├── adapters/
│   │   ├── models/
│   │   │   ├── sklearn/       # LinearRegression, Ridge, RF, GB, SVM
│   │   │   ├── xgboost/      # XGBoost
│   │   │   ├── pytorch/      # PyTorch neural networks
│   │   │   └── tensorflow/   # TensorFlow/Keras neural networks
│   │   ├── trackers/          # MLflow tracker
│   │   ├── preprocessors/     # Sklearn preprocessor
│   │   └── data_loaders/      # CSV data loader
│   └── config/
│       └── settings.py        # Centralized settings
└── cli.py                     # CLI entry point
```

## Quick Start

### 1. Setup Infrastructure

```bash
make start      # Start PostgreSQL + MinIO (Docker)
make install    # Install Python dependencies
```

### 2. Prepare Data

Place competition data in `data/raw/`, then preprocess:
```bash
# Customize src/infrastructure/adapters/preprocessors/sklearn_preprocessor.py
# for your competition's specific feature engineering
```

### 3. Train Models

```bash
# Train with auto-optimized trials based on model complexity
make train MODEL=xgboost

# Train with custom trial count
make train MODEL=ridge TRIALS=50

# Train and generate submission file
make train MODEL=xgboost SUBMIT=true

# Train with all options
make train MODEL=gradient_boosting TASK=regression TRIALS=200 CV_FOLDS=10 SUBMIT=true
```

**Model-Specific Default Trials** (automatically set when `TRIALS=auto` or omitted):
- `linear`: 10 trials (simple model, 2×2 parameter combinations)
- `ridge`: 50 trials (moderate search space)
- `svm`: 50 trials (medium complexity)
- `random_forest`: 100 trials (large search space)
- `gradient_boosting`: 150 trials (6 continuous/integer parameters)
- `xgboost`: 200 trials (9 hyperparameters, very large space)
- `pytorch`: 150 trials (dynamic architecture space)
- `tensorflow`: 150 trials (dynamic architecture space)

Available models: `linear`, `ridge`, `random_forest`, `gradient_boosting`, `svm`, `xgboost`, `pytorch`, `tensorflow`

Available tasks: `regression`, `binary_classification`, `multiclass_classification`

### 4. View Results & Artifacts

All artifacts (models, submission files, trial logs) are automatically stored in MinIO (S3-compatible storage).

```bash
make ui         # Open MLflow UI at http://localhost:5000
```

Access MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

**What's stored in MinIO:**
- Trained model artifacts (`s3://mlflow-artifacts/`)
- Submission CSV files (when `SUBMIT=true`)
- Optuna trial logs (`optuna_trials.csv`)
- Model input examples for validation

### 5. Data Versioning

```bash
make dvc-push   # Push data to MinIO
make dvc-pull   # Pull data from MinIO
make dvc-status # Check DVC status
```

## Adding a New Model

1. Create a new adapter in `src/infrastructure/adapters/models/` implementing `ModelPort`
2. Register it in `src/cli.py` → `_register_models()`
3. Train: `make train MODEL=my_new_model`

## Adding a New Framework

1. Create a new folder under `src/infrastructure/adapters/models/<framework>/`
2. Implement the `ModelPort` interface
3. Register in `src/cli.py` with a `try/except ImportError` guard
4. Add the dependency to `pyproject.toml` under `[project.optional-dependencies]`

## Stack

- **Experiment Tracking**: MLflow (PostgreSQL backend + MinIO artifacts)
- **Data Versioning**: DVC (MinIO remote storage)
- **Hyperparameter Optimization**: Optuna (model-specific trial optimization)
- **Infrastructure**: Docker Compose (PostgreSQL + MinIO)

## Environment Configuration

All configuration is centralized in `src/infrastructure/config/settings.py` with environment variable overrides.

Copy `.env.example` to `.env` and customize if needed:
```bash
cp .env.example .env
```

**Key Environment Variables:**
- `MLFLOW_S3_ENDPOINT_URL`: MinIO endpoint (default: `http://localhost:9000`)
- `AWS_ACCESS_KEY_ID`: MinIO access key (default: `minioadmin`)
- `AWS_SECRET_ACCESS_KEY`: MinIO secret key (default: `minioadmin`)
- `MLFLOW_ARTIFACT_LOCATION`: S3 bucket for artifacts (default: `s3://mlflow-artifacts/`)

The `make train` command automatically sets these variables for proper MinIO integration.

## Troubleshooting

**Issue: "No such file or directory" when training**
- Ensure Docker containers are running: `make status`
- Check MinIO is accessible: `curl http://localhost:9000/minio/health/live`

**Issue: Artifacts not appearing in MLflow UI**
- Verify environment variables are set correctly when running `make ui`
- Check MinIO console at http://localhost:9001 to verify bucket exists

**Issue: "Connection refused" to PostgreSQL**
- Wait for containers to fully start: `docker compose logs -f`
- Check PostgreSQL is ready: `docker exec mlflow-postgres pg_isready -U mlflow`

**Issue: Submission file not generated**
- Ensure `data/raw/train.csv` and `data/raw/test.csv` exist
- Use `SUBMIT=true` flag: `make train MODEL=xgboost SUBMIT=true`

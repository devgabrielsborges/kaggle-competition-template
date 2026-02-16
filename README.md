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
# Train with defaults (xgboost, regression, 100 trials)
make train

# Train specific models
make train MODEL=ridge TASK=regression TRIALS=200
make train MODEL=xgboost TASK=regression TRIALS=300
make train MODEL=pytorch TASK=binary_classification TRIALS=50
make train MODEL=tensorflow TASK=multiclass_classification TRIALS=50
```

Available models: `linear`, `ridge`, `random_forest`, `gradient_boosting`, `svm`, `xgboost`, `pytorch`, `tensorflow`

Available tasks: `regression`, `binary_classification`, `multiclass_classification`

### 4. View Results

```bash
make ui         # Open http://localhost:5000
```

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
- **Hyperparameter Optimization**: Optuna
- **Infrastructure**: Docker Compose (PostgreSQL + MinIO)

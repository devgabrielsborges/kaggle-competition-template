"""TensorFlow/Keras neural network adapter.

Implements the ModelPort interface using TensorFlow/Keras.
Customize the network architecture in build() for your competition.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.domain.entities.experiment_config import TaskType
from src.domain.ports.model_port import ModelPort


class TensorFlowAdapter(ModelPort):
    """Adapter for TensorFlow/Keras models."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION):
        self._model = None
        self._task_type = task_type
        self._params: dict = {}

    @property
    def name(self) -> str:
        return "TensorFlowNN"

    def build(self, hyperparameters: dict) -> None:
        self._params = hyperparameters
        # Model will be fully built in fit() when we know input_dim

    def _build_network(self, input_dim: int) -> Any:
        import tensorflow as tf

        hidden_dims = self._params.get("hidden_dims", [128, 64])
        dropout = self._params.get("dropout", 0.2)
        lr = self._params.get("learning_rate", 1e-3)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        for units in hidden_dims:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout))

        if self._task_type == TaskType.REGRESSION:
            model.add(tf.keras.layers.Dense(1))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss="mse",
                metrics=["mae"],
            )
        elif self._task_type == TaskType.BINARY_CLASSIFICATION:
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        else:
            n_classes = self._params.get("n_classes", 10)
            model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

        return model

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values

        input_dim = X.shape[1]
        self._model = self._build_network(input_dim)

        epochs = self._params.get("epochs", 100)
        batch_size = self._params.get("batch_size", 256)

        self._model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = self._model.predict(X, verbose=0)

        if self._task_type == TaskType.REGRESSION:
            return predictions.squeeze()
        elif self._task_type == TaskType.BINARY_CLASSIFICATION:
            return (predictions.squeeze() > 0.5).astype(int)
        else:
            return predictions.argmax(axis=1)

    def get_search_space(self, trial: Any) -> dict:
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_dims = [
            trial.suggest_int(
                f"hidden_dim_{i}",
                32,
                512,
            )
            for i in range(n_layers)
        ]
        return {
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float(
                "dropout",
                0.0,
                0.5,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                1e-5,
                1e-2,
                log=True,
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                [64, 128, 256, 512],
            ),
            "epochs": trial.suggest_int("epochs", 10, 100),
        }

    def get_model(self) -> Any:
        return self._model

    def get_default_trials(self) -> int:
        """TensorFlow NN has dynamic architecture space similar to PyTorch."""
        return 150

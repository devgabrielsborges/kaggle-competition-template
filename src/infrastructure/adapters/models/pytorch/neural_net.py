"""PyTorch neural network adapter.

Implements the ModelPort interface using PyTorch.
Customize the network architecture in _build_network() for your competition.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.domain.entities.experiment_config import TaskType
from src.domain.ports.model_port import ModelPort


class _NeuralNet(nn.Module):
    """Default feedforward neural network. Customize per competition."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PyTorchAdapter(ModelPort):
    """Adapter for PyTorch neural networks."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION):
        self._model: _NeuralNet | None = None
        self._task_type = task_type
        self._params: dict = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "PyTorchNN"

    def build(self, hyperparameters: dict) -> None:
        self._params = hyperparameters
        # Model will be fully built in fit() when we know input_dim

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
        output_dim = 1 if self._task_type == TaskType.REGRESSION else len(np.unique(y))

        hidden_dims = self._params.get("hidden_dims", [128, 64])
        dropout = self._params.get("dropout", 0.2)
        lr = self._params.get("learning_rate", 1e-3)
        epochs = self._params.get("epochs", 100)
        batch_size = self._params.get("batch_size", 256)

        self._model = _NeuralNet(
            input_dim,
            hidden_dims,
            output_dim,
            dropout,
        ).to(self._device)

        X_tensor = torch.FloatTensor(X).to(self._device)
        y_tensor = torch.FloatTensor(y).to(self._device)
        if self._task_type != TaskType.REGRESSION:
            y_tensor = torch.LongTensor(y).to(self._device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if self._task_type == TaskType.REGRESSION:
            criterion = nn.MSELoss()
        elif self._task_type == TaskType.BINARY_CLASSIFICATION:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        self._model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self._model(batch_X)
                if self._task_type == TaskType.REGRESSION:
                    outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            outputs = self._model(X_tensor)
            if self._task_type == TaskType.REGRESSION:
                return outputs.squeeze().cpu().numpy()
            else:
                return outputs.argmax(dim=1).cpu().numpy()

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

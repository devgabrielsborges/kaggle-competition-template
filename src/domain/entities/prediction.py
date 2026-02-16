"""Prediction entity holding model outputs and metadata."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Prediction:
    """Holds model predictions along with metadata."""

    values: np.ndarray
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)

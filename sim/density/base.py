"""Abstract base class for density estimators."""

from abc import ABC, abstractmethod

import numpy as np


class DensityEstimator(ABC):
    """Base class for per-agent local density estimation."""

    @abstractmethod
    def estimate(self, positions: np.ndarray, **kwargs) -> np.ndarray:
        """Estimate local density at each agent position.

        Args:
            positions: Agent positions, shape (N, 2).
            **kwargs: Estimator-specific parameters.

        Returns:
            Per-agent density, shape (N,).
        """
        ...

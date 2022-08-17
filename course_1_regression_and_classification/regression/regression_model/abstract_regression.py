from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from course_1_regression_and_classification.regression.dataset_processing.dataset_data_model import Dataset
from course_1_regression_and_classification.regression.regression_model.predictor import Predictor


class AbstractRegression(ABC):
    @abstractmethod
    def _calculate_model_predictions(self) -> np.ndarray:
        """Generate array of model predictions based on best weight and bias."""
        pass

    @abstractmethod
    def _compute_gradients(self, model_predictions: np.ndarray) -> Tuple[float, float] :
        """Compute partial derivatives of cost with respect to bias and weight."""
        pass

    @abstractmethod
    def _compute_total_cost(self, model_predictions: np.ndarray) -> float:
        """Calculate total cost based on current model predictions."""
        pass

    @abstractmethod
    def _gradient_descent_iteration(self, learning_rate: float) -> float:
        """Updates weight and bias in place, returns total cost at iteration"""
        pass

    @abstractmethod
    def fit(
            self,
            training_dataset: Dataset,
            learning_rate: float = 0.01,
            number_of_iterations: int = 1000
    ) -> Predictor:
        pass

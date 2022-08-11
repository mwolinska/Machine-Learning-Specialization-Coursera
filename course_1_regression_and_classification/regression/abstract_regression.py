from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from course_1_regression_and_classification.regression.dataset_processing.dataset_data_model import Dataset


class AbstractRegression(ABC):
    @abstractmethod
    def calculate_model_predictions(self) -> np.ndarray:
        """Generate array of model predictions based on best weight and bias."""
        pass

    @abstractmethod
    def compute_gradients(self, model_predictions: np.ndarray) -> Tuple[float, float] :
        """Compute partial derivatives of cost with respect to bias and weight."""
        pass

    @abstractmethod
    def compute_total_cost(self, model_predictions: np.ndarray) -> float:
        """Calculate total cost based on current model predictions."""
        pass

    @abstractmethod
    def gradient_descent(self, learning_rate: float, number_of_iterations: int) -> Tuple[List, List]:
        """Perform gradient descent, return cost over time and best model predictions. """
        pass

    @abstractmethod
    def return_training_data_for_plotting(self) -> Dataset:
        """Return relevant training data used for regression as Dataset. """
        pass

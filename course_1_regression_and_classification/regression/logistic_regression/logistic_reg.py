from typing import Tuple

import numpy as np

from course_1_regression_and_classification.regression.dataset_processing.dataset_data_model import Dataset
from course_1_regression_and_classification.regression.regression_model.abstract_regression import AbstractRegression
from course_1_regression_and_classification.regression.regression_model.predictor import Predictor


class LogisticRegression(AbstractRegression):
    def _calculate_model_predictions(self) -> np.ndarray:
        pass

    def _compute_gradients(self, model_predictions: np.ndarray) -> Tuple[float, float]:
        pass

    def _compute_total_cost(self, model_predictions: np.ndarray) -> float:
        pass

    def _gradient_descent_iteration(self, learning_rate: float) -> float:
        pass

    def fit(
        self,
        training_dataset: Dataset,
        learning_rate: float = 0.01,
        number_of_iterations: int = 1000,
    ) -> Predictor:
        pass

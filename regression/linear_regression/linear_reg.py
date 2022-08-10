from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from regression.dataset_processing.dataset_data_model import Dataset

class LinearRegression:
    def __init__(
        self,
        training_dataset: Dataset,
        validate_dataset: Dataset,
        test_dataset: Dataset,
        number_of_iterations: int = 1500,
        learning_rate: float = 0.01
    ):
        self.training_data = training_dataset
        self.test_data = test_dataset
        self.validation_data = validate_dataset
        self.feature_idx_for_regression = 0
        self.current_weight = 0
        self.current_bias = 0
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate

    def model_prediction_array(self) -> np.ndarray:
        prediction = self.current_weight \
                     * self.training_data.feature_data[:, self.feature_idx_for_regression] \
                     + self.current_bias
        return prediction

    def compute_total_cost(self):
        model_predictions = self.model_prediction_array()
        dataset_size = self.training_data.feature_data.shape[0]
        cost_array = (model_predictions - self.training_data.label_data) ** 2
        total_cost = np.sum(cost_array) / (2 * dataset_size)
        return total_cost

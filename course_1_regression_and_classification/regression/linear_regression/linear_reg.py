from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from course_1_regression_and_classification.regression.dataset_processing.dataset_data_model import Dataset

class LinearRegression:
    def __init__(
        self,
        training_dataset: Dataset,
        initial_weight: float = 0.0,
        initial_bias: float = 0.0,
        feature_index_for_regression: int = 0,
    ):
        self.training_data = training_dataset
        self.feature_index = feature_index_for_regression
        self.best_weight = initial_weight
        self.best_bias = initial_bias
        self.dataset_size = self.training_data.feature_data.shape[0]
        self.cost_over_time = []

    def model_prediction_array(self) -> np.ndarray:
        prediction = self.best_weight \
                     * self.training_data.feature_data[:, self.feature_index] \
                     + self.best_bias
        return prediction

    def compute_total_cost(self, model_predictions: np.ndarray) -> np.ndarray:
        cost_array = (model_predictions - self.training_data.label_data) ** 2
        total_cost = np.sum(cost_array) / (2 * self.dataset_size)
        return total_cost

    def compute_gradients(self, model_predictions: np.ndarray) -> Tuple[float, float]:
        array_of_derivatives_cost_wrt_bias, array_of_derivatives_cost_wrt_weight = \
            self.compute_derivative_arrays(model_predictions)

        derivative_cost_wrt_bias = np.sum(array_of_derivatives_cost_wrt_bias) / self.dataset_size
        derivative_cost_wrt_weight = np.sum(array_of_derivatives_cost_wrt_weight) / self.dataset_size

        return derivative_cost_wrt_bias, derivative_cost_wrt_weight

    def compute_derivative_arrays(self, model_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        array_of_derivatives_cost_wrt_bias = model_predictions - \
                                               self.training_data.label_data
        array_of_derivatives_cost_wrt_weight = array_of_derivatives_cost_wrt_bias \
            * self.training_data.feature_data[:, self.feature_index]
        return array_of_derivatives_cost_wrt_bias, array_of_derivatives_cost_wrt_weight

    def gradient_descent(self, learning_rate: float = 0.01, number_of_iterations: int = 1000) -> Tuple[List, List]:
        cost_over_time = []
        model_predictions_array = np.empty(0)

        for i in range(number_of_iterations):
            model_predictions_array = self.model_prediction_array()
            derivative_cost_wrt_bias, derivative_cost_wrt_weight = \
                self.compute_gradients(model_predictions_array)

            self.best_weight = self.best_weight - learning_rate * derivative_cost_wrt_weight
            self.best_bias = self.best_bias - learning_rate * derivative_cost_wrt_bias

            print(f"weight {self.best_weight} bias {self.best_bias}")
            total_cost = self.compute_total_cost(model_predictions_array)
            cost_over_time.append(total_cost)

        plt.plot(cost_over_time)
        plt.show()
        # plt.scatter(self.training_data.feature_data[:, self.feature_index], self.training_data.label_data)
        # plt.plot(self.training_data.feature_data[:, self.feature_index], model_predictions_array, color="red")
        # plt.show()

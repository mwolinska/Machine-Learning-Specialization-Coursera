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

    def compute_gradients(self, model_predictions: np.ndarray) -> Tuple[float, float]:
        array_of_derivatives_cost_wrt_bias = model_predictions - \
                                               self.training_data.label_data
        array_of_derivatives_cost_wrt_weight = array_of_derivatives_cost_wrt_bias \
            * self.training_data.feature_data[:, self.feature_idx_for_regression]

        sum_array_of_derivatives_cost_wrt_weight = np.sum(array_of_derivatives_cost_wrt_weight)
        sum_array_of_derivatives_cost_wrt_bias = np.sum(array_of_derivatives_cost_wrt_bias)

        dataset_size = self.training_data.feature_data.shape[0]

        derivative_cost_wrt_weight = sum_array_of_derivatives_cost_wrt_weight / dataset_size
        derivative_cost_wrt_bias = sum_array_of_derivatives_cost_wrt_bias / dataset_size

        return derivative_cost_wrt_weight, derivative_cost_wrt_bias

    def gradient_descent(self):
        cost_over_time = []
        weight_over_time = []

        for i in range(self.number_of_iterations):
            model_predictions_array = self.model_prediction_array()
            derivative_cost_wrt_weight, derivative_cost_wrt_bias = \
                self.compute_gradients(model_predictions_array)

            self.current_weight = self.current_weight - self.learning_rate * derivative_cost_wrt_weight
            self.current_bias = self.current_bias - self.learning_rate * derivative_cost_wrt_bias

            print(f"weight {self.current_weight} bias {self.current_bias}")
            total_cost = self.compute_total_cost()
            cost_over_time.append(total_cost)
            weight_over_time.append(self.current_weight)

        # plt.plot(cost_over_time)
        # plt.show()
        plt.scatter(self.training_data.feature_data[:, self.feature_idx_for_regression], self.training_data.label_data)
        plt.plot(self.training_data.feature_data[:, self.feature_idx_for_regression], model_predictions_array, color="red")
        plt.show()

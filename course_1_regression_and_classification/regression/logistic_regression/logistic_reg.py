import math
from math import exp, log10
from typing import Tuple

import numpy as np

from course_1_regression_and_classification.regression.dataset_processing.dataset_data_model import Dataset
from course_1_regression_and_classification.regression.dataset_processing.dataset_utils import prepare_datasets
from course_1_regression_and_classification.regression.regression_model.abstract_regression import AbstractRegression
from course_1_regression_and_classification.regression.regression_model.predictor import Predictor


class LogisticRegression(AbstractRegression):
    def __init__(
            self,
            initial_weight: np.ndarray,
            initial_bias: float = 0.0,
    ):
        self.training_data = None  # update to None
        self.best_weight = initial_weight
        self.best_bias = initial_bias

    def _calculate_model_predictions(self) -> np.ndarray:
        linear_predictions = self._sum_weighted_features() \
                             + self.best_bias
        sigmoid_predictions = self._sigmoid(linear_predictions)

        return sigmoid_predictions

    def _sum_weighted_features(self) -> np.ndarray:
        weighted_features = self.training_data.feature_data * self.best_weight
        return weighted_features.sum(axis=1)

    @staticmethod
    def _sigmoid(linear_predictions: np.ndarray) -> np.ndarray:
        sigmoid_predictions = np.empty(0)
        for element in linear_predictions:
            sigmoid_element = 1 / (1 + exp(-element))
            sigmoid_predictions = np.append(sigmoid_predictions, sigmoid_element)

        return sigmoid_predictions

    def _compute_total_cost(self, model_predictions: np.ndarray) -> float:
        array_of_loss = self._compute_loss(model_predictions)
        total_cost = array_of_loss.sum() / self.training_data.size
        return total_cost

    def _compute_loss(self, model_predictions) -> np.ndarray:
        loss_element_1 = - self.training_data.label_data * np.log(model_predictions)
        loss_element_2 = (np.ones(self.training_data.size) - self.training_data.label_data) \
                         * np.log((np.ones(self.training_data.size) - model_predictions))

        array_of_individual_loss = loss_element_1 - loss_element_2
        return array_of_individual_loss

    def _compute_gradients(self, model_predictions: np.ndarray) -> Tuple[float, float]:
        """Compute partial derivatives of cost with respect to bias and weight."""
        array_of_derivatives_cost_wrt_bias, array_of_derivatives_cost_wrt_weight = \
            self._compute_derivative_arrays(model_predictions)

        derivative_cost_wrt_bias = array_of_derivatives_cost_wrt_bias.sum() / self.training_data.size
        derivative_cost_wrt_weight = array_of_derivatives_cost_wrt_weight.sum(axis=0) / self.training_data.size

        return derivative_cost_wrt_bias, derivative_cost_wrt_weight

    def _compute_derivative_arrays(self, model_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute individual partial derivatives of cost for each datapoint with respect to bias and weight. """
        array_of_derivatives_cost_wrt_bias = model_predictions - \
                                             self.training_data.label_data
        array_of_derivatives_cost_wrt_bias_for_multiplication = \
            self._dummy_array_of_bias_derivatives(array_of_derivatives_cost_wrt_bias)
        array_of_derivatives_cost_wrt_weight = array_of_derivatives_cost_wrt_bias_for_multiplication \
                                               * self.training_data.feature_data
        return array_of_derivatives_cost_wrt_bias, array_of_derivatives_cost_wrt_weight

    def _dummy_array_of_bias_derivatives(self, array_of_derivatives_cost_wrt_bias) -> np.ndarray:
        number_of_features = self.training_data.feature_data.shape[1]
        dummy_array = np.tile(array_of_derivatives_cost_wrt_bias, [number_of_features, 1])
        transposed_dummy_array = dummy_array.transpose()
        return transposed_dummy_array

    def _gradient_descent_iteration(self, learning_rate: float) -> float:
        """Updates weight and bias in place, returns total cost at iteration"""
        model_predictions_array = self._calculate_model_predictions()
        derivative_cost_wrt_bias, derivative_cost_wrt_weight = \
            self._compute_gradients(model_predictions_array)

        self.best_weight = self.best_weight - learning_rate * derivative_cost_wrt_weight
        self.best_bias = self.best_bias - learning_rate * derivative_cost_wrt_bias

        return self._compute_total_cost(model_predictions_array)

    def fit(
            self,
            training_dataset: Dataset,
            learning_rate: float = 0.01,
            number_of_iterations: int = 1000
    ) -> Predictor:
        cost_over_time = []

        self.training_data = training_dataset
        # self.best_weight = np.zeros(self.training_data.feature_data.shape[1])

        for i in range(number_of_iterations):
            total_cost_at_iteration = self._gradient_descent_iteration(learning_rate=learning_rate)
            cost_over_time.append(total_cost_at_iteration)

            if i % math.ceil(number_of_iterations / 10) == 0 or i == (number_of_iterations - 1):
                print(f"Iteration {i:4}: Cost {float(cost_over_time[-1]):8.2f}   ")

        model_predictions = list(self._calculate_model_predictions())

        return Predictor(
            weight=self.best_weight,
            bias=self.best_bias,
            training_data=self.training_data,
            cost_over_time=cost_over_time,
            model_predictions_for_training=model_predictions,
        )

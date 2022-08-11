from typing import Tuple, List

import numpy as np

from course_1_regression_and_classification.regression.dataset_processing.dataset_data_model import Dataset
from course_1_regression_and_classification.regression.regression_model.predictor import Predictor


class LinearRegression:
    def __init__(
            self,
            initial_weight: float = 0.0,
            initial_bias: float = 0.0,
            feature_index_for_regression: int = 0,
    ):
        self.training_data = None
        self.feature_index = feature_index_for_regression
        self.best_weight = initial_weight
        self.best_bias = initial_bias

    def _model_prediction_array(self) -> np.ndarray:
        """Generate array of model predictions based on best weight and bias."""
        prediction = self.best_weight \
            * self.training_data.feature_data[:, self.feature_index] \
            + self.best_bias
        return prediction

    def _compute_total_cost(self, model_predictions: np.ndarray) -> float:
        """Calculate total cost based on current model predictions."""
        cost_array = (model_predictions - self.training_data.label_data) ** 2
        total_cost = np.sum(cost_array) / (2 * self.training_data.size)
        return total_cost

    def _compute_gradients(self, model_predictions: np.ndarray) -> Tuple[float, float]:
        """Compute partial derivatives of cost with respect to bias and weight."""
        array_of_derivatives_cost_wrt_bias, array_of_derivatives_cost_wrt_weight = \
            self._compute_derivative_arrays(model_predictions)

        derivative_cost_wrt_bias = np.sum(array_of_derivatives_cost_wrt_bias) / self.training_data.size
        derivative_cost_wrt_weight = np.sum(array_of_derivatives_cost_wrt_weight) / self.training_data.size

        return derivative_cost_wrt_bias, derivative_cost_wrt_weight

    def _compute_derivative_arrays(self, model_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute individual partial derivatives of cost for each datapoint with respect to bias and weight. """
        array_of_derivatives_cost_wrt_bias = model_predictions - \
                                             self.training_data.label_data
        array_of_derivatives_cost_wrt_weight = array_of_derivatives_cost_wrt_bias \
                                               * self.training_data.feature_data[:, self.feature_index]
        return array_of_derivatives_cost_wrt_bias, array_of_derivatives_cost_wrt_weight

    def _gradient_descent_iteration(self, learning_rate: float) -> float:
        """Updates weight and bias in place, returns total cost at iteration"""
        model_predictions_array = self._model_prediction_array()
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

        self.training_data = Dataset.from_dataset(
            training_dataset,
            feature_row_tuple=(None, None),
            feature_column_tuple=(self.feature_index, self.feature_index + 1)
        )

        for i in range(number_of_iterations):
            total_cost_at_iteration = self._gradient_descent_iteration(learning_rate=learning_rate)
            cost_over_time.append(total_cost_at_iteration)

        model_predictions = list(self._model_prediction_array())

        return Predictor(
            weight=self.best_weight,
            bias=self.best_bias,
            training_data=self.training_data,
            cost_over_time=cost_over_time,
            model_predictions_for_training=model_predictions,
        )

    def return_training_data_for_plotting(self) -> Dataset:
        """Return relevant training data used for regression as Dataset. """
        return Dataset.from_dataset(
            self.training_data,
            feature_row_tuple=(None, None),
            feature_column_tuple=(self.feature_index, self.feature_index + 1))

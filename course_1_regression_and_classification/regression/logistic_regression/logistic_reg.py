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

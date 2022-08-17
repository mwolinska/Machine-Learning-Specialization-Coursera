from dataclasses import dataclass
from typing import List, Union

from course_1_regression_and_classification.regression.dataset_processing.dataset_data_model import Dataset

class Predictor:
    def __init__(
        self,
        weight: Union[float, List[float]],
        bias: float,
        cost_over_time: List[float],
        training_data: Dataset,
        model_predictions_for_training: List[float],
    ):
        self.weight = weight
        self.bias = bias
        self.cost_over_time = cost_over_time
        self.training_data = training_data
        self.model_predictions_for_training = model_predictions_for_training

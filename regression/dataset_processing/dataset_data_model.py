from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Dataset:
    feature_data: np.ndarray
    feature_names: List
    label_data: np.ndarray
    label_name: str

    @classmethod
    def from_array(cls, dataset_array: np.ndarray) -> "Dataset":
        label_name = dataset_array[0][-1]
        feature_names = list(dataset_array[0][:-1])
        label_data = dataset_array[1:, -1].astype(float)
        feature_data = dataset_array[1:, :-1].astype(float)
        return cls(
            feature_data=feature_data,
            feature_names=feature_names,
            label_data=label_data,
            label_name=label_name,
        )

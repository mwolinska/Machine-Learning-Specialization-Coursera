from dataclasses import dataclass
from typing import List, Tuple, Optional

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

    @classmethod
    def from_dataset(
        cls,
        base_dataset: "Dataset",
        feature_row_tuple: Tuple[Optional[int], Optional[int]],
        feature_column_tuple: Tuple[Optional[int], Optional[int]] = (None, None),
    ) -> "Dataset":
        new_feature_data = base_dataset.feature_data[
            feature_row_tuple[0]:feature_row_tuple[1],
            feature_column_tuple[0]:feature_column_tuple[1]
        ]
        new_label_data = base_dataset.label_data[feature_row_tuple[0]:feature_row_tuple[1]]

        new_feature_names = base_dataset.feature_names[feature_column_tuple[0]:feature_column_tuple[1]]

        return cls(
            feature_data=new_feature_data,
            feature_names=new_feature_names,
            label_data=new_label_data,
            label_name=base_dataset.label_name,
        )

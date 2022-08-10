import csv
from typing import Tuple

import numpy as np

from course_1_regression_and_classification.regression.dataset_processing.dataset_data_model import Dataset


def create_array_from_csv_file(filename: str, delimiter: str = ",") -> np.ndarray:
    dataset_list = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC, delimiter=delimiter)  # change contents to floats
        for row in reader:  # each row is a list
            dataset_list.append(row)

    dataset_array = np.asarray(dataset_list)
    np.random.shuffle(dataset_array[1:, ])
    return dataset_array


def prepare_datasets(
        filename: str,
        delimiter: str = ",",
        training_set_ratio: float = 0.6,
) -> Tuple["Dataset", "Dataset", "Dataset"]:
    # TODO(marta): create interface for datasets
    full_dataset_array = create_array_from_csv_file(filename, delimiter=delimiter)
    full_dataset = Dataset.from_array(full_dataset_array)
    # split dataset into train, validate, test datasets
    dataset_size = full_dataset.feature_data.shape[0]
    test_set_ratio = (1 - training_set_ratio) / 2
    split_index_train_dataset = int(dataset_size * training_set_ratio)
    split_index_test_dataset = split_index_train_dataset + int(dataset_size * test_set_ratio)

    training_dataset = Dataset.from_dataset(
        base_dataset=full_dataset,
        feature_row_tuple=(0, split_index_train_dataset),
    )
    test_dataset = Dataset.from_dataset(
        base_dataset=full_dataset,
        feature_row_tuple=(split_index_train_dataset, split_index_test_dataset),
    )
    validate_dataset = Dataset.from_dataset(
        base_dataset=full_dataset,
        feature_row_tuple=(split_index_test_dataset, None),
    )
    return training_dataset, validate_dataset, test_dataset

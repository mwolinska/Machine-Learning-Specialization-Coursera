import csv
from typing import Tuple

import numpy as np

from regression.dataset_processing.dataset_data_model import Dataset


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

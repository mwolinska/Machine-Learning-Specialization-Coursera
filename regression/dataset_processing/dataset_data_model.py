from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Dataset:
    feature_data: np.ndarray
    feature_names: List
    label_data: np.ndarray
    label_name: str

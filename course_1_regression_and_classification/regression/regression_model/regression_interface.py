from enum import Enum

from course_1_regression_and_classification.regression.regression_model.abstract_regression import AbstractRegression
from course_1_regression_and_classification.regression.linear_regression.linear_reg import LinearRegression
from course_1_regression_and_classification.regression.logistic_regression.logistic_reg import LogisticRegression


class Regressions(str, Enum):
    LINEAR = "linear"
    LOGISTIC = "logistic"


class Regression:
    regression_dict = {
        Regressions.LINEAR: LinearRegression,
        Regressions.LOGISTIC: LogisticRegression,
    }

    @classmethod
    def get_regression(cls, regression_type: Regressions) -> type(AbstractRegression):
        return cls.regression_dict[regression_type]

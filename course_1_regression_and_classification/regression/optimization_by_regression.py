import matplotlib.pyplot as plt

from course_1_regression_and_classification.regression.dataset_processing.dataset_data_model import Dataset
from course_1_regression_and_classification.regression.regression_model.abstract_regression import AbstractRegression

class OptimizationByRegression:
    def __init__(self, regression_object: AbstractRegression):
        self.regression = regression_object
        self.predictor = None

    def gradient_descent(self, training_data: Dataset, learning_rate: float = 0.01, number_of_iterations: int = 1000):
        self.predictor = self.regression.fit(
            training_data,
            learning_rate=learning_rate,
            number_of_iterations=number_of_iterations,
        )

    def plot_cost_over_time(self):
        plt.plot(self.predictor.cost_over_time)
        plt.title(f"Regression Training Data and Model Fit")
        plt.xlabel("Iteration")
        plt.ylabel("Total Cost at Iteration")
        plt.show()

    def plot_data_and_fit(self):
        plt.scatter(self.predictor.training_data.feature_data, self.predictor.training_data.label_data, label="Training data")
        plt.plot(self.predictor.training_data.feature_data, self.predictor.model_predictions_for_training, label="Model fit", color="red")
        plt.title(f"Regression Training Data and Model Fit")
        plt.xlabel(self.predictor.training_data.feature_names[0])
        plt.ylabel(self.predictor.training_data.label_name)
        plt.legend()
        plt.show()

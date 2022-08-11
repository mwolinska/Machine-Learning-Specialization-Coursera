import matplotlib.pyplot as plt

from course_1_regression_and_classification.regression.regression_model.abstract_regression import AbstractRegression

class OptimizationByRegression:
    def __init__(self, regression_object: AbstractRegression):
        self.regression = regression_object
        self.cost_over_time = []
        self.best_model_predictions = []

    def gradient_descent(self, learning_rate: float = 0.01, number_of_iterations: int = 1000):
        self.cost_over_time, self.best_model_predictions = self.regression.gradient_descent(
            learning_rate=learning_rate,
            number_of_iterations=number_of_iterations,
        )

    def plot_cost_over_time(self):
        plt.plot(self.cost_over_time)
        plt.title(f"Regression Training Data and Model Fit")
        plt.xlabel("Iteration")
        plt.ylabel("Total Cost at Iteration")
        plt.show()

    def plot_data_and_fit(self):
        training_dataset = self.regression.return_training_data_for_plotting()
        plt.scatter(training_dataset.feature_data, training_dataset.label_data, label="Training data")
        plt.plot(training_dataset.feature_data, self.best_model_predictions, label="Model fit", color="red")
        plt.title(f"Regression Training Data and Model Fit")
        plt.xlabel(training_dataset.feature_names[0])
        plt.ylabel(training_dataset.label_name)
        plt.legend()
        plt.show()

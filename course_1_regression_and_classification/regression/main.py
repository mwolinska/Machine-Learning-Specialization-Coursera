from course_1_regression_and_classification.regression.dataset_processing.dataset_utils import prepare_datasets
from course_1_regression_and_classification.regression.optimization_by_regression import OptimizationByRegression
from course_1_regression_and_classification.regression.regression_interface import Regression, Regressions

if __name__ == '__main__':
    train, validate, test = prepare_datasets("../../datasets/coursera_lin_reg_dataset.csv", delimiter=",")
    regression_class = Regression.get_regression(Regressions.LINEAR)
    regression_object = regression_class(
        training_dataset=train,
    )

    optimization = OptimizationByRegression(regression_object)
    optimization.gradient_descent(
        learning_rate=0.01,
        number_of_iterations=1000,
    )

    optimization.plot_data_and_fit()
    optimization.plot_cost_over_time()

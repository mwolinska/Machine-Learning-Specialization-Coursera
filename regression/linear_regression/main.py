from regression.dataset_processing.dataset_utils import prepare_datasets
from regression.linear_regression.linear_reg import LinearRegression

if __name__ == '__main__':
    train, validate, test = prepare_datasets("../../datasets/coursera_lin_reg_dataset.csv", delimiter=",")
    lin_reg = LinearRegression(train, validate, test)
    # lin_reg.plot_data(lin_reg.training_data.feature_data[:, lin_reg.feature_idx_for_regression],
    #                   lin_reg.training_data.label_data)
    lin_reg.gradient_descent()

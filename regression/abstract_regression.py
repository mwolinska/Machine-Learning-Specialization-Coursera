from abc import ABC, abstractmethod


class AbstractRegression(ABC):
    @abstractmethod
    def calculate_model_predictions(self):
        pass

    @abstractmethod
    def compute_gradients(self):
        pass

    @abstractmethod
    def compute_total_cost(self):
        pass

    @abstractmethod
    def gradient_descent(self):
        pass

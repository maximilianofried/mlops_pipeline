import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearRegressionModel(Model):
    """
    Linear regression model
    """
    def train(self, X_train, y_train, **kwargs):
        logging.info("Training model")
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model trained successfully")
            return reg
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise e

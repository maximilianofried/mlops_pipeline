import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy to evaluate our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores of the model on the test data
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        Returns:
            scores: Dictionary of scores
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean squared error evaluation
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores of the model on the test data
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        Returns:
            scores: Dictionary of scores
        """
        try:
            logging.info("Calculating MSE scores")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE scores: {e}")
            raise e

class R2(Evaluation):
    """
    Evaluation Strategy that uses R2 evaluation
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores of the model on the test data
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        Returns:
            scores: Dictionary of scores
        """
        try:
            logging.info("Calculating R2 scores")
            r2 = r2_score(y_true, y_pred)
            logging.info("The R2 score value is: " + str(r2))
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2 scores: {e}")
            raise e

class RMSE(Evaluation):
    """
    Evaluation Strategy that uses RMSE evaluation
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores of the model on the test data
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        """
        try:
            logging.info("Calculating RMSE scores")
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE scores: {e}")
            raise e
import logging
import time
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError
from typing import List, Any, Union
from custom_exceptions import TrainingError

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class AnomalyDetector:
    """Detects anomalies in data using Isolation Forest."""

    def __init__(
        self, 
        contamination: float = 0.1, 
        random_state: int = 42, 
        retry_attempts: int = 3, 
        retry_delay: float = 0.5
    ) -> None:
        """
        Initialize the Anomaly Detector with an Isolation Forest model.

        :param contamination: Proportion of outliers in the dataset (0 < contamination <= 0.5).
        :param random_state: Random seed for reproducibility.
        :param retry_attempts: Number of attempts for fitting in case of failure.
        :param retry_delay: Delay between retry attempts in seconds.
        :raises ValueError: For invalid contamination or retry settings.
        :raises TypeError: For invalid random_state type.
        """
        if not (0 < contamination <= 0.5):
            raise ValueError("contamination must be a float between 0 and 0.5.")
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer.")
        if not isinstance(retry_attempts, int) or retry_attempts <= 0:
            raise ValueError("retry_attempts must be a positive integer.")
        if not isinstance(retry_delay, (int, float)) or retry_delay <= 0:
            raise ValueError("retry_delay must be a positive number.")

        self.contamination: float = contamination
        self.random_state: int = random_state
        self.retry_attempts: int = retry_attempts
        self.retry_delay: float = retry_delay

        self.model: IsolationForest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        logging.debug("IsolationForest model initialized successfully.")

    def fit(self, X: Union[List[List[float]], Any]) -> None:
        """
        Fit the Isolation Forest model on the provided dataset.

        :param X: 2D array-like structure for training data.
        :raises TrainingError: If fitting fails after retries.
        """
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if not hasattr(X, 'shape') or len(X.shape) != 2:
                    raise TypeError("X must be a 2D array-like structure.")
                if X.shape[0] == 0:
                    raise ValueError("X must contain at least one sample.")

                self.model.fit(X)
                logging.info("Anomaly Detector model fitted successfully.")
                return
            except (ValueError, TypeError) as e:
                logging.error(f"Error fitting Anomaly Detector (Attempt {attempt}): {e}")
                time.sleep(self.retry_delay)
            except Exception as e:
                logging.error(f"Unexpected error during fitting (Attempt {attempt}): {e}")
                raise TrainingError(message="Failed to fit Anomaly Detector.", details=str(e))
        raise TrainingError(message="Max retries reached during model fitting.", details={"retry_attempts": self.retry_attempts})

    def predict(self, X: Union[List[List[float]], Any]) -> List[int]:
        """
        Predict anomalies in the dataset.

        :param X: 2D array-like structure for data to predict.
        :return: List of predictions (-1 for anomalies, 1 for normal points).
        :raises TrainingError: If prediction fails.
        """
        try:
            if not hasattr(self.model, 'estimators_'):
                raise NotFittedError("This IsolationForest instance is not fitted yet.")
            if not hasattr(X, 'shape') or len(X.shape) != 2:
                raise TypeError("X must be a 2D array-like structure.")
            if X.shape[0] == 0:
                raise ValueError("X must contain at least one sample.")

            predictions = self.model.predict(X)
            logging.debug(f"Anomaly predictions completed on {X.shape[0]} samples.")
            return predictions.tolist()
        except Exception as e:
            logging.error(f"Error predicting anomalies: {e}")
            raise TrainingError(message="Error predicting anomalies.", details=str(e))

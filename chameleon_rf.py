import logging
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, Union
from custom_exceptions import ModelCreationError

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class ChameleonRandomForest:
    """
    Creates a RandomForestRegressor with adaptive parameters and random perturbations.
    Includes dynamic error handling and retry logic for robustness.
    """

    def __init__(self, random_state: int = 42, retry_attempts: int = 3, retry_delay: float = 0.5) -> None:
        """
        Initialize the ChameleonRandomForest.

        :param random_state: Random seed for reproducibility.
        :param retry_attempts: Number of attempts to create the model in case of failure.
        :param retry_delay: Delay between retry attempts in seconds.
        :raises ValueError: For invalid retry settings.
        :raises TypeError: For invalid random_state type.
        """
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer.")
        if not isinstance(retry_attempts, int) or retry_attempts <= 0:
            raise ValueError("retry_attempts must be a positive integer.")
        if not isinstance(retry_delay, (int, float)) or retry_delay <= 0:
            raise ValueError("retry_delay must be a positive number.")

        self.random_state: int = random_state
        self.retry_attempts: int = retry_attempts
        self.retry_delay: float = retry_delay
        logging.debug("ChameleonRandomForest initialized successfully.")

    def create_model(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        """
        Create an adaptive RandomForestRegressor based on dataset characteristics.

        :param X: Feature matrix as a numpy array.
        :param y: Target vector as a numpy array.
        :return: Configured RandomForestRegressor instance.
        :raises ModelCreationError: If model creation fails after retries.
        """
        for attempt in range(1, self.retry_attempts + 1):
            try:
                # Validate inputs
                if not isinstance(X, np.ndarray):
                    raise TypeError("X must be a numpy.ndarray.")
                if not isinstance(y, np.ndarray):
                    raise TypeError("y must be a numpy.ndarray.")
                if X.shape[0] != y.shape[0]:
                    raise ValueError("X and y must have the same number of samples.")
                if X.ndim != 2 or y.ndim != 1:
                    raise ValueError("X must be 2D and y must be 1D.")

                # Calculate dataset characteristics
                data_size = X.shape[0]
                feature_variance = np.var(X, axis=0).mean()
                target_variance = np.var(y)

                # Dynamic parameter adjustments
                n_estimators = 100 if data_size < 5000 else 200
                max_depth: Optional[int] = None if feature_variance < 0.5 else int(10 + feature_variance * 10)
                min_samples_split = max(2, int(2 + target_variance / 10))

                # Log selected parameters
                logging.info(
                    f"Chameleon Model Parameters: "
                    f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}"
                )

                # Create RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=self.random_state
                )
            except (TypeError, ValueError) as e:
                logging.error(
                    f"Error creating RandomForestRegressor (Attempt {attempt}/{self.retry_attempts}): {e}"
                )
                time.sleep(self.retry_delay)
            except Exception as e:
                logging.error(
                    f"Unexpected error creating RandomForestRegressor (Attempt {attempt}/{self.retry_attempts}): {e}"
                )
                raise ModelCreationError(
                    message="Failed to create RandomForestRegressor.",
                    details={"attempt": attempt, "error": str(e)}
                )

        # If all retries fail, raise a ModelCreationError
        logging.error("Failed to create RandomForestRegressor after maximum retry attempts.")
        raise ModelCreationError(
            message="Maximum retries reached for creating RandomForestRegressor.",
            details={"retry_attempts": self.retry_attempts}
        )

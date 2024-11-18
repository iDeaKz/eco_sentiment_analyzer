import logging
import time
import numpy as np
import pandas as pd
from typing import Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from custom_exceptions import TrainingError, DataLoaderError
from error_management import retry, secure_error_handler

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class Trainer:
    """
    Handles model training and evaluation with support for small datasets, anomaly detection,
    and synthetic data generation.
    """

    def __init__(self, transformer: Any, model_creator: Any, anomaly_detector: Any):
        """
        Initializes the Trainer with required components.

        :param transformer: Data transformation utility.
        :param model_creator: Model creation utility.
        :param anomaly_detector: Anomaly detection utility.
        """
        if not transformer or not model_creator or not anomaly_detector:
            raise ValueError("Trainer requires valid transformer, model_creator, and anomaly_detector instances.")

        self.transformer = transformer
        self.model_creator = model_creator
        self.anomaly_detector = anomaly_detector
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Trainer initialized successfully.")

    @secure_error_handler
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(TrainingError,))
    def train_and_evaluate(self, data_size: int, noise_level: float) -> float:
        """
        Trains the model and evaluates it on a test dataset.

        :param data_size: Size of the training dataset.
        :param noise_level: Noise level in the dataset.
        :return: Mean squared error (MSE) of the test predictions.
        :raises TrainingError: If training or evaluation fails.
        """
        self.logger.debug(f"Starting training with data_size={data_size}, noise_level={noise_level}.")
        try:
            # Generate synthetic data
            X, y = self.generate_synthetic_data(data_size, noise_level)
            self.logger.info(f"Generated synthetic data with size={data_size}.")

            # Transform and validate data
            X = self.transformer.apply_math_transformations(X)
            self.logger.debug("Applied mathematical transformations to synthetic data.")
            X_train, X_test, y_train, y_test = self.handle_small_datasets(X, y)
            self.logger.info("Handled dataset partitioning.")

            # Normalize features
            X_train = self.transformer.scaler.fit_transform(X_train)
            X_test = self.transformer.scaler.transform(X_test)
            self.logger.debug("Normalized training and test data.")

            # Detect and remove anomalies
            self.anomaly_detector.fit(X_train)
            anomalies = self.anomaly_detector.detect_anomalies(X_test)
            self.logger.info(f"Anomaly detection completed. Detected {sum(anomalies == -1)} anomalies.")

            # Filter out anomalies from training data
            normal_indices = self.anomaly_detector.model.predict(X_train) == 1
            X_train = X_train[normal_indices]
            y_train = y_train[normal_indices]
            self.logger.info(f"Filtered anomalies from training data. Remaining samples: {len(X_train)}.")

            if len(X_train) < 2:
                raise TrainingError(
                    message="Insufficient normal data after filtering anomalies.",
                    details={"remaining_samples": len(X_train)},
                    resolution="Ensure training data is sufficient or reduce anomaly sensitivity."
                )

            # Train the model
            model = self.model_creator.create_model(X_train, y_train)
            model.fit(X_train, y_train)
            self.logger.info("Model training completed successfully.")

            # Evaluate the model
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            self.logger.info(f"Model evaluation completed with Test MSE: {mse}.")
            return mse
        except Exception as e:
            self.logger.error(f"Training and evaluation failed: {e}", exc_info=True)
            raise TrainingError(
                message="Training and evaluation process failed.",
                details={"data_size": data_size, "noise_level": noise_level, "error": str(e)},
                resolution="Review training data, anomaly detection, and model configurations."
            )

    @secure_error_handler
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(DataLoaderError,))
    def generate_synthetic_data(self, size: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates synthetic regression data.

        :param size: Number of samples in the dataset.
        :param noise: Noise level for data generation.
        :return: Tuple of feature matrix X and target vector y.
        :raises DataLoaderError: If data generation fails.
        """
        self.logger.debug(f"Generating synthetic regression data with size={size}, noise={noise}.")
        try:
            X, y = make_regression(
                n_samples=size, n_features=5, noise=noise, random_state=42
            )
            return X, y
        except Exception as e:
            self.logger.error(f"Failed to generate synthetic data: {e}", exc_info=True)
            raise DataLoaderError(
                message="Synthetic data generation failed.",
                details={"size": size, "noise": noise, "error": str(e)},
                resolution="Ensure make_regression parameters are valid and sklearn is installed."
            )

    def handle_small_datasets(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Handles small datasets by duplicating data if necessary or partitioning normally.

        :param X: Feature matrix.
        :param y: Target vector.
        :return: Partitioned training and test data.
        :raises TrainingError: If input data is invalid.
        """
        self.logger.debug("Handling small datasets.")
        try:
            if len(X) < 5:
                X_train, X_test, y_train, y_test = X, X, y, y
                self.logger.warning("Small dataset used for both training and testing.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error handling small datasets: {e}", exc_info=True)
            raise TrainingError(
                message="Error occurred while handling small datasets.",
                details={"dataset_size": len(X)},
                resolution="Verify dataset structure and ensure sufficient data samples."
            )

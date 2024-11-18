import logging
import numpy as np
import pandas as pd
from typing import Any, List, Union
from custom_exceptions import DataTransformationError, InvalidParameterError
from error_management import retry, secure_error_handler

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class DataTransformer:
    """
    Handles data preprocessing and transformation operations.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("DataTransformer initialized.")

    @secure_error_handler
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(DataTransformationError,))
    
    def apply_math_transformations(self, X: np.ndarray) -> np.ndarray:
        """
        Applies mathematical transformations to the input feature matrix.

        :param X: Input feature matrix.
        :return: Transformed feature matrix.
        """
        try:
            return np.sqrt(np.abs(X))  # Example: Square root of absolute values
        except Exception as e:
            raise ValueError(f"Error during mathematical transformations: {e}")
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:

        """
        Cleans the data by handling missing values and duplicates.

        :param data: Input DataFrame.
        :return: Cleaned DataFrame.
        :raises DataTransformationError: If data cleaning fails.
        """
        self.logger.debug("Starting data cleaning.")
        try:
            if not isinstance(data, pd.DataFrame):
                raise InvalidParameterError(
                    message="Invalid input type for data cleaning.",
                    details={"expected_type": "pd.DataFrame", "received_type": type(data).__name__},
                    resolution="Ensure that the input is a pandas DataFrame."
                )

            data = data.drop_duplicates()
            data = data.fillna(method='ffill').fillna(method='bfill')
            self.logger.info("Data cleaned successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}", exc_info=True)
            raise DataTransformationError(
                message="Data cleaning operation failed.",
                details=str(e),
                resolution="Check the input DataFrame for unsupported formats or corrupted data."
            )

    @secure_error_handler
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(DataTransformationError,))
    def scale_features(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Scales numerical features to a range of [0, 1].

        :param data: Input DataFrame.
        :param feature_columns: List of columns to scale.
        :return: DataFrame with scaled features.
        :raises DataTransformationError: If feature scaling fails.
        """
        self.logger.debug(f"Scaling features: {feature_columns}")
        try:
            if not isinstance(data, pd.DataFrame):
                raise InvalidParameterError(
                    message="Invalid input type for feature scaling.",
                    details={"expected_type": "pd.DataFrame", "received_type": type(data).__name__},
                    resolution="Ensure that the input is a pandas DataFrame."
                )
            if not isinstance(feature_columns, list) or not all(isinstance(col, str) for col in feature_columns):
                raise InvalidParameterError(
                    message="Invalid feature column specification.",
                    details={"expected_type": "list of str", "received_type": type(feature_columns).__name__},
                    resolution="Ensure that feature_columns is a list of string column names."
                )

            for column in feature_columns:
                if column in data.columns:
                    min_val = data[column].min()
                    max_val = data[column].max()
                    if min_val == max_val:
                        self.logger.warning(f"Skipping scaling for column '{column}' due to zero range.")
                        data[column] = 0.5  # Assign neutral value for zero-range features
                    else:
                        data[column] = (data[column] - min_val) / (max_val - min_val)

            self.logger.info("Feature scaling completed successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Feature scaling failed: {e}", exc_info=True)
            raise DataTransformationError(
                message="Feature scaling operation failed.",
                details=str(e),
                resolution="Check the feature columns for invalid or non-numerical data."
            )

    @secure_error_handler
    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Validates the DataFrame to ensure required columns are present.

        :param data: Input DataFrame.
        :param required_columns: List of required column names.
        :raises DataTransformationError: If validation fails.
        """
        self.logger.debug(f"Validating data for required columns: {required_columns}")
        try:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise DataTransformationError(
                    message="Data validation failed due to missing columns.",
                    details={"missing_columns": missing_columns},
                    resolution="Ensure that the input DataFrame includes all required columns."
                )
            self.logger.info("Data validation completed successfully.")
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}", exc_info=True)
            raise

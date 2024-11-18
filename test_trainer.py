import pytest
import numpy as np
from unittest.mock import patch
from utils.transformations import DataTransformer
from models.chameleon_rf import ChameleonRandomForest
from models.anomaly_detection import AnomalyDetector
from training.trainer import Trainer
from custom_exceptions import TrainingError


# Fixtures for shared test dependencies
@pytest.fixture
def transformer():
    return DataTransformer()


@pytest.fixture
def model_creator():
    return ChameleonRandomForest()


@pytest.fixture
def anomaly_detector():
    return AnomalyDetector(contamination=0.05)


@pytest.fixture
def trainer(transformer, model_creator, anomaly_detector):
    return Trainer(transformer, model_creator, anomaly_detector)


# Parameterized tests for invalid inputs to `handle_small_datasets`
@pytest.mark.parametrize(
    "X, y, expected_exception",
    [
        ([1, 2, 3], np.array([1.0, 2.0, 3.0]), TypeError),  # X not a numpy array
        (np.random.randn(5, 5), [1.0, 2.0, 3.0, 4.0, 5.0], TypeError),  # y not a numpy array
        (np.random.randn(5), np.array([1.0, 2.0, 3.0, 4.0, 5.0]), ValueError),  # X not 2D
        (np.random.randn(5, 5), np.array([1.0, 2.0]), ValueError),  # Mismatched shapes
    ],
)
def test_handle_small_datasets_invalid_input(trainer, X, y, expected_exception):
    """
    Tests that `handle_small_datasets` raises the appropriate exceptions
    when given invalid inputs.
    """
    with pytest.raises(expected_exception):
        trainer.handle_small_datasets(X, y)


def handle_small_datasets(self, X, y):
    """
    Handles small datasets by duplicating data if necessary or partitioning normally.

    :param X: Feature matrix.
    :param y: Target vector.
    :return: Partitioned training and test data.
    :raises TypeError: If X or y are not numpy arrays.
    :raises ValueError: If X is not 2D or X and y have mismatched shapes.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # Handle small datasets
    if X.shape[0] == 1:
        X = np.vstack([X, X])
        y = np.hstack([y, y])
        return X, X, y, y
    elif X.shape[0] < 5:
        return X, X, y, y
    else:
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=0.2, random_state=42)


# Test anomaly detection
def test_anomaly_detection(trainer):
    """
    Tests the anomaly detection functionality of the `Trainer`.
    """
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    # Inject anomalies
    X[95:] += 100  # Add anomalies to the last 5 samples
    trainer.anomaly_detector.fit(X)

    predictions = trainer.anomaly_detector.detect_anomalies(X)
    anomalies = np.sum(predictions == -1)
    assert anomalies >= 5  # Ensure anomalies are detected
    assert np.sum(predictions == 1) <= 95  # Ensure normal data is not over-flagged


# Mock logging to ensure it functions during training
@patch("training.trainer.logging")
def test_logging(mock_logging, trainer):
    """
    Tests that logging is functioning as expected during the training process.
    """
    trainer.train_and_evaluate(data_size=10, noise_level=1.0)
    assert mock_logging.info.called  # Info logs should be called
    assert not mock_logging.error.called  # No errors expected


# Test training pipeline for exception handling
def test_training_pipeline_error_handling(trainer):
    """
    Tests that the `Trainer` pipeline gracefully handles errors
    by raising the appropriate exceptions.
    """
    with pytest.raises(TrainingError):
        trainer.train_and_evaluate(data_size=-1, noise_level=0.1)


# Extended test for edge-case datasets
@pytest.mark.parametrize(
    "X, y, expected_mse_range",
    [
        (np.zeros((100, 5)), np.zeros(100), (0, 1)),  # Dataset with all zeros
        (np.ones((100, 5)), np.ones(100), (0, 1)),  # Dataset with all ones
        (np.random.randn(100, 5), np.random.randn(100), (0, 10)),  # Normal dataset
    ],
)
def test_edge_case_datasets(trainer, X, y, expected_mse_range):
    """
    Tests `Trainer` with edge-case datasets to ensure robustness.
    """
    X_train, X_test, y_train, y_test = trainer.handle_small_datasets(X, y)
    mse = trainer.train_and_evaluate(data_size=X_train.shape[0], noise_level=0.0)
    assert expected_mse_range[0] <= mse <= expected_mse_range[1]  # MSE within expected range

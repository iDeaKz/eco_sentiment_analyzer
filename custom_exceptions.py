import logging
import traceback

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("errors.log"),
        logging.StreamHandler()
    ]
)

class BaseCustomException(Exception):
    """Base class for all custom exceptions."""
    def __init__(self, message=None, details=None, resolution=None):
        """
        Initialize the exception with a message, optional details, and a possible resolution hint.
        
        :param message: The primary error message.
        :param details: Additional details for debugging or context.
        :param resolution: Suggestion for resolving the error.
        """
        super().__init__(message)
        self.details = details
        self.resolution = resolution
        self.log_exception()

    def log_exception(self):
        """Log exception details for traceability."""
        logging.error(f"{self.__class__.__name__}: {self}")
        if self.details:
            logging.debug(f"Details: {self.details}")
        if self.resolution:
            logging.info(f"Suggested Resolution: {self.resolution}")
        logging.debug(traceback.format_exc())

# Specific custom exceptions with deploy-ready logic
class DataTransformationError(BaseCustomException):
    """Custom exception for data transformation errors."""
    def __init__(self, message="Data transformation failed.", details=None, resolution="Check input data format and transformation logic."):
        super().__init__(message, details, resolution)

class ModelCreationError(BaseCustomException):
    """Custom exception for model creation errors."""
    def __init__(self, message="Model creation failed.", details=None, resolution="Validate model parameters and architecture."):
        super().__init__(message, details, resolution)

class TrainingError(BaseCustomException):
    """Custom exception for training errors."""
    def __init__(self, message="Model training failed.", details=None, resolution="Ensure sufficient data quality and validate hyperparameters."):
        super().__init__(message, details, resolution)

class DataLoaderError(BaseCustomException):
    """Custom exception for data loading errors."""
    def __init__(self, message="Data loading failed.", details=None, resolution="Check dataset paths and accessibility."):
        super().__init__(message, details, resolution)

import logging
import traceback

# Configure logging for errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("errors.log"),
        logging.StreamHandler()
    ]
)

class BaseCustomException(Exception):
    """Base class for all custom exceptions."""
    def __init__(self, message=None, details=None, resolution=None):
        """
        Initialize the exception with a message, optional details, and a possible resolution hint.

        :param message: The primary error message.
        :param details: Additional details for debugging or context.
        :param resolution: Suggestion for resolving the error.
        """
        super().__init__(message)
        self.details = details
        self.resolution = resolution
        self.log_exception()

    def log_exception(self):
        """Log exception details for traceability."""
        logging.error(f"{self.__class__.__name__}: {self}")
        if self.details:
            logging.debug(f"Details: {self.details}")
        if self.resolution:
            logging.info(f"Suggested Resolution: {self.resolution}")
        logging.debug(traceback.format_exc())

# Specific custom exceptions
class DataTransformationError(BaseCustomException):
    """Custom exception for data transformation errors."""
    def __init__(self, message="Data transformation failed.", details=None, resolution="Check input data format and transformation logic."):
        super().__init__(message, details, resolution)

class ModelCreationError(BaseCustomException):
    """Custom exception for model creation errors."""
    def __init__(self, message="Model creation failed.", details=None, resolution="Validate model parameters and architecture."):
        super().__init__(message, details, resolution)

class TrainingError(BaseCustomException):
    """Custom exception for training errors."""
    def __init__(self, message="Model training failed.", details=None, resolution="Ensure sufficient data quality and validate hyperparameters."):
        super().__init__(message, details, resolution)

class DataLoaderError(BaseCustomException):
    """Custom exception for data loading errors."""
    def __init__(self, message="Data loading failed.", details=None, resolution="Check dataset paths and accessibility."):
        super().__init__(message, details, resolution)

# Add more exceptions as needed (e.g., InvalidParameterError, APIConnectionError)

class InvalidParameterError(BaseCustomException):
    """Custom exception for invalid parameter errors."""
    def __init__(self, message="Invalid parameter provided.", details=None, resolution="Verify the input parameters against expected values."):
        super().__init__(message, details, resolution)

class APIConnectionError(BaseCustomException):
    """Custom exception for API connection errors."""
    def __init__(self, message="Failed to connect to the API.", details=None, resolution="Check API endpoint and network connectivity."):
        super().__init__(message, details, resolution)

class TimeoutError(BaseCustomException):
    """Custom exception for timeout errors."""
    def __init__(self, message="Operation timed out.", details=None, resolution="Increase timeout duration or check system responsiveness."):
        super().__init__(message, details, resolution)

class SecurityViolationError(BaseCustomException):
    """Custom exception for security policy violations."""
    def __init__(self, message="Security violation detected.", details=None, resolution="Review security policies and access controls."):
        super().__init__(message, details, resolution)

class DependencyError(BaseCustomException):
    """Custom exception for missing or failed dependencies."""
    def __init__(self, message="Missing or failed dependency.", details=None, resolution="Install and verify the required dependencies."):
        super().__init__(message, details, resolution)

class FileProcessingError(BaseCustomException):
    """Custom exception for file processing errors."""
    def __init__(self, message="File processing failed.", details=None, resolution="Check file format, accessibility, and permissions."):
        super().__init__(message, details, resolution)

class ResourceAllocationError(BaseCustomException):
    """Custom exception for resource allocation errors."""
    def __init__(self, message="Resource allocation failed.", details=None, resolution="Ensure sufficient resources (CPU, memory, etc.) are available."):
        super().__init__(message, details, resolution)

# Add additional exceptions as necessary

import matplotlib.pyplot as plt
import numpy as np
import logging
from custom_exceptions import VisualizationError
from error_management import retry, secure_error_handler

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class RealTimeVisualization:
    """
    Handles real-time visualization of metrics like MSE progression.
    """

    def __init__(self):
        """
        Initializes the RealTimeVisualization instance.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("RealTimeVisualization initialized successfully.")

    @secure_error_handler
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(VisualizationError,))
    def plot_mse_progression(self, mse_progression: list):
        """
        Plots the progression of MSE over time.

        :param mse_progression: List of tuples (level, mse).
        :raises VisualizationError: If plotting fails.
        """
        self.logger.debug("Starting MSE progression plotting.")
        try:
            # Validate input
            if not mse_progression or not isinstance(mse_progression, list):
                raise VisualizationError(
                    message="Invalid input for MSE progression plotting.",
                    details={"received_type": type(mse_progression).__name__, "value": mse_progression},
                    resolution="Ensure that mse_progression is a list of (level, mse) tuples."
                )

            levels, mses = zip(*mse_progression)
            if not all(isinstance(level, (int, float)) and isinstance(mse, (int, float)) for level, mse in mse_progression):
                raise VisualizationError(
                    message="MSE progression data contains invalid types.",
                    details={"invalid_data": mse_progression},
                    resolution="Ensure all elements in mse_progression are numeric tuples (level, mse)."
                )

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(levels, mses, marker='o', label="MSE Progression", color="blue")
            plt.title("MSE Progression Over Levels")
            plt.xlabel("Level")
            plt.ylabel("Mean Squared Error (MSE)")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

            self.logger.info("MSE progression plot displayed successfully.")
        except Exception as e:
            self.logger.error(f"Failed to plot MSE progression: {e}", exc_info=True)
            raise VisualizationError(
                message="Failed to plot MSE progression.",
                details=str(e),
                resolution="Check the input data and ensure Matplotlib is properly installed."
            )

    @secure_error_handler
    def save_plot(self, mse_progression: list, filename: str = "mse_progression.png"):
        """
        Saves the progression of MSE over time as a PNG image.

        :param mse_progression: List of tuples (level, mse).
        :param filename: Name of the file to save the plot.
        :raises VisualizationError: If saving the plot fails.
        """
        self.logger.debug(f"Saving MSE progression plot to {filename}.")
        try:
            # Validate input
            if not filename or not filename.endswith(".png"):
                raise VisualizationError(
                    message="Invalid filename for saving plot.",
                    details={"filename": filename},
                    resolution="Ensure the filename ends with '.png'."
                )

            self.plot_mse_progression(mse_progression)

            plt.savefig(filename, format="png")
            self.logger.info(f"MSE progression plot saved successfully as {filename}.")
        except Exception as e:
            self.logger.error(f"Failed to save MSE progression plot: {e}", exc_info=True)
            raise VisualizationError(
                message="Failed to save MSE progression plot.",
                details=str(e),
                resolution="Ensure the directory is writable and the filename is valid."
            )

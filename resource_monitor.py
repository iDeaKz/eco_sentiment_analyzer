import logging
import psutil
from custom_exceptions import ResourceAllocationError
from error_management import retry, secure_error_handler

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class ResourceMonitor:
    """
    Monitors system resources like CPU, memory, and disk usage.
    """

    def __init__(self, context: str):
        """
        Initializes the ResourceMonitor for a specific context.

        :param context: A descriptive name for the monitoring context (e.g., "Training Step 1").
        """
        self.context = context
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"ResourceMonitor initialized for context: {self.context}")

    @secure_error_handler
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(ResourceAllocationError,))
    def log_usage(self) -> None:
        """
        Logs current CPU, memory, and disk usage.

        :raises ResourceAllocationError: If resource usage cannot be monitored.
        """
        try:
            # Capture system resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')

            # Log resource usage details
            self.logger.info(
                f"{self.context} - CPU Usage: {cpu_percent}%, "
                f"Memory Usage: {memory_info.percent}%, "
                f"Disk Usage: {disk_usage.percent}%"
            )

            # Warn if critical thresholds are exceeded
            if cpu_percent > 90:
                self.logger.warning(f"{self.context} - High CPU Usage: {cpu_percent}%")
            if memory_info.percent > 90:
                self.logger.warning(f"{self.context} - High Memory Usage: {memory_info.percent}%")
            if disk_usage.percent > 90:
                self.logger.warning(f"{self.context} - High Disk Usage: {disk_usage.percent}%")
        except Exception as e:
            self.logger.error(f"Resource monitoring failed: {e}", exc_info=True)
            raise ResourceAllocationError(
                message="Failed to monitor system resources.",
                details={"context": self.context, "error": str(e)},
                resolution="Ensure psutil is correctly installed and the system is accessible."
            )

    @secure_error_handler
    def check_critical_resources(self, cpu_threshold: float = 90.0, memory_threshold: float = 90.0) -> bool:
        """
        Checks if critical resource thresholds are exceeded.

        :param cpu_threshold: Threshold for CPU usage as a percentage.
        :param memory_threshold: Threshold for memory usage as a percentage.
        :return: True if resources are within limits, False otherwise.
        :raises ResourceAllocationError: If resource usage cannot be monitored.
        """
        self.logger.debug(
            f"Checking critical resources with CPU threshold={cpu_threshold}% and memory threshold={memory_threshold}%."
        )
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()

            if cpu_percent > cpu_threshold or memory_info.percent > memory_threshold:
                self.logger.warning(
                    f"Critical resource thresholds exceeded - CPU: {cpu_percent}%, Memory: {memory_info.percent}%"
                )
                return False

            self.logger.info("Critical resource usage is within acceptable limits.")
            return True
        except Exception as e:
            self.logger.error(f"Critical resource check failed: {e}", exc_info=True)
            raise ResourceAllocationError(
                message="Failed to check critical system resources.",
                details={"cpu_threshold": cpu_threshold, "memory_threshold": memory_threshold, "error": str(e)},
                resolution="Ensure psutil is correctly installed and the system is accessible."
            )

    def monitor_resources(self, interval: int = 5, duration: int = 60) -> None:
        """
        Continuously monitors and logs resource usage over a specified duration.

        :param interval: Time interval (in seconds) between resource checks.
        :param duration: Total duration (in seconds) for monitoring.
        """
        self.logger.debug(
            f"Starting resource monitoring for duration={duration}s with interval={interval}s."
        )
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                self.log_usage()
                time.sleep(interval)
        except Exception as e:
            self.logger.error(f"Continuous resource monitoring failed: {e}", exc_info=True)
            raise

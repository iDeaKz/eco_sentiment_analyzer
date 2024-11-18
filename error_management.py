import time
import logging

# Retry decorator with exponential backoff
def retry(max_retries=3, delay=2, backoff=2, exceptions=(Exception,)):
    """
    Retry decorator with exponential backoff.

    :param max_retries: Maximum number of retries.
    :param delay: Initial delay between retries in seconds.
    :param backoff: Multiplier for delay on each retry.
    :param exceptions: Tuple of exceptions to catch and retry.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    logging.debug(f"Attempting {func.__name__} (try {retries + 1})")
                    result = func(*args, **kwargs)
                    logging.info(f"Successful execution of {func.__name__}")
                    return result
                except exceptions as e:
                    retries += 1
                    logging.warning(f"Retry {retries}/{max_retries} for {func.__name__}: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    if retries == max_retries:
                        logging.error(f"Max retries reached for {func.__name__}. Raising exception.")
                        raise
        return wrapper
    return decorator

# Secure error handler
def secure_error_handler(func):
    """
    Secure error handler decorator with logging.

    :param func: Function to wrap.
    """
    def wrapper(*args, **kwargs):
        try:
            logging.debug(f"Starting execution of {func.__name__}")
            result = func(*args, **kwargs)
            logging.debug(f"Finished execution of {func.__name__}")
            return result
        except Exception as e:
            logging.critical(f"Unhandled exception in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper

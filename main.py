# main.py

import asyncio
import logging
import time
from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    CHECKPOINT_INTERVAL,
    TOTAL_LEVELS,
    MAX_DATA_SIZE,
    MAX_NOISE_LEVEL,
    ANOMALY_CONTAMINATION
)
from utils.resource_monitor import ResourceMonitor
from utils.transformations import DataTransformer
from models.chameleon_rf import ChameleonRandomForest
from models.anomaly_detection import AnomalyDetector
from training.trainer import Trainer
from custom_exceptions import TrainingError

def setup_logging():
    """Sets up logging configuration."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    logging.debug("Logging configured successfully.")

async def asynchronous_training(
    trainer: Trainer, 
    level: int, 
    data_size: int, 
    noise_level: float
) -> tuple:
    """Handles asynchronous training for a single level with error handling and retries."""
    attempts = 0
    max_attempts = 2
    while attempts < max_attempts:
        try:
            # Log resource usage before training
            monitor_before = ResourceMonitor(f"Before training on level {level}")
            monitor_before.log_usage()

            # Record start time
            start_time = time.time()

            # Train and evaluate
            mse = trainer.train_and_evaluate(data_size, noise_level)

            # Record end time
            end_time = time.time()

            # Log resource usage after training
            monitor_after = ResourceMonitor(f"After training on level {level}")
            monitor_after.log_usage()

            # Log training completion
            logging.info(
                f"Training on level {level} completed in {end_time - start_time:.2f} seconds with Test MSE: {mse}"
            )

            return (level, mse)
        except RuntimeError as e:
            attempts += 1
            delay = 0.5 * attempts  # Example: linear backoff
            logging.error(f"RuntimeError at level {level}: {e}. Retrying after {delay:.2f} seconds... ({attempts}/{max_attempts})")
            await asyncio.sleep(delay)
        except TrainingError as e:
            logging.error(f"TrainingError at level {level}: {e}.")
            return (level, None)
        except Exception as e:
            logging.error(f"Unexpected error at level {level}: {e}")
            return (level, None)
    else:
        logging.error(f"Failed at level {level} after multiple attempts.")
        return (level, None)

async def main_progressive_learning():
    """Runs the model on datasets of increasing difficulty asynchronously with error handling."""
    try:
        # Initialize components
        transformer = DataTransformer()
        model_creator = ChameleonRandomForest()
        anomaly_detector = AnomalyDetector(contamination=ANOMALY_CONTAMINATION)
        trainer = Trainer(transformer, model_creator, anomaly_detector)
        logging.debug("Initialized Transformer, Model Creator, Anomaly Detector, and Trainer.")

        mse_progression = []

        tasks = []
        for level in range(1, TOTAL_LEVELS + 1):
            # Incrementally increase dataset size and noise
            data_size = min(level, MAX_DATA_SIZE)
            noise_level = min(0.001 * level, MAX_NOISE_LEVEL)

            # Create a task for asynchronous training
            task = asyncio.create_task(asynchronous_training(trainer, level, data_size, noise_level))
            tasks.append(task)
            logging.debug(f"Created training task for level {level} with data_size={data_size}, noise_level={noise_level}.")

            # Control the number of concurrent tasks to prevent resource exhaustion
            if len(tasks) >= 10:  # Adjust concurrency level as needed
                completed, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in completed:
                    result = t.result()
                    if result[1] is not None:
                        mse_progression.append(result)
                        logging.debug(f"Appended MSE for level {result[0]}: {result[1]}")
                tasks = list(pending)

        # Await remaining tasks
        for task in asyncio.as_completed(tasks):
            result = await task
            if result[1] is not None:
                mse_progression.append(result)
                logging.debug(f"Appended MSE for level {result[0]}: {result[1]}")

        # Log MSE progression at the end
        for level, mse in sorted(mse_progression, key=lambda x: x[0]):
            logging.info(f"Level: {level} -> Test MSE: {mse}")

    except Exception as e:
        logging.error(f"An error occurred during progressive learning: {e}")

if __name__ == "__main__":
    setup_logging()
    try:
        asyncio.run(main_progressive_learning())
    except KeyboardInterrupt:
        logging.info("Progressive learning interrupted by user.")
    except Exception as e:
        logging.critical(f"Unhandled exception in main: {e}")

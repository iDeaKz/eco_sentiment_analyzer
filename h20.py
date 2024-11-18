import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd
from sklearn.base import BaseEstimator
from datetime import datetime
import os
from typing import List, Type, Union, Optional
from collections import deque
import joblib
import time

# Set up logging configuration for console and file output
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)

# File handler
f_handler = logging.FileHandler('eco_sentiment_analyzer.log')
f_handler.setLevel(logging.DEBUG)
f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

class EcoSentimentAnalyzer:
    """
    An adaptive sentiment analysis model with visualization capabilities.
    """

    def __init__(self,
                 initial_data: List[str],
                 initial_labels: Optional[List[float]] = None,
                 model_class: Type[BaseEstimator] = BaseEstimator,
                 dynamic_threshold: float = 0.1,
                 learning_rate: float = 0.05,
                 rolling_window_size: int = 100,
                 retrain_frequency: int = 100,
                 save_stats_frequency: int = 1000,
                 outlier_deviation_threshold: float = 1.5,
                 max_retries: int = 3,
                 retry_delay: float = 2.0,
                 realization_threshold: float = 0.8,
                 ignore_threshold: float = 0.2) -> None:
        """
        Initialize the EcoSentimentAnalyzer with initial training data.
        """
        self.data = deque(maxlen=rolling_window_size)
        self.labels = deque(maxlen=rolling_window_size) if initial_labels else None
        self.model_class = model_class
        self.model = self._train_model(initial_data, initial_labels)
        self.dynamic_threshold = dynamic_threshold
        self.learning_rate = learning_rate
        self.rolling_window_size = rolling_window_size
        self.retrain_frequency = retrain_frequency
        self.save_stats_frequency = save_stats_frequency
        self.outlier_deviation_threshold = outlier_deviation_threshold
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.new_data_count = 0
        self.feedback_buffer = []
        self.realization_threshold = realization_threshold
        self.ignore_threshold = ignore_threshold
        self._baseline_sentiment = self._calculate_rolling_baseline()
        self.stats = {
            'predictions': [],
            'thresholds': [],
            'deviations': [],
            'baseline_sentiment': [],
            'accuracy': [],
            'quality_scores': [],
            'stability_scores': []
        }
        self._create_directories()

    # Include the other methods (_train_model, process_new_data, etc.) here with appropriate modifications

    def process_new_data(self, new_text: str) -> Union[float, None]:
        """
        Processes a new text data point, predicts sentiment, and updates stats.
        """
        try:
            prediction = self.model.predict([new_text])[0]
            deviation = prediction - self._baseline_sentiment

            # Outlier Detection
            if self._is_outlier(deviation):
                logger.warning(f"Outlier detected. Original Deviation: {deviation}")
                deviation = np.clip(deviation, -self.outlier_deviation_threshold, self.outlier_deviation_threshold)
                logger.info(f"Deviation after clipping: {deviation}")

            # Compute quality score (e.g., absolute deviation)
            quality_score = abs(deviation)
            self.stats['quality_scores'].append(quality_score)

            # Compute stability score (e.g., change in deviation)
            if len(self.stats['deviations']) > 0:
                stability_score = abs(deviation - self.stats['deviations'][-1])
            else:
                stability_score = 0.0
            self.stats['stability_scores'].append(stability_score)

            # Update stats
            self.stats['predictions'].append(prediction)
            self.stats['deviations'].append(deviation)
            self.stats['thresholds'].append(self.dynamic_threshold)
            self.stats['baseline_sentiment'].append(self._baseline_sentiment)
            logger.debug(
                f"Prediction: {prediction}, Deviation: {deviation}, Threshold: {self.dynamic_threshold}"
            )

            # Adjust dynamic threshold
            if abs(deviation) > self.dynamic_threshold:
                self.dynamic_threshold = max(
                    0.05, min(0.5, self.dynamic_threshold + self.learning_rate * deviation)
                )
                logger.info(f"Dynamic threshold adjusted to {self.dynamic_threshold}")

            # Update data and labels
            self.data.append(new_text)
            if self.labels is not None:
                self.labels.append(prediction)
            self._baseline_sentiment = self._calculate_rolling_baseline()
            self.new_data_count += 1

            # Retrain model if needed
            if self.new_data_count >= self.retrain_frequency:
                self.model = self._train_model(list(self.data), list(self.labels) if self.labels else None)
                self.new_data_count = 0
                logger.info("Model retrained based on retrain frequency.")

            # Save stats periodically
            if len(self.stats['predictions']) % self.save_stats_frequency == 0:
                self._save_stats_to_file()

            return prediction
        except Exception as e:
            logger.error("Prediction failed.", exc_info=True)
            return None

    # Other methods remain the same, with logging adjusted to use 'logger' instead of 'logging'

    # Visualization method
    def plot_eagle_eye_overview(self):
        """
        Generates a comprehensive visualization of the model's performance.
        """
        # Initialize a new figure with a large size for panoramic visualization
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Data Validation
        required_stats = ['predictions', 'baseline_sentiment', 'thresholds', 'quality_scores', 'stability_scores']
        for stat in required_stats:
            if stat not in self.stats or len(self.stats[stat]) == 0:
                logger.error(f"Missing or empty '{stat}' in stats.")
                raise ValueError(f"Missing or empty '{stat}' in stats.")
        
        # Ensure all lists have the same length
        stats_lengths = [len(self.stats[stat]) for stat in required_stats]
        if len(set(stats_lengths)) != 1:
            logger.error("Stats lists have mismatched lengths.")
            raise ValueError("Stats lists have mismatched lengths.")
        
        # Generate a time series for the x-axis
        time_series = np.arange(len(self.stats['predictions']))

        # Plot Sentiment Predictions
        ax.plot(time_series, self.stats['predictions'], label="Sentiment Predictions", linewidth=1.5)
        
        # Plot Baseline Sentiment
        ax.plot(time_series, self.stats['baseline_sentiment'], label="Baseline Sentiment", linestyle="--", linewidth=1.5)

        # Threshold Range Shading
        upper_threshold = np.array(self.stats['baseline_sentiment']) + np.array(self.stats['thresholds'])
        lower_threshold = np.array(self.stats['baseline_sentiment']) - np.array(self.stats['thresholds'])
        ax.fill_between(time_series, lower_threshold, upper_threshold, color='orange', alpha=0.2, label="Dynamic Threshold Range")

        # Realization and Ignored Events
        realization_events = [i for i, score in enumerate(self.stats['quality_scores']) if score > self.realization_threshold]
        ignored_events = [i for i, score in enumerate(self.stats['quality_scores']) if score < self.ignore_threshold]
        
        if realization_events:
            ax.scatter(realization_events, [self.stats['predictions'][i] for i in realization_events],
                       color='green', label='Realization Event', marker='o')
        else:
            logger.info("No realization events detected above the 'realization_threshold'.")

        if ignored_events:
            ax.scatter(ignored_events, [self.stats['predictions'][i] for i in ignored_events],
                       color='red', label='Ignored Event', marker='x')
        else:
            logger.info("No ignored events detected below the 'ignore_threshold'.")

        # Volatility Zones
        stability_threshold = 0.05  # Define stability threshold
        volatility_zones = [i for i, stability in enumerate(self.stats['stability_scores']) if stability > stability_threshold]
        
        if volatility_zones:
            for start, end in self.find_continuous_regions(volatility_zones):
                ax.axvspan(start, end, color='gray', alpha=0.3, label="High Volatility Zone" if start == volatility_zones[0] else "")
        else:
            logger.info("No high volatility zones detected above the stability threshold.")

        # Annotation of Baseline Adjustments
        for i, (baseline, threshold) in enumerate(zip(self.stats['baseline_sentiment'], self.stats['thresholds'])):
            if i % 50 == 0:  # Adjust annotation frequency as needed
                ax.annotate(f"{threshold:.2f}", (i, baseline + threshold), textcoords="offset points", xytext=(0, 10), ha='center')

        # Final adjustments for plot aesthetics
        ax.set_title("Eagle Eye Overview of Adaptive Sentiment Analysis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Sentiment Value")
        ax.legend()

        # Display the plot
        plt.show()

    def find_continuous_regions(self, indices):
        """
        Groups consecutive indices into start-end pairs for shading regions.
        """
        if not indices:
            logger.info("No volatility zones detected, skipping continuous region shading.")
            return []
        
        regions = []
        start = indices[0]
        
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                regions.append((start, indices[i - 1]))
                start = indices[i]
        regions.append((start, indices[-1]))
        
        return regions

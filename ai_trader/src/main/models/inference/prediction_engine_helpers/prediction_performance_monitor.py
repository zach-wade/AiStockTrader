# File: src/ai_trader/models/inference/prediction_engine_helpers/prediction_performance_monitor.py

import logging
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PredictionPerformanceMonitor:
    """
    Monitors and tracks performance metrics for model predictions and
    feature retrieval. Provides aggregated statistics (mean, median, percentiles).
    """

    def __init__(self, max_history_size: int = 1000):
        """
        Initializes the PredictionPerformanceMonitor.

        Args:
            max_history_size: Maximum number of recent prediction/feature times to store.
        """
        self.prediction_times: deque[float] = deque(maxlen=max_history_size) # Stores prediction latencies in seconds
        self.feature_times: deque[float] = deque(maxlen=max_history_size)    # Stores feature retrieval latencies in seconds
        logger.debug(f"PredictionPerformanceMonitor initialized with max_history_size: {max_history_size}")

    def record_prediction_latency(self, duration_seconds: float):
        """Records the time taken for a single prediction."""
        self.prediction_times.append(duration_seconds)
        logger.debug(f"Recorded prediction latency: {duration_seconds:.4f}s")

    def record_feature_latency(self, duration_seconds: float):
        """Records the time taken for feature retrieval."""
        self.feature_times.append(duration_seconds)
        logger.debug(f"Recorded feature latency: {duration_seconds:.4f}s")

    def get_performance_stats(self, current_cache_size: int = 0, models_loaded_count: int = 0) -> Dict[str, Any]:
        """
        Retrieves aggregated performance statistics.

        Args:
            current_cache_size: The current size of the feature cache (for reporting).
            models_loaded_count: The number of models currently loaded (for reporting).

        Returns:
            A dictionary containing various performance metrics.
        """
        def calculate_stats(times_deque: deque) -> Dict[str, float]:
            if not times_deque:
                return {'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0}
            
            times_array = np.array(list(times_deque))
            return {
                'mean': float(np.mean(times_array)),
                'median': float(np.median(times_array)),
                'p95': float(np.percentile(times_array, 95)),
                'p99': float(np.percentile(times_array, 99))
            }

        return {
            'prediction_latencies_seconds': calculate_stats(self.prediction_times),
            'feature_retrieval_latencies_seconds': calculate_stats(self.feature_times),
            'current_feature_cache_size': current_cache_size,
            'models_loaded_count': models_loaded_count,
            'total_predictions_tracked': len(self.prediction_times)
        }

    def clear_all_latencies(self):
        """Clears all recorded prediction and feature latencies."""
        self.prediction_times.clear()
        self.feature_times.clear()
        logger.info("All prediction and feature latencies cleared.")
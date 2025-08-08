# File: src/ai_trader/models/monitoring/monitor_helpers/prediction_data_collector.py

import logging
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class PredictionDataCollector:
    """
    Collects and manages historical prediction and feature data for monitoring.
    Maintains time-windowed deques to store recent data.
    """

    def __init__(self, history_maxlen: int = 10000, feature_history_maxlen: int = 1000):
        """
        Initializes the PredictionDataCollector.

        Args:
            history_maxlen: Maximum number of prediction records to keep.
            feature_history_maxlen: Maximum number of individual feature values to keep per feature.
        """
        # Stores full prediction records: {model_id: deque[Dict]}
        self.prediction_history: Dict[str, deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=history_maxlen))
        
        # Stores individual feature values: {f"{model_id}_{feature_name}": deque[value]}
        self.feature_history: Dict[str, deque[Any]] = defaultdict(lambda: deque(maxlen=feature_history_maxlen))
        
        logger.debug(f"PredictionDataCollector initialized. Prediction history maxlen: {history_maxlen}, Feature history maxlen: {feature_history_maxlen}")

    def record_prediction(self, model_id: str, features: Dict[str, float], 
                          prediction: float, actual: Optional[float] = None):
        """
        Records a new prediction data point, including features and actual outcome if available.
        Updates internal history deques.

        Args:
            model_id: The ID of the model that made the prediction.
            features: Dictionary of feature values used for the prediction.
            prediction: The predicted value from the model.
            actual: Optional. The actual outcome observed (for performance calculation).
        """
        record = {
            'timestamp': datetime.now(timezone.utc), # Ensure UTC for consistency
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'error': None # Calculated later by performance calculator if needed
        }
        
        self.prediction_history[model_id].append(record)
        
        # Update feature value history for drift detection
        for feature, value in features.items():
            self.feature_history[f"{model_id}_{feature}"].append(value)
        
        logger.debug(f"Recorded prediction for {model_id}. History size: {len(self.prediction_history[model_id])}")

    def get_recent_predictions(self, model_id: str, window_hours: int, min_predictions: int) -> List[Dict[str, Any]]:
        """
        Retrieves recent prediction records (within a time window and minimum count)
        for performance calculation.

        Args:
            model_id: The ID of the model.
            window_hours: The time window in hours to look back.
            min_predictions: Minimum number of predictions required for valid data.

        Returns:
            A list of recent prediction records, or empty list if criteria not met.
        """
        history = list(self.prediction_history[model_id]) # Convert deque to list for slicing/filtering
        
        if len(history) < min_predictions:
            logger.debug(f"Insufficient prediction history ({len(history)} < {min_predictions}) for model {model_id}.")
            return []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        recent_predictions = [h for h in history if h['timestamp'] > cutoff_time]
        
        if len(recent_predictions) < min_predictions:
            logger.debug(f"Insufficient recent predictions ({len(recent_predictions)} < {min_predictions}) for model {model_id}.")
            return []
        
        logger.debug(f"Retrieved {len(recent_predictions)} recent predictions for {model_id}.")
        return recent_predictions

    def get_feature_history_for_drift(self, model_id: str, feature_name: str, min_points: int) -> List[Any]:
        """
        Retrieves the historical values for a specific feature for drift detection.

        Args:
            model_id: The ID of the model.
            feature_name: The name of the feature.
            min_points: Minimum data points required.

        Returns:
            A list of feature values, or empty list if criteria not met.
        """
        feature_key = f"{model_id}_{feature_name}"
        history = list(self.feature_history[feature_key])
        
        if len(history) < min_points:
            logger.debug(f"Insufficient feature history ({len(history)} < {min_points}) for feature {feature_name} of model {model_id}.")
            return []
        
        return history

    def clean_old_data(self, cutoff_days: int = 7):
        """
        Cleans old prediction and feature history data beyond a specified cutoff.
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=cutoff_days)
        
        # Clean prediction history
        for model_id in list(self.prediction_history.keys()):
            history = self.prediction_history[model_id]
            while history and history[0]['timestamp'] < cutoff_time:
                history.popleft()
            if not history: # Remove empty deques
                del self.prediction_history[model_id]
        
        # Clean feature history
        for feature_key in list(self.feature_history.keys()):
            history = self.feature_history[feature_key]
            # Assumes values in feature_history are just numbers, not records with timestamps
            # If feature_history stored timestamps, more complex logic is needed.
            # Given it stores 'value', we rely on `maxlen` to manage size.
            # Time-based cleaning is harder here without timestamps per value.
            # For this context, `maxlen` is the primary size management.
            pass # Maxlen handles this implicitly. If time-based prune, feature_history needs timestamps
        
        logger.debug(f"Cleaned old monitoring data older than {cutoff_days} days (for prediction_history).")
# File: src/ai_trader/models/monitoring/monitor_helpers/performance_calculator.py

import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PerformanceCalculator:
    """
    Calculates various performance metrics for model predictions,
    such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
    Mean Absolute Percentage Error (MAPE), and directional accuracy.
    """

    def __init__(self):
        logger.debug("PerformanceCalculator initialized.")

    def calculate_metrics(self, recent_predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculates performance metrics from a list of recent prediction records.
        Each record must contain 'prediction' and 'actual'.

        Args:
            recent_predictions: A list of dictionaries, each with 'prediction' and 'actual' values.

        Returns:
            A dictionary of calculated performance metrics (e.g., MAE, RMSE, MAPE).
            Returns empty dict if unable to calculate.
        """
        metrics: Dict[str, float] = {}
        
        # Filter for predictions where both 'prediction' and 'actual' are valid numbers
        valid_data = [
            r for r in recent_predictions 
            if r.get('prediction') is not None and np.isfinite(r.get('prediction')) and
               r.get('actual') is not None and np.isfinite(r.get('actual'))
        ]

        if not valid_data:
            logger.warning("No valid prediction-actual pairs to calculate performance metrics.")
            return {'status': 'no_valid_data', 'count': 0}
        
        predictions = np.array([r['prediction'] for r in valid_data])
        actuals = np.array([r['actual'] for r in valid_data])
        
        if len(predictions) < 2: # Need at least 2 points for some metrics like directional accuracy consistency
            logger.debug("Too few valid data points for comprehensive performance metrics. Returning basic stats.")
            return {'status': 'insufficient_points', 'count': len(predictions)}

        errors = np.abs(predictions - actuals)
        
        metrics['mae'] = float(np.mean(errors))
        metrics['rmse'] = float(np.sqrt(np.mean(np.square(errors))))
        
        # Avoid division by zero in MAPE if actual is 0. Handle potential warnings.
        # Use a small epsilon to prevent division by zero or large values near zero.
        with np.errstate(divide='ignore', invalid='ignore'):
            percentage_errors = np.abs((actuals - predictions) / (actuals + np.finfo(float).eps))
            metrics['mape'] = float(np.mean(percentage_errors)) * 100 # In percentage form
            if np.isinf(metrics['mape']) or np.isnan(metrics['mape']):
                 metrics['mape'] = 0.0 # Handle cases where MAPE blows up for zero actuals


        # Directional accuracy (assuming sign prediction)
        # Note: np.sign(0) is 0. If both are 0, it's 0==0 -> True.
        metrics['directional_accuracy'] = float(np.mean(np.sign(predictions) == np.sign(actuals)))
        metrics['count'] = len(valid_data)
        metrics['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
        
        logger.debug(f"Calculated performance metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}.")
        return metrics

    def check_performance_degradation(self, current_mae: float, baseline_mae: float, degradation_threshold: float = 0.20) -> bool:
        """
        Checks if current performance (MAE) has degraded significantly compared to a baseline.

        Args:
            current_mae: The current Mean Absolute Error.
            baseline_mae: The baseline (e.g., training) Mean Absolute Error.
            degradation_threshold: The percentage increase in MAE that constitutes degradation (e.g., 0.20 for 20%).

        Returns:
            True if performance has degraded beyond the threshold, False otherwise.
        """
        if not np.isfinite(current_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
            logger.warning(f"Invalid MAE values for degradation check: current={current_mae}, baseline={baseline_mae}. Cannot determine degradation.")
            return False
            
        degradation = (current_mae - baseline_mae) / baseline_mae
        is_degraded = degradation > degradation_threshold
        logger.debug(f"Performance degradation check: current MAE={current_mae:.4f}, baseline MAE={baseline_mae:.4f}, degradation={degradation:.2%}. Degraded: {is_degraded}.")
        return is_degraded
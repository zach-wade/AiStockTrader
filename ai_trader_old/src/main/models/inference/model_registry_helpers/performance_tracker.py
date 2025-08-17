# File: src/ai_trader/models/inference/model_registry_helpers/performance_tracker.py

# Standard library imports
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np

# Local imports
# Corrected absolute import for ModelVersion
from main.models.inference.model_registry_types import ModelVersion

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks and updates the real-time or rolling performance metrics of deployed models.
    Updates ModelVersion objects with consolidated metrics.
    """

    def __init__(self, ema_alpha: float = 0.1):
        """
        Initializes the PerformanceTracker.

        Args:
            ema_alpha: The smoothing factor (alpha) for Exponential Moving Average (EMA)
                       when updating rolling metrics (0.0 to 1.0).
        """
        self.ema_alpha = ema_alpha
        # In a real system, performance history might be stored externally (e.g., DB, time-series DB)
        # For this component, it primarily pushes to ModelVersion's internal metrics dict.
        # If storing history, would need a mechanism here:
        # self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list) # {model_id_version: [metrics_snapshot]}
        logger.debug(f"PerformanceTracker initialized with EMA alpha: {ema_alpha}")

    def update_model_performance(
        self, model_version: ModelVersion, performance_data: Dict[str, float]
    ):
        """
        Updates the performance metrics of a ModelVersion object based on new incoming data.
        Applies Exponential Moving Average (EMA) for rolling metrics.

        Args:
            model_version: The ModelVersion object to update.
            performance_data: A dictionary of new performance metrics (e.g., {'accuracy': 0.92, 'sharpe': 1.5}).
        """
        if not model_version:
            logger.warning("Attempted to update performance for a None ModelVersion object.")
            return

        for metric_name, new_value in performance_data.items():
            if (
                not isinstance(new_value, (int, float))
                or np.isnan(new_value)
                or np.isinf(new_value)
            ):
                logger.warning(
                    f"Skipping non-numeric or invalid value for metric '{metric_name}': {new_value}."
                )
                continue

            # Update existing metrics using EMA
            if metric_name in model_version.metrics:
                model_version.metrics[metric_name] = (
                    self.ema_alpha * new_value
                    + (1 - self.ema_alpha) * model_version.metrics[metric_name]
                )
            else:
                # Add new metric directly
                model_version.metrics[metric_name] = new_value
            logger.debug(
                f"Updated '{model_version.model_id}' v'{model_version.version}' metric '{metric_name}' to {model_version.metrics[metric_name]:.4f}."
            )

        # Optional: Add a timestamped snapshot to a more granular performance history if needed
        # self.performance_history[f"{model_version.model_id}_{model_version.version}"].append({
        #     'timestamp': datetime.now(timezone.utc).isoformat(),
        #     **performance_data
        # })
        logger.info(
            f"Performance updated for model '{model_version.model_id}' version '{model_version.version}'."
        )

    # Optional: Method to retrieve historical performance snapshots if collected
    # def get_performance_history(self, model_id: str, version: str, limit: int = 100) -> List[Dict[str, Any]]:
    #     """Retrieves historical performance snapshots for a specific model version."""
    #     return self.performance_history.get(f"{model_id}_{version}", [])[-limit:]

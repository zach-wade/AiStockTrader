# File: src/ai_trader/models/monitoring/monitor_helpers/drift_detector.py

# Standard library imports
import logging
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects statistical drift in model predictions and feature distributions
    using methods like Kolmogorov-Smirnov test.
    """

    def __init__(self, drift_threshold: float = 0.05):
        """
        Initializes the DriftDetector.

        Args:
            drift_threshold: The threshold for the drift statistic (e.g., KS statistic)
                             above which drift is considered detected.
        """
        self.drift_threshold = drift_threshold
        logger.debug(f"DriftDetector initialized with drift_threshold: {drift_threshold}")

    def calculate_prediction_drift(
        self, prediction_history: List[float], min_predictions_for_drift: int
    ) -> Tuple[float, Optional[float]]:
        """
        Calculates prediction distribution drift using the Kolmogorov-Smirnov (KS) test.
        Compares two halves of the prediction history.

        Args:
            prediction_history: A list of raw prediction values (floats).
            min_predictions_for_drift: Minimum total predictions needed to perform drift test.

        Returns:
            A tuple: (ks_statistic: float, p_value: Optional[float]).
            Returns (0.0, None) if insufficient data.
        """
        if (
            len(prediction_history) < min_predictions_for_drift * 2
        ):  # Need at least two windows of min_predictions
            logger.debug(
                f"Insufficient prediction history ({len(prediction_history)} < {min_predictions_for_drift * 2}) for drift calculation."
            )
            return 0.0, None

        # Split history into two windows (e.g., reference and current)
        mid_point = len(prediction_history) // 2
        reference_data = prediction_history[:mid_point]
        current_data = prediction_history[mid_point:]

        # Ensure non-empty and non-constant data for KS test
        if (
            not reference_data
            or not current_data
            or (np.std(reference_data) == 0 and np.std(current_data) == 0)
        ):
            logger.warning(
                "Prediction history for drift is empty or constant. Cannot perform KS test."
            )
            return 0.0, None

        try:
            # Kolmogorov-Smirnov test for distribution drift
            ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
            logger.debug(
                f"Prediction drift KS statistic: {ks_statistic:.4f}, p-value: {p_value:.4f}."
            )
            return float(ks_statistic), float(p_value)  # Ensure float return
        except Exception as e:
            logger.error(f"Error calculating prediction drift: {e}", exc_info=True)
            return 0.0, None

    def check_feature_drift(
        self,
        model_id: str,
        feature_history_data: Dict[str, List[Any]],
        model_features_list: List[str],
        min_points_for_drift: int,
    ) -> Dict[str, float]:
        """
        Checks for statistical drift in individual feature distributions.

        Args:
            model_id: The ID of the model.
            feature_history_data: Dictionary of {f"{model_id}_{feature_name}": List[value]}.
            model_features_list: List of feature names used by the model.
            min_points_for_drift: Minimum data points required per feature for drift test.

        Returns:
            A dictionary where keys are feature names and values are their KS statistic
            if drift is detected above the threshold.
        """
        drifted_features: Dict[str, float] = {}

        # Iterate over features used by the model
        for feature_name in model_features_list:
            feature_key = f"{model_id}_{feature_name}"
            history = feature_history_data.get(
                feature_key, []
            )  # Get feature history for this specific feature

            if len(history) < min_points_for_drift * 2:
                logger.debug(
                    f"Insufficient history ({len(history)}) for feature '{feature_name}' drift check."
                )
                continue

            # Split into reference and current windows
            mid_point = len(history) // 2
            reference_data = history[:mid_point]
            current_data = history[mid_point:]

            # Ensure data is not constant for KS test
            if (
                not reference_data
                or not current_data
                or (
                    isinstance(reference_data[0], (int, float))
                    and np.std(reference_data) == 0
                    and np.std(current_data) == 0
                )
            ):
                logger.debug(
                    f"Feature '{feature_name}' data is empty or constant. Skipping drift check."
                )
                continue

            try:
                ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)

                if ks_statistic > self.drift_threshold:
                    drifted_features[feature_name] = float(ks_statistic)
                    logger.warning(
                        f"Drift detected for feature '{feature_name}' of model {model_id}: KS={ks_statistic:.4f}, p={p_value:.4f}"
                    )
            except Exception as e:
                logger.error(
                    f"Error calculating drift for feature '{feature_name}' of model {model_id}: {e}",
                    exc_info=True,
                )

        return drifted_features

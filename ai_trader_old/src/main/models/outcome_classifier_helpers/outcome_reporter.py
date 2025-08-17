# File: src/ai_trader/models/outcome_classifier_helpers/outcome_reporter.py

# Standard library imports
import logging
from typing import Any, Dict, List

# Third-party imports
import numpy as np

# Local imports
# Corrected absolute imports for OutcomeLabel and OutcomeMetrics
from main.models.outcome_classifier_types import OutcomeLabel, OutcomeMetrics

logger = logging.getLogger(__name__)


class OutcomeReporter:
    """
    Generates comprehensive reports and summaries from lists of classified OutcomeMetrics.
    Provides insights into the distribution and performance of different outcome labels.
    """

    def __init__(self):
        logger.debug("OutcomeReporter initialized.")

    def generate_classification_report(self, outcomes: List[OutcomeMetrics]) -> Dict[str, Any]:
        """
        Generates a comprehensive report summarizing the classification results.

        Args:
            outcomes: A list of OutcomeMetrics objects.

        Returns:
            A dictionary containing various statistics and summaries of the classifications.
            Returns an error dict if no outcomes are provided.
        """
        if not outcomes:
            logger.warning("No outcomes provided for report generation.")
            return {"error": "No outcomes to analyze", "total_outcomes": 0}

        # Filter for valid outcomes (exclude NO_DATA and CALCULATION_ERROR)
        valid_outcomes = [
            o
            for o in outcomes
            if o.outcome_label not in [OutcomeLabel.NO_DATA, OutcomeLabel.CALCULATION_ERROR]
        ]

        total_outcomes = len(outcomes)
        total_valid_outcomes = len(valid_outcomes)

        # 1. Label Distribution
        label_counts: Dict[str, int] = defaultdict(int)
        for outcome in valid_outcomes:
            label_counts[outcome.outcome_label.value] += 1

        label_percentages = (
            {k: (v / total_valid_outcomes) * 100 for k, v in label_counts.items()}
            if total_valid_outcomes > 0
            else {}
        )

        # 2. Performance Metrics (e.g., average return for successful breakouts)
        successful_breakouts = [
            o for o in valid_outcomes if o.outcome_label == OutcomeLabel.SUCCESSFUL_BREAKOUT
        ]
        # Filter out None values before calculating mean
        successful_returns_3d = [
            o.return_3d for o in successful_breakouts if o.return_3d is not None
        ]
        avg_successful_return_3d = np.mean(successful_returns_3d) if successful_returns_3d else 0.0

        # 3. Confidence Distribution
        confidences = [o.confidence_score for o in valid_outcomes if o.confidence_score is not None]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_outcomes_processed": total_outcomes,
            "valid_classifications_count": total_valid_outcomes,
            "classification_success_rate_pct": (
                (total_valid_outcomes / total_outcomes) * 100 if total_outcomes > 0 else 0.0
            ),
            "label_distribution": label_counts,
            "label_percentages_of_valid_outcomes": label_percentages,
            "performance_metrics": {
                "successful_breakout_count": len(successful_breakouts),
                "successful_breakout_rate_of_valid": (
                    (len(successful_breakouts) / total_valid_outcomes) * 100
                    if total_valid_outcomes > 0
                    else 0.0
                ),
                "avg_successful_return_3d": float(avg_successful_return_3d),  # Ensure float type
                "avg_confidence_score": float(avg_confidence),  # Ensure float type
            },
            "data_quality_issues_in_classification": {
                "no_data_count": sum(
                    1 for o in outcomes if o.outcome_label == OutcomeLabel.NO_DATA
                ),
                "calculation_error_count": sum(
                    1 for o in outcomes if o.outcome_label == OutcomeLabel.CALCULATION_ERROR
                ),
            },
            "confidence_bins": self._get_confidence_bins(confidences),
        }

        logger.info(f"Generated classification report for {total_outcomes} outcomes.")
        return report

    def _get_confidence_bins(self, confidences: List[float]) -> Dict[str, int]:
        """Helper to bin confidence scores."""
        bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        if not confidences:
            return bins

        for conf in confidences:
            if conf >= 0.8:
                bins["0.8-1.0"] += 1
            elif conf >= 0.6:
                bins["0.6-0.8"] += 1
            elif conf >= 0.4:
                bins["0.4-0.6"] += 1
            elif conf >= 0.2:
                bins["0.2-0.4"] += 1
            else:
                bins["0.0-0.2"] += 1

        return bins

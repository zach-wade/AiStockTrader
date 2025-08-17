# File: src/main/events/feature_pipeline_helpers/feature_handler_stats_tracker.py

# Standard library imports
from typing import Any

# Local imports
from main.utils.core import get_logger
from main.utils.monitoring import MetricsCollector, record_metric

logger = get_logger(__name__)


class FeatureHandlerStatsTracker:
    """
    Tracks and provides statistics specific to the Feature Pipeline Handler,
    such as requests received, processed, features computed, and errors.

    Now uses utils.monitoring.MetricsCollector for consistency.
    """

    def __init__(self):
        """Initializes the FeatureHandlerStatsTracker with MetricsCollector."""
        self.metrics = MetricsCollector()
        logger.debug("FeatureHandlerStatsTracker initialized with MetricsCollector.")

    def increment_requests_received(self):
        """Increments the count of feature requests received."""
        self.metrics.increment_counter("feature_pipeline.requests_received")
        record_metric("feature_pipeline.requests_received", 1)

    def increment_requests_processed(self):
        """Increments the count of feature requests successfully processed."""
        self.metrics.increment_counter("feature_pipeline.requests_processed")
        record_metric("feature_pipeline.requests_processed", 1)

    def increment_features_computed(self, count: int):
        """Increments the total count of individual features computed."""
        self.metrics.increment_counter("feature_pipeline.features_computed", count)
        record_metric("feature_pipeline.features_computed", count)

    def increment_computation_errors(self):
        """Increments the count of feature computation errors."""
        self.metrics.increment_counter("feature_pipeline.computation_errors")
        record_metric("feature_pipeline.computation_errors", 1)

    def get_stats(self, queue_size: int = 0, active_workers: int = 0) -> dict[str, Any]:
        """
        Retrieves current statistics for the Feature Pipeline Handler.

        Args:
            queue_size: Current size of the internal request queue.
            active_workers: Number of workers currently processing requests.

        Returns:
            A dictionary containing handler-specific statistics.
        """
        # Get metrics from collector
        received_stats = self.metrics.get_metric_stats("feature_pipeline.requests_received") or {}
        processed_stats = self.metrics.get_metric_stats("feature_pipeline.requests_processed") or {}
        computed_stats = self.metrics.get_metric_stats("feature_pipeline.features_computed") or {}
        error_stats = self.metrics.get_metric_stats("feature_pipeline.computation_errors") or {}

        return {
            "requests_received": int(received_stats.get("latest", 0)),
            "requests_processed": int(processed_stats.get("latest", 0)),
            "features_computed_total": int(computed_stats.get("latest", 0)),
            "computation_errors_total": int(error_stats.get("latest", 0)),
            "queue_size": queue_size,
            "active_workers": active_workers,
        }

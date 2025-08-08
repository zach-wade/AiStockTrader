# File: src/main/events/scanner_bridge_helpers/bridge_stats_tracker.py

from typing import Dict, Any, Set
from main.utils.core import get_logger
from main.utils.monitoring import MetricsCollector, record_metric

logger = get_logger(__name__)

class BridgeStatsTracker:
    """
    Tracks and provides statistics specific to the Scanner-Feature Bridge,
    such as alerts received, feature requests sent, and unique symbols processed.
    
    Now uses utils.monitoring.MetricsCollector for consistency.
    """

    def __init__(self):
        """Initializes the BridgeStatsTracker with MetricsCollector."""
        self.metrics = MetricsCollector()
        self._unique_symbols = set()  # Track unique symbols in memory
        logger.debug("BridgeStatsTracker initialized with MetricsCollector.")

    def increment_alerts_received(self):
        """Increments the count of alerts received by the bridge."""
        self.metrics.increment_counter("scanner_bridge.alerts_received")
        record_metric("scanner_bridge.alerts_received", 1)

    def increment_feature_requests_sent(self):
        """Increments the count of feature requests published by the bridge."""
        self.metrics.increment_counter("scanner_bridge.feature_requests_sent")
        record_metric("scanner_bridge.feature_requests_sent", 1)

    def add_symbol_processed(self, symbol: str):
        """Adds a symbol to the set of unique symbols processed by the bridge."""
        self._unique_symbols.add(symbol)
        # Update gauge with current count
        self.metrics.record_gauge("scanner_bridge.unique_symbols", len(self._unique_symbols))
        record_metric("scanner_bridge.symbol_processed", 1, tags={"symbol": symbol})

    def get_stats(self, pending_batches_count: int = 0, pending_symbols_count: int = 0) -> Dict[str, Any]:
        """
        Retrieves current statistics for the Scanner-Feature Bridge.

        Args:
            pending_batches_count: Number of feature request batches currently pending.
            pending_symbols_count: Total number of symbols across all pending batches.

        Returns:
            A dictionary containing bridge-specific statistics.
        """
        # Get metrics from collector
        alerts_stats = self.metrics.get_metric_stats("scanner_bridge.alerts_received") or {}
        requests_stats = self.metrics.get_metric_stats("scanner_bridge.feature_requests_sent") or {}
        
        return {
            'alerts_received_total': int(alerts_stats.get('latest', 0)),
            'feature_requests_sent_total': int(requests_stats.get('latest', 0)),
            'unique_symbols_processed_count': len(self._unique_symbols),
            'pending_batches_count': pending_batches_count,
            'pending_symbols_count': pending_symbols_count
        }
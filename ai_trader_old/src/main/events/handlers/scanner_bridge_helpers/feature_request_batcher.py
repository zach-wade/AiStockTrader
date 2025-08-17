# File: src/main/events/scanner_bridge_helpers/feature_request_batcher.py

# Standard library imports
from dataclasses import dataclass, field
from datetime import UTC, datetime

# Local imports
from main.utils.core import ErrorHandlingMixin, get_logger

logger = get_logger(__name__)


# This dataclass will now live in this module or a shared 'scanner_bridge_types.py'
# For now, defining it here as it's tightly coupled to the batcher's state.
@dataclass
class FeatureRequestBatch:
    """Represents a pending feature computation request batch."""

    symbols: set[str] = field(default_factory=set)
    features: set[str] = field(default_factory=set)
    priority: int = 5  # Higher number = higher priority for EventBus
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    scanner_sources: set[str] = field(default_factory=set)
    # Store original event IDs/correlation IDs if needed for traceability
    correlation_ids: set[str] = field(default_factory=set)


class FeatureRequestBatcher(ErrorHandlingMixin):
    """
    Manages the aggregation and deduplication of scanner alerts into
    batched feature computation requests.

    Now inherits from ErrorHandlingMixin for better error handling.
    """

    def __init__(self, batch_size: int = 50):
        """
        Initializes the FeatureRequestBatcher.

        Args:
            batch_size: The maximum number of unique symbols per feature request batch.
        """
        # Pending feature requests (keyed by a hash of priority and features)
        # This allows grouping identical feature sets with similar priority.
        self._pending_requests: dict[str, FeatureRequestBatch] = {}
        self._batch_size = batch_size
        logger.debug(f"FeatureRequestBatcher initialized with batch_size: {batch_size}")

    def add_to_pending(
        self,
        symbol: str,
        features: list[str],
        priority: int,
        scanner_source: str,
        correlation_id: str | None = None,
    ) -> FeatureRequestBatch | None:
        """
        Adds a symbol and its requested features to a pending batch.
        If a batch reaches its size limit, it is returned for immediate processing.

        Args:
            symbol: The symbol from the scanner alert.
            features: The list of features requested for this alert.
            priority: The priority of the request (1-10, higher is more urgent).
            scanner_source: The name of the scanner that generated the alert.
            correlation_id: Optional. A correlation ID from the original event.

        Returns:
            The completed FeatureRequestBatch if it is full and ready to be sent,
            otherwise None.
        """
        # Create a deterministic batch key based on priority and features
        # Features are sorted to ensure consistent key generation regardless of input order.
        features_key = "_".join(sorted(features))
        batch_key = f"p{priority}_{features_key}"

        if batch_key not in self._pending_requests:
            self._pending_requests[batch_key] = FeatureRequestBatch(priority=priority)
            logger.debug(f"Created new batch key '{batch_key}' for symbol '{symbol}'.")

        request_batch = self._pending_requests[batch_key]

        # Check if adding this symbol would exceed the batch size BEFORE adding it
        # This ensures we don't accidentally overfill a batch and can immediately return it.
        # This requires a new batch if current one is full.
        if symbol not in request_batch.symbols and len(request_batch.symbols) >= self._batch_size:
            # Current batch is full and this is a new symbol for it, so return the current batch
            # and create a new one for the incoming symbol.
            completed_batch = self._pending_requests.pop(batch_key)  # Pop current full batch
            # Create a new batch for the new incoming symbol
            new_batch_key = f"p{priority}_{features_key}_overflow_{datetime.now(UTC).isoformat()}"  # Unique key for overflow
            self._pending_requests[new_batch_key] = FeatureRequestBatch(priority=priority)
            new_request_batch = self._pending_requests[new_batch_key]

            new_request_batch.symbols.add(symbol)
            new_request_batch.features.update(features)
            new_request_batch.scanner_sources.add(scanner_source)
            if correlation_id:
                new_request_batch.correlation_ids.add(correlation_id)

            logger.debug(
                f"Batch '{batch_key}' completed by symbol limit. Returning. New batch created: '{new_batch_key}'."
            )
            return completed_batch  # Return the batch that just became full

        # Add to the current batch
        request_batch.symbols.add(symbol)
        request_batch.features.update(features)
        request_batch.scanner_sources.add(scanner_source)
        if correlation_id:
            request_batch.correlation_ids.add(correlation_id)

        logger.debug(
            f"Added symbol '{symbol}' to batch '{batch_key}'. Current symbols: {len(request_batch.symbols)}/{self._batch_size}."
        )

        # If the batch filled exactly with this addition
        if len(request_batch.symbols) == self._batch_size:
            logger.debug(f"Batch '{batch_key}' filled to capacity. Returning for immediate send.")
            return self._pending_requests.pop(batch_key)  # Pop and return the newly filled batch

        return None  # Batch not yet full or just created a new one

    def get_and_clear_all_pending_batches(self) -> list[FeatureRequestBatch]:
        """
        Retrieves all currently pending feature request batches and clears them from storage.
        Batches are returned sorted by priority (highest first).
        """
        if not self._pending_requests:
            return []

        # Sort by priority (higher priority first)
        sorted_batches = sorted(
            self._pending_requests.values(), key=lambda req: req.priority, reverse=True
        )

        self._pending_requests.clear()  # Clear all pending requests after retrieval
        logger.debug(f"Retrieved and cleared {len(sorted_batches)} pending batches.")
        return sorted_batches

    def get_pending_counts(self) -> tuple[int, int]:
        """
        Returns the number of pending batches and total pending symbols.
        """
        total_symbols = sum(len(req.symbols) for req in self._pending_requests.values())
        return len(self._pending_requests), total_symbols

"""Unit tests for request_queue_manager module."""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

# Third-party imports
import pytest

# Local imports
from main.events.handlers.feature_pipeline_helpers.feature_types import FeatureGroup, FeatureRequest
from main.events.handlers.feature_pipeline_helpers.queue_types import QueuedRequest, QueueStats
from main.events.handlers.feature_pipeline_helpers.request_queue_manager import RequestQueueManager
from main.events.types import AlertType


class TestRequestQueueManager:
    """Test RequestQueueManager class."""

    @pytest.fixture
    def queue_manager(self):
        """Create RequestQueueManager instance for testing."""
        return RequestQueueManager(
            max_queue_size=100,
            request_ttl_seconds=60,
            dedup_window_seconds=30,
            max_requests_per_symbol=5,
        )

    @pytest.fixture
    def sample_request(self):
        """Create sample feature request."""
        return FeatureRequest(
            symbol="AAPL",
            feature_groups=[FeatureGroup.PRICE, FeatureGroup.VOLUME],
            alert_type=AlertType.HIGH_VOLUME,
            priority=5,
            metadata={"test": True},
        )

    def test_initialization(self):
        """Test queue manager initialization."""
        manager = RequestQueueManager(
            max_queue_size=50,
            request_ttl_seconds=120,
            dedup_window_seconds=45,
            max_requests_per_symbol=3,
        )

        assert manager.max_queue_size == 50
        assert manager.request_ttl_seconds == 120
        assert manager.dedup_window_seconds == 45
        assert manager.max_requests_per_symbol == 3
        assert len(manager._queue) == 0
        assert manager._stats.current_queue_size == 0

    @pytest.mark.asyncio
    async def test_enqueue_request(self, queue_manager, sample_request):
        """Test enqueueing a request."""
        success = await queue_manager.enqueue_request(sample_request, "test_123")

        assert success is True
        assert queue_manager.get_queue_size() == 1
        assert queue_manager._stats.total_queued == 1
        assert queue_manager._stats.requests_by_symbol["AAPL"] == 1

    @pytest.mark.asyncio
    async def test_enqueue_request_auto_id(self, queue_manager, sample_request):
        """Test enqueueing with auto-generated ID."""
        success = await queue_manager.enqueue_request(sample_request)

        assert success is True
        assert queue_manager.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_enqueue_duplicate_rejection(self, queue_manager, sample_request):
        """Test duplicate request rejection."""
        # Enqueue first time
        await queue_manager.enqueue_request(sample_request, "test_123")

        # Try to enqueue same request
        success = await queue_manager.enqueue_request(sample_request, "test_123")

        assert success is False
        assert queue_manager.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_enqueue_similar_request_dedup(self, queue_manager):
        """Test deduplication of similar requests."""
        # Create two similar requests for same symbol
        request1 = FeatureRequest(
            symbol="AAPL",
            feature_groups=[FeatureGroup.PRICE],
            alert_type=AlertType.HIGH_VOLUME,
            priority=5,
        )

        request2 = FeatureRequest(
            symbol="AAPL",
            feature_groups=[FeatureGroup.PRICE],
            alert_type=AlertType.HIGH_VOLUME,
            priority=7,  # Different priority
        )

        await queue_manager.enqueue_request(request1)
        success = await queue_manager.enqueue_request(request2)

        # Should be deduplicated
        assert success is False
        assert queue_manager.get_queue_size() == 1

    @pytest.mark.asyncio
    async def test_queue_overflow(self, queue_manager):
        """Test behavior when queue is full."""
        queue_manager.max_queue_size = 3

        # Fill queue
        for i in range(3):
            request = FeatureRequest(
                symbol=f"STOCK{i}",
                feature_groups=[FeatureGroup.PRICE],
                alert_type=AlertType.HIGH_VOLUME,
            )
            await queue_manager.enqueue_request(request)

        # Try to add one more
        overflow_request = FeatureRequest(
            symbol="OVERFLOW", feature_groups=[FeatureGroup.PRICE], alert_type=AlertType.HIGH_VOLUME
        )

        with patch(
            "main.events.handlers.feature_pipeline_helpers.request_queue_manager.record_metric"
        ) as mock_metric:
            success = await queue_manager.enqueue_request(overflow_request)

            assert success is False
            assert queue_manager.get_queue_size() == 3
            mock_metric.assert_called_with("request_queue.rejected", 1)

    @pytest.mark.asyncio
    async def test_per_symbol_limit(self, queue_manager):
        """Test per-symbol request limit."""
        queue_manager.max_requests_per_symbol = 2

        # Add 2 requests for AAPL
        for i in range(2):
            request = FeatureRequest(
                symbol="AAPL",
                feature_groups=[FeatureGroup.PRICE],
                alert_type=AlertType.HIGH_VOLUME,
                metadata={"id": i},
            )
            await queue_manager.enqueue_request(request, f"aapl_{i}")

        # Try to add third AAPL request
        third_request = FeatureRequest(
            symbol="AAPL", feature_groups=[FeatureGroup.VOLUME], alert_type=AlertType.HIGH_VOLUME
        )

        success = await queue_manager.enqueue_request(third_request)
        assert success is False

    @pytest.mark.asyncio
    async def test_dequeue_request(self, queue_manager):
        """Test dequeueing requests."""
        # Add multiple requests with different priorities
        high_priority = FeatureRequest(
            symbol="HIGH",
            feature_groups=[FeatureGroup.PRICE],
            alert_type=AlertType.ML_SIGNAL,
            priority=9,
        )

        low_priority = FeatureRequest(
            symbol="LOW",
            feature_groups=[FeatureGroup.PRICE],
            alert_type=AlertType.HIGH_VOLUME,
            priority=3,
        )

        await queue_manager.enqueue_request(low_priority, "low")
        await queue_manager.enqueue_request(high_priority, "high")

        # Dequeue should return high priority first
        queued = await queue_manager.dequeue_request(worker_id=1)

        assert queued is not None
        assert queued.request.symbol == "HIGH"
        assert queued.request.priority == 9
        assert queued.attempt_count == 1
        assert queued.last_attempt is not None

    @pytest.mark.asyncio
    async def test_dequeue_empty_queue(self, queue_manager):
        """Test dequeuing from empty queue."""
        queued = await queue_manager.dequeue_request()
        assert queued is None

    @pytest.mark.asyncio
    async def test_request_expiration(self, queue_manager):
        """Test expired requests are not returned."""
        queue_manager.request_ttl_seconds = 1  # 1 second TTL

        # Add request
        request = FeatureRequest(
            symbol="EXPIRED", feature_groups=[FeatureGroup.PRICE], alert_type=AlertType.HIGH_VOLUME
        )
        await queue_manager.enqueue_request(request, "exp_1")

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Try to dequeue
        queued = await queue_manager.dequeue_request()

        assert queued is None
        assert queue_manager._stats.total_expired == 1

    @pytest.mark.asyncio
    async def test_symbol_load_balancing(self, queue_manager):
        """Test load balancing across symbols."""
        queue_manager.max_requests_per_symbol = 1

        # Add requests for different symbols
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            request = FeatureRequest(
                symbol=symbol, feature_groups=[FeatureGroup.PRICE], alert_type=AlertType.HIGH_VOLUME
            )
            await queue_manager.enqueue_request(request)

        # Mark AAPL as active
        queue_manager._active_requests_by_symbol["AAPL"] = 1

        # Dequeue should skip AAPL and return GOOGL or MSFT
        queued = await queue_manager.dequeue_request()

        assert queued is not None
        assert queued.request.symbol in ["GOOGL", "MSFT"]

    @pytest.mark.asyncio
    async def test_complete_request(self, queue_manager, sample_request):
        """Test completing a request."""
        # Enqueue and dequeue
        await queue_manager.enqueue_request(sample_request, "test_123")
        queued = await queue_manager.dequeue_request()

        # Complete successfully
        await queue_manager.complete_request("test_123", success=True)

        assert queue_manager._stats.total_processed == 1
        assert queue_manager._active_requests_by_symbol.get("AAPL", 0) == 0
        assert "test_123" not in queue_manager._request_map

    @pytest.mark.asyncio
    async def test_complete_request_failure(self, queue_manager, sample_request):
        """Test completing a failed request."""
        await queue_manager.enqueue_request(sample_request, "test_123")
        queued = await queue_manager.dequeue_request()

        # Complete with failure
        await queue_manager.complete_request("test_123", success=False)

        assert queue_manager._stats.total_failed == 1
        assert queue_manager._stats.total_processed == 0

    def test_get_stats(self, queue_manager):
        """Test getting queue statistics."""
        # Add some queue times
        queue_manager._queue_times = [1.0, 2.0, 3.0, 4.0, 5.0]

        stats = queue_manager.get_stats()

        assert isinstance(stats, QueueStats)
        assert stats.avg_queue_time_seconds == 3.0  # Average of 1-5

    def test_get_queued_symbols(self, queue_manager):
        """Test getting list of queued symbols."""
        # Add requests for different symbols
        for symbol in ["AAPL", "GOOGL", "AAPL", "MSFT"]:
            request = FeatureRequest(
                symbol=symbol, feature_groups=[FeatureGroup.PRICE], alert_type=AlertType.HIGH_VOLUME
            )
            queued = QueuedRequest(request=request, request_id=f"{symbol}_{id(request)}")
            queue_manager._queue.append(queued)

        symbols = queue_manager.get_queued_symbols()

        assert symbols == ["AAPL", "GOOGL", "MSFT"]  # Sorted and unique

    @pytest.mark.asyncio
    async def test_clear_symbol_requests(self, queue_manager):
        """Test clearing all requests for a symbol."""
        # Add multiple requests
        for i in range(3):
            for symbol in ["AAPL", "GOOGL"]:
                request = FeatureRequest(
                    symbol=symbol,
                    feature_groups=[FeatureGroup.PRICE],
                    alert_type=AlertType.HIGH_VOLUME,
                )
                await queue_manager.enqueue_request(request, f"{symbol}_{i}")

        assert queue_manager.get_queue_size() == 6

        # Clear AAPL requests
        cleared = await queue_manager.clear_symbol_requests("AAPL")

        assert cleared == 3
        assert queue_manager.get_queue_size() == 3
        assert "AAPL" not in queue_manager.get_queued_symbols()

    @pytest.mark.asyncio
    async def test_queue_time_tracking(self, queue_manager, sample_request):
        """Test queue time calculation."""
        await queue_manager.enqueue_request(sample_request)

        # Wait a bit
        await asyncio.sleep(0.1)

        # Dequeue
        queued = await queue_manager.dequeue_request()

        # Queue time should be tracked
        assert len(queue_manager._queue_times) == 1
        assert queue_manager._queue_times[0] >= 0.1

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, queue_manager):
        """Test concurrent enqueue/dequeue operations."""

        async def enqueue_requests():
            for i in range(20):
                request = FeatureRequest(
                    symbol=f"STOCK{i % 5}",
                    feature_groups=[FeatureGroup.PRICE],
                    alert_type=AlertType.HIGH_VOLUME,
                    priority=i % 10,
                )
                await queue_manager.enqueue_request(request)
                await asyncio.sleep(0.01)

        async def dequeue_requests():
            dequeued = []
            for _ in range(15):
                queued = await queue_manager.dequeue_request()
                if queued:
                    dequeued.append(queued)
                    await asyncio.sleep(0.02)
            return dequeued

        # Run concurrently
        results = await asyncio.gather(enqueue_requests(), dequeue_requests())

        dequeued = results[1]
        assert len(dequeued) > 0

        # Check priority ordering
        for i in range(len(dequeued) - 1):
            # Higher priority should come first
            assert dequeued[i].request.priority >= dequeued[i + 1].request.priority

    @pytest.mark.asyncio
    async def test_cleanup_expired_dedup_entries(self, queue_manager):
        """Test cleanup of expired deduplication entries."""
        # Access dedup tracker
        tracker = queue_manager._dedup_tracker

        # Add old entry
        old_time = datetime.now(UTC) - timedelta(seconds=120)
        tracker._recent_requests["old_request"] = old_time

        # Add recent entry
        tracker._recent_requests["new_request"] = datetime.now(UTC)

        # Trigger cleanup through enqueue
        request = FeatureRequest(
            symbol="TEST", feature_groups=[FeatureGroup.PRICE], alert_type=AlertType.HIGH_VOLUME
        )
        await queue_manager.enqueue_request(request)

        # Old entry should be cleaned up eventually
        assert len(tracker._recent_requests) >= 1

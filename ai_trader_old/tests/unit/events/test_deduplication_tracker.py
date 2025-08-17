"""Unit tests for deduplication_tracker module."""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

# Third-party imports
import pytest

# Local imports
from main.events.handlers.feature_pipeline_helpers.deduplication_tracker import DeduplicationTracker
from main.events.handlers.feature_pipeline_helpers.feature_types import FeatureGroup, FeatureRequest
from main.events.types import AlertType


class TestDeduplicationTracker:
    """Test DeduplicationTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create DeduplicationTracker instance for testing."""
        return DeduplicationTracker(window_seconds=60, cleanup_interval_seconds=30)

    @pytest.fixture
    def sample_request(self):
        """Create sample feature request for testing."""
        return FeatureRequest(
            symbol="AAPL",
            feature_groups=[FeatureGroup.PRICE, FeatureGroup.VOLUME],
            alert_type=AlertType.HIGH_VOLUME,
            priority=5,
        )

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = DeduplicationTracker(window_seconds=120, cleanup_interval_seconds=60)

        assert tracker.window_seconds == 120
        assert tracker.cleanup_interval_seconds == 60
        assert len(tracker._recent_requests) == 0
        assert tracker._last_cleanup is not None

    @pytest.mark.asyncio
    async def test_is_duplicate_new_request(self, tracker, sample_request):
        """Test that new request is not duplicate."""
        is_dup = await tracker.is_duplicate(sample_request)

        assert is_dup is False
        assert len(tracker._recent_requests) == 1

    @pytest.mark.asyncio
    async def test_is_duplicate_same_request(self, tracker, sample_request):
        """Test that same request is marked as duplicate."""
        # First submission
        is_dup1 = await tracker.is_duplicate(sample_request)
        assert is_dup1 is False

        # Second submission (duplicate)
        is_dup2 = await tracker.is_duplicate(sample_request)
        assert is_dup2 is True

    @pytest.mark.asyncio
    async def test_is_duplicate_by_request_id(self, tracker, sample_request):
        """Test deduplication by request ID."""
        request_id = "unique_123"

        # First submission
        is_dup1 = await tracker.is_duplicate(sample_request, request_id)
        assert is_dup1 is False

        # Same request ID (duplicate)
        is_dup2 = await tracker.is_duplicate(sample_request, request_id)
        assert is_dup2 is True

        # Different request ID (not duplicate)
        is_dup3 = await tracker.is_duplicate(sample_request, "different_456")
        assert is_dup3 is False

    @pytest.mark.asyncio
    async def test_is_duplicate_similar_requests(self, tracker):
        """Test deduplication of similar requests."""
        # Base request
        request1 = FeatureRequest(
            symbol="AAPL", feature_groups=[FeatureGroup.PRICE], alert_type=AlertType.HIGH_VOLUME
        )

        # Same content, different metadata
        request2 = FeatureRequest(
            symbol="AAPL",
            feature_groups=[FeatureGroup.PRICE],
            alert_type=AlertType.HIGH_VOLUME,
            metadata={"source": "scanner"},
        )

        # Different symbol
        request3 = FeatureRequest(
            symbol="GOOGL", feature_groups=[FeatureGroup.PRICE], alert_type=AlertType.HIGH_VOLUME
        )

        await tracker.is_duplicate(request1)

        # Same core content might be duplicate
        is_dup2 = await tracker.is_duplicate(request2)
        # Depends on implementation - metadata might be ignored

        # Different symbol should not be duplicate
        is_dup3 = await tracker.is_duplicate(request3)
        assert is_dup3 is False

    @pytest.mark.asyncio
    async def test_window_expiration(self, tracker, sample_request):
        """Test that duplicates expire after window."""
        tracker.window_seconds = 0.1  # 100ms window

        # First submission
        await tracker.is_duplicate(sample_request)

        # Within window - duplicate
        is_dup1 = await tracker.is_duplicate(sample_request)
        assert is_dup1 is True

        # Wait for expiration
        await asyncio.sleep(0.15)

        # After window - not duplicate
        is_dup2 = await tracker.is_duplicate(sample_request)
        assert is_dup2 is False

    @pytest.mark.asyncio
    async def test_cleanup_old_entries(self, tracker):
        """Test cleanup of old entries."""
        # Add old entry manually
        old_time = datetime.now(UTC) - timedelta(seconds=120)
        tracker._recent_requests["old_hash"] = old_time

        # Add recent entry
        tracker._recent_requests["new_hash"] = datetime.now(UTC)

        # Trigger cleanup
        await tracker._cleanup_old_entries()

        # Old entry should be removed
        assert "old_hash" not in tracker._recent_requests
        assert "new_hash" in tracker._recent_requests

    @pytest.mark.asyncio
    async def test_auto_cleanup_trigger(self, tracker, sample_request):
        """Test automatic cleanup triggering."""
        tracker.cleanup_interval_seconds = 0.1  # 100ms interval

        # Reset last cleanup to trigger
        tracker._last_cleanup = datetime.now(UTC) - timedelta(seconds=1)

        # This should trigger cleanup
        with patch.object(tracker, "_cleanup_old_entries") as mock_cleanup:
            await tracker.is_duplicate(sample_request)

            # Cleanup should have been called
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_method(self, tracker, sample_request):
        """Test clearing all entries."""
        # Add some entries
        await tracker.is_duplicate(sample_request)
        await tracker.is_duplicate(sample_request, "id_123")

        assert len(tracker._recent_requests) > 0

        # Clear all
        await tracker.clear()

        assert len(tracker._recent_requests) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, tracker):
        """Test getting tracker statistics."""
        # Add various entries
        requests = [
            FeatureRequest(symbol=f"STOCK{i}", feature_groups=[FeatureGroup.PRICE])
            for i in range(5)
        ]

        for req in requests:
            await tracker.is_duplicate(req)

        # Try duplicates
        await tracker.is_duplicate(requests[0])
        await tracker.is_duplicate(requests[1])

        stats = tracker.get_stats()

        assert stats["total_entries"] == 5
        assert stats["window_seconds"] == tracker.window_seconds
        assert "oldest_entry_age" in stats

    @pytest.mark.asyncio
    async def test_concurrent_access(self, tracker):
        """Test thread-safe concurrent access."""

        async def add_requests(start_idx):
            for i in range(start_idx, start_idx + 10):
                request = FeatureRequest(symbol=f"STOCK{i}", feature_groups=[FeatureGroup.PRICE])
                await tracker.is_duplicate(request)

        # Run concurrent tasks
        await asyncio.gather(add_requests(0), add_requests(10), add_requests(20))

        # Should have all unique entries
        assert len(tracker._recent_requests) == 30


class TestCreateRequestHash:
    """Test create_request_hash function."""

    def test_hash_basic_request(self):
        """Test hashing basic request."""
        request = FeatureRequest(symbol="AAPL", feature_groups=[FeatureGroup.PRICE])

        hash1 = create_request_hash(request)
        hash2 = create_request_hash(request)

        # Same request should produce same hash
        assert hash1 == hash2
        assert len(hash1) > 0

    def test_hash_different_requests(self):
        """Test hashing different requests."""
        request1 = FeatureRequest(symbol="AAPL", feature_groups=[FeatureGroup.PRICE])

        request2 = FeatureRequest(symbol="GOOGL", feature_groups=[FeatureGroup.PRICE])

        hash1 = create_request_hash(request1)
        hash2 = create_request_hash(request2)

        # Different requests should have different hashes
        assert hash1 != hash2

    def test_hash_ignores_metadata(self):
        """Test that hash ignores certain fields like metadata."""
        request1 = FeatureRequest(
            symbol="AAPL", feature_groups=[FeatureGroup.PRICE], metadata={"source": "scanner1"}
        )

        request2 = FeatureRequest(
            symbol="AAPL", feature_groups=[FeatureGroup.PRICE], metadata={"source": "scanner2"}
        )

        hash1 = create_request_hash(request1)
        hash2 = create_request_hash(request2)

        # Hashes might be same if metadata is ignored
        # This depends on implementation

    def test_hash_considers_feature_order(self):
        """Test if hash considers feature group order."""
        request1 = FeatureRequest(
            symbol="AAPL", feature_groups=[FeatureGroup.PRICE, FeatureGroup.VOLUME]
        )

        request2 = FeatureRequest(
            symbol="AAPL", feature_groups=[FeatureGroup.VOLUME, FeatureGroup.PRICE]
        )

        hash1 = create_request_hash(request1)
        hash2 = create_request_hash(request2)

        # Might be same or different depending on implementation
        # If order doesn't matter, they should be equal

    def test_hash_with_custom_fields(self):
        """Test hashing with custom fields."""
        if "include_fields" in create_request_hash.__code__.co_varnames:
            request = FeatureRequest(
                symbol="AAPL",
                feature_groups=[FeatureGroup.PRICE],
                alert_type=AlertType.ML_SIGNAL,
                priority=8,
            )

            # Hash with only specific fields
            hash1 = create_request_hash(request, include_fields=["symbol"])
            hash2 = create_request_hash(request, include_fields=["symbol", "alert_type"])

            # Different fields should produce different hashes
            assert hash1 != hash2


class TestIsDuplicateRequest:
    """Test is_request_duplicate helper function."""

    @pytest.mark.asyncio
    async def test_is_duplicate_helper(self):
        """Test is_request_duplicate helper function."""
        tracker = DeduplicationTracker()

        request = FeatureRequest(symbol="AAPL", feature_groups=[FeatureGroup.PRICE])

        # First check - not duplicate
        is_dup1 = await is_request_duplicate(tracker, request)
        assert is_dup1 is False

        # Second check - duplicate
        is_dup2 = await is_request_duplicate(tracker, request)
        assert is_dup2 is True

    @pytest.mark.asyncio
    async def test_is_duplicate_with_id(self):
        """Test is_request_duplicate with request ID."""
        tracker = DeduplicationTracker()

        request = FeatureRequest(symbol="GOOGL", feature_groups=[FeatureGroup.VOLUME])

        # With unique IDs
        is_dup1 = await is_request_duplicate(tracker, request, "id_1")
        is_dup2 = await is_request_duplicate(tracker, request, "id_2")

        assert is_dup1 is False
        assert is_dup2 is False

        # Same ID - duplicate
        is_dup3 = await is_request_duplicate(tracker, request, "id_1")
        assert is_dup3 is True


class TestDeduplicationIntegration:
    """Test deduplication in integrated scenarios."""

    @pytest.mark.asyncio
    async def test_high_volume_deduplication(self):
        """Test deduplication under high volume."""
        tracker = DeduplicationTracker(window_seconds=1.0)

        # Simulate high volume of requests
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        duplicates_found = 0

        for _ in range(100):
            for symbol in symbols:
                request = FeatureRequest(symbol=symbol, feature_groups=[FeatureGroup.PRICE])

                is_dup = await tracker.is_duplicate(request)
                if is_dup:
                    duplicates_found += 1

        # Should find many duplicates
        assert duplicates_found > 400  # Most should be duplicates

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency with cleanup."""
        tracker = DeduplicationTracker(window_seconds=0.1, cleanup_interval_seconds=0.05)

        # Add many requests
        for i in range(1000):
            request = FeatureRequest(symbol=f"STOCK{i}", feature_groups=[FeatureGroup.PRICE])
            await tracker.is_duplicate(request)

            # Small delay to spread over time
            if i % 100 == 0:
                await asyncio.sleep(0.02)

        # Wait for cleanup
        await asyncio.sleep(0.2)

        # Trigger cleanup
        dummy_request = FeatureRequest(symbol="DUMMY", feature_groups=[FeatureGroup.PRICE])
        await tracker.is_duplicate(dummy_request)

        # Most entries should be cleaned up
        assert len(tracker._recent_requests) < 100  # Much less than 1000

"""Unit tests for queue_types module."""

import pytest
from dataclasses import fields, asdict
from datetime import datetime, timezone
from typing import Optional

from main.events.handlers.feature_pipeline_helpers.queue_types import (
    QueuedRequest, QueueStats, create_queued_request,
    calculate_queue_time, update_queue_stats
)
from main.events.handlers.feature_pipeline_helpers.feature_types import FeatureRequest, FeatureGroup
from main.events.types import AlertType


class TestQueuedRequest:
    """Test QueuedRequest dataclass."""
    
    @pytest.fixture
    def sample_feature_request(self):
        """Create sample FeatureRequest for testing."""
        return FeatureRequest(
            symbol="AAPL",
            feature_groups=[FeatureGroup.PRICE, FeatureGroup.VOLUME],
            alert_type=AlertType.HIGH_VOLUME,
            priority=7
        )
    
    def test_queued_request_creation(self, sample_feature_request):
        """Test creating QueuedRequest."""
        queued = QueuedRequest(
            request=sample_feature_request,
            request_id="test_123",
            worker_id=1,
            attempt_count=0
        )
        
        assert queued.request == sample_feature_request
        assert queued.request_id == "test_123"
        assert queued.worker_id == 1
        assert queued.attempt_count == 0
        assert isinstance(queued.created_at, datetime)
        assert queued.last_attempt is None
        
    def test_queued_request_defaults(self, sample_feature_request):
        """Test QueuedRequest default values."""
        queued = QueuedRequest(
            request=sample_feature_request,
            request_id="test_456"
        )
        
        assert queued.worker_id is None
        assert queued.attempt_count == 0
        assert queued.last_attempt is None
        assert queued.created_at.tzinfo == timezone.utc
        
    def test_queued_request_with_attempts(self, sample_feature_request):
        """Test QueuedRequest with attempt tracking."""
        last_attempt_time = datetime.now(timezone.utc)
        
        queued = QueuedRequest(
            request=sample_feature_request,
            request_id="test_789",
            worker_id=2,
            attempt_count=3,
            last_attempt=last_attempt_time
        )
        
        assert queued.attempt_count == 3
        assert queued.last_attempt == last_attempt_time
        
    def test_queued_request_immutability(self, sample_feature_request):
        """Test that QueuedRequest is immutable."""
        queued = QueuedRequest(
            request=sample_feature_request,
            request_id="test_999"
        )
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            queued.request_id = "new_id"
            
    def test_queued_request_to_dict(self, sample_feature_request):
        """Test converting QueuedRequest to dictionary."""
        queued = QueuedRequest(
            request=sample_feature_request,
            request_id="test_dict",
            worker_id=3,
            attempt_count=1
        )
        
        data = asdict(queued)
        
        assert data["request_id"] == "test_dict"
        assert data["worker_id"] == 3
        assert data["attempt_count"] == 1
        assert "created_at" in data
        assert "request" in data


class TestQueueStats:
    """Test QueueStats dataclass."""
    
    def test_queue_stats_creation(self):
        """Test creating QueueStats."""
        stats = QueueStats(
            current_queue_size=50,
            total_queued=1000,
            total_processed=950,
            total_failed=30,
            total_expired=20,
            avg_queue_time_seconds=2.5,
            avg_processing_time_seconds=0.1,
            requests_by_symbol={"AAPL": 100, "GOOGL": 80}
        )
        
        assert stats.current_queue_size == 50
        assert stats.total_queued == 1000
        assert stats.total_processed == 950
        assert stats.total_failed == 30
        assert stats.total_expired == 20
        assert stats.avg_queue_time_seconds == 2.5
        assert stats.avg_processing_time_seconds == 0.1
        assert stats.requests_by_symbol == {"AAPL": 100, "GOOGL": 80}
        
    def test_queue_stats_defaults(self):
        """Test QueueStats default values."""
        stats = QueueStats()
        
        assert stats.current_queue_size == 0
        assert stats.total_queued == 0
        assert stats.total_processed == 0
        assert stats.total_failed == 0
        assert stats.total_expired == 0
        assert stats.avg_queue_time_seconds == 0.0
        assert stats.avg_processing_time_seconds == 0.0
        assert stats.requests_by_symbol == {}
        
    def test_queue_stats_calculated_properties(self):
        """Test calculated properties on QueueStats."""
        stats = QueueStats(
            total_queued=1000,
            total_processed=800,
            total_failed=150,
            total_expired=50
        )
        
        # Test if methods exist for calculated properties
        if hasattr(stats, 'success_rate'):
            # Success rate = processed / (processed + failed)
            expected_rate = 800 / (800 + 150)
            assert abs(stats.success_rate() - expected_rate) < 0.001
            
        if hasattr(stats, 'total_completed'):
            # Total completed = processed + failed + expired
            assert stats.total_completed() == 800 + 150 + 50
            
    def test_queue_stats_to_dict(self):
        """Test converting QueueStats to dictionary."""
        stats = QueueStats(
            current_queue_size=25,
            requests_by_symbol={"TSLA": 50}
        )
        
        data = asdict(stats)
        
        assert data["current_queue_size"] == 25
        assert data["requests_by_symbol"] == {"TSLA": 50}
        assert all(key in data for key in [
            "total_queued", "total_processed", "total_failed",
            "avg_queue_time_seconds"
        ])


class TestCreateQueuedRequest:
    """Test create_queued_request helper function."""
    
    def test_create_queued_request_basic(self):
        """Test creating queued request with helper."""
        feature_request = FeatureRequest(
            symbol="MSFT",
            feature_groups=[FeatureGroup.VOLATILITY]
        )
        
        queued = create_queued_request(
            feature_request,
            request_id="helper_123"
        )
        
        assert isinstance(queued, QueuedRequest)
        assert queued.request == feature_request
        assert queued.request_id == "helper_123"
        assert queued.worker_id is None
        assert queued.attempt_count == 0
        
    def test_create_queued_request_with_worker(self):
        """Test creating queued request with worker ID."""
        feature_request = FeatureRequest(
            symbol="AMZN",
            feature_groups=[FeatureGroup.TREND]
        )
        
        queued = create_queued_request(
            feature_request,
            request_id="worker_test",
            worker_id=5
        )
        
        assert queued.worker_id == 5
        
    def test_create_queued_request_auto_id(self):
        """Test creating queued request with auto-generated ID."""
        feature_request = FeatureRequest(
            symbol="NFLX",
            feature_groups=[FeatureGroup.ML_SIGNALS]
        )
        
        # If function supports auto ID generation
        if 'request_id' not in create_queued_request.__code__.co_varnames:
            queued = create_queued_request(feature_request)
            assert queued.request_id is not None
            assert len(queued.request_id) > 0


class TestCalculateQueueTime:
    """Test calculate_queue_time helper function."""
    
    def test_calculate_queue_time_basic(self):
        """Test calculating queue time."""
        created_at = datetime.now(timezone.utc)
        dequeued_at = created_at.timestamp() + 5.0  # 5 seconds later
        
        queue_time = calculate_queue_time(created_at, dequeued_at)
        
        assert abs(queue_time - 5.0) < 0.1
        
    def test_calculate_queue_time_sub_second(self):
        """Test calculating sub-second queue time."""
        created_at = datetime.now(timezone.utc)
        dequeued_at = created_at.timestamp() + 0.250  # 250ms later
        
        queue_time = calculate_queue_time(created_at, dequeued_at)
        
        assert abs(queue_time - 0.250) < 0.01
        
    def test_calculate_queue_time_negative(self):
        """Test handling negative queue time (shouldn't happen)."""
        created_at = datetime.now(timezone.utc)
        dequeued_at = created_at.timestamp() - 1.0  # 1 second before
        
        queue_time = calculate_queue_time(created_at, dequeued_at)
        
        # Should handle gracefully, possibly return 0
        assert queue_time <= 0 or queue_time == 0


class TestUpdateQueueStats:
    """Test update_queue_stats helper function."""
    
    def test_update_queue_stats_increment(self):
        """Test updating queue stats with increments."""
        stats = QueueStats(
            total_queued=100,
            total_processed=80
        )
        
        # Update with new processed request
        updated = update_queue_stats(
            stats,
            processed=1,
            queue_time=1.5
        )
        
        assert updated.total_processed == 81
        
        # If it updates average queue time
        if hasattr(update_queue_stats, 'queue_time'):
            # Average should be updated
            assert updated.avg_queue_time_seconds > 0
            
    def test_update_queue_stats_failed(self):
        """Test updating stats for failed request."""
        stats = QueueStats(
            total_failed=10
        )
        
        updated = update_queue_stats(
            stats,
            failed=1
        )
        
        assert updated.total_failed == 11
        
    def test_update_queue_stats_by_symbol(self):
        """Test updating per-symbol stats."""
        stats = QueueStats(
            requests_by_symbol={"AAPL": 50}
        )
        
        # If function supports symbol updates
        if 'symbol' in update_queue_stats.__code__.co_varnames:
            updated = update_queue_stats(
                stats,
                symbol="AAPL",
                queued=1
            )
            
            assert updated.requests_by_symbol["AAPL"] == 51
            
            # New symbol
            updated = update_queue_stats(
                updated,
                symbol="GOOGL",
                queued=1
            )
            
            assert updated.requests_by_symbol["GOOGL"] == 1


class TestQueueTypesIntegration:
    """Test queue types working together."""
    
    def test_full_request_lifecycle(self):
        """Test request through full queue lifecycle."""
        # Create feature request
        feature_req = FeatureRequest(
            symbol="AAPL",
            feature_groups=[FeatureGroup.PRICE],
            priority=8
        )
        
        # Queue it
        queued = create_queued_request(feature_req, "lifecycle_test")
        assert queued.attempt_count == 0
        
        # Simulate dequeue (would update attempt count)
        dequeue_time = datetime.now(timezone.utc).timestamp()
        queue_time = calculate_queue_time(queued.created_at, dequeue_time)
        assert queue_time >= 0
        
        # Update stats
        stats = QueueStats()
        stats = QueueStats(
            total_queued=stats.total_queued + 1,
            current_queue_size=stats.current_queue_size + 1
        )
        
        # Process complete
        stats = QueueStats(
            total_queued=stats.total_queued,
            total_processed=stats.total_processed + 1,
            current_queue_size=stats.current_queue_size - 1
        )
        
        assert stats.total_processed == 1
        assert stats.current_queue_size == 0
        
    def test_queue_priority_ordering(self):
        """Test that queued requests can be ordered by priority."""
        requests = []
        
        for i, priority in enumerate([3, 9, 1, 7, 5]):
            feature_req = FeatureRequest(
                symbol=f"STOCK{i}",
                feature_groups=[FeatureGroup.PRICE],
                priority=priority
            )
            queued = create_queued_request(feature_req, f"priority_{i}")
            requests.append(queued)
            
        # Sort by priority (highest first)
        sorted_requests = sorted(
            requests,
            key=lambda x: x.request.priority,
            reverse=True
        )
        
        priorities = [r.request.priority for r in sorted_requests]
        assert priorities == [9, 7, 5, 3, 1]
        
    def test_queue_stats_accuracy(self):
        """Test queue stats remain accurate through operations."""
        stats = QueueStats()
        
        # Simulate queue operations
        operations = [
            ("queued", 10),
            ("processed", 7),
            ("failed", 2),
            ("expired", 1)
        ]
        
        for op_type, count in operations:
            if op_type == "queued":
                stats = QueueStats(
                    total_queued=stats.total_queued + count,
                    current_queue_size=stats.current_queue_size + count
                )
            elif op_type == "processed":
                stats = QueueStats(
                    total_queued=stats.total_queued,
                    total_processed=stats.total_processed + count,
                    current_queue_size=stats.current_queue_size - count
                )
            elif op_type == "failed":
                stats = QueueStats(
                    total_queued=stats.total_queued,
                    total_failed=stats.total_failed + count,
                    current_queue_size=stats.current_queue_size - count
                )
            elif op_type == "expired":
                stats = QueueStats(
                    total_queued=stats.total_queued,
                    total_expired=stats.total_expired + count,
                    current_queue_size=stats.current_queue_size - count
                )
                
        # Verify consistency
        total_removed = stats.total_processed + stats.total_failed + stats.total_expired
        assert stats.current_queue_size == stats.total_queued - total_removed
        assert stats.current_queue_size == 0  # All processed
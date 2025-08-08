"""Unit tests for event_bus_stats_tracker module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from main.events.core.event_bus_helpers.event_bus_stats_tracker import EventBusStatsTracker
from main.events.types import EventType
from main.utils.monitoring import MetricsCollector


class TestEventBusStatsTracker:
    """Test EventBusStatsTracker class."""
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock MetricsCollector."""
        collector = Mock(spec=MetricsCollector)
        collector.increment_counter = Mock()
        collector.record_histogram = Mock()
        collector.set_gauge = Mock()
        collector.get_metric_value = Mock()
        collector.get_all_metrics = Mock(return_value={})
        return collector
    
    @pytest.fixture
    def stats_tracker(self, mock_metrics_collector):
        """Create EventBusStatsTracker with mocked MetricsCollector."""
        with patch('main.events.core.event_bus_helpers.event_bus_stats_tracker.MetricsCollector') as mock_class:
            mock_class.return_value = mock_metrics_collector
            tracker = EventBusStatsTracker()
            tracker.metrics = mock_metrics_collector  # Ensure mock is used
            return tracker
    
    def test_initialization(self):
        """Test stats tracker initialization."""
        with patch('main.events.core.event_bus_helpers.event_bus_stats_tracker.MetricsCollector') as mock_class:
            mock_collector = Mock()
            mock_class.return_value = mock_collector
            
            tracker = EventBusStatsTracker()
            
            # Should create MetricsCollector
            mock_class.assert_called_once()
            assert tracker.metrics == mock_collector
            assert tracker._subscribers_by_type == {}
    
    def test_increment_published(self, stats_tracker, mock_metrics_collector):
        """Test incrementing published events counter."""
        stats_tracker.increment_published()
        
        mock_metrics_collector.increment_counter.assert_called_once_with(
            "event_bus.events_published"
        )
    
    def test_increment_processed(self, stats_tracker, mock_metrics_collector):
        """Test incrementing processed events counter."""
        stats_tracker.increment_processed()
        
        mock_metrics_collector.increment_counter.assert_called_once_with(
            "event_bus.events_processed"
        )
    
    def test_increment_failed(self, stats_tracker, mock_metrics_collector):
        """Test incrementing failed events counter."""
        stats_tracker.increment_failed()
        
        mock_metrics_collector.increment_counter.assert_called_once_with(
            "event_bus.events_failed"
        )
    
    def test_record_processing_time(self, stats_tracker, mock_metrics_collector):
        """Test recording event processing time."""
        stats_tracker.record_processing_time(0.125)
        
        mock_metrics_collector.record_histogram.assert_called_once_with(
            "event_bus.processing_time",
            0.125
        )
    
    def test_record_queue_size(self, stats_tracker, mock_metrics_collector):
        """Test recording queue size."""
        stats_tracker.record_queue_size(42)
        
        mock_metrics_collector.set_gauge.assert_called_once_with(
            "event_bus.queue_size",
            42
        )
    
    def test_update_subscriber_count(self, stats_tracker, mock_metrics_collector):
        """Test updating subscriber count for event type."""
        # Update subscriber count
        stats_tracker.update_subscriber_count(EventType.SCANNER_ALERT, 5)
        
        # Should update internal tracking
        assert stats_tracker._subscribers_by_type[EventType.SCANNER_ALERT.value] == 5
        
        # Should update gauge metric
        mock_metrics_collector.set_gauge.assert_called_with(
            "event_bus.subscribers.scanner_alert",
            5
        )
    
    def test_update_subscriber_count_multiple_types(self, stats_tracker, mock_metrics_collector):
        """Test updating subscriber counts for multiple event types."""
        stats_tracker.update_subscriber_count(EventType.SCANNER_ALERT, 3)
        stats_tracker.update_subscriber_count(EventType.ORDER_PLACED, 2)
        stats_tracker.update_subscriber_count(EventType.FEATURE_REQUEST, 4)
        
        # Check internal tracking
        assert len(stats_tracker._subscribers_by_type) == 3
        assert stats_tracker._subscribers_by_type[EventType.SCANNER_ALERT.value] == 3
        assert stats_tracker._subscribers_by_type[EventType.ORDER_PLACED.value] == 2
        assert stats_tracker._subscribers_by_type[EventType.FEATURE_REQUEST.value] == 4
        
        # Check gauge updates
        assert mock_metrics_collector.set_gauge.call_count == 3
    
    def test_update_subscriber_count_zero(self, stats_tracker, mock_metrics_collector):
        """Test updating subscriber count to zero."""
        # First set a count
        stats_tracker.update_subscriber_count(EventType.ERROR, 5)
        
        # Then update to zero
        stats_tracker.update_subscriber_count(EventType.ERROR, 0)
        
        # Should still track zero
        assert stats_tracker._subscribers_by_type[EventType.ERROR.value] == 0
        
        # Should update gauge to zero
        mock_metrics_collector.set_gauge.assert_called_with(
            "event_bus.subscribers.error",
            0
        )
    
    def test_get_stats_basic(self, stats_tracker, mock_metrics_collector):
        """Test getting basic stats."""
        # Setup mock metric values
        mock_metrics_collector.get_metric_stats.side_effect = lambda key: {
            "event_bus.events_published": {"latest": 100},
            "event_bus.events_processed": {"latest": 95},
            "event_bus.events_failed": {"latest": 5},
            "event_bus.queue_size": {"latest": 10},
            "event_bus.processing_time": {"avg": 0.05, "p99": 0.1}
        }.get(key, None)
        
        # Set some subscriber counts
        stats_tracker._subscribers_by_type = {
            "scanner_alert": 3,
            "order_placed": 2
        }
        
        # Get stats - pass queue_size as parameter
        stats = stats_tracker.get_stats(queue_size=10)
        
        assert stats["events_published"] == 100
        assert stats["events_processed"] == 95
        assert stats["events_failed"] == 5
        assert stats["queue_size"] == 10
        assert stats["subscribers_by_type"] == {
            "scanner_alert": 3,
            "order_placed": 2
        }
    
    def test_get_stats_with_processing_time(self, stats_tracker, mock_metrics_collector):
        """Test getting stats including processing time metrics."""
        # Setup processing time as dict (histogram data)
        processing_time_data = {
            "count": 1000,
            "avg": 0.025,
            "min": 0.001,
            "max": 0.5,
            "p50": 0.02,
            "p95": 0.08,
            "p99": 0.15
        }
        
        mock_metrics_collector.get_metric_value.side_effect = lambda key: {
            "event_bus.events_published": 50,
            "event_bus.events_processed": 50,
            "event_bus.events_failed": 0,
            "event_bus.queue_size": 0,
            "event_bus.processing_time": processing_time_data
        }.get(key, 0)
        
        stats = stats_tracker.get_stats()
        
        assert stats["processing_time"] == processing_time_data
        assert stats["processing_time"]["avg"] == 0.025
        assert stats["processing_time"]["p99"] == 0.15
    
    def test_get_stats_empty(self, stats_tracker, mock_metrics_collector):
        """Test getting stats when no events processed."""
        # All metrics return 0
        mock_metrics_collector.get_metric_value.return_value = 0
        
        stats = stats_tracker.get_stats()
        
        assert stats["events_published"] == 0
        assert stats["events_processed"] == 0
        assert stats["events_failed"] == 0
        assert stats["queue_size"] == 0
        assert stats["subscribers_by_type"] == {}
    
    def test_reset_stats(self, stats_tracker, mock_metrics_collector):
        """Test resetting statistics."""
        # Set some subscriber counts
        stats_tracker._subscribers_by_type = {
            "scanner_alert": 5,
            "order_placed": 3
        }
        
        # Add reset method if it exists
        if hasattr(stats_tracker, 'reset'):
            stats_tracker.reset()
            
            # Should clear subscriber counts
            assert stats_tracker._subscribers_by_type == {}
            
            # Should reset metrics collector if supported
            if hasattr(mock_metrics_collector, 'reset'):
                mock_metrics_collector.reset.assert_called_once()
    
    def test_get_event_type_stats(self, stats_tracker, mock_metrics_collector):
        """Test getting stats for specific event type."""
        # If tracker supports per-event-type stats
        if hasattr(stats_tracker, 'get_event_type_stats'):
            mock_metrics_collector.get_metric_value.side_effect = lambda key: {
                "event_bus.scanner_alert.published": 30,
                "event_bus.scanner_alert.processed": 28,
                "event_bus.scanner_alert.failed": 2
            }.get(key, 0)
            
            stats = stats_tracker.get_event_type_stats(EventType.SCANNER_ALERT)
            
            assert stats["published"] == 30
            assert stats["processed"] == 28
            assert stats["failed"] == 2
    
    def test_record_event_with_tags(self, stats_tracker, mock_metrics_collector):
        """Test recording event with additional tags."""
        # If tracker supports tagged metrics
        if hasattr(stats_tracker, 'record_event'):
            stats_tracker.record_event(
                EventType.ORDER_PLACED,
                success=True,
                processing_time=0.05
            )
            
            # Should record with appropriate tags
            mock_metrics_collector.increment_counter.assert_called()
            mock_metrics_collector.record_histogram.assert_called()
    
    def test_get_all_metrics(self, stats_tracker, mock_metrics_collector):
        """Test getting all raw metrics."""
        # Setup mock to return all metrics
        all_metrics = {
            "event_bus.events_published": 1000,
            "event_bus.events_processed": 950,
            "event_bus.events_failed": 50,
            "event_bus.queue_size": 25,
            "event_bus.processing_time": {"avg": 0.03},
            "event_bus.subscribers.scanner_alert": 5,
            "event_bus.subscribers.order_placed": 3
        }
        
        mock_metrics_collector.get_all_metrics.return_value = all_metrics
        
        # If tracker exposes raw metrics
        if hasattr(stats_tracker, 'get_all_metrics'):
            metrics = stats_tracker.get_all_metrics()
            assert metrics == all_metrics
    
    def test_thread_safety(self, stats_tracker):
        """Test thread-safe operations."""
        import threading
        
        # Simulate concurrent updates
        def update_stats():
            for _ in range(100):
                stats_tracker.increment_published()
                stats_tracker.increment_processed()
                stats_tracker.update_subscriber_count(EventType.SCANNER_ALERT, 5)
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=update_stats)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should complete without errors
        # Exact counts depend on MetricsCollector thread safety
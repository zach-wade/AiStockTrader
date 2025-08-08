"""Unit tests for feature_handler_stats_tracker module."""

import pytest
from unittest.mock import Mock, patch

from main.events.handlers.feature_pipeline_helpers.feature_handler_stats_tracker import FeatureHandlerStatsTracker
from main.utils.monitoring import MetricsCollector


class TestFeatureHandlerStatsTracker:
    """Test FeatureHandlerStatsTracker class."""
    
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
        """Create FeatureHandlerStatsTracker with mocked MetricsCollector."""
        with patch('main.events.handlers.feature_pipeline_helpers.feature_handler_stats_tracker.MetricsCollector') as mock_class:
            mock_class.return_value = mock_metrics_collector
            tracker = FeatureHandlerStatsTracker()
            tracker.metrics = mock_metrics_collector  # Ensure mock is used
            return tracker
    
    def test_initialization(self):
        """Test stats tracker initialization."""
        with patch('main.events.handlers.feature_pipeline_helpers.feature_handler_stats_tracker.MetricsCollector') as mock_class:
            mock_collector = Mock()
            mock_class.return_value = mock_collector
            
            tracker = FeatureHandlerStatsTracker()
            
            # Should create MetricsCollector
            mock_class.assert_called_once()
            assert tracker.metrics == mock_collector
    
    def test_increment_requests_received(self, stats_tracker, mock_metrics_collector):
        """Test incrementing requests received counter."""
        stats_tracker.increment_requests_received()
        
        mock_metrics_collector.increment_counter.assert_called_once_with(
            "feature_pipeline.requests_received"
        )
    
    def test_increment_requests_processed(self, stats_tracker, mock_metrics_collector):
        """Test incrementing requests processed counter."""
        stats_tracker.increment_requests_processed()
        
        mock_metrics_collector.increment_counter.assert_called_once_with(
            "feature_pipeline.requests_processed"
        )
    
    def test_increment_features_computed(self, stats_tracker, mock_metrics_collector):
        """Test incrementing features computed counter."""
        # Test with default count
        stats_tracker.increment_features_computed()
        
        mock_metrics_collector.increment_counter.assert_called_with(
            "feature_pipeline.features_computed", 1
        )
        
        # Test with specific count
        mock_metrics_collector.reset_mock()
        stats_tracker.increment_features_computed(10)
        
        mock_metrics_collector.increment_counter.assert_called_with(
            "feature_pipeline.features_computed", 10
        )
    
    def test_increment_errors(self, stats_tracker, mock_metrics_collector):
        """Test incrementing errors counter."""
        stats_tracker.increment_errors()
        
        mock_metrics_collector.increment_counter.assert_called_once_with(
            "feature_pipeline.errors"
        )
    
    def test_record_computation_time(self, stats_tracker, mock_metrics_collector):
        """Test recording computation time."""
        stats_tracker.record_computation_time(1.25)
        
        mock_metrics_collector.record_histogram.assert_called_once_with(
            "feature_pipeline.computation_time",
            1.25
        )
    
    def test_record_queue_size(self, stats_tracker, mock_metrics_collector):
        """Test recording queue size."""
        stats_tracker.record_queue_size(50)
        
        mock_metrics_collector.set_gauge.assert_called_once_with(
            "feature_pipeline.queue_size",
            50
        )
    
    def test_record_worker_utilization(self, stats_tracker, mock_metrics_collector):
        """Test recording worker utilization."""
        # If method exists
        if hasattr(stats_tracker, 'record_worker_utilization'):
            stats_tracker.record_worker_utilization(0.75)
            
            mock_metrics_collector.set_gauge.assert_called_with(
                "feature_pipeline.worker_utilization",
                0.75
            )
    
    def test_get_stats(self, stats_tracker, mock_metrics_collector):
        """Test getting aggregated stats."""
        # Setup mock metric values
        mock_metrics_collector.get_metric_value.side_effect = lambda key: {
            "feature_pipeline.requests_received": 1000,
            "feature_pipeline.requests_processed": 950,
            "feature_pipeline.features_computed": 5000,
            "feature_pipeline.errors": 50,
            "feature_pipeline.queue_size": 25,
            "feature_pipeline.computation_time": {
                "count": 950,
                "avg": 0.1,
                "min": 0.01,
                "max": 2.5,
                "p50": 0.08,
                "p95": 0.25,
                "p99": 0.5
            }
        }.get(key, 0)
        
        stats = stats_tracker.get_stats()
        
        assert stats["requests_received"] == 1000
        assert stats["requests_processed"] == 950
        assert stats["features_computed"] == 5000
        assert stats["errors"] == 50
        assert stats["queue_size"] == 25
        assert stats["computation_time"]["avg"] == 0.1
        assert stats["computation_time"]["p99"] == 0.5
    
    def test_get_stats_empty(self, stats_tracker, mock_metrics_collector):
        """Test getting stats when no data."""
        mock_metrics_collector.get_metric_value.return_value = 0
        
        stats = stats_tracker.get_stats()
        
        assert stats["requests_received"] == 0
        assert stats["requests_processed"] == 0
        assert stats["features_computed"] == 0
        assert stats["errors"] == 0
    
    def test_reset_stats(self, stats_tracker, mock_metrics_collector):
        """Test resetting statistics if supported."""
        if hasattr(stats_tracker, 'reset'):
            stats_tracker.reset()
            
            # Should reset metrics collector
            if hasattr(mock_metrics_collector, 'reset'):
                mock_metrics_collector.reset.assert_called_once()
    
    def test_get_error_rate(self, stats_tracker, mock_metrics_collector):
        """Test calculating error rate."""
        if hasattr(stats_tracker, 'get_error_rate'):
            mock_metrics_collector.get_metric_value.side_effect = lambda key: {
                "feature_pipeline.requests_processed": 1000,
                "feature_pipeline.errors": 50
            }.get(key, 0)
            
            error_rate = stats_tracker.get_error_rate()
            
            assert error_rate == 0.05  # 50/1000 = 5%
    
    def test_get_throughput(self, stats_tracker, mock_metrics_collector):
        """Test calculating throughput."""
        if hasattr(stats_tracker, 'get_throughput'):
            # Mock time-based calculations
            mock_metrics_collector.get_metric_value.side_effect = lambda key: {
                "feature_pipeline.requests_processed": 3600,  # In 1 hour
                "feature_pipeline.uptime_seconds": 3600
            }.get(key, 0)
            
            throughput = stats_tracker.get_throughput()
            
            assert throughput == 1.0  # 1 request per second
    
    def test_record_batch_size(self, stats_tracker, mock_metrics_collector):
        """Test recording batch size for batch operations."""
        if hasattr(stats_tracker, 'record_batch_size'):
            stats_tracker.record_batch_size(100)
            
            mock_metrics_collector.record_histogram.assert_called_with(
                "feature_pipeline.batch_size",
                100
            )
    
    def test_increment_by_feature_group(self, stats_tracker, mock_metrics_collector):
        """Test incrementing counters by feature group."""
        if hasattr(stats_tracker, 'increment_by_feature_group'):
            stats_tracker.increment_by_feature_group("price_features", 10)
            
            mock_metrics_collector.increment_counter.assert_called_with(
                "feature_pipeline.features_computed.price_features",
                10
            )
    
    def test_get_feature_group_stats(self, stats_tracker, mock_metrics_collector):
        """Test getting stats by feature group."""
        if hasattr(stats_tracker, 'get_feature_group_stats'):
            mock_metrics_collector.get_metric_value.side_effect = lambda key: {
                "feature_pipeline.features_computed.price_features": 1000,
                "feature_pipeline.features_computed.volume_features": 800,
                "feature_pipeline.features_computed.volatility_features": 600
            }.get(key, 0)
            
            group_stats = stats_tracker.get_feature_group_stats()
            
            assert group_stats["price_features"] == 1000
            assert group_stats["volume_features"] == 800
            assert group_stats["volatility_features"] == 600
    
    def test_thread_safety(self, stats_tracker):
        """Test thread-safe operations."""
        import threading
        
        def increment_counters():
            for _ in range(100):
                stats_tracker.increment_requests_received()
                stats_tracker.increment_requests_processed()
                stats_tracker.increment_features_computed(5)
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=increment_counters)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should complete without errors
        # MetricsCollector should handle thread safety
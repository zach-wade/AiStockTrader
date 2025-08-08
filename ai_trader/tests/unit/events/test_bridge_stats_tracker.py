"""Unit tests for bridge_stats_tracker module."""

import pytest
from unittest.mock import Mock, patch

from main.events.handlers.scanner_bridge_helpers.bridge_stats_tracker import BridgeStatsTracker
from main.utils.monitoring import MetricsCollector


class TestBridgeStatsTracker:
    """Test BridgeStatsTracker class."""
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock MetricsCollector."""
        collector = Mock(spec=MetricsCollector)
        collector.increment_counter = Mock()
        collector.record_gauge = Mock()
        collector.get_metric_stats = Mock()
        return collector
    
    @pytest.fixture
    def tracker(self, mock_metrics_collector):
        """Create BridgeStatsTracker instance with mocked MetricsCollector."""
        with patch('main.events.handlers.scanner_bridge_helpers.bridge_stats_tracker.MetricsCollector') as mock_class:
            mock_class.return_value = mock_metrics_collector
            tracker = BridgeStatsTracker()
            tracker.metrics = mock_metrics_collector  # Ensure mock is used
            return tracker
    
    def test_initialization(self):
        """Test stats tracker initialization."""
        with patch('main.events.handlers.scanner_bridge_helpers.bridge_stats_tracker.MetricsCollector') as mock_class:
            mock_collector = Mock()
            mock_class.return_value = mock_collector
            
            tracker = BridgeStatsTracker()
            
            # Should create MetricsCollector
            mock_class.assert_called_once()
            assert tracker.metrics == mock_collector
            assert isinstance(tracker._unique_symbols, set)
            assert len(tracker._unique_symbols) == 0
    
    def test_increment_alerts_received(self, tracker, mock_metrics_collector):
        """Test incrementing alerts received counter."""
        with patch('main.events.handlers.scanner_bridge_helpers.bridge_stats_tracker.record_metric') as mock_record:
            tracker.increment_alerts_received()
            
            # Should increment counter in MetricsCollector
            mock_metrics_collector.increment_counter.assert_called_once_with(
                "scanner_bridge.alerts_received"
            )
            
            # Should record metric
            mock_record.assert_called_once_with(
                "scanner_bridge.alerts_received", 1
            )
    
    def test_increment_feature_requests_sent(self, tracker, mock_metrics_collector):
        """Test incrementing feature requests sent counter."""
        with patch('main.events.handlers.scanner_bridge_helpers.bridge_stats_tracker.record_metric') as mock_record:
            tracker.increment_feature_requests_sent()
            
            # Should increment counter in MetricsCollector
            mock_metrics_collector.increment_counter.assert_called_once_with(
                "scanner_bridge.feature_requests_sent"
            )
            
            # Should record metric
            mock_record.assert_called_once_with(
                "scanner_bridge.feature_requests_sent", 1
            )
    
    def test_add_symbol_processed_new_symbol(self, tracker, mock_metrics_collector):
        """Test adding a new symbol to processed set."""
        with patch('main.events.handlers.scanner_bridge_helpers.bridge_stats_tracker.record_metric') as mock_record:
            tracker.add_symbol_processed("AAPL")
            
            # Should add to unique symbols set
            assert "AAPL" in tracker._unique_symbols
            assert len(tracker._unique_symbols) == 1
            
            # Should update gauge
            mock_metrics_collector.record_gauge.assert_called_once_with(
                "scanner_bridge.unique_symbols", 1
            )
            
            # Should record metric
            mock_record.assert_called_once_with(
                "scanner_bridge.symbol_processed", 
                1, 
                tags={"symbol": "AAPL"}
            )
    
    def test_add_symbol_processed_duplicate_symbol(self, tracker, mock_metrics_collector):
        """Test adding a duplicate symbol (should still update gauge)."""
        # Add symbol first time
        tracker.add_symbol_processed("GOOGL")
        mock_metrics_collector.reset_mock()
        
        with patch('main.events.handlers.scanner_bridge_helpers.bridge_stats_tracker.record_metric') as mock_record:
            # Add same symbol again
            tracker.add_symbol_processed("GOOGL")
            
            # Should still be only one unique symbol
            assert "GOOGL" in tracker._unique_symbols
            assert len(tracker._unique_symbols) == 1
            
            # Should update gauge (even though count is same)
            mock_metrics_collector.record_gauge.assert_called_once_with(
                "scanner_bridge.unique_symbols", 1
            )
            
            # Should still record metric
            mock_record.assert_called_once_with(
                "scanner_bridge.symbol_processed", 
                1, 
                tags={"symbol": "GOOGL"}
            )
    
    def test_add_multiple_symbols(self, tracker, mock_metrics_collector):
        """Test adding multiple different symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        for i, symbol in enumerate(symbols):
            tracker.add_symbol_processed(symbol)
            
            # Verify count is correct
            assert len(tracker._unique_symbols) == i + 1
            assert symbol in tracker._unique_symbols
    
    def test_get_stats_empty(self, tracker, mock_metrics_collector):
        """Test getting stats when no activity."""
        # Mock metric stats to return None (no data)
        mock_metrics_collector.get_metric_stats.return_value = None
        
        stats = tracker.get_stats()
        
        assert stats == {
            'alerts_received_total': 0,
            'feature_requests_sent_total': 0,
            'unique_symbols_processed_count': 0,
            'pending_batches_count': 0,
            'pending_symbols_count': 0
        }
        
        # Should query metrics
        assert mock_metrics_collector.get_metric_stats.call_count == 2
        mock_metrics_collector.get_metric_stats.assert_any_call("scanner_bridge.alerts_received")
        mock_metrics_collector.get_metric_stats.assert_any_call("scanner_bridge.feature_requests_sent")
    
    def test_get_stats_with_data(self, tracker, mock_metrics_collector):
        """Test getting stats with actual data."""
        # Add some symbols
        tracker.add_symbol_processed("AAPL")
        tracker.add_symbol_processed("GOOGL")
        
        # Mock metric stats
        def mock_get_stats(metric_name):
            if metric_name == "scanner_bridge.alerts_received":
                return {'latest': 150, 'count': 150}
            elif metric_name == "scanner_bridge.feature_requests_sent":
                return {'latest': 75, 'count': 75}
            return None
        
        mock_metrics_collector.get_metric_stats.side_effect = mock_get_stats
        
        # Get stats with pending counts
        stats = tracker.get_stats(pending_batches_count=5, pending_symbols_count=25)
        
        assert stats == {
            'alerts_received_total': 150,
            'feature_requests_sent_total': 75,
            'unique_symbols_processed_count': 2,
            'pending_batches_count': 5,
            'pending_symbols_count': 25
        }
    
    def test_get_stats_with_float_values(self, tracker, mock_metrics_collector):
        """Test getting stats when metrics return float values."""
        # Mock metric stats with float values
        mock_metrics_collector.get_metric_stats.side_effect = lambda name: {
            'latest': 123.45 if "alerts" in name else 67.89
        }
        
        stats = tracker.get_stats()
        
        # Should convert to int
        assert stats['alerts_received_total'] == 123
        assert stats['feature_requests_sent_total'] == 67
        assert isinstance(stats['alerts_received_total'], int)
        assert isinstance(stats['feature_requests_sent_total'], int)
    
    def test_stats_consistency(self, tracker):
        """Test that stats remain consistent through operations."""
        # Simulate activity
        for _ in range(10):
            tracker.increment_alerts_received()
        
        for _ in range(5):
            tracker.increment_feature_requests_sent()
        
        symbols = ["AAPL", "GOOGL", "AAPL", "MSFT", "GOOGL", "TSLA"]
        for symbol in symbols:
            tracker.add_symbol_processed(symbol)
        
        # Should have 4 unique symbols (AAPL, GOOGL, MSFT, TSLA)
        assert len(tracker._unique_symbols) == 4
    
    def test_concurrent_symbol_additions(self, tracker):
        """Test thread safety of symbol additions."""
        import threading
        
        def add_symbols(symbol_prefix):
            for i in range(100):
                tracker.add_symbol_processed(f"{symbol_prefix}_{i}")
        
        threads = []
        prefixes = ["A", "B", "C", "D", "E"]
        
        for prefix in prefixes:
            t = threading.Thread(target=add_symbols, args=(prefix,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have 500 unique symbols (5 threads * 100 symbols each)
        assert len(tracker._unique_symbols) == 500
    
    def test_stats_with_no_metrics_collector_data(self, tracker, mock_metrics_collector):
        """Test stats when MetricsCollector returns empty dict."""
        mock_metrics_collector.get_metric_stats.return_value = {}
        
        stats = tracker.get_stats()
        
        # Should handle empty dict gracefully
        assert stats['alerts_received_total'] == 0
        assert stats['feature_requests_sent_total'] == 0
    
    def test_large_symbol_set(self, tracker, mock_metrics_collector):
        """Test performance with large number of unique symbols."""
        # Add 10,000 unique symbols
        for i in range(10000):
            tracker.add_symbol_processed(f"SYM_{i:05d}")
        
        assert len(tracker._unique_symbols) == 10000
        
        # Getting stats should still be fast
        stats = tracker.get_stats()
        assert stats['unique_symbols_processed_count'] == 10000
    
    def test_symbol_memory_efficiency(self, tracker):
        """Test that symbols are stored efficiently (as strings in set)."""
        # Add symbol
        tracker.add_symbol_processed("TEST")
        
        # Verify it's stored as string in set
        assert isinstance(tracker._unique_symbols, set)
        for symbol in tracker._unique_symbols:
            assert isinstance(symbol, str)
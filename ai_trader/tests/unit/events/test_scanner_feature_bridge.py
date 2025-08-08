"""Unit tests for scanner_feature_bridge module."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from main.events.handlers.scanner_feature_bridge import ScannerFeatureBridge
from main.events.core.event_bus import EventBus
from main.events.types import EventType, ScanAlert, AlertType, Event

from tests.fixtures.events.mock_events import (
    create_scan_alert,
    create_multiple_alerts,
    create_feature_request_event
)
from tests.fixtures.events.mock_configs import create_scanner_bridge_config


class TestScannerFeatureBridge:
    """Test ScannerFeatureBridge class."""
    
    @pytest.fixture
    async def mock_event_bus(self):
        """Create mock event bus."""
        bus = AsyncMock(spec=EventBus)
        bus.subscribe = AsyncMock()
        bus.publish = AsyncMock()
        return bus
    
    @pytest.fixture
    async def bridge(self, mock_event_bus):
        """Create ScannerFeatureBridge instance for testing."""
        config = create_scanner_bridge_config()["scanner_feature_bridge"]
        bridge = ScannerFeatureBridge(
            event_bus=mock_event_bus,
            batch_size=config["batch_size"],
            batch_timeout_seconds=config["batch_timeout_seconds"],
            max_symbols_per_batch=config["max_symbols_per_batch"],
            rate_limit_per_second=config["rate_limit_per_second"],
            dedup_window_seconds=config["dedup_window_seconds"]
        )
        return bridge
    
    def test_initialization(self, mock_event_bus):
        """Test bridge initialization with various configurations."""
        # Test with default values
        bridge = ScannerFeatureBridge(event_bus=mock_event_bus)
        
        assert bridge.event_bus == mock_event_bus
        assert bridge.batch_size == 50
        assert bridge.batch_timeout_seconds == 5.0
        assert bridge.max_symbols_per_batch == 100
        assert bridge.dedup_window_seconds == 60
        assert bridge._running is False
        
        # Test with custom values
        bridge2 = ScannerFeatureBridge(
            event_bus=mock_event_bus,
            batch_size=20,
            batch_timeout_seconds=2.0,
            max_symbols_per_batch=50,
            rate_limit_per_second=20,
            dedup_window_seconds=30,
            config={"custom": "value"}
        )
        
        assert bridge2.batch_size == 20
        assert bridge2.batch_timeout_seconds == 2.0
        assert bridge2.max_symbols_per_batch == 50
        assert bridge2.dedup_window_seconds == 30
        assert bridge2.config == {"custom": "value"}
    
    @pytest.mark.asyncio
    async def test_start_stop(self, bridge, mock_event_bus):
        """Test starting and stopping the bridge."""
        # Initially not running
        assert bridge._running is False
        assert bridge._batch_task is None
        
        # Start the bridge
        await bridge.start()
        
        # Should be running
        assert bridge._running is True
        assert bridge._batch_task is not None
        
        # Should subscribe to scanner alerts
        mock_event_bus.subscribe.assert_called_once()
        call_args = mock_event_bus.subscribe.call_args[0]
        assert call_args[0] == EventType.SCANNER_ALERT
        
        # Stop the bridge
        await bridge.stop()
        
        # Should not be running
        assert bridge._running is False
        assert bridge._batch_task is None
    
    @pytest.mark.asyncio
    async def test_handle_scanner_alert(self, bridge):
        """Test handling scanner alerts."""
        # Create mock methods
        bridge.alert_mapper.map_alert_to_features = Mock(
            return_value=["price_features", "volume_features"]
        )
        bridge.request_batcher.add_symbol_features = Mock()
        
        # Handle alert
        alert = create_scan_alert(symbol="AAPL", score=0.85)
        await bridge._handle_scanner_alert(alert)
        
        # Should map alert to features
        bridge.alert_mapper.map_alert_to_features.assert_called_once_with("high_volume")
        
        # Should add to batcher
        bridge.request_batcher.add_symbol_features.assert_called_once()
        call_args = bridge.request_batcher.add_symbol_features.call_args[0]
        assert call_args[0] == "AAPL"
        assert "price_features" in call_args[1]
        assert "volume_features" in call_args[1]
    
    @pytest.mark.asyncio
    async def test_deduplication(self, bridge):
        """Test symbol deduplication within window."""
        bridge.alert_mapper.map_alert_to_features = Mock(
            return_value=["price_features"]
        )
        bridge.request_batcher.add_symbol_features = Mock()
        
        # Handle same symbol multiple times
        alert1 = create_scan_alert(symbol="AAPL", score=0.8)
        alert2 = create_scan_alert(symbol="AAPL", score=0.9)
        
        await bridge._handle_scanner_alert(alert1)
        await bridge._handle_scanner_alert(alert2)
        
        # Should only process once due to deduplication
        assert bridge.request_batcher.add_symbol_features.call_count == 1
        
        # Stats should reflect skipped duplicate
        stats = bridge.get_stats()
        assert stats["duplicates_skipped_total"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, bridge):
        """Test batch processing of requests."""
        # Mock components
        bridge.alert_mapper.map_alert_to_features = Mock(
            return_value=["price_features"]
        )
        bridge.priority_calculator.calculate_base_priority = Mock(return_value=5)
        bridge.request_dispatcher.send_feature_request_batch = AsyncMock()
        
        # Manually create batch
        bridge.request_batcher._current_batch.symbols = {"AAPL", "GOOGL", "MSFT"}
        bridge.request_batcher._current_batch.features = {"price_features", "volume_features"}
        bridge.request_batcher._current_batch.priority = 5
        
        # Process batch
        await bridge._process_batch()
        
        # Should send batch via dispatcher
        bridge.request_dispatcher.send_feature_request_batch.assert_called_once()
        batch_sent = bridge.request_dispatcher.send_feature_request_batch.call_args[0][0]
        assert len(batch_sent.symbols) == 3
        assert "AAPL" in batch_sent.symbols
    
    @pytest.mark.asyncio
    async def test_batch_timeout(self, bridge):
        """Test batch timeout triggers processing."""
        bridge._running = True
        bridge.batch_timeout_seconds = 0.1  # Short timeout
        
        # Mock batch processing
        bridge._process_batch = AsyncMock()
        
        # Start batch task
        bridge._batch_task = asyncio.create_task(bridge._batch_processor())
        
        # Add something to batch
        bridge.request_batcher._current_batch.symbols.add("AAPL")
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        
        # Should have processed batch
        assert bridge._process_batch.called
        
        # Stop
        bridge._running = False
        await bridge._batch_task
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, bridge):
        """Test rate limiting functionality."""
        # Configure rate limiter
        bridge.rate_limiter = AsyncMock()
        bridge.rate_limiter.acquire = AsyncMock()
        
        # Process batch
        bridge.request_batcher._current_batch.symbols = {"AAPL"}
        bridge.request_dispatcher.send_feature_request_batch = AsyncMock()
        
        await bridge._process_batch()
        
        # Should acquire rate limit
        bridge.rate_limiter.acquire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_priority_calculation(self, bridge):
        """Test priority calculation for batches."""
        # Mock priority calculator
        bridge.priority_calculator.calculate_base_priority = Mock(return_value=3)
        bridge.priority_calculator.apply_boosts = Mock(return_value=5)
        
        # Create alerts with different scores
        alerts = [
            create_scan_alert(symbol="AAPL", score=0.9),
            create_scan_alert(symbol="GOOGL", score=0.7)
        ]
        
        # Calculate priority
        priority = bridge._calculate_batch_priority(alerts)
        
        # Should use priority calculator
        bridge.priority_calculator.calculate_base_priority.assert_called()
        bridge.priority_calculator.apply_boosts.assert_called()
        assert priority == 5
    
    @pytest.mark.asyncio
    async def test_get_stats(self, bridge):
        """Test getting bridge statistics."""
        # Process some alerts
        bridge.alert_mapper.map_alert_to_features = Mock(
            return_value=["price_features"]
        )
        bridge.request_batcher.add_symbol_features = Mock()
        
        # Handle alerts
        await bridge._handle_scanner_alert(create_scan_alert(symbol="AAPL"))
        await bridge._handle_scanner_alert(create_scan_alert(symbol="GOOGL"))
        
        # Get stats
        stats = bridge.get_stats()
        
        assert "alerts_received_total" in stats
        assert "feature_requests_sent_total" in stats
        assert "unique_symbols_processed_count" in stats
        assert "duplicates_skipped_total" in stats
        assert "current_batch_size" in stats
        
        assert stats["alerts_received_total"] == 2
        assert stats["unique_symbols_processed_count"] == 2
    
    @pytest.mark.asyncio
    async def test_error_handling_in_alert_processing(self, bridge):
        """Test error handling when processing alerts."""
        # Make mapper raise error
        bridge.alert_mapper.map_alert_to_features = Mock(
            side_effect=Exception("Mapping failed")
        )
        
        # Should not raise exception
        alert = create_scan_alert()
        await bridge._handle_scanner_alert(alert)
        
        # Stats should reflect error
        stats = bridge.get_stats()
        # Implementation may track errors
    
    @pytest.mark.asyncio
    async def test_split_large_batches(self, bridge):
        """Test splitting large batches."""
        bridge.max_symbols_per_batch = 3
        bridge.request_dispatcher.send_feature_request_batch = AsyncMock()
        
        # Create large batch
        for i in range(7):
            bridge.request_batcher._current_batch.symbols.add(f"STOCK{i}")
        bridge.request_batcher._current_batch.features = {"price_features"}
        
        # Process batch
        await bridge._process_batch()
        
        # Should have sent multiple batches
        assert bridge.request_dispatcher.send_feature_request_batch.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_cleanup_old_symbols(self, bridge):
        """Test cleanup of old deduplicated symbols."""
        # Add old symbol
        old_time = datetime.now(timezone.utc).timestamp() - 120  # 2 minutes ago
        bridge._processed_symbols["OLD_STOCK"] = old_time
        
        # Add recent symbol
        bridge._processed_symbols["NEW_STOCK"] = datetime.now(timezone.utc).timestamp()
        
        # Process alert to trigger cleanup
        bridge.alert_mapper.map_alert_to_features = Mock(return_value=[])
        await bridge._handle_scanner_alert(create_scan_alert(symbol="AAPL"))
        
        # Old symbol should be cleaned up
        assert "OLD_STOCK" not in bridge._processed_symbols
        assert "NEW_STOCK" in bridge._processed_symbols
    
    @pytest.mark.asyncio
    async def test_integration_with_helpers(self, bridge):
        """Test integration with all helper components."""
        # Create alert
        alert = create_scan_alert(
            symbol="AAPL",
            alert_type=AlertType.VOLATILITY_SPIKE,
            score=0.95
        )
        
        # Mock helper responses
        bridge.alert_mapper.map_alert_to_features = Mock(
            return_value=["volatility_features", "risk_metrics"]
        )
        bridge.priority_calculator.calculate_base_priority = Mock(return_value=7)
        bridge.priority_calculator.apply_boosts = Mock(return_value=9)
        bridge.request_dispatcher.send_feature_request_batch = AsyncMock()
        
        # Process alert
        await bridge._handle_scanner_alert(alert)
        
        # Force batch processing
        await bridge._process_batch()
        
        # Verify integration
        bridge.alert_mapper.map_alert_to_features.assert_called_with("volatility_spike")
        bridge.priority_calculator.calculate_base_priority.assert_called()
        bridge.request_dispatcher.send_feature_request_batch.assert_called()
        
        # Check stats
        stats = bridge.stats_tracker.get_stats()
        assert stats["alerts_received_total"] >= 1
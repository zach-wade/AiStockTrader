"""Unit tests for request_dispatcher module."""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from main.events.handlers.scanner_bridge_helpers.request_dispatcher import RequestDispatcher
from main.events.handlers.scanner_bridge_helpers.feature_request_batcher import FeatureRequestBatch
from main.events.core.event_bus import EventBus
from main.events.types import FeatureRequestEvent


class TestRequestDispatcher:
    """Test RequestDispatcher class."""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock EventBus."""
        bus = Mock(spec=EventBus)
        bus.publish = AsyncMock()
        return bus
    
    @pytest.fixture
    def dispatcher(self, mock_event_bus):
        """Create RequestDispatcher instance."""
        return RequestDispatcher(event_bus=mock_event_bus)
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample FeatureRequestBatch for testing."""
        batch = FeatureRequestBatch(
            symbols={"AAPL", "GOOGL", "MSFT"},
            features={"price_action", "volume_profile", "momentum"},
            priority=7,
            scanner_sources={"scanner1", "scanner2"},
            correlation_ids={"corr_123", "corr_456"}
        )
        # Set created_at to a known time for testing
        batch.created_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        return batch
    
    def test_initialization(self, mock_event_bus):
        """Test dispatcher initialization."""
        dispatcher = RequestDispatcher(event_bus=mock_event_bus)
        
        assert dispatcher.event_bus == mock_event_bus
    
    @pytest.mark.asyncio
    async def test_send_feature_request_batch_success(self, dispatcher, mock_event_bus, sample_batch):
        """Test successful sending of feature request batch."""
        with patch('main.events.handlers.scanner_bridge_helpers.request_dispatcher.record_metric') as mock_metric:
            await dispatcher.send_feature_request_batch(sample_batch)
            
            # Verify event was published
            mock_event_bus.publish.assert_called_once()
            
            # Get the published event
            published_event = mock_event_bus.publish.call_args[0][0]
            
            # Verify event type
            assert isinstance(published_event, FeatureRequestEvent)
            
            # Verify event data
            assert set(published_event.symbols) == {"AAPL", "GOOGL", "MSFT"}
            assert set(published_event.features) == {"price_action", "volume_profile", "momentum"}
            assert published_event.requester == 'scanner_bridge'
            assert published_event.priority == 7
            
            # Verify metadata
            assert set(published_event.metadata['scanner_sources']) == {"scanner1", "scanner2"}
            assert published_event.metadata['batch_size_symbols'] == 3
            assert published_event.metadata['request_age_seconds'] >= 5
            assert set(published_event.metadata['correlation_ids']) == {"corr_123", "corr_456"}
            
            # Verify metric was recorded
            mock_metric.assert_called_once_with(
                "scanner_bridge.request_dispatched",
                1,
                tags={
                    "symbols": 3,
                    "features": 3,
                    "priority": 7
                }
            )
    
    @pytest.mark.asyncio
    async def test_send_empty_batch(self, dispatcher, mock_event_bus):
        """Test handling of empty batch."""
        empty_batch = FeatureRequestBatch()  # No symbols
        
        await dispatcher.send_feature_request_batch(empty_batch)
        
        # Should not publish event
        mock_event_bus.publish.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_send_batch_with_single_symbol(self, dispatcher, mock_event_bus):
        """Test sending batch with single symbol."""
        single_batch = FeatureRequestBatch(
            symbols={"AAPL"},
            features={"price_action"},
            priority=10
        )
        
        await dispatcher.send_feature_request_batch(single_batch)
        
        # Should publish event
        mock_event_bus.publish.assert_called_once()
        
        published_event = mock_event_bus.publish.call_args[0][0]
        assert published_event.symbols == ["AAPL"]
        assert published_event.features == ["price_action"]
        assert published_event.priority == 10
    
    @pytest.mark.asyncio
    async def test_send_batch_no_correlation_ids(self, dispatcher, mock_event_bus):
        """Test sending batch without correlation IDs."""
        batch = FeatureRequestBatch(
            symbols={"AAPL"},
            features={"price"},
            priority=5
        )
        
        await dispatcher.send_feature_request_batch(batch)
        
        published_event = mock_event_bus.publish.call_args[0][0]
        assert published_event.metadata['correlation_ids'] == []
    
    @pytest.mark.asyncio
    async def test_request_age_calculation(self, dispatcher, mock_event_bus):
        """Test accurate request age calculation."""
        batch = FeatureRequestBatch(
            symbols={"AAPL"},
            features={"price"},
            priority=5
        )
        
        # Set specific created time
        batch.created_at = datetime.now(timezone.utc) - timedelta(seconds=10.5)
        
        await dispatcher.send_feature_request_batch(batch)
        
        published_event = mock_event_bus.publish.call_args[0][0]
        age = published_event.metadata['request_age_seconds']
        
        # Should be approximately 10.5 seconds (allow small variance)
        assert 10 <= age <= 11
    
    @pytest.mark.asyncio
    async def test_error_handling_during_publish(self, dispatcher, mock_event_bus, sample_batch):
        """Test error handling when event publish fails."""
        mock_event_bus.publish.side_effect = Exception("Event bus error")
        
        # Should handle error due to ErrorHandlingMixin
        with pytest.raises(Exception, match="Event bus error"):
            await dispatcher.send_feature_request_batch(sample_batch)
    
    @pytest.mark.asyncio
    async def test_multiple_batches_sequential(self, dispatcher, mock_event_bus):
        """Test sending multiple batches sequentially."""
        batches = [
            FeatureRequestBatch(
                symbols={f"STOCK{i}"},
                features={"price"},
                priority=i
            )
            for i in range(5)
        ]
        
        for batch in batches:
            await dispatcher.send_feature_request_batch(batch)
        
        # Should have published 5 events
        assert mock_event_bus.publish.call_count == 5
        
        # Verify each event
        for i, call in enumerate(mock_event_bus.publish.call_args_list):
            event = call[0][0]
            assert event.symbols == [f"STOCK{i}"]
            assert event.priority == i
    
    @pytest.mark.asyncio
    async def test_concurrent_batch_sending(self, dispatcher, mock_event_bus):
        """Test concurrent sending of multiple batches."""
        batches = [
            FeatureRequestBatch(
                symbols={f"STOCK{i}"},
                features={"feature"},
                priority=5
            )
            for i in range(10)
        ]
        
        # Send all batches concurrently
        tasks = [dispatcher.send_feature_request_batch(batch) for batch in batches]
        await asyncio.gather(*tasks)
        
        # Should have published all events
        assert mock_event_bus.publish.call_count == 10
    
    @pytest.mark.asyncio
    async def test_large_batch_handling(self, dispatcher, mock_event_bus):
        """Test handling of large batch with many symbols."""
        large_batch = FeatureRequestBatch(
            symbols={f"SYMBOL_{i:04d}" for i in range(1000)},
            features={"price", "volume", "momentum", "volatility"},
            priority=8,
            scanner_sources={"scanner1", "scanner2", "scanner3"}
        )
        
        with patch('main.events.handlers.scanner_bridge_helpers.request_dispatcher.record_metric') as mock_metric:
            await dispatcher.send_feature_request_batch(large_batch)
            
            # Verify event was published
            published_event = mock_event_bus.publish.call_args[0][0]
            assert len(published_event.symbols) == 1000
            assert len(published_event.features) == 4
            
            # Verify metric
            mock_metric.assert_called_once_with(
                "scanner_bridge.request_dispatched",
                1,
                tags={
                    "symbols": 1000,
                    "features": 4,
                    "priority": 8
                }
            )
    
    @pytest.mark.asyncio
    async def test_metadata_preservation(self, dispatcher, mock_event_bus):
        """Test that all metadata is properly preserved."""
        batch = FeatureRequestBatch(
            symbols={"TEST"},
            features={"feature"},
            priority=5,
            scanner_sources={"source1", "source2", "source3"},
            correlation_ids={"id1", "id2", "id3", "id4", "id5"}
        )
        
        await dispatcher.send_feature_request_batch(batch)
        
        event = mock_event_bus.publish.call_args[0][0]
        
        # All metadata should be preserved
        assert len(event.metadata['scanner_sources']) == 3
        assert len(event.metadata['correlation_ids']) == 5
        assert all(s in event.metadata['scanner_sources'] for s in ["source1", "source2", "source3"])
        assert all(id in event.metadata['correlation_ids'] for id in ["id1", "id2", "id3", "id4", "id5"])
    
    @pytest.mark.asyncio
    async def test_feature_order_consistency(self, dispatcher, mock_event_bus):
        """Test that feature order is consistent when converting set to list."""
        # Create multiple batches with same features
        for _ in range(10):
            batch = FeatureRequestBatch(
                symbols={"TEST"},
                features={"gamma", "alpha", "beta", "delta"},
                priority=5
            )
            
            await dispatcher.send_feature_request_batch(batch)
        
        # All events should have consistent feature order
        feature_lists = []
        for call in mock_event_bus.publish.call_args_list:
            event = call[0][0]
            feature_lists.append(event.features)
        
        # All feature lists should be equal (same order)
        for features in feature_lists[1:]:
            assert features == feature_lists[0]
    
    @pytest.mark.asyncio
    async def test_error_handling_inheritance(self, dispatcher):
        """Test that dispatcher inherits from ErrorHandlingMixin."""
        assert hasattr(dispatcher, '_handle_error')
    
    @pytest.mark.asyncio
    async def test_event_properties(self, dispatcher, mock_event_bus, sample_batch):
        """Test that created event has all required properties."""
        await dispatcher.send_feature_request_batch(sample_batch)
        
        event = mock_event_bus.publish.call_args[0][0]
        
        # Verify event has all required attributes
        assert hasattr(event, 'symbols')
        assert hasattr(event, 'features')
        assert hasattr(event, 'requester')
        assert hasattr(event, 'priority')
        assert hasattr(event, 'metadata')
        assert hasattr(event, 'event_type')
        assert hasattr(event, 'timestamp')
        assert hasattr(event, 'event_id')
    
    @pytest.mark.asyncio
    async def test_logging_output(self, dispatcher, mock_event_bus, sample_batch):
        """Test that appropriate log messages are generated."""
        with patch('main.events.handlers.scanner_bridge_helpers.request_dispatcher.logger') as mock_logger:
            await dispatcher.send_feature_request_batch(sample_batch)
            
            # Should log info message
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            
            assert "3 symbols" in log_message
            assert "3 features" in log_message
            assert "priority 7" in log_message
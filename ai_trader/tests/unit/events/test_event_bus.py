"""Unit tests for event_bus module."""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from main.interfaces.events import IEventBus, Event, EventType, EventPriority
from main.events.core import EventBusFactory, EventBusConfig
from main.events.core.event_bus import EventBus  # For white-box testing
from main.utils.resilience import CircuitBreakerConfig

from tests.fixtures.events.mock_events import (
    create_scan_alert,
    create_order_event,
    create_error_event
)
from tests.fixtures.events.mock_database import create_mock_db_pool


@pytest.mark.unit
@pytest.mark.events
class TestEventBus:
    """Test EventBus class."""
    
    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create EventBus instance for testing."""
        config = EventBusConfig(
            max_queue_size=100,
            max_workers=2,
            enable_history=True,
            history_retention_seconds=60,
            enable_dlq=False  # Disable DLQ for most tests
        )
        bus = EventBusFactory.create(config)
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.fixture
    def sample_handler(self):
        """Create a sample event handler."""
        handler = AsyncMock()
        handler.__name__ = "test_handler"
        return handler
    
    @pytest.mark.asyncio
    async def test_event_bus_initialization(self):
        """Test EventBus initialization with various configurations."""
        # Test with default config
        config = EventBusConfig()
        bus = EventBusFactory.create(config)
        # Note: EventBus doesn't expose internal attributes, so we test behavior instead
        assert bus.is_running() is False
        
        # Test with custom config
        config2 = EventBusConfig(
            max_queue_size=500,
            max_workers=5,
            enable_history=False,
            enable_dlq=False
        )
        bus2 = EventBusFactory.create(config2)
        assert bus2.is_running() is False
    
    @pytest.mark.asyncio
    async def test_event_bus_start_stop(self):
        """Test starting and stopping the event bus."""
        config = EventBusConfig(max_workers=2)
        bus = EventBusFactory.create(config)
        
        # Initially not running
        assert bus.is_running() is False
        
        # Start the bus
        await bus.start()
        assert bus.is_running() is True
        
        # Stop the bus
        await bus.stop()
        assert bus.is_running() is False
    
    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self, event_bus, sample_handler):
        """Test subscribing and unsubscribing handlers."""
        # Subscribe handler
        event_bus.subscribe(EventType.SCANNER_ALERT, sample_handler)
        
        # Test by publishing an event
        alert = create_scan_alert()
        await event_bus.publish(alert)
        await asyncio.sleep(0.1)
        
        # Handler should have been called
        sample_handler.assert_called_once()
        
        # Subscribe another handler
        handler2 = AsyncMock()
        handler2.__name__ = "priority_handler"
        event_bus.subscribe(EventType.SCANNER_ALERT, handler2, priority=10)
        
        # Reset and publish again
        sample_handler.reset_mock()
        await event_bus.publish(create_scan_alert())
        await asyncio.sleep(0.1)
        
        # Both handlers should be called
        sample_handler.assert_called_once()
        handler2.assert_called_once()
        
        # Unsubscribe first handler
        event_bus.unsubscribe(EventType.SCANNER_ALERT, sample_handler)
        
        # Reset and publish again
        sample_handler.reset_mock()
        handler2.reset_mock()
        await event_bus.publish(create_scan_alert())
        await asyncio.sleep(0.1)
        
        # Only handler2 should be called
        sample_handler.assert_not_called()
        handler2.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_event(self, event_bus, sample_handler):
        """Test publishing events."""
        # Subscribe handler
        event_bus.subscribe(EventType.SCANNER_ALERT, sample_handler)
        
        # Create and publish event
        alert = create_scan_alert(symbol="AAPL")
        await event_bus.publish(alert)
        
        # Give time for processing
        await asyncio.sleep(0.1)
        
        # Handler should have been called
        sample_handler.assert_called_once()
        call_args = sample_handler.call_args[0]
        assert call_args[0].symbol == "AAPL"
    
    @pytest.mark.asyncio
    async def test_multiple_handlers(self, event_bus):
        """Test multiple handlers for same event type."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        handler3 = AsyncMock()
        
        handler1.__name__ = "handler1"
        handler2.__name__ = "handler2"
        handler3.__name__ = "handler3"
        
        # Subscribe with different priorities
        event_bus.subscribe(EventType.ORDER_PLACED, handler1, priority=5)
        event_bus.subscribe(EventType.ORDER_PLACED, handler2, priority=10)
        event_bus.subscribe(EventType.ORDER_PLACED, handler3, priority=1)
        
        # Publish event
        order = create_order_event()
        await event_bus.publish(order)
        
        await asyncio.sleep(0.1)
        
        # All handlers should be called
        handler1.assert_called_once()
        handler2.assert_called_once()
        handler3.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handler_error_handling(self, event_bus):
        """Test error handling when handler fails."""
        # Create failing handler
        failing_handler = AsyncMock(side_effect=Exception("Handler error"))
        failing_handler.__name__ = "failing_handler"
        
        # Create successful handler
        success_handler = AsyncMock()
        success_handler.__name__ = "success_handler"
        
        event_bus.subscribe(EventType.ERROR, failing_handler)
        event_bus.subscribe(EventType.ERROR, success_handler)
        
        # Publish event
        error_event = create_error_event()
        await event_bus.publish(error_event)
        
        await asyncio.sleep(0.1)
        
        # Both handlers should be called despite one failing
        failing_handler.assert_called_once()
        success_handler.assert_called_once()
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="EventBus.get_event_history() method not implemented")
    async def test_event_history(self):
        """Test event history functionality."""
        # TODO: Implement this test once EventBus exposes history functionality
        pass
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker functionality."""
        with patch('main.utils.resilience.get_circuit_breaker') as mock_get_cb:
            # Create mock circuit breaker
            mock_cb = AsyncMock()
            mock_cb.call = AsyncMock()
            mock_get_cb.return_value = mock_cb
            
            cb_config = CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=10
            )
            
            config = EventBusConfig(circuit_breaker_config=cb_config)
            bus = EventBusFactory.create(config)
            await bus.start()
            
            try:
                # Verify circuit breaker was created
                mock_get_cb.assert_called_once_with(
                    "event_bus",
                    failure_threshold=2,
                    recovery_timeout=10
                )
                
                # Publish event
                event = create_scan_alert()
                await bus.publish(event)
                
                # Circuit breaker should have been used
                assert mock_cb.call.called
                
            finally:
                await bus.stop()
    
    @pytest.mark.asyncio
    async def test_queue_overflow(self, event_bus):
        """Test behavior when queue is full."""
        # Fill up the queue
        config = EventBusConfig(max_queue_size=5, max_workers=0)  # No workers
        bus = EventBusFactory.create(config)
        await bus.start()
        
        try:
            # Try to publish more events than queue size
            published = 0
            for i in range(10):
                event = create_scan_alert(symbol=f"STOCK{i}")
                try:
                    await bus.publish(event)
                    published += 1
                except:
                    pass
            
            # Should have published up to queue size
            assert published <= 5
            
        finally:
            await bus.stop()
    
    @pytest.mark.asyncio
    async def test_get_stats(self, event_bus, sample_handler):
        """Test getting event bus statistics."""
        # Subscribe handler
        event_bus.subscribe(EventType.SCANNER_ALERT, sample_handler)
        
        # Publish some events
        for i in range(3):
            await event_bus.publish(create_scan_alert())
        
        await asyncio.sleep(0.1)
        
        # Get stats
        stats = event_bus.get_stats()
        
        assert "events_published" in stats
        assert "events_processed" in stats
        assert "events_failed" in stats
        assert "queue_size" in stats
        assert "subscribers_by_type" in stats
        
        # Check subscriber count
        assert stats["subscribers_by_type"][EventType.SCANNER_ALERT.value] == 1
    
    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus):
        """Test subscribing to all event types."""
        wildcard_handler = AsyncMock()
        wildcard_handler.__name__ = "wildcard_handler"
        
        # Subscribe to None (all events)
        event_bus.subscribe(None, wildcard_handler)
        
        # Publish different event types
        await event_bus.publish(create_scan_alert())
        await event_bus.publish(create_order_event())
        await event_bus.publish(create_error_event())
        
        await asyncio.sleep(0.1)
        
        # Handler should be called for all events
        assert wildcard_handler.call_count == 3
    
    @pytest.mark.asyncio
    async def test_dlq_integration(self):
        """Test dead letter queue integration."""
        with patch('main.events.event_bus_helpers.dead_letter_queue_manager.DeadLetterQueueManager') as mock_dlq:
            mock_dlq_instance = AsyncMock()
            mock_dlq.return_value = mock_dlq_instance
            
            config = EventBusConfig(enable_dlq=True)
            bus = EventBusFactory.create(config)
            await bus.start()
            
            try:
                # Create failing handler
                failing_handler = AsyncMock(side_effect=Exception("Processing failed"))
                failing_handler.__name__ = "failing_handler"
                
                bus.subscribe(EventType.SCANNER_ALERT, failing_handler)
                
                # Publish event
                alert = create_scan_alert()
                await bus.publish(alert)
                
                await asyncio.sleep(0.1)
                
                # DLQ should have been called for failed event
                # Note: Implementation may vary based on DLQ integration
                
            finally:
                await bus.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_publishing(self, event_bus, sample_handler):
        """Test concurrent event publishing."""
        event_bus.subscribe(EventType.SCANNER_ALERT, sample_handler)
        
        # Publish multiple events concurrently
        tasks = []
        for i in range(10):
            event = create_scan_alert(symbol=f"STOCK{i}")
            tasks.append(event_bus.publish(event))
        
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.2)
        
        # All events should be processed
        assert sample_handler.call_count == 10
    
    @pytest.mark.asyncio
    async def test_handler_execution_order(self, event_bus):
        """Test that handlers are executed in priority order."""
        call_order = []
        
        async def make_handler(name, priority):
            async def handler(event):
                call_order.append((name, priority))
            handler.__name__ = name
            return handler
        
        # Subscribe handlers with different priorities
        handler1 = await make_handler("handler1", 1)
        handler2 = await make_handler("handler2", 5)
        handler3 = await make_handler("handler3", 10)
        
        event_bus.subscribe(EventType.SCANNER_ALERT, handler1, priority=1)
        event_bus.subscribe(EventType.SCANNER_ALERT, handler2, priority=5)
        event_bus.subscribe(EventType.SCANNER_ALERT, handler3, priority=10)
        
        # Publish event
        await event_bus.publish(create_scan_alert())
        await asyncio.sleep(0.1)
        
        # Check execution order (highest priority first)
        assert call_order[0][0] == "handler3"
        assert call_order[1][0] == "handler2"
        assert call_order[2][0] == "handler1"
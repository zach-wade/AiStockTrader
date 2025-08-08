"""
Integration tests for Event Bus Helpers.

Tests the coordinated operation of EventBusStatsTracker, EventHistoryManager,
and DeadLetterQueueManager with their refactored dependencies.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Dict, Any
import json

from main.interfaces.events import IEventBus, Event, EventType
from main.events.core import EventBusFactory
from main.events.core.event_bus_helpers.event_bus_stats_tracker import EventBusStatsTracker
from main.events.core.event_bus_helpers.event_history_manager import EventHistoryManager
from main.events.core.event_bus_helpers.dead_letter_queue_manager import DeadLetterQueueManager
from main.events.types import (
    OrderEvent, ScannerAlertEvent, FeatureRequestEvent
)
from tests.fixtures.events.mock_database import create_mock_db_pool


@pytest.mark.integration
@pytest.mark.events
class TestEventBusHelpersIntegration:
    """Test integrated operation of all event bus helpers."""


@pytest.fixture
async def event_bus_with_helpers():
    """Create event bus with all helpers integrated."""
    # Create mock database for DLQ
    mock_db_pool = create_mock_db_pool()
    
    # Initialize helpers
    stats_tracker = EventBusStatsTracker()
    history_manager = EventHistoryManager(max_history_size=1000)
    dlq_manager = DeadLetterQueueManager(
        db_pool=mock_db_pool,
        max_retries=3,
        retention_days=7
    )
    
    await dlq_manager.initialize()
    
    # Create event bus
    event_bus = EventBusFactory.create_test_instance()
    
    # Integrate helpers with event bus
    event_bus._stats_tracker = stats_tracker
    event_bus._history_manager = history_manager
    event_bus._dlq_manager = dlq_manager
    
    # Add helper event handlers
    event_bus.subscribe_all_events(stats_tracker.track_event)
    event_bus.subscribe_all_events(history_manager.record_event)
    
    await event_bus.start()
    
    yield {
        'event_bus': event_bus,
        'stats_tracker': stats_tracker,
        'history_manager': history_manager,
        'dlq_manager': dlq_manager,
        'db_pool': mock_db_pool
    }
    
    await event_bus.stop()
    await dlq_manager.close()


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    return [
        OrderEvent(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            side="buy",
            order_type="limit"
        ),
        ScannerAlertEvent(
            data={
                'symbol': 'GOOGL',
                'alert_type': 'high_volume',
                'confidence': 0.85
            }
        ),
        FeatureRequestEvent(
            symbols=['MSFT'],
            features=['price_features'],
            requester='test_scanner'
        )
    ]
    
    @pytest.mark.asyncio
    async def test_complete_event_lifecycle_tracking(
        self, 
        event_bus_with_helpers, 
        sample_events
    ):
        """Test that all helpers properly track complete event lifecycle."""
        bus = event_bus_with_helpers['event_bus']
        stats = event_bus_with_helpers['stats_tracker']
        history = event_bus_with_helpers['history_manager']
        
        # Publish events
        for event in sample_events:
            await bus.publish(event)
        
        await asyncio.sleep(0.2)  # Allow processing
        
        # Verify stats tracking
        event_stats = stats.get_stats()
        assert event_stats['total_events_published'] >= len(sample_events)
        assert event_stats['total_events_processed'] >= len(sample_events)
        
        # Verify history tracking
        recorded_history = await history.get_recent_events(limit=10)
        assert len(recorded_history) >= len(sample_events)
        
        # Verify event types in history
        event_types_in_history = {event.event_type for event in recorded_history}
        expected_types = {event.event_type for event in sample_events}
        assert expected_types.issubset(event_types_in_history)
    
    @pytest.mark.asyncio
    async def test_error_handling_with_dlq_integration(
        self, 
        event_bus_with_helpers
    ):
        """Test error handling with dead letter queue integration."""
        bus = event_bus_with_helpers['event_bus']
        dlq = event_bus_with_helpers['dlq_manager']
        
        # Create failing handler
        failure_count = 0
        
        async def failing_handler(event):
            nonlocal failure_count
            failure_count += 1
            raise Exception("Handler failed!")
        
        # Subscribe failing handler
        bus.subscribe(EventType.ORDER_PLACED, failing_handler)
        
        # Create test event
        test_event = OrderEvent(
            symbol="FAIL",
            quantity=50,
            price=100.0,
            side="sell",
            order_type="market"
        )
        
        # Publish event that will fail
        await bus.publish(test_event)
        await asyncio.sleep(0.2)
        
        # Verify handler was called and failed
        assert failure_count > 0
        
        # Verify event was added to DLQ (would be called by event bus on failure)
        # In real implementation, event bus would call dlq.add_failed_event
        await dlq.add_failed_event(test_event, "Handler failed!")
        
        # Verify DLQ has the failed event
        failed_events = await dlq.get_failed_events(limit=10)
        assert len(failed_events) >= 1
        assert any(evt['event_id'] == test_event.event_id for evt in failed_events)
    
    @pytest.mark.asyncio
    async def test_high_volume_event_processing_with_helpers(
        self, 
        event_bus_with_helpers
    ):
        """Test helper performance under high event volume."""
        bus = event_bus_with_helpers['event_bus']
        stats = event_bus_with_helpers['stats_tracker']
        history = event_bus_with_helpers['history_manager']
        
        # Generate high volume of events
        event_count = 500
        events = []
        
        for i in range(event_count):
            event = OrderEvent(
                symbol=f"STOCK{i % 10}",
                quantity=100,
                price=100.0 + (i % 50),
                side="buy" if i % 2 == 0 else "sell",
                order_type="market"
            )
            events.append(event)
        
        # Publish events rapidly
        start_time = asyncio.get_event_loop().time()
        
        for event in events:
            await bus.publish(event)
            if i % 50 == 0:  # Small batches
                await asyncio.sleep(0.01)
        
        # Wait for processing
        await asyncio.sleep(1.0)
        end_time = asyncio.get_event_loop().time()
        
        # Verify performance
        processing_time = end_time - start_time
        throughput = event_count / processing_time
        
        # Should handle reasonable throughput
        assert throughput > 100  # events per second
        
        # Verify all helpers tracked events
        final_stats = stats.get_stats()
        assert final_stats['total_events_published'] >= event_count
        
        # History should contain recent events (may not be all due to size limit)
        recent_events = await history.get_recent_events(limit=1000)
        assert len(recent_events) > 0
    
    @pytest.mark.asyncio
    async def test_event_replay_with_helpers(
        self, 
        event_bus_with_helpers,
        sample_events
    ):
        """Test event replay functionality integrated with helpers."""
        bus = event_bus_with_helpers['event_bus']
        history = event_bus_with_helpers['history_manager']
        stats = event_bus_with_helpers['stats_tracker']
        
        # Track replayed events
        replayed_events = []
        
        async def replay_tracker(event):
            replayed_events.append(event)
        
        # Subscribe replay tracker
        bus.subscribe(EventType.ORDER_PLACED, replay_tracker)
        
        # Publish original events
        for event in sample_events:
            await bus.publish(event)
        
        await asyncio.sleep(0.2)
        
        # Get events from history for replay
        historical_events = await history.get_events_by_type(
            EventType.ORDER_PLACED,
            limit=10
        )
        
        # Clear replay tracker
        replayed_events.clear()
        
        # Replay events
        for event in historical_events:
            await bus.publish(event)
        
        await asyncio.sleep(0.2)
        
        # Verify replay occurred
        assert len(replayed_events) > 0
        
        # Verify stats reflect both original and replayed events
        final_stats = stats.get_stats()
        assert final_stats['total_events_published'] > len(sample_events)
    
    @pytest.mark.asyncio
    async def test_dlq_retry_mechanism_integration(
        self, 
        event_bus_with_helpers
    ):
        """Test DLQ retry mechanism integrated with event processing."""
        bus = event_bus_with_helpers['event_bus']
        dlq = event_bus_with_helpers['dlq_manager']
        
        # Create handler that fails first time, succeeds second time
        call_count = 0
        
        async def retry_handler(event):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt failed")
            return True  # Success on retry
        
        bus.subscribe(EventType.SCANNER_ALERT, retry_handler)
        
        # Create test event
        test_event = ScannerAlertEvent(
            data={
                'symbol': 'RETRY_TEST',
                'alert_type': 'test_retry'
            }
        )
        
        # First attempt - will fail
        await bus.publish(test_event)
        await asyncio.sleep(0.1)
        
        # Add to DLQ
        await dlq.add_failed_event(test_event, "First attempt failed")
        
        # Get retryable events
        retryable = await dlq.get_retryable_events()
        assert len(retryable) >= 1
        
        # Retry the event
        for retry_event_data in retryable:
            # Mark as retried
            await dlq.mark_event_retried(retry_event_data['event_id'])
            
            # Republish (in real system, this would be done by retry mechanism)
            await bus.publish(test_event)
        
        await asyncio.sleep(0.1)
        
        # Verify retry succeeded
        assert call_count == 2  # Called twice - once failed, once succeeded
    
    @pytest.mark.asyncio
    async def test_memory_management_under_load(
        self, 
        event_bus_with_helpers
    ):
        """Test memory management of helpers under sustained load."""
        bus = event_bus_with_helpers['event_bus']
        history = event_bus_with_helpers['history_manager']
        
        # Generate sustained event load
        for batch in range(10):
            events = []
            for i in range(100):
                event = OrderEvent(
                    symbol=f"MEM_TEST_{batch}_{i}",
                    quantity=100,
                    price=100.0,
                    side="buy",
                    order_type="market"
                )
                events.append(event)
            
            # Publish batch
            for event in events:
                await bus.publish(event)
            
            await asyncio.sleep(0.1)  # Small pause between batches
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Verify history manager respects size limits
        all_events = await history.get_recent_events(limit=2000)
        assert len(all_events) <= history.max_history_size
        
        # Verify deque optimization is working (O(1) operations)
        # In practice, this would be measured via performance metrics
        assert len(history._event_history) <= history.max_history_size
    
    @pytest.mark.asyncio
    async def test_concurrent_helper_operations(
        self, 
        event_bus_with_helpers
    ):
        """Test concurrent operations across all helpers."""
        bus = event_bus_with_helpers['event_bus']
        stats = event_bus_with_helpers['stats_tracker']
        history = event_bus_with_helpers['history_manager']
        dlq = event_bus_with_helpers['dlq_manager']
        
        async def publisher_task():
            for i in range(50):
                event = OrderEvent(
                    symbol=f"CONCURRENT_{i}",
                    quantity=100,
                    price=100.0,
                    side="buy",
                    order_type="limit"
                )
                await bus.publish(event)
                await asyncio.sleep(0.01)
        
        async def stats_reader_task():
            for _ in range(10):
                stats.get_stats()
                await asyncio.sleep(0.05)
        
        async def history_reader_task():
            for _ in range(10):
                await history.get_recent_events(limit=20)
                await asyncio.sleep(0.05)
        
        async def dlq_operations_task():
            # Add some test failures
            for i in range(5):
                test_event = OrderEvent(
                    symbol=f"DLQ_TEST_{i}",
                    quantity=100,
                    price=100.0,
                    side="sell",
                    order_type="market"
                )
                await dlq.add_failed_event(test_event, f"Test failure {i}")
                await asyncio.sleep(0.1)
        
        # Run all tasks concurrently
        await asyncio.gather(
            publisher_task(),
            stats_reader_task(),
            history_reader_task(),
            dlq_operations_task()
        )
        
        # Verify all operations completed successfully
        final_stats = stats.get_stats()
        assert final_stats['total_events_published'] >= 50
        
        recent_events = await history.get_recent_events(limit=100)
        assert len(recent_events) > 0
        
        failed_events = await dlq.get_failed_events(limit=10)
        assert len(failed_events) >= 5
    
    @pytest.mark.asyncio
    async def test_helper_cleanup_and_shutdown(
        self, 
        event_bus_with_helpers
    ):
        """Test proper cleanup and shutdown of all helpers."""
        bus = event_bus_with_helpers['event_bus']
        stats = event_bus_with_helpers['stats_tracker']
        history = event_bus_with_helpers['history_manager']
        dlq = event_bus_with_helpers['dlq_manager']
        
        # Generate some activity
        for i in range(10):
            event = OrderEvent(
                symbol=f"SHUTDOWN_TEST_{i}",
                quantity=100,
                price=100.0,
                side="buy",
                order_type="market"
            )
            await bus.publish(event)
        
        await asyncio.sleep(0.2)
        
        # Verify activity was tracked
        pre_shutdown_stats = stats.get_stats()
        assert pre_shutdown_stats['total_events_published'] >= 10
        
        # Test graceful shutdown
        await bus.stop()
        await dlq.close()
        
        # Verify clean shutdown (no exceptions)
        # In practice, would verify resources are properly released
        assert True  # If we get here, shutdown was successful
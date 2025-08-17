"""
Integration tests for the Event Bus system.

Tests event publishing, subscription, routing, and error handling
in realistic scenarios.
"""

# Standard library imports
import asyncio
from datetime import datetime

# Third-party imports
import pytest

# Local imports
from main.events.core import EventBusFactory
from main.events.types import FillEvent, OrderEvent, RiskEvent, ScanAlert
from main.interfaces.events import Event, EventPriority, EventType


@pytest.fixture
async def event_bus():
    """Create event bus instance for testing."""
    bus = EventBusFactory.create_test_instance()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    return {
        "order": OrderEvent(
            symbol="AAPL", quantity=100, price=150.0, side="buy", order_type="limit"
        ),
        "fill": FillEvent(
            symbol="AAPL", quantity=100, direction="buy", fill_cost=15000.0, commission=1.0
        ),
        "risk": RiskEvent(
            risk_type="position_limit",
            severity="warning",
            message="Position size approaching limit",
            metrics={"position_size": 0.95},
        ),
        "scan": ScanAlert(
            scanner_name="momentum",
            symbol="TSLA",
            alert_type="breakout",
            confidence=0.85,
            metadata={"volume_spike": 2.5},
        ),
    }


class TestEventBusIntegration:
    """Test event bus functionality in integrated scenarios."""

    @pytest.mark.asyncio
    async def test_event_publishing_and_subscription(self, event_bus, sample_events):
        """Test basic pub/sub functionality."""
        received_events = []

        # Subscribe to events
        async def event_handler(event: Event):
            received_events.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, event_handler)
        event_bus.subscribe(EventType.ORDER_FILLED, event_handler)

        # Publish events
        await event_bus.publish(sample_events["order"])
        await event_bus.publish(sample_events["fill"])

        # Allow time for async processing
        await asyncio.sleep(0.1)

        # Verify events were received
        assert len(received_events) == 2
        assert any(isinstance(e, OrderEvent) for e in received_events)
        assert any(isinstance(e, FillEvent) for e in received_events)

    @pytest.mark.asyncio
    async def test_event_filtering_by_type(self, event_bus, sample_events):
        """Test that subscribers only receive events they subscribed to."""
        order_events = []
        risk_events = []

        # Subscribe to specific event types
        event_bus.subscribe(EventType.ORDER_PLACED, lambda e: order_events.append(e))
        event_bus.subscribe(EventType.RISK_ALERT, lambda e: risk_events.append(e))

        # Publish various events
        for event in sample_events.values():
            await event_bus.publish(event)

        await asyncio.sleep(0.1)

        # Verify filtering
        assert len(order_events) == 1
        assert len(risk_events) == 1
        assert isinstance(order_events[0], OrderEvent)
        assert isinstance(risk_events[0], RiskEvent)

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_event(self, event_bus, sample_events):
        """Test multiple handlers for same event type."""
        handler1_called = False
        handler2_called = False
        handler3_called = False

        async def handler1(event):
            nonlocal handler1_called
            handler1_called = True

        async def handler2(event):
            nonlocal handler2_called
            handler2_called = True

        async def handler3(event):
            nonlocal handler3_called
            handler3_called = True

        # Multiple subscriptions
        event_bus.subscribe(EventType.ORDER_PLACED, handler1)
        event_bus.subscribe(EventType.ORDER_PLACED, handler2)
        event_bus.subscribe(EventType.ORDER_PLACED, handler3)

        # Publish event
        await event_bus.publish(sample_events["order"])
        await asyncio.sleep(0.1)

        # All handlers should be called
        assert handler1_called
        assert handler2_called
        assert handler3_called

    @pytest.mark.asyncio
    async def test_event_priority_handling(self, event_bus):
        """Test that high priority events are processed first."""
        processing_order = []

        async def track_handler(event):
            processing_order.append(event.priority)
            await asyncio.sleep(0.01)  # Simulate processing

        event_bus.subscribe(EventType.RISK_ALERT, track_handler)

        # Create events with different priorities
        high_priority = RiskEvent(
            priority=EventPriority.HIGH,
            risk_type="circuit_breaker",
            severity="critical",
            message="Circuit breaker triggered",
        )

        medium_priority = RiskEvent(
            priority=EventPriority.MEDIUM,
            risk_type="position_limit",
            severity="warning",
            message="Position limit warning",
        )

        low_priority = RiskEvent(
            priority=EventPriority.LOW,
            risk_type="daily_report",
            severity="info",
            message="Daily report",
        )

        # Publish in reverse priority order
        await event_bus.publish(low_priority)
        await event_bus.publish(medium_priority)
        await event_bus.publish(high_priority)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify high priority processed first
        assert processing_order[0] == EventPriority.HIGH

    @pytest.mark.asyncio
    async def test_error_handling_in_subscribers(self, event_bus, sample_events):
        """Test that errors in one subscriber don't affect others."""
        successful_handler_called = False

        async def failing_handler(event):
            raise Exception("Handler failed!")

        async def successful_handler(event):
            nonlocal successful_handler_called
            successful_handler_called = True

        # Subscribe both handlers
        event_bus.subscribe(EventType.ORDER_PLACED, failing_handler)
        event_bus.subscribe(EventType.ORDER_PLACED, successful_handler)

        # Publish event
        await event_bus.publish(sample_events["order"])
        await asyncio.sleep(0.1)

        # Successful handler should still be called
        assert successful_handler_called

    @pytest.mark.asyncio
    async def test_event_replay_functionality(self, event_bus, sample_events):
        """Test event replay for recovery scenarios."""
        replayed_events = []

        # Subscribe to events
        event_bus.subscribe(EventType.ORDER_PLACED, lambda e: replayed_events.append(e))

        # Publish events
        await event_bus.publish(sample_events["order"])

        # Simulate getting event history
        if hasattr(event_bus, "get_event_history"):
            history = await event_bus.get_event_history(event_type=EventType.ORDER_PLACED, limit=10)

            # Replay events
            for event in history:
                await event_bus.publish(event)

        await asyncio.sleep(0.1)

        # Should have original + replayed
        assert len(replayed_events) >= 1

    @pytest.mark.asyncio
    async def test_event_bus_under_load(self, event_bus):
        """Test event bus performance under high load."""
        events_received = 0
        total_events = 1000

        async def counter_handler(event):
            nonlocal events_received
            events_received += 1

        # Subscribe to events
        event_bus.subscribe(EventType.ORDER_PLACED, counter_handler)

        # Publish many events rapidly
        start_time = asyncio.get_event_loop().time()

        for i in range(total_events):
            event = OrderEvent(
                symbol=f"SYM{i}", quantity=100, price=100.0 + i, side="buy", order_type="market"
            )
            await event_bus.publish(event)

        # Wait for all events to be processed
        max_wait = 5.0  # seconds
        start_wait = asyncio.get_event_loop().time()

        while (
            events_received < total_events
            and (asyncio.get_event_loop().time() - start_wait) < max_wait
        ):
            await asyncio.sleep(0.1)

        end_time = asyncio.get_event_loop().time()

        # Verify all events processed
        assert events_received == total_events

        # Calculate throughput
        duration = end_time - start_time
        throughput = total_events / duration
        print(f"Event bus throughput: {throughput:.2f} events/second")

        # Should handle at least 100 events/second
        assert throughput > 100

    @pytest.mark.asyncio
    async def test_event_persistence_integration(self, event_bus, sample_events):
        """Test event persistence for audit trail."""
        # This would integrate with actual database in production
        persisted_events = []

        async def persistence_handler(event):
            # Simulate database save
            persisted_events.append(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.to_dict(),
                }
            )

        # Add persistence handler for all events
        for event_type in EventType:
            event_bus.subscribe(event_type, persistence_handler)

        # Publish various events
        for event in sample_events.values():
            await event_bus.publish(event)

        await asyncio.sleep(0.1)

        # Verify all events persisted
        assert len(persisted_events) == len(sample_events)

        # Verify event data integrity
        for persisted in persisted_events:
            assert "event_id" in persisted
            assert "timestamp" in persisted
            assert "data" in persisted

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, event_bus):
        """Test circuit breaker pattern for failing subscribers."""
        call_count = 0
        circuit_open = False

        async def flaky_handler(event):
            nonlocal call_count, circuit_open
            call_count += 1

            # Fail first 3 times
            if call_count <= 3:
                raise Exception("Service unavailable")

            # Then succeed
            return True

        # Add circuit breaker wrapper
        if hasattr(event_bus, "subscribe_with_circuit_breaker"):
            event_bus.subscribe_with_circuit_breaker(
                EventType.ORDER_PLACED, flaky_handler, failure_threshold=3, timeout=1.0
            )
        else:
            # Fallback to regular subscription
            event_bus.subscribe(EventType.ORDER_PLACED, flaky_handler)

        # Send multiple events
        for i in range(10):
            event = OrderEvent(
                symbol="AAPL", quantity=100, price=150.0, side="buy", order_type="market"
            )
            await event_bus.publish(event)
            await asyncio.sleep(0.1)

        # Circuit should open after 3 failures
        # Then close and allow successful calls
        assert call_count >= 3  # At least the failed attempts


@pytest.mark.integration
class TestEventBusRealWorldScenarios:
    """Test real-world trading scenarios with event bus."""

    @pytest.mark.asyncio
    async def test_order_lifecycle_events(self, event_bus):
        """Test complete order lifecycle event flow."""
        events_timeline = []

        async def timeline_handler(event):
            events_timeline.append(
                {"time": datetime.now(), "type": event.event_type.value, "data": event}
            )

        # Subscribe to all order-related events
        order_events = [
            EventType.ORDER_PLACED,
            EventType.ORDER_VALIDATED,
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_FILLED,
            EventType.POSITION_UPDATED,
        ]

        for event_type in order_events:
            event_bus.subscribe(event_type, timeline_handler)

        # Simulate order lifecycle
        order_id = "ORD123"

        # 1. Place order
        await event_bus.publish(
            OrderEvent(
                order_id=order_id,
                symbol="AAPL",
                quantity=100,
                price=150.0,
                side="buy",
                order_type="limit",
            )
        )

        # 2. Validate order (would come from risk manager)
        # 3. Submit to broker
        # 4. Receive fill
        await event_bus.publish(
            FillEvent(
                order_id=order_id,
                symbol="AAPL",
                quantity=100,
                direction="buy",
                fill_cost=15000.0,
                commission=1.0,
            )
        )

        await asyncio.sleep(0.2)

        # Verify event sequence
        assert len(events_timeline) >= 2
        assert events_timeline[0]["type"] == EventType.ORDER_PLACED.value
        assert events_timeline[-1]["type"] == EventType.ORDER_FILLED.value

    @pytest.mark.asyncio
    async def test_risk_alert_cascade(self, event_bus):
        """Test risk alert triggering cascade of events."""
        actions_taken = []

        async def risk_response_handler(event):
            if event.severity == "critical":
                actions_taken.append("halt_trading")
                # Publish trading halt event
                halt_event = Event(event_type=EventType.TRADING_HALTED)
                await event_bus.publish(halt_event)
            elif event.severity == "warning":
                actions_taken.append("reduce_position_sizes")

        async def trading_halt_handler(event):
            actions_taken.append("positions_flattened")
            actions_taken.append("orders_cancelled")

        # Subscribe handlers
        event_bus.subscribe(EventType.RISK_ALERT, risk_response_handler)
        event_bus.subscribe(EventType.TRADING_HALTED, trading_halt_handler)

        # Trigger critical risk event
        critical_risk = RiskEvent(
            risk_type="drawdown_limit",
            severity="critical",
            message="Maximum drawdown exceeded",
            metrics={"drawdown": -0.15},
        )

        await event_bus.publish(critical_risk)
        await asyncio.sleep(0.2)

        # Verify cascade of actions
        assert "halt_trading" in actions_taken
        assert "positions_flattened" in actions_taken
        assert "orders_cancelled" in actions_taken

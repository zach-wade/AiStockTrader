"""
Integration tests for the Event-Driven Engine.

Tests complete event-driven workflows, event ordering,
dependencies, persistence, and replay capabilities.
"""

# Standard library imports
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

# Third-party imports
import pytest

# Local imports
from main.config.config_manager import get_config
from main.events.handlers.event_driven_engine import EventDrivenEngine
from main.events.types import Event, EventType, FillEvent, OrderEvent, RiskEvent, ScanAlert


@pytest.fixture
async def event_engine():
    """Create event-driven engine for testing."""
    config = get_config()
    engine = EventDrivenEngine(config)
    await engine.start()
    yield engine
    await engine.stop()


@pytest.fixture
def mock_broker():
    """Create mock broker for testing."""
    broker = AsyncMock()
    broker.submit_order = AsyncMock(return_value="BRK123")
    broker.cancel_order = AsyncMock(return_value=True)
    broker.get_positions = AsyncMock(return_value={})
    return broker


@pytest.fixture
def mock_components():
    """Create mock system components."""
    return {
        "scanner": AsyncMock(),
        "risk_manager": AsyncMock(),
        "position_manager": AsyncMock(),
        "order_manager": AsyncMock(),
        "portfolio_manager": AsyncMock(),
    }


class TestEventDrivenEngineIntegration:
    """Test event-driven engine in integrated scenarios."""

    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, event_engine, mock_broker, mock_components):
        """Test complete event-driven trading workflow."""
        # Track workflow progress
        workflow_steps = []

        # Configure mock responses
        mock_components["risk_manager"].validate_order = AsyncMock(return_value=True)
        mock_components["order_manager"].create_order = AsyncMock(
            return_value={"order_id": "ORD123", "status": "pending"}
        )

        # Register components
        event_engine.register_component("broker", mock_broker)
        for name, component in mock_components.items():
            event_engine.register_component(name, component)

        # Define workflow handlers
        async def on_scan_alert(event: ScanAlert):
            workflow_steps.append("scan_received")
            # Generate order
            order_event = OrderEvent(
                symbol=event.symbol,
                quantity=100,
                price=150.0,
                side="buy",
                order_type="limit",
                metadata={"scan_id": event.event_id},
            )
            await event_engine.publish(order_event)

        async def on_order_placed(event: OrderEvent):
            workflow_steps.append("order_placed")
            # Risk validation would trigger here
            if await mock_components["risk_manager"].validate_order(event):
                workflow_steps.append("risk_approved")
                # Submit to broker
                broker_id = await mock_broker.submit_order(event)
                workflow_steps.append(f"broker_submitted:{broker_id}")

        async def on_order_filled(event: FillEvent):
            workflow_steps.append("order_filled")
            # Update positions
            await mock_components["position_manager"].update_position(event)
            workflow_steps.append("position_updated")

        # Register handlers
        event_engine.subscribe(EventType.SCAN_ALERT, on_scan_alert)
        event_engine.subscribe(EventType.ORDER_PLACED, on_order_placed)
        event_engine.subscribe(EventType.ORDER_FILLED, on_order_filled)

        # Start workflow with scan alert
        scan_alert = ScanAlert(
            scanner_name="momentum",
            symbol="AAPL",
            alert_type="breakout",
            confidence=0.85,
            metadata={"price": 150.0, "volume": 1000000},
        )

        await event_engine.publish(scan_alert)

        # Simulate fill after order submission
        await asyncio.sleep(0.1)

        fill_event = FillEvent(
            symbol="AAPL", quantity=100, direction="buy", fill_cost=15000.0, commission=1.0
        )
        await event_engine.publish(fill_event)

        # Allow workflow to complete
        await asyncio.sleep(0.2)

        # Verify complete workflow
        expected_steps = [
            "scan_received",
            "order_placed",
            "risk_approved",
            "broker_submitted:BRK123",
            "order_filled",
            "position_updated",
        ]

        for step in expected_steps:
            assert step in workflow_steps

    @pytest.mark.asyncio
    async def test_event_ordering_and_dependencies(self, event_engine):
        """Test that events are processed in correct order with dependencies."""
        processing_order = []
        completed_events = set()

        # Define events with dependencies
        async def process_event_with_deps(event: Event):
            # Check dependencies
            deps = event.metadata.get("depends_on", [])
            for dep in deps:
                if dep not in completed_events:
                    # Requeue event
                    await asyncio.sleep(0.05)
                    await event_engine.publish(event)
                    return

            # Process event
            processing_order.append(event.event_id)
            completed_events.add(event.event_id)

        # Subscribe handler
        event_engine.subscribe(EventType.GENERIC, process_event_with_deps)

        # Create events with dependencies
        event1 = Event(event_type=EventType.GENERIC, event_id="EVT1", metadata={"depends_on": []})

        event2 = Event(
            event_type=EventType.GENERIC, event_id="EVT2", metadata={"depends_on": ["EVT1"]}
        )

        event3 = Event(
            event_type=EventType.GENERIC, event_id="EVT3", metadata={"depends_on": ["EVT1", "EVT2"]}
        )

        # Publish in reverse order
        await event_engine.publish(event3)
        await event_engine.publish(event2)
        await event_engine.publish(event1)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Verify correct ordering
        assert processing_order == ["EVT1", "EVT2", "EVT3"]

    @pytest.mark.asyncio
    async def test_event_persistence_and_recovery(self, event_engine):
        """Test event persistence and recovery after failure."""
        # Mock persistence layer
        persisted_events = []

        async def persist_event(event: Event):
            persisted_events.append(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.to_dict(),
                }
            )

        # Add persistence handler
        event_engine.add_persistence_handler(persist_event)

        # Publish events
        events_to_persist = [
            OrderEvent(symbol="AAPL", quantity=100, price=150.0, side="buy", order_type="limit"),
            RiskEvent(
                risk_type="position_limit", severity="warning", message="Position size warning"
            ),
            FillEvent(symbol="AAPL", quantity=100, direction="buy", fill_cost=15000.0),
        ]

        for event in events_to_persist:
            await event_engine.publish(event)

        await asyncio.sleep(0.1)

        # Verify persistence
        assert len(persisted_events) == len(events_to_persist)

        # Simulate recovery - replay events
        replayed_count = 0

        async def count_replayed(event):
            nonlocal replayed_count
            replayed_count += 1

        # Subscribe counter
        for event_type in [EventType.ORDER_PLACED, EventType.RISK_ALERT, EventType.ORDER_FILLED]:
            event_engine.subscribe(event_type, count_replayed)

        # Replay persisted events
        for persisted in persisted_events:
            # Reconstruct event from persisted data
            await event_engine.replay_event(persisted["data"])

        await asyncio.sleep(0.1)

        # Verify replay
        assert replayed_count == len(events_to_persist)

    @pytest.mark.asyncio
    async def test_event_aggregation_and_windowing(self, event_engine):
        """Test event aggregation over time windows."""
        price_updates = []
        aggregated_stats = {}

        # Window aggregator
        async def aggregate_price_updates(event: Event):
            if event.event_type == EventType.MARKET_DATA:
                price_updates.append(
                    {"symbol": event.symbol, "price": event.price, "timestamp": event.timestamp}
                )

                # Aggregate over 1-second windows
                window_key = event.timestamp.replace(microsecond=0)
                if window_key not in aggregated_stats:
                    aggregated_stats[window_key] = {
                        "count": 0,
                        "sum_price": 0,
                        "min_price": float("inf"),
                        "max_price": float("-inf"),
                    }

                stats = aggregated_stats[window_key]
                stats["count"] += 1
                stats["sum_price"] += event.price
                stats["min_price"] = min(stats["min_price"], event.price)
                stats["max_price"] = max(stats["max_price"], event.price)

        # Subscribe aggregator
        event_engine.subscribe(EventType.MARKET_DATA, aggregate_price_updates)

        # Generate rapid price updates
        base_price = 100.0
        for i in range(100):
            price_event = Event(
                event_type=EventType.MARKET_DATA,
                symbol="AAPL",
                price=base_price + (i % 10) - 5,  # Oscillating price
                timestamp=datetime.now(),
            )
            await event_engine.publish(price_event)
            await asyncio.sleep(0.01)  # 10ms between updates

        # Verify aggregation
        assert len(price_updates) == 100
        assert len(aggregated_stats) >= 1  # At least one time window

        # Check aggregated statistics
        for window, stats in aggregated_stats.items():
            assert stats["count"] > 0
            assert stats["min_price"] <= stats["max_price"]
            avg_price = stats["sum_price"] / stats["count"]
            assert stats["min_price"] <= avg_price <= stats["max_price"]

    @pytest.mark.asyncio
    async def test_event_routing_rules(self, event_engine):
        """Test complex event routing based on rules."""
        routed_events = {"high_value_orders": [], "risk_alerts": [], "normal_orders": []}

        # Define routing rules
        async def route_order_events(event: OrderEvent):
            order_value = event.quantity * event.price

            if order_value > 100000:  # High value
                routed_events["high_value_orders"].append(event)
                # Could trigger additional risk checks
                risk_event = RiskEvent(
                    risk_type="order_review",
                    severity="info",
                    message=f"High value order: ${order_value:,.2f}",
                    metrics={"order_id": event.order_id, "order_value": order_value},
                )
                await event_engine.publish(risk_event)
            else:
                routed_events["normal_orders"].append(event)

        async def route_risk_events(event: RiskEvent):
            routed_events["risk_alerts"].append(event)

        # Subscribe routers
        event_engine.subscribe(EventType.ORDER_PLACED, route_order_events)
        event_engine.subscribe(EventType.RISK_ALERT, route_risk_events)

        # Test various order sizes
        test_orders = [
            ("AAPL", 100, 150.0),  # $15,000 - normal
            ("GOOGL", 500, 2500.0),  # $1,250,000 - high value
            ("MSFT", 50, 300.0),  # $15,000 - normal
            ("AMZN", 1000, 120.0),  # $120,000 - high value
        ]

        for symbol, qty, price in test_orders:
            order = OrderEvent(
                symbol=symbol, quantity=qty, price=price, side="buy", order_type="market"
            )
            await event_engine.publish(order)

        await asyncio.sleep(0.2)

        # Verify routing
        assert len(routed_events["high_value_orders"]) == 2
        assert len(routed_events["normal_orders"]) == 2
        assert len(routed_events["risk_alerts"]) == 2  # One for each high value order

    @pytest.mark.asyncio
    async def test_event_driven_state_machine(self, event_engine):
        """Test state machine driven by events."""

        class OrderStateMachine:
            def __init__(self):
                self.states = {}
                self.transitions = []

            async def handle_event(self, event: Event):
                order_id = event.metadata.get("order_id")
                if not order_id:
                    return

                current_state = self.states.get(order_id, "new")
                new_state = None

                # State transitions based on event type
                if event.event_type == EventType.ORDER_PLACED:
                    if current_state == "new":
                        new_state = "pending"

                elif event.event_type == EventType.ORDER_VALIDATED:
                    if current_state == "pending":
                        new_state = "validated"

                elif event.event_type == EventType.ORDER_SUBMITTED:
                    if current_state == "validated":
                        new_state = "submitted"

                elif event.event_type == EventType.ORDER_FILLED:
                    if current_state == "submitted":
                        new_state = "filled"

                elif event.event_type == EventType.ORDER_CANCELLED:
                    if current_state in ["pending", "validated", "submitted"]:
                        new_state = "cancelled"

                if new_state:
                    self.states[order_id] = new_state
                    self.transitions.append(
                        {
                            "order_id": order_id,
                            "from_state": current_state,
                            "to_state": new_state,
                            "event": event.event_type.value,
                            "timestamp": event.timestamp,
                        }
                    )

        # Create state machine
        state_machine = OrderStateMachine()

        # Subscribe to all order events
        order_events = [
            EventType.ORDER_PLACED,
            EventType.ORDER_VALIDATED,
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
        ]

        for event_type in order_events:
            event_engine.subscribe(event_type, state_machine.handle_event)

        # Simulate order lifecycle
        order_id = "ORD456"

        # Create events
        events = [
            Event(event_type=EventType.ORDER_PLACED, metadata={"order_id": order_id}),
            Event(event_type=EventType.ORDER_VALIDATED, metadata={"order_id": order_id}),
            Event(event_type=EventType.ORDER_SUBMITTED, metadata={"order_id": order_id}),
            Event(event_type=EventType.ORDER_FILLED, metadata={"order_id": order_id}),
        ]

        # Publish events
        for event in events:
            await event_engine.publish(event)
            await asyncio.sleep(0.05)

        # Verify state transitions
        assert state_machine.states[order_id] == "filled"
        assert len(state_machine.transitions) == 4

        # Verify transition sequence
        expected_transitions = [
            ("new", "pending"),
            ("pending", "validated"),
            ("validated", "submitted"),
            ("submitted", "filled"),
        ]

        actual_transitions = [(t["from_state"], t["to_state"]) for t in state_machine.transitions]

        assert actual_transitions == expected_transitions

    @pytest.mark.asyncio
    async def test_event_compensation_pattern(self, event_engine):
        """Test compensation/rollback pattern for failed workflows."""
        execution_log = []
        compensations_run = []

        # Define workflow steps with compensations
        async def step1_handler(event):
            execution_log.append("step1_executed")
            # Define compensation
            event.metadata["compensation"] = "step1_rollback"

        async def step2_handler(event):
            execution_log.append("step2_executed")
            event.metadata["compensation"] = "step2_rollback"
            # Simulate failure in step 2
            if event.metadata.get("should_fail"):
                raise Exception("Step 2 failed!")

        async def step3_handler(event):
            execution_log.append("step3_executed")
            event.metadata["compensation"] = "step3_rollback"

        async def compensation_handler(event):
            if event.event_type == EventType.WORKFLOW_FAILED:
                # Run compensations in reverse order
                compensations = event.metadata.get("compensations", [])
                for comp in reversed(compensations):
                    compensations_run.append(comp)

        # Subscribe handlers
        event_engine.subscribe(EventType.WORKFLOW_STEP, step1_handler)
        event_engine.subscribe(EventType.WORKFLOW_STEP, step2_handler)
        event_engine.subscribe(EventType.WORKFLOW_STEP, step3_handler)
        event_engine.subscribe(EventType.WORKFLOW_FAILED, compensation_handler)

        # Run workflow that will fail
        completed_compensations = []

        try:
            # Step 1
            event1 = Event(event_type=EventType.WORKFLOW_STEP, metadata={"step": 1})
            await event_engine.publish(event1)
            completed_compensations.append(event1.metadata.get("compensation"))

            # Step 2 (will fail)
            event2 = Event(
                event_type=EventType.WORKFLOW_STEP, metadata={"step": 2, "should_fail": True}
            )
            await event_engine.publish(event2)
            completed_compensations.append(event2.metadata.get("compensation"))

        except Exception:
            # Trigger compensation
            failure_event = Event(
                event_type=EventType.WORKFLOW_FAILED,
                metadata={"compensations": completed_compensations},
            )
            await event_engine.publish(failure_event)

        await asyncio.sleep(0.1)

        # Verify execution and compensation
        assert "step1_executed" in execution_log
        assert "step2_executed" in execution_log
        assert "step3_executed" not in execution_log  # Should not reach step 3

        # Verify compensations ran in reverse order
        assert compensations_run == ["step1_rollback"]  # Only step 1 completed successfully

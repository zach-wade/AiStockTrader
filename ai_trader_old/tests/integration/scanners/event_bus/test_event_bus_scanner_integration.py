"""
Integration tests for event bus and scanner system integration.

Tests the complete event flow from scanner alert generation to event handling.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import pytest

# Local imports
from main.events.types.event_types import AlertType, EventType, ScannerAlertEvent
from main.interfaces.events import IEventBus
from main.scanners.types import ScanAlert


@pytest.mark.integration
@pytest.mark.asyncio
class TestEventBusScannerIntegration:
    """Test complete event bus and scanner integration."""

    async def test_orchestrator_event_coordination(
        self, test_event_bus: IEventBus, event_collector, test_symbols, mock_scanner_repository
    ):
        """Test scanner orchestrator coordinates events from multiple scanners."""
        # Third-party imports
        from omegaconf import OmegaConf

        # Local imports
        from main.scanners.scanner_orchestrator_factory import ScannerOrchestratorFactory

        # Create orchestrator config
        config = OmegaConf.create(
            {
                "orchestrator": {
                    "execution_strategy": "parallel",
                    "max_concurrent_scanners": 3,
                    "timeout_seconds": 30.0,
                    "enable_health_monitoring": True,
                    "publish_events": True,
                },
                "scanners": {
                    "volume": {"enabled": True, "priority": 8},
                    "news": {"enabled": True, "priority": 9},
                    "technical": {"enabled": True, "priority": 7},
                },
            }
        )

        # Create orchestrator
        orchestrator = ScannerOrchestratorFactory.create_orchestrator(
            config=config, repository=mock_scanner_repository, event_bus=event_collector
        )

        # Mock scanner results
        mock_volume_alerts = [
            ScanAlert(
                symbol="AAPL",
                alert_type=AlertType.VOLUME_SPIKE,
                score=0.85,
                metadata={"volume_ratio": 2.5},
                timestamp=datetime.now(UTC),
                scanner_name="volume_scanner",
            )
        ]

        mock_news_alerts = [
            ScanAlert(
                symbol="GOOGL",
                alert_type=AlertType.NEWS_SENTIMENT,
                score=0.78,
                metadata={"sentiment_score": 0.8},
                timestamp=datetime.now(UTC),
                scanner_name="news_scanner",
            )
        ]

        # Mock scanner factory to return mock results
        with patch.object(orchestrator, "_create_scanner_instances") as mock_create:
            mock_volume_scanner = Mock()
            mock_volume_scanner.scan = AsyncMock(return_value=mock_volume_alerts)
            mock_volume_scanner.name = "volume_scanner"
            mock_volume_scanner.priority = 8

            mock_news_scanner = Mock()
            mock_news_scanner.scan = AsyncMock(return_value=mock_news_alerts)
            mock_news_scanner.name = "news_scanner"
            mock_news_scanner.priority = 9

            mock_create.return_value = [mock_volume_scanner, mock_news_scanner]

            # Run orchestrated scan
            results = await orchestrator.run_scan(symbols=test_symbols[:2])

            # Should have results from both scanners
            assert len(results) == 2
            assert "volume_scanner" in results
            assert "news_scanner" in results

            # Check published events
            published_events = event_collector.get_scanner_alerts()
            assert len(published_events) >= 2  # At least one from each scanner

            # Events should be from different scanners
            scanner_names = [event.scanner_name for event in published_events]
            assert "volume_scanner" in scanner_names
            assert "news_scanner" in scanner_names

    async def test_event_driven_scanner_cascade(
        self, test_event_bus: IEventBus, event_collector, test_symbols
    ):
        """Test event-driven cascade where one scanner triggers another."""
        cascade_events = []

        # Primary scanner that publishes events
        async def primary_scanner_handler(event: ScannerAlertEvent):
            """Handler that triggers secondary analysis."""
            if event.score > 0.8:  # High-confidence alerts trigger cascade
                # Simulate secondary scanner activation
                secondary_alert = ScannerAlertEvent(
                    symbol=event.symbol,
                    alert_type=AlertType.COORDINATED_ACTIVITY,
                    score=0.9,
                    scanner_name="secondary_scanner",
                    metadata={
                        "triggered_by": event.scanner_name,
                        "original_score": event.score,
                        "cascade_analysis": True,
                    },
                    timestamp=datetime.now(UTC),
                    event_type=EventType.SCANNER_ALERT,
                )
                cascade_events.append(secondary_alert)
                await test_event_bus.publish(secondary_alert)

        # Subscribe to scanner alerts
        await test_event_bus.subscribe(EventType.SCANNER_ALERT, primary_scanner_handler)

        # Create primary scanner
        # Local imports
        from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner

        primary_scanner = MockEventPublishingScanner("primary_scanner", test_event_bus)

        # Trigger high-confidence alert
        high_confidence_alert = ScanAlert(
            symbol="AAPL",
            alert_type=AlertType.VOLUME_SPIKE,
            score=0.85,  # Above threshold
            metadata={"confidence": 0.9},
            timestamp=datetime.now(UTC),
            scanner_name="primary_scanner",
        )

        await primary_scanner.publish_alerts_to_event_bus([high_confidence_alert])

        # Wait for cascade processing
        await asyncio.sleep(0.1)

        # Should have triggered cascade
        assert len(cascade_events) == 1

        cascade_event = cascade_events[0]
        assert cascade_event.scanner_name == "secondary_scanner"
        assert cascade_event.metadata["triggered_by"] == "primary_scanner"
        assert cascade_event.metadata["cascade_analysis"] == True

    async def test_event_aggregation_and_correlation(self, test_event_bus: IEventBus, test_symbols):
        """Test event aggregation and correlation across scanners."""
        collected_events = []
        correlations = []

        async def correlation_handler(event: ScannerAlertEvent):
            """Handler that looks for correlations between events."""
            collected_events.append(event)

            # Look for correlations (same symbol, different scanners, within time window)
            recent_events = [
                e
                for e in collected_events
                if (event.timestamp - e.timestamp).total_seconds() < 300  # 5 minutes
                and e.symbol == event.symbol
                and e.scanner_name != event.scanner_name
            ]

            if recent_events:
                correlation = {
                    "symbol": event.symbol,
                    "scanners": [event.scanner_name] + [e.scanner_name for e in recent_events],
                    "combined_score": max(event.score, max(e.score for e in recent_events)),
                    "event_count": len(recent_events) + 1,
                    "timestamp": event.timestamp,
                }
                correlations.append(correlation)

        await test_event_bus.subscribe(EventType.SCANNER_ALERT, correlation_handler)

        # Create multiple scanners
        # Local imports
        from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner

        volume_scanner = MockEventPublishingScanner("volume_scanner", test_event_bus)
        news_scanner = MockEventPublishingScanner("news_scanner", test_event_bus)
        technical_scanner = MockEventPublishingScanner("technical_scanner", test_event_bus)

        # Generate correlated alerts for same symbol
        base_time = datetime.now(UTC)

        volume_alert = ScanAlert(
            symbol="AAPL",
            alert_type=AlertType.VOLUME_SPIKE,
            score=0.8,
            metadata={},
            timestamp=base_time,
            scanner_name="volume_scanner",
        )

        news_alert = ScanAlert(
            symbol="AAPL",  # Same symbol
            alert_type=AlertType.NEWS_SENTIMENT,
            score=0.85,
            metadata={},
            timestamp=base_time + timedelta(minutes=2),  # Shortly after
            scanner_name="news_scanner",
        )

        technical_alert = ScanAlert(
            symbol="AAPL",  # Same symbol again
            alert_type=AlertType.TECHNICAL_BREAKOUT,
            score=0.9,
            metadata={},
            timestamp=base_time + timedelta(minutes=4),  # Shortly after
            scanner_name="technical_scanner",
        )

        # Publish alerts sequentially
        await volume_scanner.publish_alerts_to_event_bus([volume_alert])
        await asyncio.sleep(0.05)

        await news_scanner.publish_alerts_to_event_bus([news_alert])
        await asyncio.sleep(0.05)

        await technical_scanner.publish_alerts_to_event_bus([technical_alert])
        await asyncio.sleep(0.1)

        # Should have detected correlations
        assert len(correlations) >= 1

        # Check correlation details
        correlation = correlations[-1]  # Latest correlation
        assert correlation["symbol"] == "AAPL"
        assert len(correlation["scanners"]) >= 2  # Multiple scanners involved
        assert correlation["combined_score"] == 0.9  # Max score

    async def test_event_filtering_and_routing(self, test_event_bus: IEventBus, test_symbols):
        """Test event filtering and routing based on criteria."""
        high_priority_events = []
        low_priority_events = []

        async def high_priority_handler(event: ScannerAlertEvent):
            if event.score >= 0.8:
                high_priority_events.append(event)

        async def low_priority_handler(event: ScannerAlertEvent):
            if event.score < 0.8:
                low_priority_events.append(event)

        # Subscribe handlers
        await test_event_bus.subscribe(EventType.SCANNER_ALERT, high_priority_handler)
        await test_event_bus.subscribe(EventType.SCANNER_ALERT, low_priority_handler)

        # Create scanner
        # Local imports
        from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner

        scanner = MockEventPublishingScanner("filter_test_scanner", test_event_bus)

        # Generate alerts with different scores
        alerts = [
            ScanAlert(
                symbol="AAPL",
                alert_type=AlertType.VOLUME_SPIKE,
                score=0.9,  # High priority
                metadata={},
                timestamp=datetime.now(UTC),
                scanner_name="filter_test_scanner",
            ),
            ScanAlert(
                symbol="GOOGL",
                alert_type=AlertType.NEWS_SENTIMENT,
                score=0.6,  # Low priority
                metadata={},
                timestamp=datetime.now(UTC),
                scanner_name="filter_test_scanner",
            ),
            ScanAlert(
                symbol="MSFT",
                alert_type=AlertType.TECHNICAL_BREAKOUT,
                score=0.85,  # High priority
                metadata={},
                timestamp=datetime.now(UTC),
                scanner_name="filter_test_scanner",
            ),
        ]

        await scanner.publish_alerts_to_event_bus(alerts)
        await asyncio.sleep(0.1)

        # Check filtering worked
        assert len(high_priority_events) == 2  # AAPL and MSFT
        assert len(low_priority_events) == 1  # GOOGL

        # Verify scores
        high_scores = [event.score for event in high_priority_events]
        low_scores = [event.score for event in low_priority_events]

        assert all(score >= 0.8 for score in high_scores)
        assert all(score < 0.8 for score in low_scores)

    async def test_event_persistence_and_replay(
        self, test_event_bus: IEventBus, event_collector, test_symbols
    ):
        """Test event persistence and replay capabilities."""
        # Enable event persistence (if supported)
        if hasattr(test_event_bus, "enable_persistence"):
            test_event_bus.enable_persistence(True)

        # Create scanner
        # Local imports
        from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner

        scanner = MockEventPublishingScanner("persistence_scanner", test_event_bus)

        # Generate events
        alerts = [
            ScanAlert(
                symbol=symbol,
                alert_type=AlertType.VOLUME_SPIKE,
                score=0.8,
                metadata={"test": "persistence"},
                timestamp=datetime.now(UTC),
                scanner_name="persistence_scanner",
            )
            for symbol in test_symbols[:3]
        ]

        await scanner.publish_alerts_to_event_bus(alerts)
        await asyncio.sleep(0.1)

        # Check if event bus supports event history
        if hasattr(test_event_bus, "get_event_history"):
            history = await test_event_bus.get_event_history(
                event_type=EventType.SCANNER_ALERT, limit=10
            )

            # Should have persisted events
            assert len(history) >= 3

            # Check event details
            persisted_symbols = [event.symbol for event in history]
            assert all(symbol in test_symbols[:3] for symbol in persisted_symbols)

    async def test_event_metrics_and_monitoring(
        self, test_event_bus: IEventBus, test_symbols, event_performance_thresholds
    ):
        """Test event metrics collection and monitoring."""
        # Create scanner
        # Local imports
        from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner

        metrics_scanner = MockEventPublishingScanner("metrics_scanner", test_event_bus)

        # Generate batch of events
        start_time = datetime.now()

        for i in range(10):
            alert = ScanAlert(
                symbol=f"SYM{i:03d}",
                alert_type=AlertType.VOLUME_SPIKE,
                score=0.8,
                metadata={"batch_id": i},
                timestamp=datetime.now(UTC),
                scanner_name="metrics_scanner",
            )
            await metrics_scanner.publish_alerts_to_event_bus([alert])

        end_time = datetime.now()
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        # Calculate throughput
        events_per_second = 10 / (total_time_ms / 1000)

        # Check performance metrics
        min_throughput = event_performance_thresholds["event_throughput_per_second"]
        # Allow for test overhead
        assert events_per_second > min_throughput / 10

        # Check if event bus provides metrics
        if hasattr(test_event_bus, "get_metrics"):
            metrics = await test_event_bus.get_metrics()

            # Should have published event count
            assert "events_published" in metrics
            assert metrics["events_published"] >= 10

    async def test_dead_letter_queue_handling(self, test_event_bus: IEventBus, test_symbols):
        """Test dead letter queue for failed event processing."""
        failed_events = []

        async def failing_handler(event: ScannerAlertEvent):
            """Handler that always fails for testing."""
            failed_events.append(event)
            raise Exception("Simulated handler failure")

        # Subscribe failing handler
        await test_event_bus.subscribe(EventType.SCANNER_ALERT, failing_handler)

        # Create scanner
        # Local imports
        from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner

        dlq_scanner = MockEventPublishingScanner("dlq_scanner", test_event_bus)

        # Generate event that will fail
        alert = ScanAlert(
            symbol="AAPL",
            alert_type=AlertType.VOLUME_SPIKE,
            score=0.8,
            metadata={"will_fail": True},
            timestamp=datetime.now(UTC),
            scanner_name="dlq_scanner",
        )

        await dlq_scanner.publish_alerts_to_event_bus([alert])
        await asyncio.sleep(0.2)  # Allow time for retry attempts

        # Event should have been processed (and failed)
        assert len(failed_events) > 0

        # Check if event bus supports dead letter queue
        if hasattr(test_event_bus, "get_dead_letter_queue"):
            dlq_events = await test_event_bus.get_dead_letter_queue()

            # Failed event should be in DLQ
            assert len(dlq_events) >= 1

            dlq_event = dlq_events[0]
            assert dlq_event.symbol == "AAPL"
            assert dlq_event.metadata["will_fail"] == True

    async def test_event_bus_health_monitoring(self, test_event_bus: IEventBus, test_symbols):
        """Test event bus health monitoring integration."""
        # Check if event bus supports health monitoring
        if hasattr(test_event_bus, "get_health_status"):
            initial_health = await test_event_bus.get_health_status()

            # Should be healthy initially
            assert initial_health.get("status", "unknown") in ["healthy", "ok", True]

            # Generate some load
            # Local imports
            from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner

            health_scanner = MockEventPublishingScanner("health_scanner", test_event_bus)

            # Generate multiple events
            tasks = []
            for i in range(5):
                alert = ScanAlert(
                    symbol=f"SYM{i:03d}",
                    alert_type=AlertType.VOLUME_SPIKE,
                    score=0.8,
                    metadata={"health_test": True},
                    timestamp=datetime.now(UTC),
                    scanner_name="health_scanner",
                )
                tasks.append(health_scanner.publish_alerts_to_event_bus([alert]))

            await asyncio.gather(*tasks)
            await asyncio.sleep(0.1)

            # Check health after load
            post_load_health = await test_event_bus.get_health_status()

            # Should still be healthy
            assert post_load_health.get("status", "unknown") in ["healthy", "ok", True]

            # Should have processed events
            processed_count = post_load_health.get("events_processed", 0)
            assert processed_count >= 5

    async def test_scanner_event_circuit_breaker(self, test_event_bus: IEventBus, test_symbols):
        """Test circuit breaker pattern for event publishing."""
        failure_count = 0

        # Create failing event bus mock
        original_publish = test_event_bus.publish

        async def failing_publish(event):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:  # Fail first 3 attempts
                raise Exception("Event bus temporarily unavailable")
            else:  # Succeed after that
                return await original_publish(event)

        test_event_bus.publish = failing_publish

        # Create scanner with circuit breaker logic
        # Local imports
        from tests.integration.scanners.event_bus.conftest import MockEventPublishingScanner

        class CircuitBreakerScanner(MockEventPublishingScanner):
            """Scanner with circuit breaker for event publishing."""

            def __init__(self, name: str, event_bus: IEventBus):
                super().__init__(name, event_bus)
                self.failure_count = 0
                self.circuit_open = False
                self.last_failure_time = None

            async def publish_alerts_to_event_bus(self, alerts):
                """Publish with circuit breaker pattern."""
                if self.circuit_open:
                    # Check if we should try again (after timeout)
                    if (datetime.now(UTC) - self.last_failure_time).total_seconds() < 10:
                        return  # Circuit still open
                    else:
                        self.circuit_open = False  # Try to close circuit

                try:
                    await super().publish_alerts_to_event_bus(alerts)
                    self.failure_count = 0  # Reset on success
                except Exception as e:
                    self.failure_count += 1
                    if self.failure_count >= 3:
                        self.circuit_open = True
                        self.last_failure_time = datetime.now(UTC)
                    raise e

        cb_scanner = CircuitBreakerScanner("circuit_breaker_scanner", test_event_bus)

        # First few attempts should fail and eventually open circuit
        for i in range(5):
            try:
                await cb_scanner.scan(["AAPL"])
            except Exception:
                pass  # Expected failures
            await asyncio.sleep(0.01)

        # Circuit should be open now
        assert cb_scanner.circuit_open == True
        assert failure_count >= 3

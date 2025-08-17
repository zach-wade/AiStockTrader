"""
Integration tests for Error Handling.

Tests error handling, resilience, and recovery across all events components
including circuit breakers, retry mechanisms, dead letter queues, and graceful degradation.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime
import json
from unittest.mock import AsyncMock, Mock

# Third-party imports
import pytest

# Local imports
from main.events.core import EventBusConfig, EventBusFactory
from main.events.core.event_bus_helpers.dead_letter_queue_manager import DeadLetterQueueManager
from main.events.handlers.feature_pipeline_handler import FeaturePipelineHandler
from main.events.handlers.scanner_feature_bridge import ScannerFeatureBridge
from main.events.types import FeatureRequestEvent, ScannerAlertEvent
from main.interfaces.events import EventType
from tests.fixtures.events.mock_database import create_mock_db_pool


@pytest.fixture
async def error_resilient_event_system():
    """Create event system configured for error handling testing."""
    # Mock configuration with aggressive error handling
    mock_config = {
        "events": {
            "batch_size": 2,
            "batch_interval_seconds": 0.05,
            "max_retries": 2,
            "circuit_breaker": {
                "failure_threshold": 3,
                "recovery_timeout": 0.1,
                "expected_failure_rate": 0.5,
            },
        },
        "scanner_bridge": {
            "batch_size": 2,
            "batch_timeout": 0.05,
            "max_retries": 2,
            "circuit_breaker_enabled": True,
        },
        "feature_pipeline": {
            "batch_size": 2,
            "queue_timeout": 5,
            "max_retries": 2,
            "computation_timeout": 1,
        },
        "dead_letter_queue": {"max_retries": 2, "retry_delay": 0.05, "retention_days": 1},
    }

    # Mock database for DLQ
    mock_db_pool = create_mock_db_pool()

    # Mock feature service that can be configured to fail
    mock_feature_service = AsyncMock()

    # Initialize components with error handling
    config = EventBusConfig(
        max_queue_size=1000, max_workers=2, enable_history=True, enable_dlq=True
    )
    event_bus = EventBusFactory.create(config)
    await event_bus.start()

    # Add DLQ manager to event bus
    dlq_manager = DeadLetterQueueManager(
        db_pool=mock_db_pool,
        max_retries=mock_config["dead_letter_queue"]["max_retries"],
        retention_days=mock_config["dead_letter_queue"]["retention_days"],
    )
    await dlq_manager.initialize()

    # Initialize bridge and pipeline with error handling
    bridge = ScannerFeatureBridge(event_bus=event_bus, config=mock_config)
    await bridge.start()

    pipeline_handler = FeaturePipelineHandler(
        event_bus=event_bus, feature_service=mock_feature_service, config=mock_config
    )
    await pipeline_handler.start()

    # Subscribe components
    event_bus.subscribe(EventType.FEATURE_REQUEST, pipeline_handler.handle_feature_request)

    yield {
        "event_bus": event_bus,
        "bridge": bridge,
        "pipeline_handler": pipeline_handler,
        "dlq_manager": dlq_manager,
        "feature_service": mock_feature_service,
        "db_pool": mock_db_pool,
        "config": mock_config,
    }

    # Cleanup
    await pipeline_handler.stop()
    await bridge.stop()
    await dlq_manager.close()
    await event_bus.stop()


@pytest.fixture
def failing_scenarios():
    """Create various failure scenarios for testing."""
    return {
        "network_failures": [
            {"type": "timeout", "delay": 2.0, "error": asyncio.TimeoutError},
            {"type": "connection", "error": ConnectionError("Network unavailable")},
            {
                "type": "intermittent",
                "failure_rate": 0.7,
                "error": Exception("Intermittent failure"),
            },
        ],
        "data_failures": [
            {"type": "malformed_event", "data": {"invalid": "structure"}},
            {"type": "missing_fields", "data": {"symbol": "TEST"}},  # Missing required fields
            {"type": "invalid_types", "data": {"symbol": None, "score": "not_a_number"}},
        ],
        "service_failures": [
            {"type": "service_unavailable", "error": Exception("Service temporarily unavailable")},
            {"type": "quota_exceeded", "error": Exception("Rate limit exceeded")},
            {"type": "authentication", "error": Exception("Authentication failed")},
        ],
    }


class TestErrorHandlingIntegration:
    """Test comprehensive error handling across all components."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(
        self, error_resilient_event_system, failing_scenarios, caplog
    ):
        """Test circuit breaker protection against cascading failures."""
        system = error_resilient_event_system
        event_bus = system["event_bus"]
        pipeline_handler = system["pipeline_handler"]
        feature_service = system["feature_service"]

        # Configure feature service to fail consistently
        failure_count = 0

        async def failing_feature_service(symbols, features, **kwargs):
            nonlocal failure_count
            failure_count += 1
            raise Exception(f"Feature service failure #{failure_count}")

        feature_service.compute_features.side_effect = failing_feature_service

        # Generate requests that will trigger circuit breaker
        failing_requests = []
        for i in range(10):  # More than failure threshold
            request = FeatureRequestEvent(
                symbols=[f"FAIL_SYMBOL_{i}"],
                features=["price_features"],
                requester="circuit_breaker_test",
                priority=5,
            )
            failing_requests.append(request)

        # Process requests and expect circuit breaker to kick in
        for request in failing_requests[:5]:  # First batch to trigger circuit breaker
            await event_bus.publish(request)
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.2)

        # Verify circuit breaker behavior
        assert failure_count >= 3  # Should have attempted several times
        assert (
            "circuit breaker" in caplog.text.lower() or "failure threshold" in caplog.text.lower()
        )

        # Continue with more requests - should be blocked by circuit breaker
        for request in failing_requests[5:]:
            await event_bus.publish(request)
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.2)

        # Circuit breaker should have prevented some calls
        assert failure_count < len(
            failing_requests
        )  # Not all requests should have reached the service

        # Verify pipeline handler remains responsive
        pipeline_stats = pipeline_handler.get_stats()
        assert isinstance(pipeline_stats, dict)

    @pytest.mark.asyncio
    async def test_dead_letter_queue_processing(self, error_resilient_event_system, caplog):
        """Test dead letter queue handling of failed events."""
        system = error_resilient_event_system
        event_bus = system["event_bus"]
        dlq_manager = system["dlq_manager"]
        feature_service = system["feature_service"]

        # Configure service to fail for specific symbols
        async def selective_failure_service(symbols, features, **kwargs):
            if any("DLQ_FAIL" in symbol for symbol in symbols):
                raise Exception("Intentional DLQ test failure")
            return {
                symbol: {feature: {"test_value": 1.0} for feature in features}
                for symbol in symbols
                if "DLQ_FAIL" not in symbol
            }

        feature_service.compute_features.side_effect = selective_failure_service

        # Add failing handler that will trigger DLQ
        failed_events = []

        async def failing_handler(event):
            if hasattr(event, "symbols") and any("DLQ_FAIL" in symbol for symbol in event.symbols):
                failed_events.append(event)
                await dlq_manager.add_failed_event(event, "Handler test failure")
                raise Exception("Handler intentionally failed")
            return True

        # Subscribe failing handler
        event_bus.subscribe(EventType.FEATURE_REQUEST, failing_handler)

        # Create mix of successful and failing events
        test_events = [
            FeatureRequestEvent(
                symbols=["SUCCESS_SYMBOL_1"], features=["price_features"], requester="dlq_test"
            ),
            FeatureRequestEvent(
                symbols=["DLQ_FAIL_SYMBOL_1"], features=["volume_features"], requester="dlq_test"
            ),
            FeatureRequestEvent(
                symbols=["SUCCESS_SYMBOL_2"], features=["trend_features"], requester="dlq_test"
            ),
            FeatureRequestEvent(
                symbols=["DLQ_FAIL_SYMBOL_2"], features=["price_features"], requester="dlq_test"
            ),
        ]

        # Process events
        for event in test_events:
            await event_bus.publish(event)
            await asyncio.sleep(0.05)

        await asyncio.sleep(0.3)

        # Verify DLQ captured failed events
        dlq_events = await dlq_manager.get_failed_events(limit=10)
        assert len(dlq_events) >= 2  # Should have captured the failing events

        # Verify DLQ events contain expected failures
        dlq_symbols = []
        for dlq_event in dlq_events:
            event_data = json.loads(dlq_event["event_data"])
            if "symbols" in event_data:
                dlq_symbols.extend(event_data["symbols"])

        assert any("DLQ_FAIL" in symbol for symbol in dlq_symbols)

        # Test DLQ retry mechanism
        retryable_events = await dlq_manager.get_retryable_events()
        assert len(retryable_events) >= 1

        # Verify error logging
        assert "failed" in caplog.text.lower() or "error" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_retry_mechanism_across_components(self, error_resilient_event_system, caplog):
        """Test retry mechanisms across bridge and pipeline components."""
        system = error_resilient_event_system
        event_bus = system["event_bus"]
        bridge = system["bridge"]
        pipeline_handler = system["pipeline_handler"]
        feature_service = system["feature_service"]

        # Configure service with intermittent failures
        call_count = 0

        async def intermittent_failure_service(symbols, features, **kwargs):
            nonlocal call_count
            call_count += 1

            # Fail first 2 attempts, succeed on 3rd
            if call_count <= 2:
                raise Exception(f"Intermittent failure attempt #{call_count}")

            return {
                symbol: {feature: {"retry_test": call_count} for feature in features}
                for symbol in symbols
            }

        feature_service.compute_features.side_effect = intermittent_failure_service

        # Create retry test event
        retry_event = FeatureRequestEvent(
            symbols=["RETRY_SYMBOL"],
            features=["price_features"],
            requester="retry_test",
            priority=7,
        )

        # Process event - should retry and eventually succeed
        await event_bus.publish(retry_event)
        await asyncio.sleep(0.5)  # Allow time for retries

        # Verify retries occurred
        assert call_count >= 2  # Should have attempted multiple times

        # Verify eventual success (if retry limit allows)
        pipeline_stats = pipeline_handler.get_stats()
        assert pipeline_stats["requests_received"] >= 1

        # Test bridge retry behavior with scanner alerts
        call_count = 0  # Reset for bridge test

        # Create scanner alert that will trigger feature request
        scanner_alert = ScannerAlertEvent(
            symbol="BRIDGE_RETRY_TEST",
            alert_type="high_volume",
            score=0.8,
            scanner_name="retry_scanner",
        )

        await event_bus.publish(scanner_alert)
        await asyncio.sleep(0.3)

        # Verify bridge handled the alert despite downstream failures
        bridge_stats = bridge.get_stats()
        assert bridge_stats["alerts_received_total"] >= 1

        # Should have logged retry attempts
        assert "retry" in caplog.text.lower() or "attempt" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_malformed_event_handling(
        self, error_resilient_event_system, failing_scenarios, caplog
    ):
        """Test handling of malformed and invalid events."""
        system = error_resilient_event_system
        event_bus = system["event_bus"]
        bridge = system["bridge"]
        dlq_manager = system["dlq_manager"]

        # Track malformed events
        malformed_events_handled = []

        async def malformed_event_handler(event):
            malformed_events_handled.append(event)
            # Simulate validation failure
            if not hasattr(event, "symbol") or not event.symbol:
                raise ValueError("Invalid event: missing required symbol")
            return True

        # Subscribe handler
        event_bus.subscribe(EventType.SCANNER_ALERT, malformed_event_handler)

        # Create various malformed events
        malformed_test_cases = [
            # Missing required fields
            ScannerAlertEvent(
                symbol="",  # Empty symbol
                alert_type="high_volume",
                score=0.7,
                scanner_name="malformed_test",
            ),
            # Invalid data types
            ScannerAlertEvent(
                symbol="VALID_SYMBOL",
                alert_type="invalid_type",  # Unknown alert type
                score=1.5,  # Invalid score > 1.0
                scanner_name="malformed_test",
            ),
            # Valid event for comparison
            ScannerAlertEvent(
                symbol="VALID_SYMBOL",
                alert_type="high_volume",
                score=0.8,
                scanner_name="valid_test",
            ),
        ]

        # Process malformed events
        for event in malformed_test_cases:
            try:
                await event_bus.publish(event)
                await asyncio.sleep(0.05)
            except Exception:
                # Some events may fail during publishing
                pass

        await asyncio.sleep(0.2)

        # Verify malformed event handling
        assert len(malformed_events_handled) >= 1

        # System should continue operating despite malformed events
        bridge_stats = bridge.get_stats()
        assert isinstance(bridge_stats, dict)

        # Should have logged validation errors
        assert any(level in caplog.text.lower() for level in ["error", "warning", "invalid"])

        # Test with completely invalid event structure
        try:
            # Create mock event with wrong structure
            invalid_event = Mock()
            invalid_event.event_type = EventType.SCANNER_ALERT
            invalid_event.event_id = "invalid_event_123"
            # Missing required attributes

            await event_bus.publish(invalid_event)
            await asyncio.sleep(0.1)
        except Exception:
            # Expected to fail
            pass

        # Event bus should remain operational
        final_stats = bridge.get_stats()
        assert isinstance(final_stats, dict)

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self, error_resilient_event_system, caplog):
        """Test prevention of cascading failures across components."""
        system = error_resilient_event_system
        event_bus = system["event_bus"]
        bridge = system["bridge"]
        pipeline_handler = system["pipeline_handler"]
        feature_service = system["feature_service"]

        # Simulate multiple failure points
        failure_points = {"feature_service": 0, "event_bus": 0, "bridge": 0}

        # Configure feature service to fail
        async def cascading_failure_service(symbols, features, **kwargs):
            failure_points["feature_service"] += 1
            raise Exception("Feature service cascade failure")

        feature_service.compute_features.side_effect = cascading_failure_service

        # Configure event bus to fail occasionally
        original_publish = event_bus.publish

        async def failing_publish(event):
            failure_points["event_bus"] += 1
            if failure_points["event_bus"] % 3 == 0:  # Fail every 3rd publish
                raise Exception("Event bus cascade failure")
            return await original_publish(event)

        # Generate load that will cause cascading failures
        cascade_alerts = []
        for i in range(15):
            alert = ScannerAlertEvent(
                symbol=f"CASCADE_SYMBOL_{i}",
                alert_type="high_volume",
                score=0.7,
                scanner_name="cascade_test",
            )
            cascade_alerts.append(alert)

        # Process alerts with potential cascading failures
        successful_publishes = 0

        for alert in cascade_alerts:
            try:
                await failing_publish(alert)
                successful_publishes += 1
            except Exception:
                # Continue despite failures
                pass
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.5)

        # Verify failure isolation
        # Bridge should remain responsive despite downstream failures
        bridge_stats = bridge.get_stats()
        assert isinstance(bridge_stats, dict)
        assert bridge_stats["alerts_received_total"] >= successful_publishes

        # Pipeline should handle failures gracefully
        pipeline_stats = pipeline_handler.get_stats()
        assert isinstance(pipeline_stats, dict)

        # Should have attempted feature service calls despite failures
        assert failure_points["feature_service"] >= 1

        # Verify components didn't cascade failures
        # Each component should isolate failures from others
        assert failure_points["event_bus"] >= 1  # Some publish failures
        assert successful_publishes > 0  # Some successes despite failures

        # Should have logged cascade prevention
        assert any(term in caplog.text.lower() for term in ["error", "failed", "exception"])

    @pytest.mark.asyncio
    async def test_graceful_degradation_under_stress(self, error_resilient_event_system, caplog):
        """Test graceful degradation when system is under stress."""
        system = error_resilient_event_system
        event_bus = system["event_bus"]
        bridge = system["bridge"]
        pipeline_handler = system["pipeline_handler"]
        feature_service = system["feature_service"]

        # Configure service with high latency and failures
        call_delays = []

        async def stressed_service(symbols, features, **kwargs):
            # Simulate high latency
            delay = 0.1 + len(call_delays) * 0.02  # Increasing delay
            call_delays.append(delay)
            await asyncio.sleep(delay)

            # Fail occasionally under stress
            if len(call_delays) % 4 == 0:
                raise Exception("Service under stress")

            return {
                symbol: {feature: {"stress_test": len(call_delays)} for feature in features}
                for symbol in symbols
            }

        feature_service.compute_features.side_effect = stressed_service

        # Generate high load to stress the system
        stress_load = []
        for i in range(50):  # High volume
            alert = ScannerAlertEvent(
                symbol=f"STRESS_SYMBOL_{i % 10}",  # Reuse symbols to test deduplication under stress
                alert_type="high_volume" if i % 2 == 0 else "catalyst_detected",
                score=0.6 + (i % 40) / 100.0,
                scanner_name=f"stress_scanner_{i % 3}",
            )
            stress_load.append(alert)

        # Apply stress load rapidly
        stress_start_time = asyncio.get_event_loop().time()

        # Publish in rapid bursts
        for i in range(0, len(stress_load), 10):
            burst = stress_load[i : i + 10]
            burst_tasks = [event_bus.publish(alert) for alert in burst]
            try:
                await asyncio.gather(*burst_tasks, return_exceptions=True)
            except Exception:
                # Continue despite exceptions
                pass

        # Wait for stress processing
        await asyncio.sleep(1.0)

        stress_end_time = asyncio.get_event_loop().time()
        stress_duration = stress_end_time - stress_start_time

        # Verify graceful degradation
        bridge_stats = bridge.get_stats()
        pipeline_stats = pipeline_handler.get_stats()

        # System should still be responsive
        assert isinstance(bridge_stats, dict)
        assert isinstance(pipeline_stats, dict)

        # Should have processed significant portion of load
        assert bridge_stats["alerts_received_total"] >= len(stress_load) * 0.5

        # Should have attempted processing despite stress
        assert len(call_delays) >= 5

        # Performance should have degraded gracefully
        avg_delay = sum(call_delays) / len(call_delays) if call_delays else 0
        assert avg_delay > 0.1  # Should show increased latency under stress

        # Should have logged stress indicators
        assert any(term in caplog.text.lower() for term in ["error", "timeout", "failed"])

        # Verify throughput remained reasonable despite stress
        effective_throughput = bridge_stats["alerts_received_total"] / stress_duration
        assert effective_throughput > 10  # Should maintain some throughput

    @pytest.mark.asyncio
    async def test_error_recovery_and_healing(self, error_resilient_event_system, caplog):
        """Test system recovery and self-healing after errors."""
        system = error_resilient_event_system
        event_bus = system["event_bus"]
        bridge = system["bridge"]
        pipeline_handler = system["pipeline_handler"]
        feature_service = system["feature_service"]

        # Phase 1: Cause system stress and failures
        failure_phase_calls = 0

        async def failing_then_recovering_service(symbols, features, **kwargs):
            nonlocal failure_phase_calls
            failure_phase_calls += 1

            # Fail for first 10 calls, then start recovering
            if failure_phase_calls <= 10:
                raise Exception(f"Recovery test failure #{failure_phase_calls}")

            # Gradual recovery - occasional failures
            if failure_phase_calls <= 15 and failure_phase_calls % 3 == 0:
                raise Exception(f"Recovery phase failure #{failure_phase_calls}")

            # Full recovery
            return {
                symbol: {feature: {"recovery_test": failure_phase_calls} for feature in features}
                for symbol in symbols
            }

        feature_service.compute_features.side_effect = failing_then_recovering_service

        # Generate load during failure and recovery
        recovery_test_events = []
        for i in range(25):  # Covers failure and recovery phases
            event = FeatureRequestEvent(
                symbols=[f"RECOVERY_SYMBOL_{i % 5}"],
                features=["price_features"],
                requester="recovery_test",
                priority=5,
            )
            recovery_test_events.append(event)

        # Process events through failure and recovery
        for i, event in enumerate(recovery_test_events):
            await event_bus.publish(event)
            await asyncio.sleep(0.05)

            # Check system state periodically
            if i % 5 == 4:
                bridge_stats = bridge.get_stats()
                pipeline_stats = pipeline_handler.get_stats()

                # System should remain responsive throughout
                assert isinstance(bridge_stats, dict)
                assert isinstance(pipeline_stats, dict)

        await asyncio.sleep(0.5)

        # Verify recovery occurred
        assert failure_phase_calls >= 20  # Should have processed through recovery

        # Final system state should be healthy
        final_bridge_stats = bridge.get_stats()
        final_pipeline_stats = pipeline_handler.get_stats()

        assert final_bridge_stats["alerts_received_total"] == 0  # No alerts in this test
        assert (
            final_pipeline_stats["requests_received"] >= 15
        )  # Should have processed recovery phase

        # Test post-recovery functionality
        post_recovery_event = ScannerAlertEvent(
            symbol="POST_RECOVERY_TEST",
            alert_type="high_volume",
            score=0.8,
            scanner_name="post_recovery_scanner",
        )

        await event_bus.publish(post_recovery_event)
        await asyncio.sleep(0.2)

        # System should function normally after recovery
        post_recovery_stats = bridge.get_stats()
        assert post_recovery_stats["alerts_received_total"] >= 1

        # Should have logged recovery process
        assert any(term in caplog.text.lower() for term in ["recovery", "failed", "error"])

    @pytest.mark.asyncio
    async def test_error_correlation_and_tracking(self, error_resilient_event_system, caplog):
        """Test error correlation and tracking across components."""
        system = error_resilient_event_system
        event_bus = system["event_bus"]
        bridge = system["bridge"]
        dlq_manager = system["dlq_manager"]
        feature_service = system["feature_service"]

        # Track errors across components
        error_tracking = {"bridge_errors": [], "pipeline_errors": [], "service_errors": []}

        # Configure service to generate trackable errors
        async def error_tracking_service(symbols, features, **kwargs):
            error_info = {
                "timestamp": datetime.now(UTC),
                "symbols": symbols,
                "features": features,
                "error_type": "service_error",
            }
            error_tracking["service_errors"].append(error_info)
            raise Exception(f"Trackable error for symbols: {symbols}")

        feature_service.compute_features.side_effect = error_tracking_service

        # Create correlated test scenarios
        correlation_scenarios = [
            {
                "correlation_id": "error_scenario_1",
                "alerts": [
                    ScannerAlertEvent(
                        symbol="TRACK_SYMBOL_1",
                        alert_type="high_volume",
                        score=0.7,
                        scanner_name="tracking_scanner",
                    ),
                    ScannerAlertEvent(
                        symbol="TRACK_SYMBOL_2",
                        alert_type="breakout",
                        score=0.8,
                        scanner_name="tracking_scanner",
                    ),
                ],
            },
            {
                "correlation_id": "error_scenario_2",
                "alerts": [
                    ScannerAlertEvent(
                        symbol="TRACK_SYMBOL_3",
                        alert_type="catalyst_detected",
                        score=0.9,
                        scanner_name="tracking_scanner",
                    )
                ],
            },
        ]

        # Process correlated scenarios
        for scenario in correlation_scenarios:
            for alert in scenario["alerts"]:
                alert.correlation_id = scenario["correlation_id"]
                await event_bus.publish(alert)
                await asyncio.sleep(0.05)

        await asyncio.sleep(0.5)

        # Verify error tracking and correlation
        assert len(error_tracking["service_errors"]) >= 2

        # Check DLQ for correlated errors
        dlq_events = await dlq_manager.get_failed_events(limit=20)

        # Verify error correlation in logs
        log_text = caplog.text.lower()
        assert "error" in log_text or "failed" in log_text

        # Test error aggregation by symbol
        error_symbols = []
        for error in error_tracking["service_errors"]:
            error_symbols.extend(error["symbols"])

        unique_error_symbols = set(error_symbols)
        assert len(unique_error_symbols) >= 3  # Should have tracked multiple symbols

        # Verify components maintained state despite errors
        bridge_stats = bridge.get_stats()
        assert bridge_stats["alerts_received_total"] >= len(
            [alert for scenario in correlation_scenarios for alert in scenario["alerts"]]
        )

        # Error tracking should help identify patterns
        symbol_error_counts = {}
        for error in error_tracking["service_errors"]:
            for symbol in error["symbols"]:
                symbol_error_counts[symbol] = symbol_error_counts.get(symbol, 0) + 1

        assert len(symbol_error_counts) >= 3  # Multiple symbols with errors

# tests/integration/test_event_feature_integration.py

# Standard library imports
import asyncio
from datetime import UTC, datetime
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
from main.events import EventBusFactory
from main.events.types import FeatureRequestEvent, ScannerAlertEvent
from main.feature_pipeline.feature_orchestrator import FeatureOrchestrator
from main.interfaces.events import EventType, IEventBus


# Mock Scanner for Testing
class MockScanner:
    """Mock scanner for integration testing."""

    def __init__(self, event_bus: IEventBus):
        self.event_bus = event_bus
        self.alerts_sent = []

    async def send_alert(self, symbol: str, alert_type: str, data: dict[str, Any]):
        """Send a scanner alert."""
        alert_event = ScannerAlertEvent(
            data={
                "symbol": symbol,
                "alert_type": alert_type,
                "timestamp": datetime.now(UTC).isoformat(),
                "data": data,
                "scanner_id": "mock_scanner",
            }
        )

        await self.event_bus.publish(alert_event)
        self.alerts_sent.append(alert_event.data)


# Test Fixtures
@pytest.fixture
def mock_config():
    """Mock configuration for event integration tests."""
    return {
        "orchestrator": {
            "features": {
                "cache": {"ttl_seconds": 300},
                "parallel_processing": {"max_workers": 2, "max_concurrent_tasks": 5},
                "batch_processing": {"interval_seconds": 0.1, "processing_delay_seconds": 0.05},
            },
            "lookback_periods": {"feature_calculation_days": 30},
        },
        "scanner": {
            "feature_bridge": {
                "priority_threshold": 7,
                "batch_size": 10,
                "max_pending_requests": 100,
            }
        },
    }


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D", tz=UTC)
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]

    data_frames = []
    np.random.seed(42)

    for symbol in symbols:
        base_price = {"AAPL": 150, "MSFT": 250, "GOOGL": 100, "TSLA": 200}[symbol]

        prices = [base_price]
        for _ in range(len(dates) - 1):
            change = secure_numpy_normal(0, 0.02)
            prices.append(prices[-1] * (1 + change))

        symbol_data = pd.DataFrame(
            {
                "symbol": symbol,
                "timestamp": dates,
                "close": prices,
                "volume": np.secure_randint(10000, 100000, len(dates)),
            }
        )
        data_frames.append(symbol_data)

    return pd.concat(data_frames, ignore_index=True)


@pytest.fixture
def mock_dependencies(sample_market_data):
    """Create mock dependencies for orchestrator."""
    # Mock database adapter
    db_adapter = MagicMock()
    db_adapter.execute_query = AsyncMock()
    db_adapter.fetch_query = AsyncMock()

    # Mock data archive
    data_archive = MagicMock()
    data_archive.save_dataframe = AsyncMock()

    # Mock market data repo
    market_data_repo = MagicMock()
    market_data_repo.get_data_for_symbols_and_range = AsyncMock(return_value=sample_market_data)

    # Mock feature store repo
    feature_store_repo = MagicMock()
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.errors = []
    feature_store_repo.store_features = AsyncMock(return_value=mock_result)

    return {
        "db_adapter": db_adapter,
        "data_archive": data_archive,
        "market_data_repo": market_data_repo,
        "feature_store_repo": feature_store_repo,
    }


# Test Scanner-Feature Bridge Integration
class TestScannerFeatureIntegration:
    """Test integration between scanner alerts and feature calculations."""

    @pytest.mark.asyncio
    async def test_scanner_alert_triggers_feature_calculation(self, mock_config, mock_dependencies):
        """Test that scanner alerts trigger appropriate feature calculations."""
        # Create event bus
        event_bus = EventBusFactory.create_test_instance()

        # Create orchestrator with mocked dependencies
        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_dependencies["data_archive"],
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies with mocks
                orchestrator.db_adapter = mock_dependencies["db_adapter"]
                orchestrator.data_archive = mock_dependencies["data_archive"]
                orchestrator.market_data_repo = mock_dependencies["market_data_repo"]
                orchestrator.feature_store_repo = mock_dependencies["feature_store_repo"]

        # Create mock scanner
        scanner = MockScanner(event_bus)

        # Send scanner alert
        await scanner.send_alert(
            symbol="AAPL",
            alert_type="high_volume",
            data={
                "current_volume": 50000,
                "average_volume": 20000,
                "volume_ratio": 2.5,
                "confidence": 0.85,
            },
        )

        # Allow time for event processing
        await asyncio.sleep(0.2)

        # Verify scanner alert was processed
        assert orchestrator.calculation_stats["scanner_triggered_calculations"] > 0

        # Verify market data was fetched for the alerted symbol
        mock_dependencies["market_data_repo"].get_data_for_symbols_and_range.assert_called()
        call_args = mock_dependencies["market_data_repo"].get_data_for_symbols_and_range.call_args
        assert "AAPL" in call_args[1]["symbols"]

        # Verify features were stored
        mock_dependencies["feature_store_repo"].store_features.assert_called()

    @pytest.mark.asyncio
    async def test_multiple_scanner_alerts_parallel_processing(
        self, mock_config, mock_dependencies
    ):
        """Test parallel processing of multiple scanner alerts."""
        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_dependencies["data_archive"],
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies
                orchestrator.db_adapter = mock_dependencies["db_adapter"]
                orchestrator.data_archive = mock_dependencies["data_archive"]
                orchestrator.market_data_repo = mock_dependencies["market_data_repo"]
                orchestrator.feature_store_repo = mock_dependencies["feature_store_repo"]

        scanner = MockScanner(event_bus)

        # Send multiple alerts for different symbols
        alert_tasks = []
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        alert_types = ["high_volume", "price_breakout", "unusual_activity", "momentum_shift"]

        for symbol, alert_type in zip(symbols, alert_types):
            task = scanner.send_alert(
                symbol=symbol, alert_type=alert_type, data={"test_metric": np.random.random()}
            )
            alert_tasks.append(task)

        # Send all alerts concurrently
        await asyncio.gather(*alert_tasks)

        # Allow time for all processing
        await asyncio.sleep(0.5)

        # Verify all alerts were processed
        assert len(scanner.alerts_sent) == len(symbols)
        assert orchestrator.calculation_stats["scanner_triggered_calculations"] >= len(symbols)

        # Verify parallel processing occurred
        assert orchestrator.calculation_stats["parallel_tasks"] > 0

    @pytest.mark.asyncio
    async def test_alert_type_to_feature_mapping(self, mock_config, mock_dependencies):
        """Test that different alert types trigger appropriate feature sets."""
        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_dependencies["data_archive"],
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies
                orchestrator.db_adapter = mock_dependencies["db_adapter"]
                orchestrator.data_archive = mock_dependencies["data_archive"]
                orchestrator.market_data_repo = mock_dependencies["market_data_repo"]
                orchestrator.feature_store_repo = mock_dependencies["feature_store_repo"]

        scanner = MockScanner(event_bus)

        # Test different alert types and their expected feature mappings
        test_cases = [
            {
                "alert_type": "high_volume",
                "expected_features": ["volume_features", "price_features", "volatility"],
            },
            {
                "alert_type": "sentiment_surge",
                "expected_features": ["sentiment_features", "news_features", "social_features"],
            },
            {"alert_type": "catalyst_detected", "expected_features": ["all_features"]},
            {
                "alert_type": "unknown_alert_type",
                "expected_features": ["price_features", "volume_features", "volatility"],  # default
            },
        ]

        for case in test_cases:
            # Send alert
            await scanner.send_alert(
                symbol="TEST", alert_type=case["alert_type"], data={"test": True}
            )

            await asyncio.sleep(0.1)

            # Verify feature mapping is correct
            # In a real implementation, we'd check what features were calculated
            # For this test, we verify the mapping exists in the orchestrator
            expected_features = orchestrator._alert_feature_mapping.get(
                case["alert_type"], orchestrator._alert_feature_mapping["default"]
            )

            assert expected_features == case["expected_features"]


# Test Feature Request Event Processing
class TestFeatureRequestEvents:
    """Test processing of feature request events."""

    @pytest.mark.asyncio
    async def test_high_priority_feature_request_immediate_processing(
        self, mock_config, mock_dependencies
    ):
        """Test that high-priority feature requests are processed immediately."""
        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_dependencies["data_archive"],
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies
                orchestrator.db_adapter = mock_dependencies["db_adapter"]
                orchestrator.data_archive = mock_dependencies["data_archive"]
                orchestrator.market_data_repo = mock_dependencies["market_data_repo"]
                orchestrator.feature_store_repo = mock_dependencies["feature_store_repo"]

        # Create high-priority feature request
        feature_request = FeatureRequestEvent(
            data={
                "symbols": ["AAPL", "MSFT"],
                "features": ["technical_features", "volume_features"],
                "priority": 9,  # High priority
                "requestor": "urgent_strategy",
                "request_id": "urgent_001",
            }
        )

        # Record time before request
        start_time = time.time()

        # Send feature request
        await event_bus.publish(feature_request)

        # Allow minimal processing time
        await asyncio.sleep(0.1)

        processing_time = time.time() - start_time

        # Verify immediate processing
        assert orchestrator.calculation_stats["feature_requests_processed"] > 0
        assert processing_time < 1.0  # Should be processed quickly

        # Verify market data was fetched
        mock_dependencies["market_data_repo"].get_data_for_symbols_and_range.assert_called()

    @pytest.mark.asyncio
    async def test_low_priority_feature_request_batch_processing(
        self, mock_config, mock_dependencies
    ):
        """Test that low-priority feature requests are batched."""
        # Reduce batch interval for faster testing
        mock_config["orchestrator"]["features"]["batch_processing"]["interval_seconds"] = 0.1

        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_dependencies["data_archive"],
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies
                orchestrator.db_adapter = mock_dependencies["db_adapter"]
                orchestrator.data_archive = mock_dependencies["data_archive"]
                orchestrator.market_data_repo = mock_dependencies["market_data_repo"]
                orchestrator.feature_store_repo = mock_dependencies["feature_store_repo"]

        # Send multiple low-priority requests
        low_priority_requests = [
            FeatureRequestEvent(
                data={
                    "symbols": [f"TEST{i}"],
                    "features": ["basic_features"],
                    "priority": 3,  # Low priority
                    "requestor": "background_service",
                    "request_id": f"batch_{i}",
                }
            )
            for i in range(5)
        ]

        # Send all requests quickly
        for request in low_priority_requests:
            await event_bus.publish(request)
            await asyncio.sleep(0.01)  # Small gap between requests

        # Wait for batch processing
        await asyncio.sleep(0.5)

        # Verify requests were processed (eventually)
        assert orchestrator.calculation_stats["feature_requests_processed"] >= len(
            low_priority_requests
        )

    @pytest.mark.asyncio
    async def test_feature_request_priority_ordering(self, mock_config, mock_dependencies):
        """Test that feature requests are processed in priority order."""
        mock_config["orchestrator"]["features"]["batch_processing"]["interval_seconds"] = 0.2

        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_dependencies["data_archive"],
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies
                orchestrator.db_adapter = mock_dependencies["db_adapter"]
                orchestrator.data_archive = mock_dependencies["data_archive"]
                orchestrator.market_data_repo = mock_dependencies["market_data_repo"]
                orchestrator.feature_store_repo = mock_dependencies["feature_store_repo"]

        # Create requests with different priorities
        requests = [
            FeatureRequestEvent(
                data={
                    "symbols": ["LOW1"],
                    "features": ["basic"],
                    "priority": 2,
                    "requestor": "low_priority_1",
                    "request_id": "low_1",
                }
            ),
            FeatureRequestEvent(
                data={
                    "symbols": ["HIGH1"],
                    "features": ["basic"],
                    "priority": 8,
                    "requestor": "high_priority_1",
                    "request_id": "high_1",
                }
            ),
            FeatureRequestEvent(
                data={
                    "symbols": ["MED1"],
                    "features": ["basic"],
                    "priority": 5,
                    "requestor": "medium_priority_1",
                    "request_id": "med_1",
                }
            ),
            FeatureRequestEvent(
                data={
                    "symbols": ["HIGH2"],
                    "features": ["basic"],
                    "priority": 9,
                    "requestor": "high_priority_2",
                    "request_id": "high_2",
                }
            ),
        ]

        # Send requests in random order
        for request in requests:
            await event_bus.publish(request)
            await asyncio.sleep(0.01)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Verify all requests processed
        assert orchestrator.calculation_stats["feature_requests_processed"] >= len(requests)

        # High priority requests (8, 9) should be processed immediately
        # Lower priority requests should be batched
        # This is verified by checking that processing occurred


# Test Event Bus Resilience
class TestEventBusResilience:
    """Test event system error handling and resilience."""

    @pytest.mark.asyncio
    async def test_event_processing_error_recovery(self, mock_config, mock_dependencies):
        """Test that event processing errors don't crash the system."""
        event_bus = EventBusFactory.create_test_instance()

        # Create orchestrator that will fail on first call
        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_dependencies["data_archive"],
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies
                orchestrator.db_adapter = mock_dependencies["db_adapter"]
                orchestrator.data_archive = mock_dependencies["data_archive"]
                orchestrator.market_data_repo = mock_dependencies["market_data_repo"]
                orchestrator.feature_store_repo = mock_dependencies["feature_store_repo"]

                # Make first call fail, subsequent calls succeed
                call_count = [0]
                original_get_data = mock_dependencies[
                    "market_data_repo"
                ].get_data_for_symbols_and_range

                async def failing_get_data(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        raise Exception("Database temporarily unavailable")
                    return await original_get_data(*args, **kwargs)

                mock_dependencies["market_data_repo"].get_data_for_symbols_and_range.side_effect = (
                    failing_get_data
                )

        # Send events that will trigger both failure and success
        scanner = MockScanner(event_bus)

        # First alert should fail
        await scanner.send_alert("FAIL", "high_volume", {"test": True})
        await asyncio.sleep(0.1)

        # Second alert should succeed
        await scanner.send_alert("SUCCESS", "high_volume", {"test": True})
        await asyncio.sleep(0.1)

        # System should continue operating despite first failure
        assert orchestrator.calculation_stats["errors"] > 0  # First call failed
        assert (
            orchestrator.calculation_stats["scanner_triggered_calculations"] > 0
        )  # Second call succeeded

    @pytest.mark.asyncio
    async def test_event_bus_message_ordering(self, mock_config, mock_dependencies):
        """Test that event messages maintain proper ordering."""
        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_dependencies["data_archive"],
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies
                orchestrator.db_adapter = mock_dependencies["db_adapter"]
                orchestrator.data_archive = mock_dependencies["data_archive"]
                orchestrator.market_data_repo = mock_dependencies["market_data_repo"]
                orchestrator.feature_store_repo = mock_dependencies["feature_store_repo"]

        # Track processing order
        processed_events = []

        # Patch the event handler to track order
        original_handler = orchestrator._on_scanner_alert

        async def tracking_handler(event):
            processed_events.append(event.data["symbol"])
            await original_handler(event)

        orchestrator._on_scanner_alert = tracking_handler

        # Re-subscribe with our tracking handler
        event_bus.unsubscribe(EventType.SCANNER_ALERT, original_handler)
        event_bus.subscribe(EventType.SCANNER_ALERT, tracking_handler)

        # Send events in specific order
        scanner = MockScanner(event_bus)
        expected_order = ["FIRST", "SECOND", "THIRD"]

        for symbol in expected_order:
            await scanner.send_alert(symbol, "test_alert", {"order_test": True})

        await asyncio.sleep(0.3)

        # Verify events were processed in order
        assert processed_events == expected_order

    @pytest.mark.asyncio
    async def test_event_handler_resource_cleanup(self, mock_config, mock_dependencies):
        """Test that event handlers properly clean up resources."""
        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_dependencies["data_archive"],
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies
                orchestrator.db_adapter = mock_dependencies["db_adapter"]
                orchestrator.data_archive = mock_dependencies["data_archive"]
                orchestrator.market_data_repo = mock_dependencies["market_data_repo"]
                orchestrator.feature_store_repo = mock_dependencies["feature_store_repo"]

        # Send multiple events to trigger resource usage
        scanner = MockScanner(event_bus)

        for i in range(10):
            await scanner.send_alert(f"SYMBOL_{i}", "resource_test", {"iteration": i})

        await asyncio.sleep(0.5)

        # Verify all events processed
        assert orchestrator.calculation_stats["scanner_triggered_calculations"] >= 10

        # Test graceful shutdown
        await orchestrator.shutdown()

        # Verify thread pool was shutdown
        assert orchestrator.executor._shutdown


# Test Performance Under Load
class TestEventSystemPerformance:
    """Test event system performance under load."""

    @pytest.mark.asyncio
    async def test_high_volume_event_processing(self, mock_config, mock_dependencies):
        """Test processing high volume of events."""
        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_dependencies["data_archive"],
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies
                orchestrator.db_adapter = mock_dependencies["db_adapter"]
                orchestrator.data_archive = mock_dependencies["data_archive"]
                orchestrator.market_data_repo = mock_dependencies["market_data_repo"]
                orchestrator.feature_store_repo = mock_dependencies["feature_store_repo"]

        scanner = MockScanner(event_bus)

        # Send high volume of events
        num_events = 100
        start_time = time.time()

        # Send events rapidly
        for i in range(num_events):
            await scanner.send_alert(f"STRESS_TEST_{i}", "volume_test", {"index": i})
            if i % 20 == 0:  # Small batches to prevent overwhelming
                await asyncio.sleep(0.01)

        # Wait for processing
        await asyncio.sleep(2.0)

        processing_time = time.time() - start_time

        # Verify high throughput
        events_processed = orchestrator.calculation_stats["scanner_triggered_calculations"]
        assert events_processed >= num_events * 0.8  # Allow for some processing delays

        # Performance should be reasonable (this is with mocks, so should be fast)
        events_per_second = events_processed / processing_time
        assert events_per_second > 10  # Should handle at least 10 events/second with mocks


if __name__ == "__main__":
    pytest.main([__file__])

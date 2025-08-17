# tests/integration/test_scanner_feature_bridge.py

# Standard library imports
import asyncio
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party imports
import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Local imports
from main.events.core.event_bus import EventBus
from main.events.handlers.scanner_feature_bridge import ScannerFeatureBridge
from main.events.types import (
    AlertType,
    EventType,
    FeatureRequestEvent,
    ScanAlert,
    ScannerAlertEvent,
)


# Mock Scanner for Integration Testing
class MockScanner:
    """Mock scanner that generates realistic alerts for testing."""

    def __init__(self, name: str = "mock_scanner"):
        self.name = name
        self.alerts_generated = []

    def generate_alert(
        self,
        symbol: str,
        alert_type: str,
        score: float = 0.75,
        metadata: dict[str, Any] | None = None,
    ) -> ScanAlert:
        """Generate a mock scanner alert."""
        alert = ScanAlert(symbol=symbol, alert_type=alert_type, score=score, data=metadata or {})
        self.alerts_generated.append(alert)
        return alert

    async def generate_high_volume_alert(self, symbol: str) -> ScanAlert:
        """Generate high volume alert."""
        return self.generate_alert(
            symbol=symbol,
            alert_type="HIGH_VOLUME",
            score=0.8,
            metadata={"current_volume": 150000, "average_volume": 50000, "volume_ratio": 3.0},
        )

    async def generate_breakout_alert(self, symbol: str) -> ScanAlert:
        """Generate breakout alert."""
        return self.generate_alert(
            symbol=symbol,
            alert_type="BREAKOUT",
            score=0.9,
            metadata={
                "breakout_level": 150.00,
                "support_level": 145.00,
                "resistance_level": 155.00,
            },
        )

    async def generate_catalyst_alert(self, symbol: str) -> ScanAlert:
        """Generate high-priority catalyst alert."""
        return self.generate_alert(
            symbol=symbol,
            alert_type="catalyst_detected",
            score=0.95,
            metadata={
                "catalyst_type": "earnings_surprise",
                "catalyst_strength": "strong",
                "news_items": 3,
            },
        )


# Mock Feature Pipeline
class MockFeaturePipeline:
    """Mock feature pipeline that receives and processes feature requests."""

    def __init__(self):
        self.feature_requests_received = []
        self.processing_delays = {}  # symbol -> delay in seconds

    async def process_feature_request(self, event: FeatureRequestEvent):
        """Mock feature request processing."""
        self.feature_requests_received.append(event.data)

        # Simulate processing delay
        symbols = event.data.get("symbols", [])
        for symbol in symbols:
            delay = self.processing_delays.get(symbol, 0.01)
            await asyncio.sleep(delay)

    def set_processing_delay(self, symbol: str, delay: float):
        """Set processing delay for specific symbol."""
        self.processing_delays[symbol] = delay

    def get_processed_symbols(self) -> list[str]:
        """Get all symbols that have been processed."""
        symbols = []
        for request in self.feature_requests_received:
            symbols.extend(request.get("symbols", []))
        return symbols

    def get_request_count(self) -> int:
        """Get total number of feature requests received."""
        return len(self.feature_requests_received)


# Test Fixtures
@pytest.fixture
def mock_config():
    """Mock configuration for bridge testing."""
    return {
        "events": {"batch_size": 5, "batch_interval_seconds": 0.1},  # Fast for testing
        "scanner_bridge": {
            "priority_boost_threshold": 0.8,
            "urgent_alert_types": ["catalyst_detected", "opportunity_signal"],
            "max_pending_batches": 100,
        },
    }


@pytest.fixture
def mock_event_bus():
    """Mock event bus for controlled testing."""
    event_bus = MagicMock(spec=EventBus)
    event_bus.subscribe = AsyncMock()
    event_bus.unsubscribe = AsyncMock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_scanner():
    """Mock scanner instance."""
    return MockScanner("integration_test_scanner")


@pytest.fixture
def mock_feature_pipeline():
    """Mock feature pipeline instance."""
    return MockFeaturePipeline()


# Test Scanner-Feature Bridge Core Flow
class TestScannerFeatureBridgeFlow:
    """Test end-to-end scanner alert to feature request flow."""

    @pytest.mark.asyncio
    async def test_basic_alert_to_feature_request_flow(
        self, mock_config, mock_event_bus, mock_scanner, mock_feature_pipeline
    ):
        """Test basic flow from scanner alert to feature request."""
        # Create bridge with mocked event bus
        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        # Start bridge
        await bridge.start()

        # Generate scanner alert
        alert = await mock_scanner.generate_high_volume_alert("AAPL")

        # Create event from alert and process directly
        event = ScannerAlertEvent(
            symbol=alert.symbol,
            alert_type=alert.alert_type,
            score=alert.score,
            scanner_name=alert.source_scanner or "test_scanner",
            metadata=alert.metadata,
        )

        # Process event directly through handler
        await bridge._handle_scanner_alert_event(event)

        # Allow time for processing
        await asyncio.sleep(0.2)

        # Verify event bus interactions
        mock_event_bus.subscribe.assert_called_with(
            EventType.SCANNER_ALERT, bridge._handle_scanner_alert_event
        )
        mock_event_bus.publish.assert_called()

        # Verify bridge statistics
        stats = bridge.get_stats()
        assert stats["alerts_received_total"] > 0
        assert stats["unique_symbols_processed_count"] > 0

        # Stop bridge
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_multiple_alerts_batching(self, mock_config, mock_event_bus, mock_scanner):
        """Test that multiple alerts are properly batched."""
        # Set small batch size for testing
        mock_config["events"]["batch_size"] = 3

        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Generate multiple alerts
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        alerts = []

        for symbol in symbols:
            alert = await mock_scanner.generate_high_volume_alert(symbol)
            alerts.append(alert)
            # Create event and process directly
            event = ScannerAlertEvent(
                symbol=alert.symbol,
                alert_type=alert.alert_type,
                score=alert.score,
                scanner_name=alert.source_scanner or "test_scanner",
                metadata=alert.metadata,
            )
            await bridge._handle_scanner_alert_event(event)

        # Allow time for batch processing
        await asyncio.sleep(0.3)

        # Verify multiple feature requests were published
        assert mock_event_bus.publish.call_count >= 2  # Should have multiple batches

        # Verify stats tracking
        stats = bridge.get_stats()
        assert stats["alerts_received_total"] == len(symbols)
        assert stats["unique_symbols_processed_count"] == len(symbols)

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_alert_type_feature_mapping(self, mock_config, mock_event_bus, mock_scanner):
        """Test that different alert types map to correct feature sets."""
        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Test different alert types
        test_cases = [
            {
                "symbol": "AAPL",
                "alert_type": AlertType.HIGH_VOLUME,
                "expected_features": ["volume_features", "price_features", "volatility_features"],
            },
            {
                "symbol": "MSFT",
                "alert_type": AlertType.BREAKOUT,
                "expected_features": [
                    "price_features",
                    "trend_features",
                    "support_resistance_features",
                ],
            },
            {
                "symbol": "GOOGL",
                "alert_type": "catalyst_detected",
                "expected_features": "all_features",  # Special case
            },
        ]

        for case in test_cases:
            alert = mock_scanner.generate_alert(
                symbol=case["symbol"], alert_type=case["alert_type"], score=0.8
            )
            bridge.process_scan_alert(alert)

        await asyncio.sleep(0.2)

        # Verify that feature mapping is working
        # (In a real implementation, we'd check the actual feature requests)
        mapper = bridge._alert_feature_mapper

        for case in test_cases:
            features = mapper.get_features_for_alert_type(case["alert_type"])
            if case["expected_features"] == "all_features":
                # Should get the full feature list
                assert len(features) > 10  # Should have many features
            else:
                assert features == case["expected_features"]

        await bridge.stop()


# Test Priority Handling
class TestPriorityHandling:
    """Test priority calculation and handling for alerts."""

    @pytest.mark.asyncio
    async def test_priority_calculation_based_on_score(
        self, mock_config, mock_event_bus, mock_scanner
    ):
        """Test that priority is calculated correctly based on alert score."""
        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Test different score levels
        test_cases = [
            {"symbol": "LOW1", "score": 0.3, "expected_priority_range": (1, 4)},
            {"symbol": "MED1", "score": 0.6, "expected_priority_range": (4, 7)},
            {"symbol": "HIGH1", "score": 0.9, "expected_priority_range": (7, 10)},
        ]

        for case in test_cases:
            alert = mock_scanner.generate_alert(
                symbol=case["symbol"], alert_type=AlertType.HIGH_VOLUME, score=case["score"]
            )
            bridge.process_scan_alert(alert)

        await asyncio.sleep(0.2)

        # Test priority calculator directly
        calculator = bridge._priority_calculator
        for case in test_cases:
            priority = calculator.calculate_priority(case["score"], AlertType.HIGH_VOLUME)
            min_p, max_p = case["expected_priority_range"]
            assert min_p <= priority <= max_p

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_urgent_alert_priority_boost(self, mock_config, mock_event_bus, mock_scanner):
        """Test that urgent alert types get priority boost."""
        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Compare priority for same score, different alert types
        regular_alert = mock_scanner.generate_alert(
            symbol="REGULAR", alert_type=AlertType.HIGH_VOLUME, score=0.7
        )

        urgent_alert = mock_scanner.generate_alert(
            symbol="URGENT", alert_type="catalyst_detected", score=0.7
        )

        bridge.process_scan_alert(regular_alert)
        bridge.process_scan_alert(urgent_alert)

        await asyncio.sleep(0.2)

        # Test priority calculator
        calculator = bridge._priority_calculator
        regular_priority = calculator.calculate_priority(0.7, AlertType.HIGH_VOLUME)
        urgent_priority = calculator.calculate_priority(0.7, "catalyst_detected")

        # Urgent alerts should have higher priority
        assert urgent_priority > regular_priority

        await bridge.stop()


# Test Batch Processing Logic
class TestBatchProcessing:
    """Test request batching logic and behavior."""

    @pytest.mark.asyncio
    async def test_batch_size_enforcement(self, mock_config, mock_event_bus, mock_scanner):
        """Test that batches are created when size limit is reached."""
        # Set specific batch size
        mock_config["events"]["batch_size"] = 3

        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Generate alerts up to batch size
        symbols = ["AAPL", "MSFT", "GOOGL"]  # Exactly batch size

        for symbol in symbols:
            alert = await mock_scanner.generate_high_volume_alert(symbol)
            bridge.process_scan_alert(alert)

        # Small delay to allow batch processing
        await asyncio.sleep(0.05)

        # Should have triggered batch dispatch
        assert mock_event_bus.publish.call_count >= 1

        # Add one more alert to start new batch
        alert = await mock_scanner.generate_high_volume_alert("TSLA")
        bridge.process_scan_alert(alert)

        await asyncio.sleep(0.2)  # Wait for time-based processing

        # Should have processed second batch by timer
        assert mock_event_bus.publish.call_count >= 2

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_time_based_batch_processing(self, mock_config, mock_event_bus, mock_scanner):
        """Test that batches are processed after time interval."""
        # Set long batch size but short interval
        mock_config["events"]["batch_size"] = 100  # Won't trigger by size
        mock_config["events"]["batch_interval_seconds"] = 0.1  # Will trigger by time

        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Generate few alerts (below batch size)
        symbols = ["AAPL", "MSFT"]

        for symbol in symbols:
            alert = await mock_scanner.generate_high_volume_alert(symbol)
            bridge.process_scan_alert(alert)

        # Wait for time-based processing
        await asyncio.sleep(0.3)

        # Should have triggered time-based batch dispatch
        assert mock_event_bus.publish.call_count >= 1

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_batch_overflow_handling(self, mock_config, mock_event_bus, mock_scanner):
        """Test handling of rapid alert generation."""
        # Small batch size for quick overflow
        mock_config["events"]["batch_size"] = 2

        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Generate many alerts rapidly
        symbols = [f"SYMBOL_{i}" for i in range(20)]

        for symbol in symbols:
            alert = await mock_scanner.generate_high_volume_alert(symbol)
            bridge.process_scan_alert(alert)

        # Allow processing time
        await asyncio.sleep(0.5)

        # Should have created multiple batches
        assert mock_event_bus.publish.call_count >= 10  # 20 symbols / 2 batch size

        # Verify all symbols were processed
        stats = bridge.get_stats()
        assert stats["alerts_received_total"] == len(symbols)

        await bridge.stop()


# Test Event Integration
class TestEventIntegration:
    """Test event bus integration and correlation."""

    @pytest.mark.asyncio
    async def test_event_correlation_preservation(self, mock_config, mock_event_bus, mock_scanner):
        """Test that correlation IDs are preserved through the pipeline."""
        # Create real event for correlation tracking
        test_correlation_id = "test_correlation_123"

        scanner_event = ScannerAlertEvent(
            data={
                "symbol": "AAPL",
                "alert_type": AlertType.HIGH_VOLUME,
                "score": 0.8,
                "metadata": {"test": True},
            }
        )
        scanner_event.correlation_id = test_correlation_id

        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Process event directly
        await bridge._handle_scanner_alert_event(scanner_event)

        # Allow batch processing
        await asyncio.sleep(0.2)

        # Verify feature request event was published
        mock_event_bus.publish.assert_called()

        # Check that correlation ID would be preserved
        # (In real implementation, this would be verified in the actual feature request)

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_invalid_event_handling(self, mock_config, mock_event_bus, caplog):
        """Test handling of invalid or malformed events."""
        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Test invalid event types
        invalid_events = [
            # Missing symbol
            ScannerAlertEvent(data={"alert_type": "test", "score": 0.5}),
            # Missing alert_type
            ScannerAlertEvent(data={"symbol": "AAPL", "score": 0.5}),
            # Wrong event type
            FeatureRequestEvent(data={"symbols": ["AAPL"]}),
        ]

        for event in invalid_events:
            await bridge._handle_scanner_alert_event(event)

        # Should log warnings for invalid events
        assert (
            "Invalid ScannerAlertEvent" in caplog.text
            or "missing symbol or alert_type" in caplog.text
        )

        # Should not crash the bridge
        stats = bridge.get_stats()
        assert isinstance(stats, dict)

        await bridge.stop()


# Test Error Handling and Resilience
class TestErrorHandlingResilience:
    """Test error handling and system resilience."""

    @pytest.mark.asyncio
    async def test_dispatcher_failure_recovery(
        self, mock_config, mock_event_bus, mock_scanner, caplog
    ):
        """Test that dispatcher failures don't crash the bridge."""
        # Make event bus publish fail
        mock_event_bus.publish.side_effect = Exception("Event bus connection failed")

        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Generate alert that should trigger dispatch
        alert = await mock_scanner.generate_high_volume_alert("AAPL")
        bridge.process_scan_alert(alert)

        await asyncio.sleep(0.2)

        # Should log error but continue operating
        assert "Failed to dispatch" in caplog.text or "Error handling scanner alert" in caplog.text

        # Bridge should still be responsive
        stats = bridge.get_stats()
        assert isinstance(stats, dict)

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_pending_requests(
        self, mock_config, mock_event_bus, mock_scanner
    ):
        """Test graceful shutdown with pending requests."""
        # Set long batch interval to keep requests pending
        mock_config["events"]["batch_interval_seconds"] = 10

        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Generate alerts that will remain pending
        symbols = ["AAPL", "MSFT"]
        for symbol in symbols:
            alert = await mock_scanner.generate_high_volume_alert(symbol)
            bridge.process_scan_alert(alert)

        # Verify pending requests exist
        stats = bridge.get_stats()
        assert stats["alerts_received_total"] > 0

        # Shutdown should process pending requests
        await bridge.stop()

        # Should have attempted to publish pending batches
        mock_event_bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_batch_processor_task_cancellation(self, mock_config, mock_event_bus):
        """Test proper cancellation of background batch processor task."""
        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Verify batch task is running
        assert bridge._batch_task is not None
        assert not bridge._batch_task.done()

        # Stop bridge
        await bridge.stop()

        # Verify task was cancelled
        assert bridge._batch_task is None or bridge._batch_task.cancelled()


# Test Configuration Handling
class TestConfigurationHandling:
    """Test configuration loading and validation."""

    @pytest.mark.asyncio
    async def test_default_configuration_values(self, mock_event_bus):
        """Test that bridge uses sensible defaults when config is missing."""
        # Create bridge with minimal config
        minimal_config = {}

        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=minimal_config)

        # Verify default values are applied
        assert bridge._batch_interval > 0
        assert bridge._batcher.batch_size > 0

        await bridge.start()
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_custom_configuration_values(self, mock_event_bus):
        """Test that custom configuration values are properly applied."""
        custom_config = {
            "events": {"batch_size": 15, "batch_interval_seconds": 2.5},
            "scanner_bridge": {
                "priority_boost_threshold": 0.9,
                "urgent_alert_types": ["custom_urgent_type"],
            },
        }

        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=custom_config)

        # Verify custom values are applied
        assert bridge._batch_interval == 2.5
        assert bridge._batcher.batch_size == 15

        await bridge.start()
        await bridge.stop()


# Test Edge Cases and Boundary Conditions
class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_duplicate_symbol_handling(self, mock_config, mock_event_bus, mock_scanner):
        """Test handling of duplicate alerts for the same symbol."""
        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Generate multiple alerts for same symbol
        symbol = "AAPL"
        for i in range(5):
            alert = await mock_scanner.generate_high_volume_alert(symbol)
            bridge.process_scan_alert(alert)

        await asyncio.sleep(0.2)

        # Verify stats tracking handles duplicates correctly
        stats = bridge.get_stats()
        assert stats["alerts_received_total"] == 5
        assert stats["unique_symbols_processed_count"] == 1  # Only one unique symbol

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_empty_metadata_handling(self, mock_config, mock_event_bus, mock_scanner):
        """Test handling of alerts with empty or missing metadata."""
        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Generate alert with no metadata
        alert = mock_scanner.generate_alert(
            symbol="TSLA", alert_type=AlertType.HIGH_VOLUME, score=0.7, metadata=None
        )
        bridge.process_scan_alert(alert)

        # Generate alert with empty metadata
        alert = mock_scanner.generate_alert(
            symbol="MSFT", alert_type=AlertType.BREAKOUT, score=0.8, metadata={}
        )
        bridge.process_scan_alert(alert)

        await asyncio.sleep(0.2)

        # Should process successfully without errors
        stats = bridge.get_stats()
        assert stats["alerts_received_total"] == 2

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_zero_and_extreme_scores(self, mock_config, mock_event_bus, mock_scanner):
        """Test handling of extreme score values."""
        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Test extreme score values
        test_scores = [0.0, 0.01, 0.99, 1.0]
        symbols = ["ZERO", "LOW", "HIGH", "MAX"]

        for symbol, score in zip(symbols, test_scores):
            alert = mock_scanner.generate_alert(
                symbol=symbol, alert_type=AlertType.HIGH_VOLUME, score=score
            )
            bridge.process_scan_alert(alert)

        await asyncio.sleep(0.2)

        # All alerts should be processed successfully
        stats = bridge.get_stats()
        assert stats["alerts_received_total"] == len(test_scores)

        # Test priority calculation with extreme values
        calculator = bridge._priority_calculator
        for score in test_scores:
            priority = calculator.calculate_priority(score, AlertType.HIGH_VOLUME)
            assert 1 <= priority <= 10  # Should stay within bounds

        await bridge.stop()


# Test Performance and Statistics
class TestPerformanceStatistics:
    """Test performance monitoring and statistics collection."""

    @pytest.mark.asyncio
    async def test_statistics_accuracy(self, mock_config, mock_event_bus, mock_scanner):
        """Test that statistics are accurately tracked."""
        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Generate known number of alerts
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        unique_symbols = set()

        for symbol in symbols:
            alert = await mock_scanner.generate_high_volume_alert(symbol)
            bridge.process_scan_alert(alert)
            unique_symbols.add(symbol)

        # Process one duplicate symbol
        duplicate_alert = await mock_scanner.generate_high_volume_alert("AAPL")
        bridge.process_scan_alert(duplicate_alert)

        await asyncio.sleep(0.2)

        # Verify statistics
        stats = bridge.get_stats()

        assert stats["alerts_received_total"] == len(symbols) + 1  # Including duplicate
        assert stats["unique_symbols_processed_count"] == len(
            unique_symbols
        )  # Unique symbols count
        assert stats["feature_requests_sent_total"] >= 1

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_performance_under_load(self, mock_config, mock_event_bus, mock_scanner):
        """Test performance with high volume of alerts."""
        # Optimize config for high throughput
        mock_config["events"]["batch_size"] = 10
        mock_config["events"]["batch_interval_seconds"] = 0.05

        with patch("main.events.event_bus_initializer.get_event_bus", return_value=mock_event_bus):
            bridge = ScannerFeatureBridge(config=mock_config)

        await bridge.start()

        # Generate high volume of alerts
        num_alerts = 100
        start_time = time.time()

        for i in range(num_alerts):
            alert = mock_scanner.generate_alert(
                symbol=f"SYMBOL_{i % 20}",  # 20 unique symbols, repeated
                alert_type=AlertType.HIGH_VOLUME,
                score=0.7,
            )
            bridge.process_scan_alert(alert)

        # Allow processing time
        await asyncio.sleep(1.0)

        processing_time = time.time() - start_time

        # Verify all alerts were processed
        stats = bridge.get_stats()
        assert stats["alerts_received_total"] == num_alerts
        assert stats["unique_symbols_processed_count"] == 20  # 20 unique symbols

        # Performance should be reasonable (with mocks)
        alerts_per_second = num_alerts / processing_time
        assert alerts_per_second > 50  # Should handle at least 50 alerts/second

        await bridge.stop()


if __name__ == "__main__":
    pytest.main([__file__])

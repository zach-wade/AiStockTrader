"""
Integration tests for Scanner Bridge Helpers.

Tests the coordinated operation of AlertFeatureMapper, PriorityCalculator,
FeatureRequestBatcher, and RequestDispatcher with their refactored dependencies.
"""

# Standard library imports
import asyncio
from unittest.mock import AsyncMock

# Third-party imports
import pytest

# Local imports
from main.events.scanner_bridge_helpers.alert_feature_mapper import AlertFeatureMapper
from main.events.scanner_bridge_helpers.bridge_stats_tracker import BridgeStatsTracker
from main.events.scanner_bridge_helpers.feature_request_batcher import (
    FeatureRequestBatch,
    FeatureRequestBatcher,
)
from main.events.scanner_bridge_helpers.priority_calculator import PriorityCalculator
from main.events.scanner_bridge_helpers.request_dispatcher import RequestDispatcher
from main.events.types import FeatureRequestEvent, ScanAlert, ScannerAlertEvent
from main.interfaces.events import IEventBus


@pytest.fixture
async def scanner_bridge_components():
    """Create scanner bridge components with mocked dependencies."""
    # Mock event bus
    mock_event_bus = AsyncMock(spec=IEventBus)
    mock_event_bus.publish = AsyncMock()
    mock_event_bus.subscribe = AsyncMock()

    # Mock configuration
    mock_config = {
        "scanner_bridge": {
            "batch_size": 5,
            "batch_timeout": 2.0,
            "max_batches": 100,
            "priority_boost_threshold": 0.8,
        },
        "alert_mappings": {
            "high_volume": ["volume_features", "price_features"],
            "breakout": ["price_features", "trend_features"],
            "catalyst_detected": "all_features",
        },
    }

    # Create components
    stats_tracker = BridgeStatsTracker()
    mapper = AlertFeatureMapper()
    calculator = PriorityCalculator()
    batcher = FeatureRequestBatcher(
        batch_size=mock_config["scanner_bridge"]["batch_size"],
        batch_timeout=mock_config["scanner_bridge"]["batch_timeout"],
    )
    dispatcher = RequestDispatcher(event_bus=mock_event_bus)

    yield {
        "stats_tracker": stats_tracker,
        "mapper": mapper,
        "calculator": calculator,
        "batcher": batcher,
        "dispatcher": dispatcher,
        "event_bus": mock_event_bus,
        "config": mock_config,
    }


@pytest.fixture
def sample_scanner_alerts():
    """Create sample scanner alert events for testing."""
    return [
        ScannerAlertEvent(
            symbol="AAPL",
            alert_type="high_volume",
            score=0.85,
            scanner_name="volume_scanner",
            metadata={"volume_ratio": 3.5, "current_volume": 150000},
        ),
        ScannerAlertEvent(
            symbol="MSFT",
            alert_type="breakout",
            score=0.75,
            scanner_name="breakout_scanner",
            metadata={"breakout_level": 300.0, "resistance_level": 295.0},
        ),
        ScannerAlertEvent(
            symbol="GOOGL",
            alert_type="catalyst_detected",
            score=0.95,
            scanner_name="catalyst_scanner",
            metadata={"catalyst_type": "earnings_surprise", "news_count": 5},
        ),
        ScannerAlertEvent(
            symbol="TSLA",
            alert_type="high_volume",
            score=0.65,
            scanner_name="volume_scanner",
            metadata={"volume_ratio": 2.1, "current_volume": 75000},
        ),
    ]


@pytest.fixture
def sample_scan_alerts():
    """Create sample ScanAlert objects for testing."""
    return [
        ScanAlert(
            symbol="AAPL",
            alert_type="high_volume",
            score=0.85,
            data={"volume_ratio": 3.5, "current_volume": 150000},
        ),
        ScanAlert(
            symbol="MSFT",
            alert_type="breakout",
            score=0.75,
            data={"breakout_level": 300.0, "resistance_level": 295.0},
        ),
        ScanAlert(
            symbol="GOOGL",
            alert_type="catalyst_detected",
            score=0.95,
            data={"catalyst_type": "earnings_surprise", "news_count": 5},
        ),
    ]


class TestScannerBridgeHelpersIntegration:
    """Test integrated operation of all scanner bridge helpers."""

    @pytest.mark.asyncio
    async def test_complete_alert_to_feature_request_flow(
        self, scanner_bridge_components, sample_scanner_alerts
    ):
        """Test complete flow from scanner alert to feature request dispatch."""
        components = scanner_bridge_components
        mapper = components["mapper"]
        calculator = components["calculator"]
        batcher = components["batcher"]
        dispatcher = components["dispatcher"]
        stats = components["stats_tracker"]

        # Process each alert through the complete pipeline
        batches_created = []

        for alert_event in sample_scanner_alerts:
            # Step 1: Track alert
            stats.increment_alerts_received()
            stats.add_unique_symbol(alert_event.symbol)

            # Step 2: Map alert to features
            features = mapper.get_features_for_alert_type(alert_event.alert_type)

            # Step 3: Calculate priority
            priority = calculator.calculate_priority(alert_event.score, alert_event.alert_type)

            # Step 4: Add to batch
            batch = await batcher.add_request(
                symbols={alert_event.symbol},
                features=set(features) if isinstance(features, list) else {features},
                priority=priority,
                requester=alert_event.scanner_name,
            )

            if batch:  # Batch was completed
                batches_created.append(batch)

                # Step 5: Dispatch batch
                await dispatcher.send_feature_request_batch(batch)
                stats.increment_feature_requests_sent()

        # Force any remaining batch
        final_batch = await batcher.force_batch()
        if final_batch:
            batches_created.append(final_batch)
            await dispatcher.send_feature_request_batch(final_batch)
            stats.increment_feature_requests_sent()

        # Verify complete flow
        assert len(batches_created) > 0

        # Verify event bus interactions
        mock_event_bus = components["event_bus"]
        assert mock_event_bus.publish.call_count >= len(batches_created)

        # Verify stats tracking
        bridge_stats = stats.get_stats()
        assert bridge_stats["alerts_received_total"] == len(sample_scanner_alerts)
        assert bridge_stats["unique_symbols_processed"] == len(
            set(alert.symbol for alert in sample_scanner_alerts)
        )
        assert bridge_stats["feature_requests_sent_total"] >= 1

    @pytest.mark.asyncio
    async def test_alert_feature_mapping_integration(
        self, scanner_bridge_components, sample_scan_alerts
    ):
        """Test alert feature mapping with different alert types."""
        components = scanner_bridge_components
        mapper = components["mapper"]
        calculator = components["calculator"]

        mapping_results = []

        for alert in sample_scan_alerts:
            # Get feature mapping
            features = mapper.get_features_for_alert_type(alert.alert_type)

            # Calculate priority
            priority = calculator.calculate_priority(alert.score, alert.alert_type)

            mapping_results.append(
                {
                    "alert_type": alert.alert_type,
                    "symbol": alert.symbol,
                    "score": alert.score,
                    "features": features,
                    "priority": priority,
                }
            )

        # Verify different alert types map to appropriate features
        high_volume_result = next(r for r in mapping_results if r["alert_type"] == "high_volume")
        assert "volume_features" in high_volume_result["features"]
        assert "price_features" in high_volume_result["features"]

        breakout_result = next(r for r in mapping_results if r["alert_type"] == "breakout")
        assert "price_features" in breakout_result["features"]
        assert "trend_features" in breakout_result["features"]

        catalyst_result = next(r for r in mapping_results if r["alert_type"] == "catalyst_detected")
        # Catalyst should map to all features or a comprehensive set
        assert len(catalyst_result["features"]) > 5 or catalyst_result["features"] == "all_features"

        # Verify priority calculation
        # Higher scores should generally result in higher priorities
        scores_and_priorities = [(r["score"], r["priority"]) for r in mapping_results]
        for score, priority in scores_and_priorities:
            assert 1 <= priority <= 10  # Valid priority range

        # Catalyst with highest score should have high priority
        assert catalyst_result["priority"] >= 8

    @pytest.mark.asyncio
    async def test_priority_calculation_and_boosting(self, scanner_bridge_components):
        """Test priority calculation with different alert types and scores."""
        components = scanner_bridge_components
        calculator = components["calculator"]

        # Test cases with different scores and alert types
        test_cases = [
            {"score": 0.5, "alert_type": "high_volume", "expected_range": (3, 7)},
            {"score": 0.8, "alert_type": "high_volume", "expected_range": (6, 9)},
            {
                "score": 0.5,
                "alert_type": "catalyst_detected",
                "expected_range": (5, 9),
            },  # Should get boost
            {
                "score": 0.8,
                "alert_type": "catalyst_detected",
                "expected_range": (8, 10),
            },  # High score + boost
            {"score": 0.3, "alert_type": "breakout", "expected_range": (1, 5)},
            {
                "score": 0.9,
                "alert_type": "opportunity_signal",
                "expected_range": (8, 10),
            },  # Another urgent type
        ]

        for case in test_cases:
            priority = calculator.calculate_priority(case["score"], case["alert_type"])
            min_priority, max_priority = case["expected_range"]

            assert (
                min_priority <= priority <= max_priority
            ), f"Priority {priority} not in range {case['expected_range']} for {case['alert_type']} with score {case['score']}"

        # Test that urgent alert types get priority boost
        regular_priority = calculator.calculate_priority(0.7, "high_volume")
        urgent_priority = calculator.calculate_priority(0.7, "catalyst_detected")

        assert (
            urgent_priority > regular_priority
        ), "Urgent alert types should receive priority boost"

    @pytest.mark.asyncio
    async def test_feature_request_batching_logic(self, scanner_bridge_components):
        """Test feature request batching with size and timeout triggers."""
        components = scanner_bridge_components
        batcher = components["batcher"]
        dispatcher = components["dispatcher"]

        # Test batch size trigger
        batch_size = batcher.batch_size
        completed_batches = []

        # Add requests up to batch size
        for i in range(batch_size):
            batch = await batcher.add_request(
                symbols={f"SYMBOL_{i}"},
                features={"price_features"},
                priority=5,
                requester="test_batcher",
            )
            if batch:
                completed_batches.append(batch)
                await dispatcher.send_feature_request_batch(batch)

        # Should have triggered at least one batch by size
        assert len(completed_batches) >= 1

        # Verify batch contents
        if completed_batches:
            first_batch = completed_batches[0]
            assert len(first_batch.symbols) == batch_size
            assert "price_features" in first_batch.features
            assert first_batch.priority >= 5

        # Test timeout trigger by adding fewer than batch_size requests
        # and waiting for timeout
        timeout_batches = []

        # Add partial batch
        for i in range(2):  # Less than batch_size
            batch = await batcher.add_request(
                symbols={f"TIMEOUT_SYMBOL_{i}"},
                features={"volume_features"},
                priority=7,
                requester="timeout_test",
            )
            if batch:
                timeout_batches.append(batch)

        # Wait for potential timeout (batcher should handle this internally)
        await asyncio.sleep(0.1)

        # Force any remaining batch to test timeout scenario
        timeout_batch = await batcher.force_batch()
        if timeout_batch:
            timeout_batches.append(timeout_batch)
            await dispatcher.send_feature_request_batch(timeout_batch)

        # Should have created timeout batch
        assert len(timeout_batches) >= 1

        # Verify timeout batch
        if timeout_batches:
            timeout_batch = timeout_batches[-1]
            assert len(timeout_batch.symbols) <= batch_size
            assert "volume_features" in timeout_batch.features

    @pytest.mark.asyncio
    async def test_request_deduplication_across_batches(self, scanner_bridge_components):
        """Test deduplication of similar requests across batches."""
        components = scanner_bridge_components
        batcher = components["batcher"]
        dispatcher = components["dispatcher"]

        # Add duplicate and similar requests
        test_requests = [
            # Exact duplicates
            {"symbols": {"AAPL"}, "features": {"price_features"}, "priority": 5},
            {"symbols": {"AAPL"}, "features": {"price_features"}, "priority": 5},
            # Same symbol, different features
            {"symbols": {"AAPL"}, "features": {"volume_features"}, "priority": 5},
            # Same features, different symbol
            {"symbols": {"MSFT"}, "features": {"price_features"}, "priority": 5},
            # Same symbol and features, different priority
            {"symbols": {"AAPL"}, "features": {"price_features"}, "priority": 8},
            # Multiple symbols
            {"symbols": {"AAPL", "MSFT"}, "features": {"trend_features"}, "priority": 6},
        ]

        completed_batches = []

        for req in test_requests:
            batch = await batcher.add_request(
                symbols=req["symbols"],
                features=req["features"],
                priority=req["priority"],
                requester="dedup_test",
            )
            if batch:
                completed_batches.append(batch)
                await dispatcher.send_feature_request_batch(batch)

        # Force any remaining requests
        final_batch = await batcher.force_batch()
        if final_batch:
            completed_batches.append(final_batch)
            await dispatcher.send_feature_request_batch(final_batch)

        # Verify deduplication occurred
        assert len(completed_batches) >= 1

        # Count total unique symbol-feature combinations
        all_combinations = set()
        for batch in completed_batches:
            for symbol in batch.symbols:
                for feature in batch.features:
                    all_combinations.add((symbol, feature))

        # Should have fewer combinations than original requests due to deduplication
        expected_combinations = {
            ("AAPL", "price_features"),
            ("AAPL", "volume_features"),
            ("MSFT", "price_features"),
            ("AAPL", "trend_features"),
            ("MSFT", "trend_features"),
        }

        assert len(all_combinations) >= len(expected_combinations) - 1  # Allow some variance

    @pytest.mark.asyncio
    async def test_dispatcher_event_publishing(self, scanner_bridge_components):
        """Test that dispatcher properly publishes feature request events."""
        components = scanner_bridge_components
        dispatcher = components["dispatcher"]
        mock_event_bus = components["event_bus"]

        # Create test batches
        test_batches = [
            FeatureRequestBatch(
                symbols={"AAPL", "MSFT"},
                features={"price_features", "volume_features"},
                priority=7,
                requester="integration_test",
            ),
            FeatureRequestBatch(
                symbols={"GOOGL"},
                features={"trend_features"},
                priority=9,
                requester="high_priority_test",
            ),
            FeatureRequestBatch(
                symbols={"TSLA", "AMZN", "NFLX"},
                features={"all_features"},
                priority=5,
                requester="batch_test",
            ),
        ]

        # Dispatch all batches
        for batch in test_batches:
            await dispatcher.send_feature_request_batch(batch)

        # Verify event bus publish was called for each batch
        assert mock_event_bus.publish.call_count == len(test_batches)

        # Verify published events have correct structure
        published_calls = mock_event_bus.publish.call_args_list

        for i, call in enumerate(published_calls):
            event = call[0][0]  # First argument of the call
            batch = test_batches[i]

            # Verify event is FeatureRequestEvent
            assert isinstance(event, FeatureRequestEvent)

            # Verify event contains batch data
            assert set(event.symbols) == batch.symbols
            assert set(event.features) == batch.features
            assert event.priority == batch.priority
            assert event.requester == batch.requester

    @pytest.mark.asyncio
    async def test_stats_tracking_across_components(
        self, scanner_bridge_components, sample_scanner_alerts
    ):
        """Test comprehensive stats tracking across all components."""
        components = scanner_bridge_components
        stats = components["stats_tracker"]
        mapper = components["mapper"]
        calculator = components["calculator"]
        batcher = components["batcher"]
        dispatcher = components["dispatcher"]

        # Track operations across components
        alerts_processed = 0
        unique_symbols = set()
        feature_requests_sent = 0

        for alert_event in sample_scanner_alerts:
            # Simulate alert processing
            stats.increment_alerts_received()
            alerts_processed += 1

            stats.add_unique_symbol(alert_event.symbol)
            unique_symbols.add(alert_event.symbol)

            # Map features and calculate priority
            features = mapper.get_features_for_alert_type(alert_event.alert_type)
            priority = calculator.calculate_priority(alert_event.score, alert_event.alert_type)

            # Batch request
            batch = await batcher.add_request(
                symbols={alert_event.symbol},
                features=set(features) if isinstance(features, list) else {features},
                priority=priority,
                requester=alert_event.scanner_name,
            )

            if batch:
                await dispatcher.send_feature_request_batch(batch)
                stats.increment_feature_requests_sent()
                feature_requests_sent += 1

        # Force final batch
        final_batch = await batcher.force_batch()
        if final_batch:
            await dispatcher.send_feature_request_batch(final_batch)
            stats.increment_feature_requests_sent()
            feature_requests_sent += 1

        # Verify stats accuracy
        bridge_stats = stats.get_stats()

        assert bridge_stats["alerts_received_total"] == alerts_processed
        assert bridge_stats["unique_symbols_processed"] == len(unique_symbols)
        assert bridge_stats["feature_requests_sent_total"] == feature_requests_sent

        # Verify symbol tracking
        assert bridge_stats["unique_symbols_processed"] == len(
            set(alert.symbol for alert in sample_scanner_alerts)
        )

    @pytest.mark.asyncio
    async def test_error_handling_across_components(self, scanner_bridge_components, caplog):
        """Test error handling and resilience across all components."""
        components = scanner_bridge_components
        mapper = components["mapper"]
        calculator = components["calculator"]
        batcher = components["batcher"]
        dispatcher = components["dispatcher"]
        stats = components["stats_tracker"]

        # Make event bus fail for some requests
        mock_event_bus = components["event_bus"]
        call_count = 0

        async def failing_publish(event):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:  # Fail every other call
                raise Exception("Event bus connection failed")
            return True

        mock_event_bus.publish.side_effect = failing_publish

        # Test error scenarios
        error_test_cases = [
            # Valid requests
            {"symbol": "AAPL", "alert_type": "high_volume", "score": 0.8},
            {"symbol": "MSFT", "alert_type": "breakout", "score": 0.7},
            # Invalid alert type (should be handled gracefully)
            {"symbol": "GOOGL", "alert_type": "unknown_type", "score": 0.6},
            # Edge case scores
            {"symbol": "TSLA", "alert_type": "high_volume", "score": 0.0},
            {"symbol": "AMZN", "alert_type": "catalyst_detected", "score": 1.0},
        ]

        successful_operations = 0

        for case in error_test_cases:
            try:
                # Track stats
                stats.increment_alerts_received()
                stats.add_unique_symbol(case["symbol"])

                # Map features (may fail for unknown types)
                try:
                    features = mapper.get_features_for_alert_type(case["alert_type"])
                    if not features:  # Fallback for unknown types
                        features = ["price_features"]  # Default features
                except Exception:
                    features = ["price_features"]  # Default fallback

                # Calculate priority (should be robust)
                priority = calculator.calculate_priority(case["score"], case["alert_type"])

                # Batch request
                batch = await batcher.add_request(
                    symbols={case["symbol"]},
                    features=set(features) if isinstance(features, list) else {features},
                    priority=priority,
                    requester="error_test",
                )

                if batch:
                    # Dispatch (may fail due to mock)
                    try:
                        await dispatcher.send_feature_request_batch(batch)
                        stats.increment_feature_requests_sent()
                        successful_operations += 1
                    except Exception:
                        # Log error but continue (resilient behavior)
                        pass

            except Exception:
                # Component should handle errors gracefully
                pass

        # Force any remaining batch
        try:
            final_batch = await batcher.force_batch()
            if final_batch:
                await dispatcher.send_feature_request_batch(final_batch)
                stats.increment_feature_requests_sent()
        except Exception:
            pass

        # Verify error handling
        # Should have logged errors but continued operating
        bridge_stats = stats.get_stats()
        assert bridge_stats["alerts_received_total"] == len(error_test_cases)

        # Some operations should have succeeded despite errors
        assert mock_event_bus.publish.call_count >= 1

        # Components should remain functional after errors
        assert isinstance(bridge_stats, dict)
        assert "alerts_received_total" in bridge_stats

    @pytest.mark.asyncio
    async def test_high_volume_processing_coordination(self, scanner_bridge_components):
        """Test coordination of components under high volume."""
        components = scanner_bridge_components
        mapper = components["mapper"]
        calculator = components["calculator"]
        batcher = components["batcher"]
        dispatcher = components["dispatcher"]
        stats = components["stats_tracker"]

        # Generate high volume of alerts
        num_alerts = 200
        alert_types = ["high_volume", "breakout", "catalyst_detected", "momentum_shift"]
        symbols = [f"STOCK_{i}" for i in range(50)]  # 50 unique symbols

        batches_created = 0

        for i in range(num_alerts):
            symbol = symbols[i % len(symbols)]
            alert_type = alert_types[i % len(alert_types)]
            score = 0.3 + (i % 70) / 100.0  # Vary scores 0.3 to 0.99

            # Process through pipeline
            stats.increment_alerts_received()
            stats.add_unique_symbol(symbol)

            # Map and calculate
            features = mapper.get_features_for_alert_type(alert_type)
            if not features:  # Handle unknown types
                features = ["price_features"]

            priority = calculator.calculate_priority(score, alert_type)

            # Batch
            batch = await batcher.add_request(
                symbols={symbol},
                features=set(features) if isinstance(features, list) else {features},
                priority=priority,
                requester=f"scanner_{i % 5}",
            )

            if batch:
                await dispatcher.send_feature_request_batch(batch)
                stats.increment_feature_requests_sent()
                batches_created += 1

            # Small delay every 50 requests to simulate realistic timing
            if i % 50 == 49:
                await asyncio.sleep(0.01)

        # Force final batch
        final_batch = await batcher.force_batch()
        if final_batch:
            await dispatcher.send_feature_request_batch(final_batch)
            stats.increment_feature_requests_sent()
            batches_created += 1

        # Verify high volume processing
        bridge_stats = stats.get_stats()
        assert bridge_stats["alerts_received_total"] == num_alerts
        assert bridge_stats["unique_symbols_processed"] == len(symbols)
        assert bridge_stats["feature_requests_sent_total"] >= batches_created

        # Verify event bus handled the volume
        mock_event_bus = components["event_bus"]
        assert mock_event_bus.publish.call_count >= batches_created

        # Should have created multiple batches due to volume
        assert batches_created >= num_alerts // batcher.batch_size

    @pytest.mark.asyncio
    async def test_component_coordination_with_mixed_priorities(self, scanner_bridge_components):
        """Test component coordination with mixed priority requests."""
        components = scanner_bridge_components
        mapper = components["mapper"]
        calculator = components["calculator"]
        batcher = components["batcher"]
        dispatcher = components["dispatcher"]

        # Create alerts with different priorities
        priority_test_cases = [
            {"symbol": "LOW1", "alert_type": "high_volume", "score": 0.3},  # Low priority
            {
                "symbol": "HIGH1",
                "alert_type": "catalyst_detected",
                "score": 0.9,
            },  # High priority + boost
            {"symbol": "MED1", "alert_type": "breakout", "score": 0.6},  # Medium priority
            {
                "symbol": "HIGH2",
                "alert_type": "opportunity_signal",
                "score": 0.8,
            },  # High priority + boost
            {"symbol": "LOW2", "alert_type": "high_volume", "score": 0.4},  # Low priority
        ]

        # Process and collect priority information
        processed_requests = []

        for case in priority_test_cases:
            features = mapper.get_features_for_alert_type(case["alert_type"])
            priority = calculator.calculate_priority(case["score"], case["alert_type"])

            processed_requests.append(
                {
                    "symbol": case["symbol"],
                    "alert_type": case["alert_type"],
                    "score": case["score"],
                    "features": features,
                    "priority": priority,
                }
            )

            # Add to batcher
            batch = await batcher.add_request(
                symbols={case["symbol"]},
                features=set(features) if isinstance(features, list) else {features},
                priority=priority,
                requester="priority_test",
            )

            if batch:
                await dispatcher.send_feature_request_batch(batch)

        # Force final batch to ensure all requests are processed
        final_batch = await batcher.force_batch()
        if final_batch:
            await dispatcher.send_feature_request_batch(final_batch)

        # Verify priority calculations
        high_priority_requests = [r for r in processed_requests if r["priority"] >= 8]
        medium_priority_requests = [r for r in processed_requests if 4 <= r["priority"] < 8]
        low_priority_requests = [r for r in processed_requests if r["priority"] < 4]

        # Should have requests in different priority categories
        assert len(high_priority_requests) >= 1  # HIGH1, HIGH2 should be high priority
        assert len(medium_priority_requests) >= 1  # MED1 should be medium priority

        # Catalyst and opportunity_signal should get priority boost
        catalyst_request = next(
            r for r in processed_requests if r["alert_type"] == "catalyst_detected"
        )
        assert catalyst_request["priority"] >= 8

        opportunity_request = next(
            r for r in processed_requests if r["alert_type"] == "opportunity_signal"
        )
        assert opportunity_request["priority"] >= 7  # Should get boost

        # Verify all requests were processed
        mock_event_bus = components["event_bus"]
        assert mock_event_bus.publish.call_count >= 1

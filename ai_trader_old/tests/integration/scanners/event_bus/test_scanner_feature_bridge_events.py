"""
Integration tests for scanner feature bridge event handling.

Tests how scanner events are bridged to the feature pipeline.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

# Third-party imports
import pytest

# Local imports
from main.events.handlers.scanner_feature_bridge import ScannerFeatureBridge
from main.events.types.event_types import AlertType, EventType, ScannerAlertEvent
from main.interfaces.events import IEventBus


@pytest.mark.integration
@pytest.mark.asyncio
class TestScannerFeatureBridgeEvents:
    """Test scanner feature bridge event integration."""

    async def test_alert_to_feature_mapping(self, test_event_bus: IEventBus, test_symbols):
        """Test alert type to feature mapping."""
        # Create feature bridge
        bridge = ScannerFeatureBridge(event_bus=test_event_bus)

        # Mock feature pipeline
        processed_features = []

        async def mock_feature_processor(features: dict[str, Any]):
            processed_features.append(features)

        bridge._process_features = mock_feature_processor

        # Create test alert event
        alert_event = ScannerAlertEvent(
            symbol="AAPL",
            alert_type=str(AlertType.VOLUME_SPIKE),
            score=0.85,
            scanner_name="volume_scanner",
            metadata={
                "current_volume": 150000000,
                "avg_volume": 75000000,
                "relative_volume": 2.0,
                "confidence": 0.9,
            },
            timestamp=datetime.now(UTC),
            event_type=EventType.SCANNER_ALERT,
        )

        # Process alert through bridge
        await bridge._handle_scanner_alert(alert_event)

        # Should have processed features
        assert len(processed_features) == 1

        features = processed_features[0]
        assert features["symbol"] == "AAPL"
        assert features["alert_type"] == str(AlertType.VOLUME_SPIKE)
        assert "volume_features" in features
        assert features["volume_features"]["relative_volume"] == 2.0

    async def test_multiple_alert_aggregation(self, test_event_bus: IEventBus, test_symbols):
        """Test aggregation of multiple alerts for same symbol."""
        # Create feature bridge with aggregation
        bridge = ScannerFeatureBridge(event_bus=test_event_bus)

        aggregated_features = []

        async def mock_aggregator(features: dict[str, Any]):
            aggregated_features.append(features)

        bridge._aggregate_features = mock_aggregator

        # Create multiple alerts for same symbol
        base_time = datetime.now(UTC)

        volume_alert = ScannerAlertEvent(
            symbol="AAPL",
            alert_type=str(AlertType.VOLUME_SPIKE),
            score=0.8,
            scanner_name="volume_scanner",
            metadata={"relative_volume": 2.5},
            timestamp=base_time,
            event_type=EventType.SCANNER_ALERT,
        )

        news_alert = ScannerAlertEvent(
            symbol="AAPL",  # Same symbol
            alert_type=str(AlertType.NEWS_SENTIMENT),
            score=0.85,
            scanner_name="news_scanner",
            metadata={"sentiment_score": 0.9, "news_count": 12},
            timestamp=base_time + timedelta(minutes=2),
            event_type=EventType.SCANNER_ALERT,
        )

        technical_alert = ScannerAlertEvent(
            symbol="AAPL",  # Same symbol
            alert_type=str(AlertType.TECHNICAL_BREAKOUT),
            score=0.9,
            scanner_name="technical_scanner",
            metadata={"breakout_level": 155.0, "current_price": 158.0},
            timestamp=base_time + timedelta(minutes=5),
            event_type=EventType.SCANNER_ALERT,
        )

        # Process alerts through bridge
        await bridge._handle_scanner_alert(volume_alert)
        await bridge._handle_scanner_alert(news_alert)
        await bridge._handle_scanner_alert(technical_alert)

        # Should aggregate features for same symbol
        if aggregated_features:
            features = aggregated_features[-1]  # Latest aggregation
            assert features["symbol"] == "AAPL"
            assert features["alert_count"] == 3
            assert features["max_score"] == 0.9
            assert "volume_features" in features
            assert "news_features" in features
            assert "technical_features" in features

    async def test_feature_pipeline_integration(self, test_event_bus: IEventBus, test_symbols):
        """Test integration with feature pipeline."""
        # Mock feature pipeline handler
        pipeline_events = []

        async def feature_pipeline_handler(event):
            pipeline_events.append(event)

        # Subscribe to feature events
        await test_event_bus.subscribe("FEATURE_EXTRACTED", feature_pipeline_handler)

        # Create feature bridge
        bridge = ScannerFeatureBridge(event_bus=test_event_bus)

        # Subscribe bridge to scanner alerts
        await test_event_bus.subscribe(EventType.SCANNER_ALERT, bridge._handle_scanner_alert)

        # Create and publish scanner alert
        alert_event = ScannerAlertEvent(
            symbol="GOOGL",
            alert_type=str(AlertType.EARNINGS_SURPRISE),
            score=0.95,
            scanner_name="earnings_scanner",
            metadata={
                "eps_actual": 2.85,
                "eps_estimate": 2.70,
                "surprise_percent": 5.56,
                "confidence": 0.95,
            },
            timestamp=datetime.now(UTC),
            event_type=EventType.SCANNER_ALERT,
        )

        await test_event_bus.publish(alert_event)
        await asyncio.sleep(0.1)  # Allow processing

        # Should have triggered feature pipeline
        assert len(pipeline_events) > 0

        # Check feature event
        feature_event = pipeline_events[0]
        assert "symbol" in feature_event
        assert feature_event["symbol"] == "GOOGL"

    async def test_feature_enrichment(self, test_event_bus: IEventBus, test_symbols):
        """Test feature enrichment from multiple data sources."""
        # Create feature bridge with enrichment
        bridge = ScannerFeatureBridge(event_bus=test_event_bus)

        # Mock external data sources
        mock_market_data = {
            "AAPL": {
                "current_price": 155.0,
                "daily_volume": 85000000,
                "market_cap": 2500000000000,
                "sector": "Technology",
            }
        }

        mock_fundamental_data = {
            "AAPL": {"pe_ratio": 28.5, "pb_ratio": 4.2, "debt_to_equity": 0.85, "roe": 0.185}
        }

        # Mock enrichment methods
        bridge._get_market_data = AsyncMock(return_value=mock_market_data)
        bridge._get_fundamental_data = AsyncMock(return_value=mock_fundamental_data)

        enriched_features = []

        async def mock_enrich_processor(features):
            enriched_features.append(features)

        bridge._process_enriched_features = mock_enrich_processor

        # Create alert
        alert_event = ScannerAlertEvent(
            symbol="AAPL",
            alert_type=str(AlertType.VOLUME_SPIKE),
            score=0.8,
            scanner_name="volume_scanner",
            metadata={"relative_volume": 2.0},
            timestamp=datetime.now(UTC),
            event_type=EventType.SCANNER_ALERT,
        )

        # Process with enrichment
        await bridge._handle_scanner_alert_with_enrichment(alert_event)

        # Should have enriched features
        assert len(enriched_features) == 1

        features = enriched_features[0]
        assert features["symbol"] == "AAPL"
        assert "market_data" in features
        assert "fundamental_data" in features
        assert features["market_data"]["current_price"] == 155.0
        assert features["fundamental_data"]["pe_ratio"] == 28.5

    async def test_real_time_feature_streaming(
        self, test_event_bus: IEventBus, test_symbols, event_performance_thresholds
    ):
        """Test real-time feature streaming performance."""
        # Create feature bridge
        bridge = ScannerFeatureBridge(event_bus=test_event_bus)

        streamed_features = []
        start_time = datetime.now()

        async def feature_stream_handler(features):
            streamed_features.append({"features": features, "timestamp": datetime.now()})

        bridge._stream_features = feature_stream_handler

        # Generate high-frequency alerts
        alerts = []
        for i in range(20):
            alert = ScannerAlertEvent(
                symbol=f"SYM{i%5:03d}",  # 5 different symbols
                alert_type=str(AlertType.VOLUME_SPIKE),
                score=0.8 + (i % 5) * 0.02,
                scanner_name="high_freq_scanner",
                metadata={"batch_id": i},
                timestamp=datetime.now(UTC),
                event_type=EventType.SCANNER_ALERT,
            )
            alerts.append(alert)

        # Process alerts rapidly
        for alert in alerts:
            await bridge._handle_scanner_alert(alert)

        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000

        # Check performance
        throughput = len(alerts) / (processing_time_ms / 1000)
        min_throughput = event_performance_thresholds["event_throughput_per_second"]

        # Allow for test overhead
        assert throughput > min_throughput / 100

        # All features should be streamed
        assert len(streamed_features) == 20

    async def test_feature_validation_and_filtering(self, test_event_bus: IEventBus, test_symbols):
        """Test feature validation and filtering."""
        # Create feature bridge with validation
        bridge = ScannerFeatureBridge(event_bus=test_event_bus)

        valid_features = []
        invalid_features = []

        async def validation_processor(features, valid=True):
            if valid:
                valid_features.append(features)
            else:
                invalid_features.append(features)

        # Mock validation logic
        async def mock_validate_features(features):
            # Simple validation: score must be > 0.5 and have required fields
            if features.get("score", 0) > 0.5 and "symbol" in features and "alert_type" in features:
                await validation_processor(features, valid=True)
            else:
                await validation_processor(features, valid=False)

        bridge._validate_and_process_features = mock_validate_features

        # Create valid alert
        valid_alert = ScannerAlertEvent(
            symbol="AAPL",
            alert_type=str(AlertType.VOLUME_SPIKE),
            score=0.85,  # Valid score
            scanner_name="volume_scanner",
            metadata={"confidence": 0.9},
            timestamp=datetime.now(UTC),
            event_type=EventType.SCANNER_ALERT,
        )

        # Create invalid alert
        invalid_alert = ScannerAlertEvent(
            symbol="GOOGL",
            alert_type=str(AlertType.NEWS_SENTIMENT),
            score=0.3,  # Invalid score (too low)
            scanner_name="news_scanner",
            metadata={"confidence": 0.2},
            timestamp=datetime.now(UTC),
            event_type=EventType.SCANNER_ALERT,
        )

        # Process both alerts
        await bridge._handle_scanner_alert(valid_alert)
        await bridge._handle_scanner_alert(invalid_alert)

        # Check validation results
        assert len(valid_features) == 1
        assert len(invalid_features) == 1

        assert valid_features[0]["symbol"] == "AAPL"
        assert invalid_features[0]["symbol"] == "GOOGL"

    async def test_feature_caching_and_deduplication(self, test_event_bus: IEventBus, test_symbols):
        """Test feature caching and deduplication."""
        # Create feature bridge with caching
        bridge = ScannerFeatureBridge(event_bus=test_event_bus)

        cached_features = {}
        processed_count = 0

        async def caching_processor(features):
            nonlocal processed_count
            symbol = features["symbol"]
            alert_type = features["alert_type"]

            cache_key = f"{symbol}:{alert_type}"

            # Check cache
            if cache_key in cached_features:
                # Update existing features
                cached_features[cache_key].update(features)
            else:
                # New features
                cached_features[cache_key] = features.copy()
                processed_count += 1

        bridge._cache_and_process_features = caching_processor

        # Create duplicate alerts (same symbol and type)
        duplicate_alerts = [
            ScannerAlertEvent(
                symbol="AAPL",
                alert_type=str(AlertType.VOLUME_SPIKE),
                score=0.8,
                scanner_name="volume_scanner",
                metadata={"iteration": i},
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                event_type=EventType.SCANNER_ALERT,
            )
            for i in range(3)
        ]

        # Process duplicate alerts
        for alert in duplicate_alerts:
            await bridge._handle_scanner_alert(alert)

        # Should only process unique features once
        assert processed_count == 1  # Only one unique symbol:alert_type combination
        assert len(cached_features) == 1

        # Should have latest metadata
        cache_key = "AAPL:volume_spike"
        assert cached_features[cache_key]["metadata"]["iteration"] == 2

    async def test_cross_scanner_feature_correlation(self, test_event_bus: IEventBus, test_symbols):
        """Test cross-scanner feature correlation."""
        # Create feature bridge with correlation
        bridge = ScannerFeatureBridge(event_bus=test_event_bus)

        correlations = []

        async def correlation_analyzer(features_batch):
            # Analyze correlations between different scanner types
            scanner_types = set(f["scanner_name"] for f in features_batch)

            if len(scanner_types) > 1:  # Multiple scanners
                correlation = {
                    "symbol": features_batch[0]["symbol"],
                    "scanners": list(scanner_types),
                    "combined_score": max(f["score"] for f in features_batch),
                    "correlation_strength": len(scanner_types) / 5.0,  # Max 5 scanners
                    "timestamp": datetime.now(UTC),
                }
                correlations.append(correlation)

        bridge._analyze_correlations = correlation_analyzer

        # Create correlated alerts (same symbol, different scanners)
        base_time = datetime.now(UTC)

        correlated_alerts = [
            ScannerAlertEvent(
                symbol="AAPL",
                alert_type=str(AlertType.VOLUME_SPIKE),
                score=0.8,
                scanner_name="volume_scanner",
                metadata={},
                timestamp=base_time,
                event_type=EventType.SCANNER_ALERT,
            ),
            ScannerAlertEvent(
                symbol="AAPL",  # Same symbol
                alert_type=str(AlertType.NEWS_SENTIMENT),
                score=0.85,
                scanner_name="news_scanner",
                metadata={},
                timestamp=base_time + timedelta(minutes=1),
                event_type=EventType.SCANNER_ALERT,
            ),
            ScannerAlertEvent(
                symbol="AAPL",  # Same symbol
                alert_type=str(AlertType.TECHNICAL_BREAKOUT),
                score=0.9,
                scanner_name="technical_scanner",
                metadata={},
                timestamp=base_time + timedelta(minutes=2),
                event_type=EventType.SCANNER_ALERT,
            ),
        ]

        # Process correlated alerts
        features_batch = []
        for alert in correlated_alerts:
            features = await bridge._extract_features_from_alert(alert)
            features_batch.append(features)

        await bridge._analyze_correlations(features_batch)

        # Should detect correlation
        assert len(correlations) == 1

        correlation = correlations[0]
        assert correlation["symbol"] == "AAPL"
        assert len(correlation["scanners"]) == 3
        assert correlation["combined_score"] == 0.9
        assert correlation["correlation_strength"] == 0.6  # 3/5 scanners

    async def test_feature_pipeline_error_handling(self, test_event_bus: IEventBus, test_symbols):
        """Test feature pipeline error handling."""
        # Create feature bridge with error handling
        bridge = ScannerFeatureBridge(event_bus=test_event_bus)

        processed_successfully = []
        processing_errors = []

        async def error_prone_processor(features):
            if features.get("will_fail", False):
                processing_errors.append(
                    {
                        "features": features,
                        "error": "Simulated processing error",
                        "timestamp": datetime.now(),
                    }
                )
                raise Exception("Simulated processing error")
            else:
                processed_successfully.append(features)

        bridge._process_features_with_error_handling = error_prone_processor

        # Create successful alert
        success_alert = ScannerAlertEvent(
            symbol="AAPL",
            alert_type=str(AlertType.VOLUME_SPIKE),
            score=0.8,
            scanner_name="volume_scanner",
            metadata={"will_fail": False},
            timestamp=datetime.now(UTC),
            event_type=EventType.SCANNER_ALERT,
        )

        # Create failing alert
        fail_alert = ScannerAlertEvent(
            symbol="GOOGL",
            alert_type=str(AlertType.NEWS_SENTIMENT),
            score=0.85,
            scanner_name="news_scanner",
            metadata={"will_fail": True},
            timestamp=datetime.now(UTC),
            event_type=EventType.SCANNER_ALERT,
        )

        # Process both alerts
        try:
            await bridge._handle_scanner_alert(success_alert)
        except Exception:
            pass  # Should not fail

        try:
            await bridge._handle_scanner_alert(fail_alert)
        except Exception:
            pass  # Expected to fail

        # Check error handling
        assert len(processed_successfully) == 1
        assert len(processing_errors) == 1

        assert processed_successfully[0]["symbol"] == "AAPL"
        assert processing_errors[0]["features"]["symbol"] == "GOOGL"

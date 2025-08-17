"""
End-to-End Integration Test
Tests the complete flow from data ingestion through feature generation.
"""

# Standard library imports
import asyncio
from datetime import datetime
import json
from pathlib import Path

# Third-party imports
import pandas as pd
import pytest

# Local imports
from main.data_pipeline.validation import ValidationPipeline
from main.events import EventBusFactory
from main.interfaces.events import Event, EventType
from main.scanners import ScanAlert


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture
    def sample_market_data(self):
        """Load sample market data from fixtures."""
        fixture_path = Path(__file__).parent.parent / "fixtures/market_data/alpaca_bars_sample.json"
        with open(fixture_path) as f:
            data = json.load(f)
        return data

    @pytest.fixture
    def sample_alerts(self):
        """Load sample scanner alerts from fixtures."""
        fixture_path = Path(__file__).parent.parent / "fixtures/scanner_alerts/sample_alerts.json"
        with open(fixture_path) as f:
            data = json.load(f)
        return [ScanAlert(**alert) for alert in data]

    @pytest.mark.asyncio
    async def test_data_validation_flow(self, sample_market_data):
        """Test data flows through validation pipeline correctly."""
        # Initialize validation pipeline
        pipeline = ValidationPipeline()

        # Convert sample data to DataFrame
        aapl_bars = sample_market_data["bars"]["AAPL"]
        df = pd.DataFrame(aapl_bars)

        # Stage 1: Ingest validation
        ingest_result = await pipeline.validate_ingest(
            df, source_type="alpaca", data_type="market_data"
        )
        assert ingest_result.passed, f"Ingest validation failed: {ingest_result.errors}"

        # Transform data (simulate ETL)
        df["timestamp"] = pd.to_datetime(df["t"])
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df.set_index("timestamp")

        # Stage 2: Post-ETL validation
        post_etl_result = await pipeline.validate_post_etl(
            df, aggregation_level="1min", expected_freq="T"
        )
        assert post_etl_result.passed, f"Post-ETL validation failed: {post_etl_result.errors}"

        # Stage 3: Feature-ready validation
        feature_ready_result = await pipeline.validate_feature_ready(
            df, required_columns=["open", "high", "low", "close", "volume"], min_rows=2
        )
        assert (
            feature_ready_result.passed
        ), f"Feature-ready validation failed: {feature_ready_result.errors}"

        # Check validation summary
        summary = pipeline.get_validation_summary()
        assert summary["total_validations"] == 3
        assert summary["passed"] == 3
        assert summary["failed"] == 0

    @pytest.mark.asyncio
    async def test_scanner_to_feature_flow(self, sample_alerts):
        """Test scanner alerts trigger feature computation."""
        # Initialize event bus
        event_bus = EventBusFactory.create_test_instance()

        # Track feature requests
        feature_requests = []

        async def feature_handler(event: Event):
            """Handler to capture feature requests."""
            if event.event_type == EventType.FEATURE_REQUEST:
                feature_requests.append(event.data)

        # Subscribe to feature requests
        await event_bus.subscribe(EventType.FEATURE_REQUEST, feature_handler)

        # Simulate scanner publishing alerts
        for alert in sample_alerts[:2]:  # Test with first 2 alerts
            event = Event(
                event_type=EventType.SCANNER_ALERT, source="test_scanner", data=alert.dict()
            )
            await event_bus.publish(event)

        # Give time for async processing
        await asyncio.sleep(0.5)

        # Verify feature requests were generated
        assert len(feature_requests) > 0, "No feature requests generated from scanner alerts"

        # Check feature request content
        for request in feature_requests:
            assert "symbol" in request
            assert "features" in request
            assert "priority" in request

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, sample_market_data):
        """Test data quality score calculation."""
        pipeline = ValidationPipeline()

        # Create data with known issues
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, None, 103.0],  # Missing value
                "high": [102.0, 103.0, 104.0, 99.0],  # Invalid: high < low
                "low": [99.0, 100.0, 101.0, 100.0],
                "close": [101.0, 102.0, 103.0, 101.0],
                "volume": [1000000, 1100000, 0, 1200000],  # Zero volume
            }
        )

        # Validate data
        result = await pipeline.validate_ingest(df, source_type="test", data_type="market_data")

        # Calculate quality score
        quality_score = pipeline._calculate_quality_score(result)

        # Score should be reduced due to issues
        assert (
            0 < quality_score < 1.0
        ), f"Quality score should be between 0 and 1, got {quality_score}"
        assert (
            quality_score < 0.8
        ), f"Quality score should be low due to data issues, got {quality_score}"

    @pytest.mark.asyncio
    async def test_validation_metrics_tracking(self, sample_market_data):
        """Test that validation metrics are properly tracked."""
        pipeline = ValidationPipeline()

        # Process some data
        df = pd.DataFrame(sample_market_data["bars"]["AAPL"])

        # Run validation
        result = await pipeline.validate_ingest(df, source_type="alpaca", data_type="market_data")

        # Check that metrics were recorded
        assert result.duration_ms > 0
        assert "row_count" in result.metrics
        assert result.metrics["row_count"] == len(df)

        # Check metrics collector was updated
        if pipeline.metrics_collector.enabled:
            # Metrics should be recorded but may not be pushed yet
            assert hasattr(pipeline.metrics_collector, "validation_success_rate")

    def test_error_handling_decision_tree(self):
        """Test error handling decision tree logic."""
        pipeline = ValidationPipeline()

        # Create mock validation results
        # Local imports
        from main.data_pipeline.validation import ValidationResult, ValidationStage

        # Test INGEST stage error handling
        ingest_error_result = ValidationResult(
            stage=ValidationStage.INGEST,
            passed=False,
            errors=["Missing required fields"],
            warnings=[],
            metrics={},
            timestamp=datetime.now(),
            duration_ms=10.0,
        )

        response = pipeline.handle_validation_failure(ingest_error_result)
        assert response["action"] == "DROP_ROW"
        assert response["severity"] == "ERROR"

        # Test POST_ETL stage error handling
        post_etl_error_result = ValidationResult(
            stage=ValidationStage.POST_ETL,
            passed=False,
            errors=["OHLC relationship violation"],
            warnings=[],
            metrics={},
            timestamp=datetime.now(),
            duration_ms=15.0,
        )

        response = pipeline.handle_validation_failure(post_etl_error_result)
        assert response["action"] == "SKIP_SYMBOL"
        assert response["should_alert"] == True

        # Test FEATURE_READY stage error handling
        feature_ready_error_result = ValidationResult(
            stage=ValidationStage.FEATURE_READY,
            passed=False,
            errors=["Insufficient history"],
            warnings=[],
            metrics={},
            timestamp=datetime.now(),
            duration_ms=20.0,
        )

        response = pipeline.handle_validation_failure(feature_ready_error_result)
        assert response["action"] == "USE_LAST_GOOD_SNAPSHOT"
        assert response["should_alert"] == True


@pytest.mark.asyncio
async def test_complete_pipeline_flow():
    """Test complete flow from ingestion to features."""
    # This would be a more comprehensive test with actual components
    # For now, just verify imports work
    # Local imports
    from main.data_pipeline import DataPipelineOrchestrator
    from main.features import FeaturePipelineOrchestrator
    from main.scanners import Layer2CatalystOrchestrator

    assert DataPipelineOrchestrator is not None
    assert Layer2CatalystOrchestrator is not None
    assert FeaturePipelineOrchestrator is not None

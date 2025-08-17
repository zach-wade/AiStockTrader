"""Unit tests for feature_computation_worker module."""

# Standard library imports
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

# Third-party imports
import pandas as pd
import pytest
import yaml

# Local imports
from main.events.core.event_bus import EventBus
from main.events.handlers.feature_pipeline_helpers.feature_computation_worker import (
    FeatureComputationWorker,
)
from main.events.types import EventType
from main.feature_pipeline.orchestrator import FeatureOrchestrator
from tests.fixtures.events.mock_events import create_feature_request_event


class TestFeatureComputationWorker:
    """Test FeatureComputationWorker class."""

    @pytest.fixture
    def mock_feature_orchestrator(self):
        """Create mock feature orchestrator."""
        orchestrator = Mock(spec=FeatureOrchestrator)
        orchestrator.compute_features = AsyncMock()
        return orchestrator

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = Mock(spec=EventBus)
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def mock_stats_tracker(self):
        """Create mock stats tracker."""
        tracker = Mock()
        tracker.increment_processed = Mock()
        tracker.increment_features_computed = Mock()
        tracker.increment_failed = Mock()
        return tracker

    @pytest.fixture
    def mock_feature_group_mapper(self):
        """Create mock feature group mapper."""
        mapper = Mock()
        mapper.group_features_by_type = Mock(
            return_value={
                "price_features": ["price", "returns"],
                "volume_features": ["volume", "vwap"],
            }
        )
        return mapper

    @pytest.fixture
    def mock_yaml_config(self):
        """Create mock YAML configuration."""
        return {
            "feature_group_mappings": {
                "price_features": ["ohlcv", "returns"],
                "volume_features": ["volume_profile", "vwap"],
                "volatility_features": {"use_features_in_group": True},
            }
        }

    @pytest.fixture
    async def worker(
        self,
        mock_feature_orchestrator,
        mock_event_bus,
        mock_stats_tracker,
        mock_feature_group_mapper,
        mock_yaml_config,
    ):
        """Create FeatureComputationWorker instance for testing."""
        # Mock file operations
        yaml_content = yaml.dump(mock_yaml_config)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.dirname") as mock_dirname:
                with patch("os.path.join") as mock_join:
                    mock_dirname.return_value = "/test/dir"
                    mock_join.return_value = "/test/config.yaml"

                    worker = FeatureComputationWorker(
                        worker_id=1,
                        feature_orchestrator=mock_feature_orchestrator,
                        event_bus=mock_event_bus,
                        stats_tracker=mock_stats_tracker,
                        feature_group_mapper=mock_feature_group_mapper,
                    )

                    # Set the loaded config
                    worker.feature_group_config = mock_yaml_config
                    return worker

    def test_initialization(
        self,
        mock_feature_orchestrator,
        mock_event_bus,
        mock_stats_tracker,
        mock_feature_group_mapper,
    ):
        """Test worker initialization."""
        with patch("builtins.open", mock_open(read_data="{}")):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    worker = FeatureComputationWorker(
                        worker_id=42,
                        feature_orchestrator=mock_feature_orchestrator,
                        event_bus=mock_event_bus,
                        stats_tracker=mock_stats_tracker,
                        feature_group_mapper=mock_feature_group_mapper,
                    )

                    assert worker.worker_id == 42
                    assert worker.feature_orchestrator == mock_feature_orchestrator
                    assert worker.event_bus == mock_event_bus
                    assert worker.stats_tracker == mock_stats_tracker
                    assert worker.feature_group_mapper == mock_feature_group_mapper

    @pytest.mark.asyncio
    async def test_process_request_success(
        self, worker, mock_feature_orchestrator, mock_event_bus, mock_stats_tracker
    ):
        """Test successful request processing."""
        # Create request
        request = {
            "symbols": ["AAPL", "GOOGL"],
            "features": ["price", "volume", "returns"],
            "event": create_feature_request_event(),
        }

        # Mock orchestrator response
        mock_df = pd.DataFrame({"price": [150.0, 155.0], "returns": [0.01, 0.02]})
        mock_feature_orchestrator.compute_features.return_value = mock_df

        # Process request
        await worker.process_request(request)

        # Check feature computation called
        assert (
            mock_feature_orchestrator.compute_features.call_count == 4
        )  # 2 symbols * 2 feature groups

        # Check stats updated
        mock_stats_tracker.increment_processed.assert_called_once()
        # Features computed: 2 symbols * 3 features
        mock_stats_tracker.increment_features_computed.assert_called_once_with(6)

        # Check completion event published
        mock_event_bus.publish.assert_called_once()
        published_event = mock_event_bus.publish.call_args[0][0]

        assert published_event.event_type == EventType.FEATURE_COMPUTED
        assert published_event.data["symbols_processed"] == ["AAPL", "GOOGL"]
        assert published_event.data["requested_features"] == ["price", "volume", "returns"]
        assert published_event.data["results_available"] is True
        assert published_event.data["computation_time_seconds"] > 0

    @pytest.mark.asyncio
    async def test_process_request_with_grouping(self, worker, mock_feature_group_mapper):
        """Test request processing with feature grouping."""
        request = {
            "symbols": ["AAPL"],
            "features": ["price", "volume", "returns", "vwap"],
            "event": create_feature_request_event(),
        }

        # Verify grouping
        await worker.process_request(request)

        mock_feature_group_mapper.group_features_by_type.assert_called_once_with(
            ["price", "volume", "returns", "vwap"]
        )

    @pytest.mark.asyncio
    async def test_compute_features_for_group(self, worker, mock_feature_orchestrator):
        """Test computing features for a specific group."""
        # Test with mapped feature sets
        mock_df = pd.DataFrame({"col1": [1, 2]})
        mock_feature_orchestrator.compute_features.return_value = mock_df

        result = await worker._compute_features_for_group(
            "AAPL", "price_features", ["price", "returns"]
        )

        assert result is not None
        mock_feature_orchestrator.compute_features.assert_called_with(
            symbols=["AAPL"], feature_sets=["ohlcv", "returns"]
        )

    @pytest.mark.asyncio
    async def test_compute_features_use_features_in_group(self, worker, mock_feature_orchestrator):
        """Test computing features when use_features_in_group is True."""
        mock_df = pd.DataFrame({"vol": [0.2]})
        mock_feature_orchestrator.compute_features.return_value = mock_df

        result = await worker._compute_features_for_group(
            "TSLA", "volatility_features", ["realized_vol", "implied_vol"]
        )

        # Should use features directly
        mock_feature_orchestrator.compute_features.assert_called_with(
            symbols=["TSLA"], feature_sets=["realized_vol", "implied_vol"]
        )

    @pytest.mark.asyncio
    async def test_compute_features_unknown_group(self, worker):
        """Test computing features for unknown group."""
        result = await worker._compute_features_for_group(
            "AAPL", "unknown_group", ["feature1", "feature2"]
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_process_request_partial_failure(
        self, worker, mock_feature_orchestrator, mock_event_bus
    ):
        """Test processing when some symbols fail."""
        request = {
            "symbols": ["AAPL", "INVALID"],
            "features": ["price"],
            "event": create_feature_request_event(),
        }

        # Mock orchestrator to fail for INVALID
        async def mock_compute(symbols, feature_sets):
            if "INVALID" in symbols:
                return None
            return pd.DataFrame({"price": [150.0]})

        mock_feature_orchestrator.compute_features.side_effect = mock_compute

        await worker.process_request(request)

        # Should still publish completion event
        mock_event_bus.publish.assert_called_once()
        published_event = mock_event_bus.publish.call_args[0][0]

        # Only AAPL processed successfully
        assert published_event.data["symbols_processed"] == ["AAPL"]

    @pytest.mark.asyncio
    async def test_process_request_complete_failure(
        self, worker, mock_event_bus, mock_stats_tracker
    ):
        """Test processing when computation completely fails."""
        request = {
            "symbols": ["AAPL"],
            "features": ["price"],
            "event": create_feature_request_event(),
        }

        # Make worker fail
        worker._compute_features_for_group = AsyncMock(side_effect=Exception("Computation failed"))

        await worker.process_request(request)

        # Should increment failed counter
        mock_stats_tracker.increment_failed.assert_called_once()

        # Should publish error event
        mock_event_bus.publish.assert_called_once()
        error_event = mock_event_bus.publish.call_args[0][0]

        assert error_event.event_type == EventType.ERROR_OCCURRED
        assert error_event.data["error_type"] == "feature_computation_failed"
        assert "Computation failed" in error_event.data["error_message"]

    @pytest.mark.asyncio
    async def test_metrics_recording(self, worker, mock_feature_orchestrator):
        """Test metrics recording during computation."""
        with patch(
            "main.events.handlers.feature_pipeline_helpers.feature_computation_worker.record_metric"
        ) as mock_metric:
            with patch(
                "main.events.handlers.feature_pipeline_helpers.feature_computation_worker.timer"
            ) as mock_timer:
                mock_context = MagicMock()
                mock_context.__enter__ = Mock(return_value=mock_context)
                mock_context.__exit__ = Mock(return_value=None)
                mock_context.elapsed_seconds = 0.5
                mock_timer.return_value = mock_context

                await worker._compute_features_for_group("AAPL", "price_features", ["price"])

                # Should record timing metric
                mock_metric.assert_any_call(
                    "feature_worker.orchestrator_compute_time",
                    0.5,
                    metric_type="histogram",
                    tags={
                        "symbol": "AAPL",
                        "feature_group": "price_features",
                        "feature_count": 2,  # From config
                    },
                )

    @pytest.mark.asyncio
    async def test_error_handling_with_metadata(self, worker, mock_event_bus):
        """Test error event includes proper metadata."""
        original_event = create_feature_request_event()
        original_event.correlation_id = "test_correlation_123"

        request = {"symbols": ["AAPL"], "features": ["bad_feature"], "event": original_event}

        # Force error
        worker.feature_group_mapper.group_features_by_type.side_effect = Exception("Mapping error")

        await worker.process_request(request)

        # Check error event
        error_event = mock_event_bus.publish.call_args[0][0]
        assert error_event.correlation_id == "test_correlation_123"
        assert error_event.data["worker_id"] == 1

    @pytest.mark.asyncio
    async def test_empty_results_handling(self, worker, mock_feature_orchestrator, mock_event_bus):
        """Test handling when orchestrator returns empty results."""
        request = {
            "symbols": ["AAPL"],
            "features": ["price"],
            "event": create_feature_request_event(),
        }

        # Return empty DataFrame
        mock_feature_orchestrator.compute_features.return_value = pd.DataFrame()

        await worker.process_request(request)

        # Should still publish completion
        published_event = mock_event_bus.publish.call_args[0][0]
        assert published_event.data["results_available"] is False
        assert published_event.data["symbols_processed"] == []

    @pytest.mark.asyncio
    async def test_configuration_reload(self, worker):
        """Test configuration can be reloaded."""
        # If worker supports config reload
        if hasattr(worker, "reload_config"):
            new_config = {"feature_group_mappings": {"new_group": ["new_features"]}}

            with patch("builtins.open", mock_open(read_data=yaml.dump(new_config))):
                worker.reload_config()

                assert "new_group" in worker.feature_group_config["feature_group_mappings"]

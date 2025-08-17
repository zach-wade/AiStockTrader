# tests/integration/test_feature_pipeline_integration.py

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.storage.repositories.market_data import MarketDataRepository
from main.events import EventBusFactory
from main.events.types import FeatureRequestEvent, ScannerAlertEvent
from main.feature_pipeline.feature_adapter import create_feature_adapter
from main.feature_pipeline.feature_orchestrator import FeatureCache, FeatureOrchestrator
from main.feature_pipeline.feature_store import FeatureStoreRepository
from main.interfaces.events import IEventBus
from main.models.strategies.base_strategy import BaseStrategy


# Mock Strategy for Testing
class MockTradingStrategy(BaseStrategy):
    """Mock trading strategy for integration testing."""

    def __init__(self, name="mock_strategy"):
        super().__init__()
        self.name = name

    def get_required_feature_sets(self) -> list[str]:
        """Return required feature sets."""
        return ["technical", "volume"]

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Mock signal generation."""
        signals = {}
        for symbol, features in data.items():
            if not features.empty and "sma_10" in features.columns:
                signals[symbol] = {
                    "signal": (
                        "BUY"
                        if features["sma_10"].iloc[-1] > features["close"].iloc[-1]
                        else "HOLD"
                    ),
                    "confidence": 0.75,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
        return signals


# Test Fixtures and Utilities
@pytest.fixture
def mock_config():
    """Mock configuration for integration tests."""
    return {
        "orchestrator": {
            "features": {
                "cache": {"ttl_seconds": 300},
                "parallel_processing": {"max_workers": 2, "max_concurrent_tasks": 5},
                "batch_processing": {
                    "default_batch_size": 10,
                    "interval_seconds": 1,
                    "processing_delay_seconds": 0.1,
                },
                "technical_indicators": {
                    "sma_short_window": 5,
                    "sma_long_window": 10,
                    "volume_sma_window": 5,
                },
            },
            "lookback_periods": {"feature_calculation_days": 30},
        },
        "paths": {"features": "test_features", "archive": "test_archive"},
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db",
            "user": "test_user",
            "password": "test_password",
        },
        "redis": {"host": "localhost", "port": 6379, "db": 1},
        "prediction": {"use_redis": False},
    }


@pytest.fixture
def sample_market_data():
    """Generate sample market data for multiple symbols."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz=UTC)

    data_frames = []
    np.random.seed(42)  # For reproducible tests

    for symbol in symbols:
        base_price = {"AAPL": 150, "MSFT": 250, "GOOGL": 100}[symbol]

        # Generate realistic price movements
        returns = secure_numpy_normal(0, 0.02, len(dates))
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        symbol_data = pd.DataFrame(
            {
                "symbol": symbol,
                "timestamp": dates,
                "open": np.array(prices) * (1 + secure_numpy_normal(0, 0.001, len(dates))),
                "high": np.array(prices) * (1 + np.abs(secure_numpy_normal(0, 0.01, len(dates)))),
                "low": np.array(prices) * (1 - np.abs(secure_numpy_normal(0, 0.01, len(dates)))),
                "close": prices,
                "volume": np.secure_randint(10000, 100000, len(dates)),
            }
        )
        data_frames.append(symbol_data)

    return pd.concat(data_frames, ignore_index=True)


@pytest.fixture
def mock_database_adapter():
    """Mock database adapter."""
    adapter = MagicMock(spec=AsyncDatabaseAdapter)
    adapter.execute_query = AsyncMock()
    adapter.fetch_query = AsyncMock()
    adapter.create_connection = AsyncMock()
    return adapter


@pytest.fixture
def mock_data_archive():
    """Mock data archive."""
    archive = MagicMock(spec=DataArchive)
    archive.save_dataframe = AsyncMock()
    archive.load_dataframe = AsyncMock()
    archive.exists = AsyncMock(return_value=True)
    return archive


@pytest.fixture
def mock_market_data_repo():
    """Mock market data repository."""
    repo = MagicMock(spec=MarketDataRepository)
    repo.get_data_for_symbols_and_range = AsyncMock()
    return repo


@pytest.fixture
def mock_feature_store_repo():
    """Mock feature store repository."""
    repo = MagicMock(spec=FeatureStoreRepository)
    repo.store_features = AsyncMock()
    repo.get_features = AsyncMock()

    # Mock successful storage result
    store_result = MagicMock()
    store_result.success = True
    store_result.errors = []
    repo.store_features.return_value = store_result

    return repo


@pytest.fixture
def mock_event_bus():
    """Mock event bus."""
    event_bus = MagicMock(spec=IEventBus)
    event_bus.subscribe = Mock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def temp_directory():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# Test Feature Pipeline Core Flow
class TestFeaturePipelineFlow:
    """Test end-to-end feature pipeline data flow."""

    @pytest.mark.asyncio
    async def test_complete_feature_pipeline_flow(
        self,
        mock_config,
        sample_market_data,
        mock_database_adapter,
        mock_data_archive,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_event_bus,
    ):
        """Test complete end-to-end feature pipeline flow."""
        # Setup mock market data repository to return sample data
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        # Create orchestrator with mocked dependencies
        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=mock_event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)

                # Replace dependencies with mocks
                orchestrator.db_adapter = mock_database_adapter
                orchestrator.data_archive = mock_data_archive
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo

        # Test feature calculation for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)

        # Calculate features
        features = await orchestrator.calculate_features(
            symbols=symbols, start_time=start_time, end_time=end_time
        )

        # Verify results
        assert isinstance(features, dict)
        assert len(features) == len(symbols)

        for symbol in symbols:
            assert symbol in features
            assert isinstance(features[symbol], pd.DataFrame)
            assert not features[symbol].empty

            # Check that basic features are calculated
            feature_df = features[symbol]
            assert "sma_10" in feature_df.columns
            assert "sma_20" in feature_df.columns
            assert "volume_sma" in feature_df.columns

        # Verify market data was fetched
        mock_market_data_repo.get_data_for_symbols_and_range.assert_called()

        # Verify features were stored in both hot and cold storage
        assert mock_data_archive.save_dataframe.call_count >= len(symbols)
        assert mock_feature_store_repo.store_features.call_count >= len(symbols)

    @pytest.mark.asyncio
    async def test_feature_calculation_with_caching(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
        mock_event_bus,
    ):
        """Test feature calculation caching mechanism."""
        # Setup mocks
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=mock_event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        symbols = ["AAPL"]
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)

        # First calculation - should hit database
        features1 = await orchestrator.calculate_features(
            symbols=symbols, start_time=start_time, end_time=end_time
        )

        initial_cache_misses = orchestrator.calculation_stats["cache_misses"]
        initial_database_calls = mock_market_data_repo.get_data_for_symbols_and_range.call_count

        # Second calculation with same parameters - should hit cache
        features2 = await orchestrator.calculate_features(
            symbols=symbols, start_time=start_time, end_time=end_time
        )

        # Verify cache behavior
        assert orchestrator.calculation_stats["cache_hits"] > 0
        assert orchestrator.calculation_stats["cache_misses"] == initial_cache_misses

        # Database should not be called again
        assert (
            mock_market_data_repo.get_data_for_symbols_and_range.call_count
            == initial_database_calls
        )

        # Results should be identical
        pd.testing.assert_frame_equal(features1["AAPL"], features2["AAPL"])

    @pytest.mark.asyncio
    async def test_parallel_feature_calculation(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
        mock_event_bus,
    ):
        """Test parallel feature calculation for multiple symbols."""
        # Setup mocks
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=mock_event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        # Test with multiple symbols to trigger parallel processing
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)

        start_time_calc = time.time()

        features = await orchestrator.calculate_features(
            symbols=symbols, start_time=start_time, end_time=end_time
        )

        calc_duration = time.time() - start_time_calc

        # Verify all symbols processed
        assert len(features) == len(symbols)
        for symbol in symbols:
            assert symbol in features
            assert not features[symbol].empty

        # Verify parallel processing stats
        assert orchestrator.calculation_stats["parallel_tasks"] > 0

        # Should complete in reasonable time due to parallelization
        assert calc_duration < 10.0  # Should be much faster with mocks

    @pytest.mark.asyncio
    async def test_feature_calculation_error_handling(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
        mock_event_bus,
    ):
        """Test error handling during feature calculation."""

        # Setup market data repo to fail for one symbol
        def mock_get_data(symbols, start_time, end_time):
            if "FAIL" in symbols:
                raise Exception("Database connection failed")
            return sample_market_data

        mock_market_data_repo.get_data_for_symbols_and_range.side_effect = mock_get_data

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=mock_event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        symbols = ["AAPL", "FAIL", "MSFT"]
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)

        features = await orchestrator.calculate_features(
            symbols=symbols, start_time=start_time, end_time=end_time
        )

        # Should continue processing despite errors
        assert "AAPL" in features
        assert "MSFT" in features
        # Failed symbol should return empty DataFrame
        assert "FAIL" in features
        assert features["FAIL"].empty

        # Error count should be incremented
        assert orchestrator.calculation_stats["errors"] > 0


# Test Feature Storage Integration
class TestFeatureStorageIntegration:
    """Test feature storage integration with hot/cold storage."""

    @pytest.mark.asyncio
    async def test_hot_cold_storage_integration(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
        mock_event_bus,
    ):
        """Test integration between hot and cold storage systems."""
        # Setup mocks
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=mock_event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        symbols = ["AAPL"]
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)

        # Calculate features
        features = await orchestrator.calculate_features(
            symbols=symbols, start_time=start_time, end_time=end_time
        )

        # Verify hot storage (PostgreSQL) was called
        mock_feature_store_repo.store_features.assert_called()
        store_call = mock_feature_store_repo.store_features.call_args
        assert store_call[1]["symbol"] == "AAPL"
        assert "features" in store_call[1]
        assert "timestamp" in store_call[1]

        # Verify cold storage (Data Lake) was called
        mock_data_archive.save_dataframe.assert_called()
        archive_call = mock_data_archive.save_dataframe.call_args
        assert "features/AAPL/" in archive_call[0][0]  # Feature key path
        assert isinstance(archive_call[0][1], pd.DataFrame)  # Feature DataFrame
        assert "symbol" in archive_call[1]["metadata"]  # Metadata

    @pytest.mark.asyncio
    async def test_storage_failure_resilience(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
        mock_event_bus,
        caplog,
    ):
        """Test resilience when storage operations fail."""
        # Setup mocks - make storage operations fail
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data
        mock_feature_store_repo.store_features.side_effect = Exception(
            "PostgreSQL connection failed"
        )
        mock_data_archive.save_dataframe.side_effect = Exception("S3 upload failed")

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=mock_event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        symbols = ["AAPL"]
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)

        # Calculate features - should not raise despite storage failures
        features = await orchestrator.calculate_features(
            symbols=symbols, start_time=start_time, end_time=end_time
        )

        # Features should still be calculated and returned
        assert "AAPL" in features
        assert not features["AAPL"].empty

        # Should log storage errors
        assert "Failed to store features" in caplog.text


# Test Event-Driven Feature Updates
class TestEventDrivenFeatures:
    """Test event-driven feature pipeline integration."""

    @pytest.mark.asyncio
    async def test_scanner_alert_triggers_feature_calculation(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
    ):
        """Test that scanner alerts trigger feature calculations."""
        # Setup mocks
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        # Create real event bus for this test
        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        # Publish scanner alert event
        alert_event = ScannerAlertEvent(
            data={
                "symbol": "AAPL",
                "alert_type": "high_volume",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {"volume_ratio": 2.5},
            }
        )

        # Publish event and give time for processing
        await event_bus.publish(alert_event)
        await asyncio.sleep(0.1)  # Allow event processing

        # Verify feature calculation was triggered
        assert orchestrator.calculation_stats["scanner_triggered_calculations"] > 0

        # Verify market data was fetched for the alerted symbol
        mock_market_data_repo.get_data_for_symbols_and_range.assert_called()
        call_args = mock_market_data_repo.get_data_for_symbols_and_range.call_args
        assert "AAPL" in call_args[1]["symbols"]

    @pytest.mark.asyncio
    async def test_feature_request_event_processing(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
    ):
        """Test processing of feature request events."""
        # Setup mocks
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        # Create real event bus
        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        # Create high-priority feature request
        feature_request = FeatureRequestEvent(
            data={
                "symbols": ["AAPL", "MSFT"],
                "features": ["price_features", "volume_features"],
                "priority": 9,  # High priority
                "requestor": "test_strategy",
            }
        )

        # Publish event
        await event_bus.publish(feature_request)
        await asyncio.sleep(0.1)

        # Verify feature request was processed
        assert orchestrator.calculation_stats["feature_requests_processed"] > 0

        # High priority requests should be processed immediately
        mock_market_data_repo.get_data_for_symbols_and_range.assert_called()

    @pytest.mark.asyncio
    async def test_batch_processing_low_priority_requests(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
    ):
        """Test batch processing of low-priority feature requests."""
        # Setup mocks
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        # Reduce batch interval for faster testing
        mock_config["orchestrator"]["features"]["batch_processing"]["interval_seconds"] = 0.1

        event_bus = EventBusFactory.create_test_instance()

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        # Create low-priority feature requests
        low_priority_request = FeatureRequestEvent(
            data={
                "symbols": ["GOOGL"],
                "features": ["technical_features"],
                "priority": 3,  # Low priority
                "requestor": "background_process",
            }
        )

        # Publish low-priority request
        await event_bus.publish(low_priority_request)

        # Wait for batch processing
        await asyncio.sleep(0.5)

        # Verify request was processed (eventually)
        assert orchestrator.calculation_stats["feature_requests_processed"] > 0


# Test Feature Adapter Integration
class TestFeatureAdapterIntegration:
    """Test integration between feature pipeline and strategy adapter."""

    @pytest.mark.asyncio
    async def test_end_to_end_feature_to_strategy_flow(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
        mock_event_bus,
    ):
        """Test complete flow from feature calculation to strategy consumption."""
        # Setup feature orchestrator
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=mock_event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        # Setup feature adapter
        adapter = create_feature_adapter(mock_config)

        # Register strategy
        strategy = MockTradingStrategy("test_strategy")
        adapter.register_strategy(strategy)

        # Calculate features
        symbols = ["AAPL", "MSFT"]
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)

        features = await orchestrator.calculate_features(
            symbols=symbols, start_time=start_time, end_time=end_time
        )

        # Prepare features for strategy
        strategy_features = adapter.prepare_features_for_strategy(features, "test_strategy")

        # Verify features are properly formatted for strategy
        assert isinstance(strategy_features, dict)
        assert "AAPL" in strategy_features
        assert "MSFT" in strategy_features

        for symbol in symbols:
            feature_df = strategy_features[symbol]
            assert not feature_df.empty
            assert "close" in feature_df.columns  # Required by strategy
            assert "sma_10" in feature_df.columns  # Technical feature

        # Generate signals using prepared features
        signals = strategy.generate_signals(strategy_features)

        # Verify signals generated
        assert isinstance(signals, dict)
        for symbol in symbols:
            if symbol in signals:
                assert "signal" in signals[symbol]
                assert "confidence" in signals[symbol]

    @pytest.mark.asyncio
    async def test_feature_compatibility_validation(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
        mock_event_bus,
    ):
        """Test feature compatibility validation between pipeline and strategies."""
        # Setup orchestrator
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=mock_event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        # Setup adapter with strategy
        adapter = create_feature_adapter(mock_config)
        strategy = MockTradingStrategy("compatibility_test")
        adapter.register_strategy(strategy)

        # Calculate features
        symbols = ["AAPL"]
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)

        features = await orchestrator.calculate_features(
            symbols=symbols, start_time=start_time, end_time=end_time
        )

        # Validate feature compatibility
        compatibility_report = adapter.validate_feature_compatibility(features)

        # Should have compatibility information for registered strategy
        assert "compatibility_test" in compatibility_report

        # Since our features include technical indicators, should be compatible
        missing_features = compatibility_report["compatibility_test"]
        assert len(missing_features) == 0  # No missing required features


# Test Performance and Load Scenarios
class TestFeaturePipelinePerformance:
    """Test feature pipeline performance and load scenarios."""

    @pytest.mark.asyncio
    async def test_batch_feature_calculation_performance(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
        mock_event_bus,
    ):
        """Test batch feature calculation performance."""
        # Setup mocks
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=mock_event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        # Test with larger symbol list
        symbols = [f"TEST{i}" for i in range(50)]  # 50 symbols

        start_time_test = time.time()

        # Run batch calculation
        await orchestrator.run_batch_calculation(symbols)

        duration = time.time() - start_time_test

        # Verify batch processing completed
        stats = orchestrator.get_calculation_stats()
        assert stats["total_calculations"] > 0

        # Should complete in reasonable time (with mocks)
        assert duration < 30.0  # Should be fast with mocks

        # Verify parallelization was used
        assert stats["parallel_tasks"] > 0

    @pytest.mark.asyncio
    async def test_cache_performance_and_hit_rates(
        self,
        mock_config,
        sample_market_data,
        mock_market_data_repo,
        mock_feature_store_repo,
        mock_data_archive,
        mock_event_bus,
    ):
        """Test cache performance and hit rates."""
        # Setup mocks
        mock_market_data_repo.get_data_for_symbols_and_range.return_value = sample_market_data

        with patch(
            "main.feature_pipeline.feature_orchestrator.get_event_bus", return_value=mock_event_bus
        ):
            with patch(
                "main.feature_pipeline.feature_orchestrator.get_archive",
                return_value=mock_data_archive,
            ):
                orchestrator = FeatureOrchestrator(config=mock_config)
                orchestrator.market_data_repo = mock_market_data_repo
                orchestrator.feature_store_repo = mock_feature_store_repo
                orchestrator.data_archive = mock_data_archive

        symbols = ["AAPL", "MSFT"]
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)

        # First calculation - all cache misses
        await orchestrator.calculate_features(symbols, start_time, end_time)
        initial_stats = orchestrator.get_calculation_stats()

        # Second calculation - should be cache hits
        await orchestrator.calculate_features(symbols, start_time, end_time)
        final_stats = orchestrator.get_calculation_stats()

        # Verify cache hit rate improved
        assert final_stats["cache_hits"] > initial_stats["cache_hits"]
        assert final_stats["cache_hit_rate"] > 0


# Test Cache Functionality
class TestFeatureCache:
    """Test feature cache functionality in isolation."""

    def test_feature_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = FeatureCache(ttl_seconds=60)

        # Test cache miss
        result = cache.get(
            "AAPL", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC)
        )
        assert result is None

        # Test cache set and hit
        test_data = pd.DataFrame({"feature": [1, 2, 3]})
        cache.set(
            "AAPL", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC), test_data
        )

        cached_result = cache.get(
            "AAPL", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC)
        )

        assert cached_result is not None
        pd.testing.assert_frame_equal(cached_result, test_data)

    def test_feature_cache_expiration(self):
        """Test cache expiration functionality."""
        cache = FeatureCache(ttl_seconds=0.1)  # Very short TTL

        test_data = pd.DataFrame({"feature": [1, 2, 3]})
        cache.set(
            "AAPL", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC), test_data
        )

        # Should hit cache immediately
        result = cache.get(
            "AAPL", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC)
        )
        assert result is not None

        # Wait for expiration
        time.sleep(0.2)

        # Should miss cache after expiration
        result = cache.get(
            "AAPL", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC)
        )
        assert result is None

    def test_cache_cleanup_expired_entries(self):
        """Test cleanup of expired cache entries."""
        cache = FeatureCache(ttl_seconds=0.1)

        # Add multiple entries
        test_data = pd.DataFrame({"feature": [1, 2, 3]})
        for i in range(5):
            cache.set(
                f"SYMBOL{i}",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 31, tzinfo=UTC),
                test_data,
            )

        assert len(cache.cache) == 5

        # Wait for expiration
        time.sleep(0.2)

        # Cleanup expired entries
        cache.cleanup_expired()

        # All entries should be cleaned up
        assert len(cache.cache) == 0


if __name__ == "__main__":
    pytest.main([__file__])

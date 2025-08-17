# tests/unit/test_dataloader.py

# Standard library imports
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
from main.feature_pipeline.dataloader import (
    AlternativeDataSource,
    DataLoader,
    DataRequest,
    FeatureDataSource,
    MarketDataSource,
    load_combined_data,
    load_features,
    load_market_data,
)


# Test Fixtures
@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz=UTC)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * 10,
            "open": np.secure_uniform(90, 110, 10),
            "high": np.secure_uniform(95, 115, 10),
            "low": np.secure_uniform(85, 105, 10),
            "close": np.secure_uniform(90, 110, 10),
            "volume": np.secure_randint(10000, 100000, 10),
        }
    )


@pytest.fixture
def sample_feature_data():
    """Sample feature data for testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz=UTC)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * 10,
            "rsi_14": np.secure_uniform(0, 100, 10),
            "sma_20": np.secure_uniform(95, 105, 10),
            "volume_ratio": np.secure_uniform(0.5, 2.0, 10),
            "sentiment_score": np.secure_uniform(-1, 1, 10),
        }
    )


@pytest.fixture
def mock_historical_manager():
    """Mock HistoricalManager."""
    manager = MagicMock()
    manager.get_data = AsyncMock()
    return manager


@pytest.fixture
def mock_market_data_repo():
    """Mock MarketDataRepository."""
    repo = MagicMock()
    repo.get_data = AsyncMock()
    repo.get_available_symbols = MagicMock(return_value=["AAPL", "MSFT", "GOOGL"])
    repo.get_data_range = MagicMock(
        return_value=(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC))
    )
    return repo


@pytest.fixture
def mock_feature_store():
    """Mock FeatureStoreRepository."""
    store = MagicMock()
    store.get_features = AsyncMock()
    store.get_all_features = AsyncMock()
    store.get_available_symbols = MagicMock(return_value=["AAPL", "MSFT"])
    store.get_data_range = MagicMock(
        return_value=(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC))
    )
    return store


@pytest.fixture
def mock_preprocessor():
    """Mock DataPreprocessor."""
    preprocessor = MagicMock()
    preprocessor.preprocess_market_data = MagicMock(side_effect=lambda x, *args: x)
    preprocessor.preprocess_feature_data = MagicMock(side_effect=lambda x, *args: x)
    preprocessor.preprocess_alternative_data = MagicMock(side_effect=lambda x, *args: x)
    return preprocessor


# Test DataRequest
class TestDataRequest:
    """Test DataRequest dataclass."""

    def test_basic_data_request(self):
        """Test basic DataRequest creation."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

        request = DataRequest(symbols=["AAPL", "MSFT"], start_date=start_date, end_date=end_date)

        assert request.symbols == ["AAPL", "MSFT"]
        assert request.data_types == ["market_data"]  # Default
        assert request.interval == "1D"  # Default
        assert not request.include_features  # Default
        assert request.feature_names is None  # Default
        assert request.preprocessing is True  # Default

        # Dates should be timezone aware
        assert request.start_date.tzinfo is not None
        assert request.end_date.tzinfo is not None

    def test_custom_data_request(self):
        """Test DataRequest with custom parameters."""
        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 12, 31, tzinfo=UTC)

        request = DataRequest(
            symbols=["AAPL"],
            start_date=start_date,
            end_date=end_date,
            data_types=["market_data", "news"],
            interval="1H",
            include_features=True,
            feature_names=["rsi_14", "sma_20"],
            preprocessing=False,
        )

        assert request.data_types == ["market_data", "news"]
        assert request.interval == "1H"
        assert request.include_features is True
        assert request.feature_names == ["rsi_14", "sma_20"]
        assert request.preprocessing is False


# Test MarketDataSource
class TestMarketDataSource:
    """Test MarketDataSource functionality."""

    @pytest.mark.asyncio
    async def test_load_data_from_repo(
        self, mock_market_data_repo, mock_historical_manager, sample_market_data
    ):
        """Test loading data from repository."""
        # Setup mocks
        mock_market_data_repo.get_data.return_value = sample_market_data

        with patch(
            "main.feature_pipeline.dataloader.HistoricalManager",
            return_value=mock_historical_manager,
        ):
            with patch(
                "main.feature_pipeline.dataloader.MarketDataRepository",
                return_value=mock_market_data_repo,
            ):
                source = MarketDataSource()

                request = DataRequest(
                    symbols=["AAPL"],
                    start_date=datetime(2024, 1, 1, tzinfo=UTC),
                    end_date=datetime(2024, 1, 10, tzinfo=UTC),
                )

                result = await source.load_data(request)

                assert not result.empty
                assert "symbol" in result.columns
                assert result["symbol"].iloc[0] == "AAPL"
                mock_market_data_repo.get_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_data_fallback_to_historical(
        self, mock_market_data_repo, mock_historical_manager, sample_market_data
    ):
        """Test fallback to historical manager when repo is empty."""
        # Setup mocks - repo returns empty, historical manager returns data
        mock_market_data_repo.get_data.return_value = pd.DataFrame()
        mock_historical_manager.get_data.return_value = sample_market_data

        with patch(
            "main.feature_pipeline.dataloader.HistoricalManager",
            return_value=mock_historical_manager,
        ):
            with patch(
                "main.feature_pipeline.dataloader.MarketDataRepository",
                return_value=mock_market_data_repo,
            ):
                source = MarketDataSource()

                request = DataRequest(
                    symbols=["AAPL"],
                    start_date=datetime(2024, 1, 1, tzinfo=UTC),
                    end_date=datetime(2024, 1, 10, tzinfo=UTC),
                )

                result = await source.load_data(request)

                assert not result.empty
                mock_market_data_repo.get_data.assert_called_once()
                mock_historical_manager.get_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_data_multiple_symbols(self, mock_market_data_repo, mock_historical_manager):
        """Test loading data for multiple symbols."""
        # Create separate data for each symbol
        aapl_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz=UTC),
                "close": [100, 101, 102, 103, 104],
            }
        )
        msft_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz=UTC),
                "close": [200, 201, 202, 203, 204],
            }
        )

        def mock_get_data(symbol, **kwargs):
            if symbol == "AAPL":
                return aapl_data
            elif symbol == "MSFT":
                return msft_data
            else:
                return pd.DataFrame()

        mock_market_data_repo.get_data.side_effect = mock_get_data

        with patch(
            "main.feature_pipeline.dataloader.HistoricalManager",
            return_value=mock_historical_manager,
        ):
            with patch(
                "main.feature_pipeline.dataloader.MarketDataRepository",
                return_value=mock_market_data_repo,
            ):
                source = MarketDataSource()

                request = DataRequest(
                    symbols=["AAPL", "MSFT"],
                    start_date=datetime(2024, 1, 1, tzinfo=UTC),
                    end_date=datetime(2024, 1, 5, tzinfo=UTC),
                )

                result = await source.load_data(request)

                assert not result.empty
                assert "symbol" in result.columns
                symbols_in_result = result["symbol"].unique()
                assert "AAPL" in symbols_in_result
                assert "MSFT" in symbols_in_result

    @pytest.mark.asyncio
    async def test_load_data_error_handling(self, mock_market_data_repo, mock_historical_manager):
        """Test error handling during data loading."""
        # Make repo raise an exception
        mock_market_data_repo.get_data.side_effect = Exception("Database error")

        with patch(
            "main.feature_pipeline.dataloader.HistoricalManager",
            return_value=mock_historical_manager,
        ):
            with patch(
                "main.feature_pipeline.dataloader.MarketDataRepository",
                return_value=mock_market_data_repo,
            ):
                source = MarketDataSource()

                request = DataRequest(
                    symbols=["AAPL", "MSFT"],  # Two symbols, one will fail
                    start_date=datetime(2024, 1, 1, tzinfo=UTC),
                    end_date=datetime(2024, 1, 5, tzinfo=UTC),
                )

                result = await source.load_data(request)

                # Should return empty DataFrame when all symbols fail
                assert result.empty

    def test_get_available_symbols(self, mock_market_data_repo, mock_historical_manager):
        """Test getting available symbols."""
        with patch(
            "main.feature_pipeline.dataloader.HistoricalManager",
            return_value=mock_historical_manager,
        ):
            with patch(
                "main.feature_pipeline.dataloader.MarketDataRepository",
                return_value=mock_market_data_repo,
            ):
                source = MarketDataSource()

                symbols = source.get_available_symbols()

                assert symbols == ["AAPL", "MSFT", "GOOGL"]
                mock_market_data_repo.get_available_symbols.assert_called_once()

    def test_get_data_range(self, mock_market_data_repo, mock_historical_manager):
        """Test getting data range for a symbol."""
        with patch(
            "main.feature_pipeline.dataloader.HistoricalManager",
            return_value=mock_historical_manager,
        ):
            with patch(
                "main.feature_pipeline.dataloader.MarketDataRepository",
                return_value=mock_market_data_repo,
            ):
                source = MarketDataSource()

                start, end = source.get_data_range("AAPL")

                assert start == datetime(2024, 1, 1, tzinfo=UTC)
                assert end == datetime(2024, 12, 31, tzinfo=UTC)
                mock_market_data_repo.get_data_range.assert_called_once_with("AAPL")


# Test FeatureDataSource
class TestFeatureDataSource:
    """Test FeatureDataSource functionality."""

    @pytest.mark.asyncio
    async def test_load_specific_features(self, mock_feature_store, sample_feature_data):
        """Test loading specific features."""
        mock_feature_store.get_features.return_value = sample_feature_data

        with patch(
            "main.feature_pipeline.dataloader.FeatureStoreRepository",
            return_value=mock_feature_store,
        ):
            source = FeatureDataSource()

            request = DataRequest(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 10, tzinfo=UTC),
                include_features=True,
                feature_names=["rsi_14", "sma_20"],
            )

            result = await source.load_data(request)

            assert not result.empty
            mock_feature_store.get_features.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_all_features(self, mock_feature_store, sample_feature_data):
        """Test loading all available features."""
        mock_feature_store.get_all_features.return_value = sample_feature_data

        with patch(
            "main.feature_pipeline.dataloader.FeatureStoreRepository",
            return_value=mock_feature_store,
        ):
            source = FeatureDataSource()

            request = DataRequest(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 10, tzinfo=UTC),
                include_features=True,
            )

            result = await source.load_data(request)

            assert not result.empty
            mock_feature_store.get_all_features.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_features_requested(self, mock_feature_store):
        """Test when no features are requested."""
        with patch(
            "main.feature_pipeline.dataloader.FeatureStoreRepository",
            return_value=mock_feature_store,
        ):
            source = FeatureDataSource()

            request = DataRequest(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 10, tzinfo=UTC),
                include_features=False,
            )

            result = await source.load_data(request)

            assert result.empty
            mock_feature_store.get_features.assert_not_called()
            mock_feature_store.get_all_features.assert_not_called()


# Test AlternativeDataSource
class TestAlternativeDataSource:
    """Test AlternativeDataSource functionality."""

    def test_news_source_initialization(self):
        """Test news data source initialization."""
        source = AlternativeDataSource("news")

        assert source.source_type == "news"
        assert hasattr(source, "news_repo")

    def test_social_source_initialization(self):
        """Test social media data source initialization."""
        source = AlternativeDataSource("social")

        assert source.source_type == "social"
        assert hasattr(source, "social_repo")

    def test_economic_source_initialization(self):
        """Test economic data source initialization."""
        source = AlternativeDataSource("economic")

        assert source.source_type == "economic"
        assert hasattr(source, "economic_repo")

    @pytest.mark.asyncio
    async def test_load_data_placeholder(self):
        """Test that load_data returns empty DataFrame (placeholder implementation)."""
        source = AlternativeDataSource("news")

        request = DataRequest(
            symbols=["AAPL"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 10, tzinfo=UTC),
        )

        result = await source.load_data(request)

        # Placeholder implementation should return empty DataFrame
        assert result.empty

    def test_get_available_symbols_placeholder(self):
        """Test that get_available_symbols returns empty list (placeholder implementation)."""
        source = AlternativeDataSource("news")
        symbols = source.get_available_symbols()
        assert symbols == []

    def test_get_data_range_placeholder(self):
        """Test that get_data_range returns None (placeholder implementation)."""
        source = AlternativeDataSource("news")
        start, end = source.get_data_range("AAPL")
        assert start is None
        assert end is None


# Test DataLoader
class TestDataLoader:
    """Test DataLoader main functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for DataLoader."""
        return {
            "enable_preprocessing": True,
            "max_workers": 2,
            "market_data": {},
            "features": {},
            "preprocessing": {},
        }

    @pytest.mark.asyncio
    async def test_load_market_data_only(self, mock_config, sample_market_data):
        """Test loading only market data."""
        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            with patch.object(MarketDataSource, "load_data", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = sample_market_data

                with patch(
                    "main.feature_pipeline.dataloader.DataPreprocessor"
                ) as mock_preprocessor_class:
                    mock_preprocessor = MagicMock()
                    mock_preprocessor.preprocess_market_data.return_value = sample_market_data
                    mock_preprocessor_class.return_value = mock_preprocessor

                    loader = DataLoader(mock_config)

                    request = DataRequest(
                        symbols=["AAPL"],
                        start_date=datetime(2024, 1, 1, tzinfo=UTC),
                        end_date=datetime(2024, 1, 10, tzinfo=UTC),
                        data_types=["market_data"],
                    )

                    results = await loader.load_data(request)

                    assert "market_data" in results
                    assert not results["market_data"].empty
                    mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_with_features(self, mock_config, sample_market_data, sample_feature_data):
        """Test loading market data with features."""
        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            with patch.object(
                MarketDataSource, "load_data", new_callable=AsyncMock
            ) as mock_market_load:
                with patch.object(
                    FeatureDataSource, "load_data", new_callable=AsyncMock
                ) as mock_feature_load:
                    mock_market_load.return_value = sample_market_data
                    mock_feature_load.return_value = sample_feature_data

                    with patch(
                        "main.feature_pipeline.dataloader.DataPreprocessor"
                    ) as mock_preprocessor_class:
                        mock_preprocessor = MagicMock()
                        mock_preprocessor.preprocess_market_data.return_value = sample_market_data
                        mock_preprocessor.preprocess_feature_data.return_value = sample_feature_data
                        mock_preprocessor_class.return_value = mock_preprocessor

                        loader = DataLoader(mock_config)

                        request = DataRequest(
                            symbols=["AAPL"],
                            start_date=datetime(2024, 1, 1, tzinfo=UTC),
                            end_date=datetime(2024, 1, 10, tzinfo=UTC),
                            data_types=["market_data"],
                            include_features=True,
                            feature_names=["rsi_14"],
                        )

                        results = await loader.load_data(request)

                        assert "market_data" in results
                        assert "features" in results
                        assert not results["market_data"].empty
                        assert not results["features"].empty

    @pytest.mark.asyncio
    async def test_load_data_validation_errors(self, mock_config):
        """Test data request validation errors."""
        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            loader = DataLoader(mock_config)

            # Test empty symbols
            with pytest.raises(ValueError, match="No symbols specified"):
                request = DataRequest(
                    symbols=[],
                    start_date=datetime(2024, 1, 1, tzinfo=UTC),
                    end_date=datetime(2024, 1, 10, tzinfo=UTC),
                )
                await loader.load_data(request)

            # Test invalid date range
            with pytest.raises(ValueError, match="Start date must be before end date"):
                request = DataRequest(
                    symbols=["AAPL"],
                    start_date=datetime(2024, 1, 10, tzinfo=UTC),
                    end_date=datetime(2024, 1, 1, tzinfo=UTC),
                )
                await loader.load_data(request)

    @pytest.mark.asyncio
    async def test_load_data_unknown_type_warning(self, mock_config, caplog):
        """Test warning for unknown data types."""
        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            loader = DataLoader(mock_config)

            request = DataRequest(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 10, tzinfo=UTC),
                data_types=["unknown_type"],
            )

            results = await loader.load_data(request)

            assert "Unknown data type requested: unknown_type" in caplog.text
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_preprocessing_disabled(self, mock_config, sample_market_data):
        """Test loading data with preprocessing disabled."""
        mock_config["enable_preprocessing"] = False

        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            with patch.object(MarketDataSource, "load_data", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = sample_market_data

                loader = DataLoader(mock_config)

                request = DataRequest(
                    symbols=["AAPL"],
                    start_date=datetime(2024, 1, 1, tzinfo=UTC),
                    end_date=datetime(2024, 1, 10, tzinfo=UTC),
                    data_types=["market_data"],
                    preprocessing=True,  # Requested but disabled in config
                )

                results = await loader.load_data(request)

                # Should not have preprocessor
                assert loader.preprocessor is None

    @pytest.mark.asyncio
    async def test_convenience_load_market_data(self, mock_config, sample_market_data):
        """Test convenience method for loading market data."""
        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            with patch.object(MarketDataSource, "load_data", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = sample_market_data

                with patch(
                    "main.feature_pipeline.dataloader.DataPreprocessor"
                ) as mock_preprocessor_class:
                    mock_preprocessor = MagicMock()
                    mock_preprocessor.preprocess_market_data.return_value = sample_market_data
                    mock_preprocessor_class.return_value = mock_preprocessor

                    loader = DataLoader(mock_config)

                    result = await loader.load_market_data(
                        symbols=["AAPL"],
                        start_date=datetime(2024, 1, 1, tzinfo=UTC),
                        end_date=datetime(2024, 1, 10, tzinfo=UTC),
                    )

                    assert not result.empty
                    assert isinstance(result, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_convenience_load_features(self, mock_config, sample_feature_data):
        """Test convenience method for loading features."""
        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            with patch.object(FeatureDataSource, "load_data", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = sample_feature_data

                with patch(
                    "main.feature_pipeline.dataloader.DataPreprocessor"
                ) as mock_preprocessor_class:
                    mock_preprocessor = MagicMock()
                    mock_preprocessor.preprocess_feature_data.return_value = sample_feature_data
                    mock_preprocessor_class.return_value = mock_preprocessor

                    loader = DataLoader(mock_config)

                    result = await loader.load_features(
                        symbols=["AAPL"],
                        start_date=datetime(2024, 1, 1, tzinfo=UTC),
                        end_date=datetime(2024, 1, 10, tzinfo=UTC),
                        feature_names=["rsi_14"],
                    )

                    assert not result.empty
                    assert isinstance(result, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_convenience_load_combined_data(
        self, mock_config, sample_market_data, sample_feature_data
    ):
        """Test convenience method for loading combined data."""
        # Ensure both DataFrames have same timestamp range for merging
        sample_feature_data["timestamp"] = sample_market_data["timestamp"]

        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            with patch.object(
                MarketDataSource, "load_data", new_callable=AsyncMock
            ) as mock_market_load:
                with patch.object(
                    FeatureDataSource, "load_data", new_callable=AsyncMock
                ) as mock_feature_load:
                    mock_market_load.return_value = sample_market_data
                    mock_feature_load.return_value = sample_feature_data

                    with patch(
                        "main.feature_pipeline.dataloader.DataPreprocessor"
                    ) as mock_preprocessor_class:
                        mock_preprocessor = MagicMock()
                        mock_preprocessor.preprocess_market_data.return_value = sample_market_data
                        mock_preprocessor.preprocess_feature_data.return_value = sample_feature_data
                        mock_preprocessor_class.return_value = mock_preprocessor

                        loader = DataLoader(mock_config)

                        result = await loader.load_combined_data(
                            symbols=["AAPL"],
                            start_date=datetime(2024, 1, 1, tzinfo=UTC),
                            end_date=datetime(2024, 1, 10, tzinfo=UTC),
                            feature_names=["rsi_14"],
                        )

                        assert not result.empty
                        assert isinstance(result, pd.DataFrame)
                        # Should have both market data and feature columns
                        assert "close" in result.columns  # From market data
                        assert "rsi_14" in result.columns  # From features

    def test_get_available_symbols(self, mock_config):
        """Test getting available symbols from a data source."""
        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            with patch.object(
                MarketDataSource, "get_available_symbols", return_value=["AAPL", "MSFT"]
            ):
                loader = DataLoader(mock_config)

                symbols = loader.get_available_symbols("market_data")

                assert symbols == ["AAPL", "MSFT"]

    def test_get_data_range(self, mock_config):
        """Test getting data range for a symbol."""
        expected_range = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC))

        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            with patch.object(MarketDataSource, "get_data_range", return_value=expected_range):
                loader = DataLoader(mock_config)

                start, end = loader.get_data_range("AAPL", "market_data")

                assert start == expected_range[0]
                assert end == expected_range[1]

    @pytest.mark.asyncio
    async def test_close_cleanup(self, mock_config):
        """Test proper cleanup when closing DataLoader."""
        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            loader = DataLoader(mock_config)

            # Mock the executor
            mock_executor = MagicMock()
            loader.executor = mock_executor

            await loader.close()

            mock_executor.shutdown.assert_called_once_with(wait=True)


# Test Standalone Convenience Functions
class TestConvenienceFunctions:
    """Test standalone convenience functions."""

    @pytest.mark.asyncio
    async def test_standalone_load_market_data(self, sample_market_data):
        """Test standalone load_market_data function."""
        with patch("main.feature_pipeline.dataloader.DataLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.load_market_data = AsyncMock(return_value=sample_market_data)
            mock_loader.close = AsyncMock()
            mock_loader_class.return_value = mock_loader

            result = await load_market_data(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 10, tzinfo=UTC),
            )

            assert not result.empty
            mock_loader.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_standalone_load_features(self, sample_feature_data):
        """Test standalone load_features function."""
        with patch("main.feature_pipeline.dataloader.DataLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.load_features = AsyncMock(return_value=sample_feature_data)
            mock_loader.close = AsyncMock()
            mock_loader_class.return_value = mock_loader

            result = await load_features(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 10, tzinfo=UTC),
                feature_names=["rsi_14"],
            )

            assert not result.empty
            mock_loader.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_standalone_load_combined_data(self, sample_market_data):
        """Test standalone load_combined_data function."""
        with patch("main.feature_pipeline.dataloader.DataLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.load_combined_data = AsyncMock(return_value=sample_market_data)
            mock_loader.close = AsyncMock()
            mock_loader_class.return_value = mock_loader

            result = await load_combined_data(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 10, tzinfo=UTC),
            )

            assert not result.empty
            mock_loader.close.assert_called_once()


# Test Error Handling and Edge Cases
class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_source_loading_failure_continues(self, mock_config, caplog):
        """Test that failure in one source doesn't stop others."""
        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            with patch.object(
                MarketDataSource, "load_data", side_effect=Exception("Market data failed")
            ):
                with patch.object(
                    FeatureDataSource, "load_data", new_callable=AsyncMock
                ) as mock_feature_load:
                    mock_feature_load.return_value = pd.DataFrame({"feature": [1, 2, 3]})

                    loader = DataLoader(mock_config)

                    request = DataRequest(
                        symbols=["AAPL"],
                        start_date=datetime(2024, 1, 1, tzinfo=UTC),
                        end_date=datetime(2024, 1, 10, tzinfo=UTC),
                        data_types=["market_data"],
                        include_features=True,
                    )

                    results = await loader.load_data(request)

                    # Market data should fail but return empty DataFrame
                    assert "market_data" in results
                    assert results["market_data"].empty

                    # Features should still load
                    assert "features" in results
                    assert not results["features"].empty

                    assert "Failed to load market_data" in caplog.text

    @pytest.mark.asyncio
    async def test_preprocessing_failure_fallback(self, mock_config, sample_market_data, caplog):
        """Test that preprocessing failure falls back to original data."""
        with patch(
            "main.feature_pipeline.dataloader.get_config", return_value={"data_loader": mock_config}
        ):
            with patch.object(MarketDataSource, "load_data", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = sample_market_data

                with patch(
                    "main.feature_pipeline.dataloader.DataPreprocessor"
                ) as mock_preprocessor_class:
                    mock_preprocessor = MagicMock()
                    mock_preprocessor.preprocess_market_data.side_effect = Exception(
                        "Preprocessing failed"
                    )
                    mock_preprocessor_class.return_value = mock_preprocessor

                    loader = DataLoader(mock_config)

                    request = DataRequest(
                        symbols=["AAPL"],
                        start_date=datetime(2024, 1, 1, tzinfo=UTC),
                        end_date=datetime(2024, 1, 10, tzinfo=UTC),
                        data_types=["market_data"],
                    )

                    results = await loader.load_data(request)

                    # Should fallback to original data
                    assert "market_data" in results
                    assert not results["market_data"].empty
                    assert "Preprocessing failed for market_data" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__])

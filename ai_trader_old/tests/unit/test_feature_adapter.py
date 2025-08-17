# tests/unit/test_feature_adapter.py

# Standard library imports
from datetime import UTC
from typing import Any
from unittest.mock import patch

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
from main.feature_pipeline.feature_adapter import (
    FeatureAdapter,
    FeatureRequirement,
    FeatureValidationError,
    StrategyFeatureMapping,
    create_feature_adapter,
)
from main.models.strategies.base_strategy import BaseStrategy


# Mock Strategy Classes for Testing
class MockTechnicalStrategy(BaseStrategy):
    """Mock technical strategy for testing."""

    def __init__(self, name="mock_technical"):
        super().__init__()
        self.name = name

    def get_required_feature_sets(self) -> list[str]:
        """Return required feature sets."""
        return ["technical"]

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Mock signal generation."""
        return {}


class MockSentimentStrategy(BaseStrategy):
    """Mock sentiment strategy for testing."""

    def __init__(self, name="mock_sentiment"):
        super().__init__()
        self.name = name

    def get_required_feature_sets(self) -> list[str]:
        """Return required feature sets."""
        return ["sentiment", "technical"]

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Mock signal generation."""
        return {}


class MockMicrostructureStrategy(BaseStrategy):
    """Mock microstructure strategy for testing."""

    def __init__(self, name="mock_microstructure"):
        super().__init__()
        self.name = name

    def get_required_feature_sets(self) -> list[str]:
        """Return required feature sets."""
        return ["microstructure", "technical"]

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Mock signal generation."""
        return {}


# Test Fixtures
@pytest.fixture
def sample_features_data():
    """Sample features data for testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D", tz=UTC)
    np.random.seed(42)

    features = {
        "AAPL": pd.DataFrame(
            {
                "close": 150 + np.cumsum(secure_numpy_normal(0, 0.02, 50)),
                "open": 150 + np.cumsum(secure_numpy_normal(0, 0.02, 50)),
                "high": 150
                + np.cumsum(secure_numpy_normal(0, 0.02, 50))
                + np.abs(secure_numpy_normal(0, 0.01, 50)),
                "low": 150
                + np.cumsum(secure_numpy_normal(0, 0.02, 50))
                - np.abs(secure_numpy_normal(0, 0.01, 50)),
                "volume": np.secure_randint(10000, 100000, 50),
                "sma_10": 150 + np.cumsum(secure_numpy_normal(0, 0.01, 50)),
                "sma_20": 150 + np.cumsum(secure_numpy_normal(0, 0.01, 50)),
                "rsi": np.secure_uniform(0, 100, 50),
                "macd": secure_numpy_normal(0, 1, 50),
                "news_sentiment": np.secure_uniform(-1, 1, 50),
                "social_sentiment": np.secure_uniform(-1, 1, 50),
                "bid_ask_spread": np.secure_uniform(0.01, 0.1, 50),
                "order_flow": secure_numpy_normal(0, 1000, 50),
            },
            index=dates,
        ),
        "MSFT": pd.DataFrame(
            {
                "close": 250 + np.cumsum(secure_numpy_normal(0, 0.02, 50)),
                "open": 250 + np.cumsum(secure_numpy_normal(0, 0.02, 50)),
                "high": 250
                + np.cumsum(secure_numpy_normal(0, 0.02, 50))
                + np.abs(secure_numpy_normal(0, 0.01, 50)),
                "low": 250
                + np.cumsum(secure_numpy_normal(0, 0.02, 50))
                - np.abs(secure_numpy_normal(0, 0.01, 50)),
                "volume": np.secure_randint(10000, 100000, 50),
                "sma_10": 250 + np.cumsum(secure_numpy_normal(0, 0.01, 50)),
                "sma_20": 250 + np.cumsum(secure_numpy_normal(0, 0.01, 50)),
                "rsi": np.secure_uniform(0, 100, 50),
                "macd": secure_numpy_normal(0, 1, 50),
                # Missing sentiment and microstructure features
            },
            index=dates,
        ),
    }

    return features


@pytest.fixture
def incomplete_features_data():
    """Features data missing required columns."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D", tz=UTC)

    return {
        "TEST": pd.DataFrame(
            {
                "volume": np.secure_randint(10000, 100000, 20),
                "sma_10": np.secure_uniform(95, 105, 20),
                # Missing required 'close' column
            },
            index=dates,
        )
    }


@pytest.fixture
def empty_features_data():
    """Empty features data for testing."""
    return {"EMPTY": pd.DataFrame()}


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {"feature_adapter": {"cache_ttl_seconds": 300}}


@pytest.fixture
def feature_adapter(mock_config):
    """Feature adapter instance for testing."""
    with patch("main.feature_pipeline.feature_adapter.get_config", return_value=mock_config):
        return FeatureAdapter(mock_config)


# Test FeatureRequirement and StrategyFeatureMapping
class TestDataClasses:
    """Test FeatureRequirement and StrategyFeatureMapping dataclasses."""

    def test_feature_requirement_defaults(self):
        """Test FeatureRequirement default values."""
        req = FeatureRequirement("test_feature")

        assert req.feature_name == "test_feature"
        assert req.required is True
        assert req.default_value is None
        assert req.description is None

    def test_feature_requirement_custom(self):
        """Test FeatureRequirement with custom values."""
        req = FeatureRequirement(
            "optional_feature", required=False, default_value=0.0, description="Test feature"
        )

        assert req.feature_name == "optional_feature"
        assert req.required is False
        assert req.default_value == 0.0
        assert req.description == "Test feature"

    def test_strategy_feature_mapping(self):
        """Test StrategyFeatureMapping creation."""
        requirements = [
            FeatureRequirement("close", required=True),
            FeatureRequirement("volume", required=False),
        ]

        mapping = StrategyFeatureMapping(
            strategy_name="test_strategy",
            required_features=requirements,
            feature_sets=["technical"],
        )

        assert mapping.strategy_name == "test_strategy"
        assert len(mapping.required_features) == 2
        assert mapping.feature_sets == ["technical"]


# Test FeatureAdapter Initialization
class TestFeatureAdapterInit:
    """Test FeatureAdapter initialization."""

    @patch("main.feature_pipeline.feature_adapter.get_config")
    def test_init_with_default_config(self, mock_get_config):
        """Test initialization with default configuration."""
        mock_get_config.return_value = {}

        adapter = FeatureAdapter()

        assert adapter.config == {}
        assert adapter.strategy_mappings == {}
        assert adapter.feature_cache == {}
        assert adapter.cache_ttl == 300  # default

    def test_init_with_custom_config(self, mock_config):
        """Test initialization with custom configuration."""
        with patch("main.feature_pipeline.feature_adapter.get_config", return_value=mock_config):
            adapter = FeatureAdapter(mock_config)

            assert adapter.config == mock_config
            assert adapter.cache_ttl == 300


# Test Strategy Registration
class TestStrategyRegistration:
    """Test strategy registration functionality."""

    def test_register_technical_strategy(self, feature_adapter, caplog):
        """Test registering a technical strategy."""
        strategy = MockTechnicalStrategy("test_tech")

        feature_adapter.register_strategy(strategy)

        assert "test_tech" in feature_adapter.strategy_mappings
        mapping = feature_adapter.strategy_mappings["test_tech"]
        assert mapping.strategy_name == "test_tech"
        assert mapping.feature_sets == ["technical"]
        assert "Registered strategy 'test_tech'" in caplog.text

    def test_register_sentiment_strategy(self, feature_adapter):
        """Test registering a sentiment strategy."""
        strategy = MockSentimentStrategy("test_sentiment")

        feature_adapter.register_strategy(strategy)

        mapping = feature_adapter.strategy_mappings["test_sentiment"]
        assert mapping.feature_sets == ["sentiment", "technical"]
        # Should have both technical and sentiment features
        feature_names = [req.feature_name for req in mapping.required_features]
        assert "close" in feature_names
        assert "sma_10" in feature_names
        assert "news_sentiment" in feature_names

    def test_register_microstructure_strategy(self, feature_adapter):
        """Test registering a microstructure strategy."""
        strategy = MockMicrostructureStrategy("test_micro")

        feature_adapter.register_strategy(strategy)

        mapping = feature_adapter.strategy_mappings["test_micro"]
        assert mapping.feature_sets == ["microstructure", "technical"]
        feature_names = [req.feature_name for req in mapping.required_features]
        assert "bid_ask_spread" in feature_names
        assert "order_flow" in feature_names

    def test_multiple_strategy_registration(self, feature_adapter):
        """Test registering multiple strategies."""
        strategy1 = MockTechnicalStrategy("tech1")
        strategy2 = MockSentimentStrategy("sentiment1")

        feature_adapter.register_strategy(strategy1)
        feature_adapter.register_strategy(strategy2)

        assert len(feature_adapter.strategy_mappings) == 2
        assert "tech1" in feature_adapter.strategy_mappings
        assert "sentiment1" in feature_adapter.strategy_mappings


# Test Feature Requirement Extraction
class TestFeatureRequirementExtraction:
    """Test feature requirement extraction."""

    def test_extract_technical_requirements(self, feature_adapter):
        """Test extraction of technical feature requirements."""
        strategy = MockTechnicalStrategy()
        requirements = feature_adapter._extract_feature_requirements(strategy, ["technical"])

        feature_names = [req.feature_name for req in requirements]

        # Should have standard features
        assert "close" in feature_names
        assert "volume" in feature_names
        assert "open" in feature_names

        # Should have technical features
        assert "sma_10" in feature_names
        assert "sma_20" in feature_names
        assert "rsi" in feature_names
        assert "macd" in feature_names

    def test_extract_sentiment_requirements(self, feature_adapter):
        """Test extraction of sentiment feature requirements."""
        strategy = MockSentimentStrategy()
        requirements = feature_adapter._extract_feature_requirements(strategy, ["sentiment"])

        feature_names = [req.feature_name for req in requirements]

        # Should have sentiment features
        assert "news_sentiment" in feature_names
        assert "social_sentiment" in feature_names

        # Should also have standard features
        assert "close" in feature_names

    def test_extract_microstructure_requirements(self, feature_adapter):
        """Test extraction of microstructure feature requirements."""
        strategy = MockMicrostructureStrategy()
        requirements = feature_adapter._extract_feature_requirements(strategy, ["microstructure"])

        feature_names = [req.feature_name for req in requirements]

        # Should have microstructure features
        assert "bid_ask_spread" in feature_names
        assert "order_flow" in feature_names

    def test_extract_combined_requirements(self, feature_adapter):
        """Test extraction with multiple feature sets."""
        strategy = MockSentimentStrategy()  # Uses both sentiment and technical
        requirements = feature_adapter._extract_feature_requirements(
            strategy, ["technical", "sentiment"]
        )

        feature_names = [req.feature_name for req in requirements]

        # Should have features from both sets
        assert "sma_10" in feature_names  # Technical
        assert "news_sentiment" in feature_names  # Sentiment
        assert "close" in feature_names  # Standard


# Test Feature Preparation
class TestFeaturePreparation:
    """Test feature preparation for strategies."""

    def test_prepare_features_success(self, feature_adapter, sample_features_data):
        """Test successful feature preparation."""
        strategy = MockTechnicalStrategy("test_tech")
        feature_adapter.register_strategy(strategy)

        result = feature_adapter.prepare_features_for_strategy(sample_features_data, "test_tech")

        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result
        assert isinstance(result["AAPL"], pd.DataFrame)
        assert not result["AAPL"].empty

    def test_prepare_features_unregistered_strategy(self, feature_adapter, sample_features_data):
        """Test preparation for unregistered strategy."""
        with pytest.raises(FeatureValidationError, match="Strategy 'unknown' not registered"):
            feature_adapter.prepare_features_for_strategy(sample_features_data, "unknown")

    def test_prepare_features_missing_required_features(
        self, feature_adapter, incomplete_features_data
    ):
        """Test preparation with missing required features."""
        strategy = MockTechnicalStrategy("test_tech")
        feature_adapter.register_strategy(strategy)

        result = feature_adapter.prepare_features_for_strategy(
            incomplete_features_data, "test_tech"
        )

        # Should skip symbols with missing required features
        assert len(result) == 0  # TEST symbol missing 'close'

    def test_prepare_features_empty_dataframe(self, feature_adapter, empty_features_data, caplog):
        """Test preparation with empty DataFrame."""
        strategy = MockTechnicalStrategy("test_tech")
        feature_adapter.register_strategy(strategy)

        result = feature_adapter.prepare_features_for_strategy(empty_features_data, "test_tech")

        # Should skip empty DataFrames
        assert len(result) == 0
        assert "Failed to prepare features for EMPTY" in caplog.text

    def test_prepare_features_adds_default_values(self, feature_adapter, sample_features_data):
        """Test that default values are added for missing optional features."""
        # Create strategy with optional features having defaults
        strategy = MockTechnicalStrategy("test_with_defaults")
        feature_adapter.register_strategy(strategy)

        # Remove optional features from one symbol
        modified_data = sample_features_data.copy()
        modified_data["MSFT"] = modified_data["MSFT"][["close", "open", "high", "low", "volume"]]

        result = feature_adapter.prepare_features_for_strategy(modified_data, "test_with_defaults")

        # MSFT should still be included, optional features should get defaults if configured
        assert "MSFT" in result


# Test Feature Validation
class TestFeatureValidation:
    """Test feature validation functionality."""

    def test_validate_and_transform_features_success(self, feature_adapter, sample_features_data):
        """Test successful validation and transformation."""
        strategy = MockTechnicalStrategy("test_tech")
        feature_adapter.register_strategy(strategy)
        mapping = feature_adapter.strategy_mappings["test_tech"]

        result = feature_adapter._validate_and_transform_features(
            sample_features_data["AAPL"], mapping, "AAPL"
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result.index.is_monotonic_increasing

    def test_validate_empty_dataframe_raises_error(self, feature_adapter):
        """Test that empty DataFrame raises validation error."""
        strategy = MockTechnicalStrategy("test_tech")
        feature_adapter.register_strategy(strategy)
        mapping = feature_adapter.strategy_mappings["test_tech"]

        empty_df = pd.DataFrame()

        with pytest.raises(FeatureValidationError, match="Empty features DataFrame"):
            feature_adapter._validate_and_transform_features(empty_df, mapping, "TEST")

    def test_validate_missing_required_features_raises_error(self, feature_adapter):
        """Test that missing required features raises validation error."""
        strategy = MockTechnicalStrategy("test_tech")
        feature_adapter.register_strategy(strategy)
        mapping = feature_adapter.strategy_mappings["test_tech"]

        # DataFrame missing required 'close' column
        incomplete_df = pd.DataFrame({"volume": [1000, 2000, 3000]})

        with pytest.raises(FeatureValidationError, match="Missing required features"):
            feature_adapter._validate_and_transform_features(incomplete_df, mapping, "TEST")

    def test_validate_sorts_unsorted_index(self, feature_adapter, sample_features_data):
        """Test that unsorted index is sorted."""
        strategy = MockTechnicalStrategy("test_tech")
        feature_adapter.register_strategy(strategy)
        mapping = feature_adapter.strategy_mappings["test_tech"]

        # Create DataFrame with unsorted index
        unsorted_data = sample_features_data["AAPL"].iloc[::-1]  # Reverse order

        result = feature_adapter._validate_and_transform_features(unsorted_data, mapping, "AAPL")

        assert result.index.is_monotonic_increasing


# Test Batch Processing
class TestBatchProcessing:
    """Test batch feature preparation."""

    def test_batch_prepare_features_success(self, feature_adapter, sample_features_data):
        """Test successful batch preparation."""
        # Register multiple strategies
        tech_strategy = MockTechnicalStrategy("tech_batch")
        sentiment_strategy = MockSentimentStrategy("sentiment_batch")

        feature_adapter.register_strategy(tech_strategy)
        feature_adapter.register_strategy(sentiment_strategy)

        result = feature_adapter.batch_prepare_features(
            sample_features_data, ["tech_batch", "sentiment_batch"]
        )

        assert isinstance(result, dict)
        assert "tech_batch" in result
        assert "sentiment_batch" in result

        # Both strategies should have AAPL data
        assert "AAPL" in result["tech_batch"]
        assert "AAPL" in result["sentiment_batch"]

        # Only AAPL should have sentiment features, MSFT missing sentiment data
        assert "AAPL" in result["sentiment_batch"]
        assert len(result["sentiment_batch"]["AAPL"]) > 0

    def test_batch_prepare_features_handles_errors(
        self, feature_adapter, sample_features_data, caplog
    ):
        """Test batch preparation handles individual strategy errors."""
        # Register one valid strategy and try to prepare for unregistered one
        tech_strategy = MockTechnicalStrategy("valid_strategy")
        feature_adapter.register_strategy(tech_strategy)

        result = feature_adapter.batch_prepare_features(
            sample_features_data, ["valid_strategy", "invalid_strategy"]
        )

        assert "valid_strategy" in result
        assert "invalid_strategy" in result
        assert len(result["valid_strategy"]) > 0  # Should have data
        assert len(result["invalid_strategy"]) == 0  # Should be empty due to error
        assert "Failed to prepare features for strategy invalid_strategy" in caplog.text


# Test Strategy Requirements
class TestStrategyRequirements:
    """Test strategy requirement queries."""

    def test_get_strategy_requirements_existing(self, feature_adapter):
        """Test getting requirements for existing strategy."""
        strategy = MockTechnicalStrategy("test_tech")
        feature_adapter.register_strategy(strategy)

        requirements = feature_adapter.get_strategy_requirements("test_tech")

        assert requirements is not None
        assert isinstance(requirements, StrategyFeatureMapping)
        assert requirements.strategy_name == "test_tech"

    def test_get_strategy_requirements_missing(self, feature_adapter):
        """Test getting requirements for non-existent strategy."""
        requirements = feature_adapter.get_strategy_requirements("nonexistent")
        assert requirements is None


# Test Feature Discovery
class TestFeatureDiscovery:
    """Test feature discovery functionality."""

    def test_get_available_features(self, feature_adapter, sample_features_data):
        """Test getting available features across all symbols."""
        available = feature_adapter.get_available_features(sample_features_data)

        assert isinstance(available, set)
        assert "close" in available
        assert "sma_10" in available
        assert "news_sentiment" in available
        assert "bid_ask_spread" in available

    def test_get_available_features_empty_data(self, feature_adapter):
        """Test getting available features with empty data."""
        available = feature_adapter.get_available_features({})
        assert available == set()


# Test Compatibility Validation
class TestCompatibilityValidation:
    """Test feature compatibility validation."""

    def test_validate_feature_compatibility_success(self, feature_adapter, sample_features_data):
        """Test compatibility validation with sufficient features."""
        # Register strategy that has all required features in sample data
        strategy = MockTechnicalStrategy("compatible_strategy")
        feature_adapter.register_strategy(strategy)

        report = feature_adapter.validate_feature_compatibility(sample_features_data)

        assert "compatible_strategy" in report
        assert len(report["compatible_strategy"]) == 0  # No missing features

    def test_validate_feature_compatibility_missing_features(
        self, feature_adapter, incomplete_features_data
    ):
        """Test compatibility validation with missing features."""
        strategy = MockTechnicalStrategy("incompatible_strategy")
        feature_adapter.register_strategy(strategy)

        report = feature_adapter.validate_feature_compatibility(incomplete_features_data)

        assert "incompatible_strategy" in report
        assert "close" in report["incompatible_strategy"]  # Missing required feature

    def test_validate_feature_compatibility_multiple_strategies(
        self, feature_adapter, sample_features_data
    ):
        """Test compatibility validation with multiple strategies."""
        tech_strategy = MockTechnicalStrategy("tech_strategy")
        sentiment_strategy = MockSentimentStrategy("sentiment_strategy")

        feature_adapter.register_strategy(tech_strategy)
        feature_adapter.register_strategy(sentiment_strategy)

        report = feature_adapter.validate_feature_compatibility(sample_features_data)

        assert "tech_strategy" in report
        assert "sentiment_strategy" in report

        # Tech strategy should be compatible
        assert len(report["tech_strategy"]) == 0

        # Sentiment strategy might be missing some features depending on data


# Test Cache Management
class TestCacheManagement:
    """Test cache management functionality."""

    def test_clear_cache(self, feature_adapter, caplog):
        """Test cache clearing."""
        # Add something to cache
        feature_adapter.feature_cache["test_key"] = pd.DataFrame({"test": [1, 2, 3]})

        feature_adapter.clear_cache()

        assert feature_adapter.feature_cache == {}
        assert "Feature cache cleared" in caplog.text

    def test_cache_initialization(self, feature_adapter):
        """Test that cache is properly initialized."""
        assert hasattr(feature_adapter, "feature_cache")
        assert isinstance(feature_adapter.feature_cache, dict)
        assert feature_adapter.cache_ttl > 0


# Test Factory Function
class TestFactoryFunction:
    """Test factory function for FeatureAdapter creation."""

    def test_create_feature_adapter_default(self):
        """Test creating FeatureAdapter with default config."""
        with patch("main.feature_pipeline.feature_adapter.get_config", return_value={}):
            adapter = create_feature_adapter()

            assert isinstance(adapter, FeatureAdapter)

    def test_create_feature_adapter_custom_config(self, mock_config):
        """Test creating FeatureAdapter with custom config."""
        with patch("main.feature_pipeline.feature_adapter.get_config", return_value=mock_config):
            adapter = create_feature_adapter(mock_config)

            assert isinstance(adapter, FeatureAdapter)
            assert adapter.config == mock_config


# Test Error Handling
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_feature_validation_error_inheritance(self):
        """Test that FeatureValidationError is proper exception."""
        error = FeatureValidationError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_prepare_features_continues_on_symbol_error(
        self, feature_adapter, sample_features_data, caplog
    ):
        """Test that preparation continues when individual symbols fail."""
        strategy = MockTechnicalStrategy("error_test")
        feature_adapter.register_strategy(strategy)

        # Corrupt one symbol's data
        corrupted_data = sample_features_data.copy()
        corrupted_data["CORRUPT"] = "not_a_dataframe"  # This will cause error

        result = feature_adapter.prepare_features_for_strategy(corrupted_data, "error_test")

        # Should still process valid symbols
        assert "AAPL" in result
        assert "MSFT" in result
        # Should log error for corrupted symbol
        assert "Failed to prepare features for CORRUPT" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__])

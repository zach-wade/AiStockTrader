# tests/unit/test_data_preprocessor.py

# Standard library imports
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
from main.feature_pipeline.data_preprocessor import DataPreprocessor, PreprocessorConfig


# Test Fixtures
@pytest.fixture
def sample_market_data():
    """Sample market data with some realistic patterns and issues."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz=UTC)
    np.random.seed(42)  # For reproducible tests

    base_price = 100
    price_changes = secure_numpy_normal(0, 0.02, 100)  # 2% daily volatility
    prices = [base_price]

    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.array(prices) * (1 + secure_numpy_normal(0, 0.001, 100)),
            "high": np.array(prices) * (1 + np.abs(secure_numpy_normal(0, 0.01, 100))),
            "low": np.array(prices) * (1 - np.abs(secure_numpy_normal(0, 0.01, 100))),
            "close": prices,
            "volume": np.secure_randint(10000, 100000, 100),
        }
    )

    # Introduce some data quality issues for testing
    data.loc[10, "high"] = data.loc[10, "low"] - 1  # Invalid OHLC
    data.loc[20, "volume"] = -1000  # Negative volume
    data.loc[30, "close"] = np.nan  # Missing data
    data.loc[40, "low"] = 0  # Zero price
    data.loc[50, "close"] = data.loc[49, "close"] * 2.5  # Extreme price movement

    return data


@pytest.fixture
def sample_feature_data():
    """Sample feature data with various data types and issues."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D", tz=UTC)
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * 50,
            "rsi_14": np.secure_uniform(0, 100, 50),
            "sma_20": np.secure_uniform(95, 105, 50),
            "volume_ratio": np.secure_uniform(0.5, 2.0, 50),
            "sentiment_score": np.secure_uniform(-1, 1, 50),
            "returns": secure_numpy_normal(0, 0.02, 50),
        }
    )

    # Introduce issues
    data.loc[10, "rsi_14"] = np.inf  # Infinite value
    data.loc[15, "sentiment_score"] = np.nan  # Missing value
    data.loc[20, "returns"] = 0.8  # Extreme outlier

    return data


@pytest.fixture
def sample_news_data():
    """Sample news data for alternative data testing."""
    dates = pd.date_range("2024-01-01", periods=20, freq="H", tz=UTC)

    data = pd.DataFrame(
        {
            "published_at": dates,
            "headline": [
                "Apple reports strong quarterly earnings",
                "Tech stocks rally on positive sentiment",
                "",  # Empty headline
                "Market volatility concerns investors",
                "Apple unveils new product line",
                "Short headline",  # Very short
                None,  # Null headline
                "Federal Reserve announces interest rate decision",
                "   Leading whitespace headline   ",
                "Duplicate news story",
                "Duplicate news story",  # Duplicate
                "Breaking: Major acquisition announced",
                "x" * 300,  # Very long headline
                "Stock market reaches new highs",
                "Economic indicators show growth",
                "Technology sector outperforms",
                "Apple stock price target raised",
                "Market correction expected soon",
                "Earnings season begins strong",
                "Investor confidence remains high",
            ],
            "sentiment": np.secure_uniform(-1, 1, 20),
            "symbol": ["AAPL"] * 20,
        }
    )

    return data


@pytest.fixture
def default_config():
    """Default preprocessor configuration."""
    return {
        "missing_threshold": 0.5,
        "imputation_method": "forward_fill",
        "outlier_method": "iqr",
        "outlier_threshold": 3.0,
        "outlier_action": "clip",
        "scaling_method": "robust",
        "scale_features": True,
        "min_data_points": 50,
        "max_gap_days": 7,
    }


@pytest.fixture
def preprocessor(default_config):
    """DataPreprocessor instance with default config."""
    return DataPreprocessor(default_config)


# Test Configuration
class TestPreprocessorConfig:
    """Test PreprocessorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PreprocessorConfig()

        assert config.missing_threshold == 0.5
        assert config.imputation_method == "forward_fill"
        assert config.outlier_method == "iqr"
        assert config.outlier_threshold == 3.0
        assert config.outlier_action == "clip"
        assert config.scaling_method == "robust"
        assert config.scale_features is True
        assert config.min_data_points == 50
        assert config.max_gap_days == 7
        assert config.price_columns == ["open", "high", "low", "close"]
        assert config.volume_columns == ["volume"]
        assert config.min_volume_threshold == 1000
        assert config.price_change_threshold == 0.5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PreprocessorConfig(
            missing_threshold=0.3, outlier_method="zscore", scaling_method="standard"
        )

        assert config.missing_threshold == 0.3
        assert config.outlier_method == "zscore"
        assert config.scaling_method == "standard"
        # Defaults should still be set
        assert config.imputation_method == "forward_fill"


# Test DataPreprocessor Initialization
class TestDataPreprocessorInit:
    """Test DataPreprocessor initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        preprocessor = DataPreprocessor()

        assert isinstance(preprocessor.config, PreprocessorConfig)
        assert preprocessor.config.missing_threshold == 0.5
        assert preprocessor.scalers == {}
        assert preprocessor.imputers == {}
        assert preprocessor.outlier_stats == {}

    def test_init_with_custom_config(self, default_config):
        """Test initialization with custom configuration."""
        custom_config = default_config.copy()
        custom_config["missing_threshold"] = 0.3

        preprocessor = DataPreprocessor(custom_config)

        assert preprocessor.config.missing_threshold == 0.3
        assert preprocessor.config.outlier_method == "iqr"


# Test Market Data Preprocessing
class TestMarketDataPreprocessing:
    """Test market data preprocessing functionality."""

    def test_preprocess_market_data_basic(self, preprocessor, sample_market_data):
        """Test basic market data preprocessing."""
        result = preprocessor.preprocess_market_data(sample_market_data, "AAPL")

        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        # Should have required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col in sample_market_data.columns:
                assert col in result.columns

        # Should add derived features
        assert "returns" in result.columns
        assert "log_returns" in result.columns
        assert "typical_price" in result.columns

    def test_preprocess_empty_data(self, preprocessor):
        """Test preprocessing with empty DataFrame."""
        empty_data = pd.DataFrame()
        result = preprocessor.preprocess_market_data(empty_data, "TEST")

        assert result.empty

    def test_ohlc_consistency_validation(self, preprocessor):
        """Test OHLC consistency validation and fixing."""
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC),
                "open": [100, 101, 102],
                "high": [99, 103, 104],  # First high is too low
                "low": [98, 99, 105],  # Last low is too high
                "close": [101, 102, 103],
                "volume": [1000, 1500, 2000],
            }
        )

        result = preprocessor.preprocess_market_data(data, "TEST")

        # High should be fixed to be >= max(open, close)
        assert result.loc[0, "high"] >= max(result.loc[0, "open"], result.loc[0, "close"])

        # Low should be fixed to be <= min(open, close)
        assert result.loc[2, "low"] <= min(result.loc[2, "open"], result.loc[2, "close"])

    def test_negative_price_cleaning(self, preprocessor):
        """Test removal of negative prices."""
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC),
                "open": [100, -50, 102],  # Negative price
                "high": [105, 55, 104],
                "low": [95, 45, 99],
                "close": [101, 50, 103],
                "volume": [1000, 1500, 2000],
            }
        )

        result = preprocessor.preprocess_market_data(data, "TEST")

        # Negative price should be NaN
        assert pd.isna(result.loc[1, "open"])

    def test_negative_volume_cleaning(self, preprocessor):
        """Test removal of negative volume."""
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC),
                "open": [100, 101, 102],
                "high": [105, 103, 104],
                "low": [95, 99, 99],
                "close": [101, 102, 103],
                "volume": [1000, -1500, 2000],  # Negative volume
            }
        )

        result = preprocessor.preprocess_market_data(data, "TEST")

        # Negative volume should be NaN
        assert pd.isna(result.loc[1, "volume"])

    def test_missing_timestamp_raises_error(self, preprocessor):
        """Test that missing timestamp column raises error."""
        data = pd.DataFrame({"open": [100, 101, 102], "close": [101, 102, 103]})

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocessor.preprocess_market_data(data, "TEST")

    def test_duplicate_timestamp_handling(self, preprocessor):
        """Test handling of duplicate timestamps."""
        timestamps = pd.date_range("2024-01-01", periods=2, freq="D", tz=UTC)
        data = pd.DataFrame(
            {
                "timestamp": [timestamps[0], timestamps[0], timestamps[1]],  # Duplicate
                "open": [100, 101, 102],
                "close": [101, 102, 103],
                "volume": [1000, 1500, 2000],
            }
        )

        result = preprocessor.preprocess_market_data(data, "TEST")

        # Should remove duplicates, keeping last
        assert len(result) == 2
        assert result.loc[0, "open"] == 101  # Last value for duplicate timestamp


# Test Missing Data Handling
class TestMissingDataHandling:
    """Test missing data handling strategies."""

    def test_forward_fill_imputation(self, default_config):
        """Test forward fill imputation."""
        config = default_config.copy()
        config["imputation_method"] = "forward_fill"
        preprocessor = DataPreprocessor(config)

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz=UTC),
                "value": [1.0, np.nan, np.nan, 4.0, 5.0],
            }
        )

        result = preprocessor._handle_missing_data(data, "test")

        # Should forward fill then backward fill
        assert result.loc[1, "value"] == 1.0
        assert result.loc[2, "value"] == 1.0

    def test_mean_imputation(self, default_config):
        """Test mean imputation."""
        config = default_config.copy()
        config["imputation_method"] = "mean"
        preprocessor = DataPreprocessor(config)

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz=UTC),
                "value": [1.0, np.nan, 3.0, np.nan, 5.0],
            }
        )

        result = preprocessor._handle_missing_data(data, "test")

        # Mean of [1, 3, 5] = 3
        assert result.loc[1, "value"] == 3.0
        assert result.loc[3, "value"] == 3.0

    def test_high_missing_threshold_drops_columns(self, default_config):
        """Test that columns with too much missing data are dropped."""
        config = default_config.copy()
        config["missing_threshold"] = 0.3  # 30% threshold
        preprocessor = DataPreprocessor(config)

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz=UTC),
                "good_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "bad_col": [1, np.nan, np.nan, np.nan, np.nan, np.nan, 7, 8, 9, 10],  # 50% missing
            }
        )

        result = preprocessor._handle_missing_data(data, "test")

        # bad_col should be dropped, good_col should remain
        assert "good_col" in result.columns
        assert "bad_col" not in result.columns

    @patch("main.feature_pipeline.data_preprocessor.KNNImputer")
    def test_knn_imputation(self, mock_knn, default_config):
        """Test KNN imputation."""
        config = default_config.copy()
        config["imputation_method"] = "knn"
        config["knn_neighbors"] = 3
        preprocessor = DataPreprocessor(config)

        # Mock the KNN imputer
        mock_imputer = MagicMock()
        mock_imputer.fit_transform.return_value = np.array([[1.0], [2.0], [3.0]])
        mock_knn.return_value = mock_imputer

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC),
                "value": [1.0, np.nan, 3.0],
            }
        )

        result = preprocessor._handle_missing_data(data, "test")

        # Should use KNN imputer
        mock_knn.assert_called_once_with(n_neighbors=3)
        mock_imputer.fit_transform.assert_called_once()


# Test Outlier Detection and Handling
class TestOutlierHandling:
    """Test outlier detection and handling."""

    def test_iqr_outlier_detection_clip(self, default_config):
        """Test IQR outlier detection with clipping."""
        config = default_config.copy()
        config["outlier_method"] = "iqr"
        config["outlier_action"] = "clip"
        config["outlier_threshold"] = 1.5
        preprocessor = DataPreprocessor(config)

        # Data with clear outliers
        data = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})  # 100 is an outlier

        result = preprocessor._handle_outliers(data, "test")

        # Outlier should be clipped
        assert result.loc[5, "value"] < 100
        assert result.loc[5, "value"] > 5  # But not too aggressive

    def test_zscore_outlier_detection(self, default_config):
        """Test Z-score outlier detection."""
        config = default_config.copy()
        config["outlier_method"] = "zscore"
        config["outlier_action"] = "flag"
        config["outlier_threshold"] = 2.0
        preprocessor = DataPreprocessor(config)

        # Data with clear outliers (mean=3, std≈1.58, so 10 has z-score > 4)
        data = pd.DataFrame({"value": [1, 2, 3, 4, 5, 10]})

        result = preprocessor._handle_outliers(data, "test")

        # Should add outlier flag column
        assert "value_outlier_flag" in result.columns
        assert result.loc[5, "value_outlier_flag"] is True

    def test_outlier_removal(self, default_config):
        """Test outlier removal action."""
        config = default_config.copy()
        config["outlier_method"] = "iqr"
        config["outlier_action"] = "remove"
        config["outlier_threshold"] = 1.5
        preprocessor = DataPreprocessor(config)

        data = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})  # 100 is an outlier

        result = preprocessor._handle_outliers(data, "test")

        # Outlier row should be removed
        assert len(result) < len(data)
        assert 100 not in result["value"].values

    def test_no_outlier_detection(self, default_config):
        """Test disabling outlier detection."""
        config = default_config.copy()
        config["outlier_method"] = "none"
        preprocessor = DataPreprocessor(config)

        data = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})  # 100 would be an outlier

        result = preprocessor._handle_outliers(data, "test")

        # Data should be unchanged
        pd.testing.assert_frame_equal(result, data)


# Test Feature Data Preprocessing
class TestFeatureDataPreprocessing:
    """Test feature data preprocessing."""

    def test_preprocess_feature_data_basic(self, preprocessor, sample_feature_data):
        """Test basic feature data preprocessing."""
        result = preprocessor.preprocess_feature_data(sample_feature_data)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        # Should handle infinite values
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

    def test_infinite_value_handling(self, preprocessor):
        """Test handling of infinite values."""
        data = pd.DataFrame(
            {"feature1": [1.0, 2.0, np.inf, 4.0], "feature2": [1.0, -np.inf, 3.0, 4.0]}
        )

        result = preprocessor._handle_infinite_values(data)

        # Infinite values should be NaN
        assert pd.isna(result.loc[2, "feature1"])
        assert pd.isna(result.loc[1, "feature2"])

    def test_feature_scaling(self, preprocessor):
        """Test feature scaling functionality."""
        data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]})

        result = preprocessor._scale_features(data, "test")

        # Should be scaled (roughly centered and scaled)
        assert abs(result["feature1"].mean()) < 0.1  # Approximately centered
        assert abs(result["feature2"].mean()) < 0.1  # Approximately centered

        # Check that scaler was stored
        assert "test_robust" in preprocessor.scalers


# Test Alternative Data Preprocessing
class TestAlternativeDataPreprocessing:
    """Test alternative data preprocessing."""

    def test_preprocess_news_data(self, preprocessor, sample_news_data):
        """Test news data preprocessing."""
        result = preprocessor._preprocess_news_data(sample_news_data)

        # Should remove empty/null headlines
        assert "" not in result["headline"].values
        assert not result["headline"].isna().any()

        # Should strip whitespace
        assert not any(
            headline.startswith(" ") or headline.endswith(" ") for headline in result["headline"]
        )

        # Should remove very short headlines
        assert all(len(headline) > 5 for headline in result["headline"])

        # Should parse timestamps
        assert pd.api.types.is_datetime64_any_dtype(result["published_at"])

    def test_preprocess_social_data(self, preprocessor):
        """Test social media data preprocessing."""
        data = pd.DataFrame(
            {
                "text": [
                    "Good post about stocks",
                    "Duplicate",
                    "Duplicate",
                    "Short",
                    "Another good post",
                ],
                "likes": ["10", "20", None, "5", "invalid"],
                "shares": [5, 10, 15, 0, 25],
            }
        )

        result = preprocessor._preprocess_social_data(data)

        # Should remove duplicates
        assert len(result) < len(data)
        assert result["text"].duplicated().sum() == 0

        # Should convert engagement metrics to numeric
        assert pd.api.types.is_numeric_dtype(result["likes"])
        assert pd.api.types.is_numeric_dtype(result["shares"])

        # Should fill NaN with 0
        assert not result["likes"].isna().any()


# Test Data Quality Validation
class TestDataQualityValidation:
    """Test data quality validation."""

    def test_insufficient_data_points_warning(self, preprocessor, caplog):
        """Test warning for insufficient data points."""
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz=UTC),
                "close": range(10),
            }
        )

        # min_data_points is 50 by default
        result = preprocessor._validate_data_quality(data, "TEST")

        assert "Insufficient data points" in caplog.text

    def test_large_time_gaps_warning(self, preprocessor, caplog):
        """Test warning for large gaps in time series."""
        timestamps = [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 15, tzinfo=UTC),  # 13-day gap
        ]

        data = pd.DataFrame({"timestamp": timestamps, "close": [100, 101, 102]})

        result = preprocessor._validate_data_quality(data, "TEST")

        assert "gaps >" in caplog.text

    def test_extreme_price_movements_warning(self, preprocessor, caplog):
        """Test warning for extreme price movements."""
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC),
                "close": [100, 200, 300],  # 100% and 50% daily moves
            }
        )

        result = preprocessor._validate_data_quality(data, "TEST")

        assert "extreme price movements" in caplog.text


# Test Utility Methods
class TestUtilityMethods:
    """Test utility and helper methods."""

    def test_get_preprocessing_summary(self, preprocessor):
        """Test preprocessing summary generation."""
        # Do some preprocessing to populate state
        data = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
        preprocessor._scale_features(data, "test")

        summary = preprocessor.get_preprocessing_summary()

        assert "config" in summary
        assert "fitted_scalers" in summary
        assert "fitted_imputers" in summary
        assert "outlier_stats_available" in summary
        assert isinstance(summary["config"], dict)
        assert "test_robust" in summary["fitted_scalers"]

    def test_reset_fitted_components(self, preprocessor):
        """Test resetting fitted components."""
        # Add some fitted components
        data = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
        preprocessor._scale_features(data, "test")

        assert len(preprocessor.scalers) > 0

        # Reset
        preprocessor.reset_fitted_components()

        assert len(preprocessor.scalers) == 0
        assert len(preprocessor.imputers) == 0
        assert len(preprocessor.outlier_stats) == 0


# Test Error Handling
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_scaling_with_no_numeric_columns(self, preprocessor):
        """Test scaling when there are no numeric columns."""
        data = pd.DataFrame(
            {
                "text_col": ["a", "b", "c"],
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC),
            }
        )

        result = preprocessor._scale_features(data, "test")

        # Should return unchanged data
        pd.testing.assert_frame_equal(result, data)

    def test_unknown_scaling_method(self, default_config, caplog):
        """Test handling of unknown scaling method."""
        config = default_config.copy()
        config["scaling_method"] = "unknown_method"
        preprocessor = DataPreprocessor(config)

        data = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
        result = preprocessor._scale_features(data, "test")

        # Should log warning and return unchanged data
        assert "Unknown scaling method" in caplog.text
        pd.testing.assert_frame_equal(result, data)

    def test_knn_imputation_fallback(self, default_config, caplog):
        """Test KNN imputation fallback to forward fill."""
        config = default_config.copy()
        config["imputation_method"] = "knn"

        with patch("main.feature_pipeline.data_preprocessor.KNNImputer") as mock_knn:
            # Make KNN imputer raise an exception
            mock_knn.return_value.fit_transform.side_effect = Exception("KNN failed")

            preprocessor = DataPreprocessor(config)
            data = pd.DataFrame({"value": [1.0, np.nan, 3.0]})

            result = preprocessor._apply_knn_imputation(data, "test")

            # Should fallback to forward fill
            assert "KNN imputation failed" in caplog.text
            assert not result["value"].isna().any()  # Should be filled


if __name__ == "__main__":
    pytest.main([__file__])

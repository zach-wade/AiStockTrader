"""
Unit tests for SectorAnalyticsCalculator

Tests cover:
- Configuration initialization and validation
- Data preprocessing and validation
- Individual feature calculation methods
- Error handling and edge cases
- Output feature validation and structure
"""

# Standard library imports
import logging
import unittest
from unittest.mock import patch

# Third-party imports
import numpy as np
import pandas as pd

# Disable logging during tests
logging.disable(logging.CRITICAL)

# Standard library imports
# Import the module directly without going through the package
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

try:
    # Local imports
    from main.feature_pipeline.calculators.sector_analytics import (
        SectorAnalyticsCalculator,
        SectorConfig,
    )
except ImportError:
    # Fallback to direct file import for testing
    # Standard library imports
    import importlib.util

    calculator_path = os.path.join(
        os.path.dirname(__file__), "../../src/main/feature_pipeline/calculators/sector_analytics.py"
    )
    spec = importlib.util.spec_from_file_location("sector_analytics", calculator_path)
    sector_analytics = importlib.util.module_from_spec(spec)

    # Mock the base calculator for testing
    class MockBaseFeatureCalculator:
        def __init__(self, config=None):
            pass

        def preprocess_data(self, data):
            return data

        def postprocess_features(self, features):
            return features

    # Inject mock into the module
    sys.modules["main.feature_pipeline.calculators.base_calculator"] = type(
        "module", (), {"BaseFeatureCalculator": MockBaseFeatureCalculator}
    )()

    try:
        spec.loader.exec_module(sector_analytics)
        SectorAnalyticsCalculator = sector_analytics.SectorAnalyticsCalculator
        SectorConfig = sector_analytics.SectorConfig
    except Exception as e:
        # Skip tests if we can't import
        # Third-party imports
        import pytest

        pytest.skip(f"Could not import sector_analytics: {e}")


class TestSectorConfig(unittest.TestCase):
    """Test SectorConfig dataclass"""

    def test_default_initialization(self):
        """Test default config initialization"""
        config = SectorConfig()

        # Check default values are set correctly
        self.assertIsNotNone(config.sector_etfs)
        self.assertIsNotNone(config.momentum_windows)
        self.assertIsNotNone(config.cyclical_sectors)
        self.assertIsNotNone(config.defensive_sectors)

        # Check specific defaults
        self.assertEqual(config.momentum_windows, [5, 20, 60])
        self.assertEqual(config.correlation_window, 60)
        self.assertEqual(config.rotation_threshold, 0.05)
        self.assertEqual(config.breadth_threshold, 0.6)

        # Check sector mappings
        self.assertIn("technology", config.sector_etfs)
        self.assertEqual(config.sector_etfs["technology"], "XLK")

    def test_custom_initialization(self):
        """Test custom config initialization"""
        custom_etfs = {"tech": "QQQ", "finance": "XLF"}
        config = SectorConfig(
            sector_etfs=custom_etfs, momentum_windows=[10, 30], rotation_threshold=0.1
        )

        self.assertEqual(config.sector_etfs, custom_etfs)
        self.assertEqual(config.momentum_windows, [10, 30])
        self.assertEqual(config.rotation_threshold, 0.1)


class TestSectorAnalyticsCalculator(unittest.TestCase):
    """Test SectorAnalyticsCalculator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "sector": {
                "momentum_windows": [5, 20],
                "correlation_window": 30,
                "rotation_threshold": 0.05,
            }
        }
        self.calculator = SectorAnalyticsCalculator(self.config)

        # Create sample market data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)  # For reproducible tests

        self.sample_data = pd.DataFrame(
            {
                "open": 100 + np.random.randn(100).cumsum(),
                "high": 102 + np.random.randn(100).cumsum(),
                "low": 98 + np.random.randn(100).cumsum(),
                "close": 100 + np.random.randn(100).cumsum(),
                "volume": 1000000 + np.secure_randint(-100000, 100000, 100),
            },
            index=dates,
        )

        # Create sample sector data
        self.sample_sector_data = pd.DataFrame(
            {
                "open": 150 + np.random.randn(100).cumsum(),
                "high": 152 + np.random.randn(100).cumsum(),
                "low": 148 + np.random.randn(100).cumsum(),
                "close": 150 + np.random.randn(100).cumsum(),
                "volume": 2000000 + np.secure_randint(-200000, 200000, 100),
            },
            index=dates,
        )

    def test_initialization(self):
        """Test calculator initialization"""
        self.assertIsInstance(self.calculator.sector_config, SectorConfig)
        self.assertEqual(self.calculator.sector_config.momentum_windows, [5, 20])
        self.assertEqual(self.calculator.sector_config.correlation_window, 30)
        self.assertIsInstance(self.calculator.sector_data, dict)
        self.assertIsInstance(self.calculator.industry_constituents, dict)

    def test_validate_inputs_valid_data(self):
        """Test input validation with valid data"""
        result = self.calculator.validate_inputs(self.sample_data)
        self.assertTrue(result)

    def test_validate_inputs_missing_columns(self):
        """Test input validation with missing required columns"""
        invalid_data = pd.DataFrame({"open": [1, 2, 3]})
        result = self.calculator.validate_inputs(invalid_data)
        self.assertFalse(result)

    def test_validate_inputs_empty_data(self):
        """Test input validation with empty DataFrame"""
        empty_data = pd.DataFrame()
        result = self.calculator.validate_inputs(empty_data)
        self.assertFalse(result)

    def test_get_required_columns(self):
        """Test getting required input columns"""
        required = self.calculator.get_required_columns()
        expected = ["open", "high", "low", "close", "volume"]
        self.assertEqual(required, expected)

    def test_get_feature_names(self):
        """Test getting feature names"""
        feature_names = self.calculator.get_feature_names()

        # Check we get a list
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)

        # Check for expected feature categories
        momentum_features = [f for f in feature_names if "momentum" in f]
        self.assertGreater(len(momentum_features), 0)

        rotation_features = [f for f in feature_names if "rotation" in f]
        self.assertGreater(len(rotation_features), 0)

        cycle_features = [f for f in feature_names if "cycle" in f]
        self.assertGreater(len(cycle_features), 0)

        # Check specific expected features
        self.assertIn("sector_relative_strength_5d", feature_names)
        self.assertIn("sector_momentum_5d", feature_names)
        self.assertIn("cyclical_defensive_ratio", feature_names)

    def test_preprocess_data_basic(self):
        """Test basic data preprocessing"""
        processed = self.calculator.preprocess_data(self.sample_data.copy())

        # Should return DataFrame with same structure
        self.assertIsInstance(processed, pd.DataFrame)
        pd.testing.assert_index_equal(processed.index, self.sample_data.index)
        self.assertEqual(list(processed.columns), list(self.sample_data.columns))

    def test_preprocess_data_with_sector_data(self):
        """Test preprocessing with sector data"""
        # Add sector data
        self.calculator.set_sector_data("technology", self.sample_sector_data.copy())

        processed = self.calculator.preprocess_data(self.sample_data.copy())

        # Should still return valid DataFrame
        self.assertIsInstance(processed, pd.DataFrame)
        # Sector data should be filtered/processed
        self.assertIn("technology", self.calculator.sector_data)

    def test_postprocess_features(self):
        """Test feature postprocessing"""
        # Create test features with various data types
        features = pd.DataFrame(
            {
                "ratio_feature": [1.0, np.inf, -np.inf, 0.5],
                "score_feature": [0.8, 1.5, -0.2, 0.5],
                "correlation_feature": [0.7, 1.2, -1.5, 0.3],
                "rotation_feature": [1, 0, 1, 0],
            },
            index=self.sample_data.index[:4],
        )

        processed = self.calculator.postprocess_features(features)

        # Check infinite values are handled
        self.assertFalse(np.isinf(processed.values).any())

        # Check score features are clipped to [0, 1]
        score_col = processed["score_feature"]
        self.assertTrue((score_col >= 0).all())
        self.assertTrue((score_col <= 1).all())

        # Check correlation features are clipped to [-1, 1]
        corr_col = processed["correlation_feature"]
        self.assertTrue((corr_col >= -1).all())
        self.assertTrue((corr_col <= 1).all())

        # Check rotation features are integers
        rotation_col = processed["rotation_feature"]
        self.assertTrue(rotation_col.dtype in [int, "int64"])

    def test_calculate_empty_data(self):
        """Test calculation with empty data"""
        empty_data = pd.DataFrame()
        result = self.calculator.calculate(empty_data)

        # Should return empty DataFrame with proper structure
        self.assertIsInstance(result, pd.DataFrame)

    def test_calculate_no_sector_data(self):
        """Test calculation with no sector data"""
        result = self.calculator.calculate(self.sample_data)

        # Should return DataFrame with basic structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_data))

    def test_calculate_with_sector_data(self):
        """Test calculation with sector data"""
        # Add some sector data
        self.calculator.set_sector_data("technology", self.sample_sector_data.copy())
        self.calculator.set_sector_data("financials", self.sample_sector_data.copy())

        result = self.calculator.calculate(self.sample_data)

        # Should return DataFrame with features
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_data))
        self.assertGreater(len(result.columns), 0)

        # Check for specific feature categories
        columns = result.columns.tolist()

        # Should have some correlation features
        corr_features = [col for col in columns if "corr_" in col]
        if len(self.calculator.sector_data) > 0:
            self.assertGreater(len(corr_features), 0)

    def test_validate_sector_data_empty(self):
        """Test sector data validation with empty data"""
        result = self.calculator._validate_sector_data()
        self.assertFalse(result)

    def test_validate_sector_data_valid(self):
        """Test sector data validation with valid data"""
        self.calculator.set_sector_data("technology", self.sample_sector_data.copy())
        result = self.calculator._validate_sector_data()
        self.assertTrue(result)

    def test_validate_sector_data_invalid(self):
        """Test sector data validation with invalid data"""
        # Add invalid sector data (missing close column)
        invalid_data = pd.DataFrame({"open": [1, 2, 3]})
        self.calculator.set_sector_data("technology", invalid_data)
        result = self.calculator._validate_sector_data()
        self.assertFalse(result)

    def test_create_empty_features(self):
        """Test creating empty features structure"""
        index = pd.date_range("2023-01-01", periods=10)
        result = self.calculator._create_empty_features(index)

        # Should return DataFrame with proper structure
        self.assertIsInstance(result, pd.DataFrame)
        pd.testing.assert_index_equal(result.index, index)

        # Should have all expected feature columns
        expected_features = self.calculator.get_feature_names()
        self.assertEqual(list(result.columns), expected_features)

        # Should have appropriate default values
        ratio_cols = [col for col in result.columns if "ratio" in col]
        for col in ratio_cols:
            self.assertTrue((result[col] == 0.0).all())

        score_cols = [col for col in result.columns if "score" in col]
        for col in score_cols:
            self.assertTrue((result[col] == 0.5).all())

    def test_validate_output_features(self):
        """Test output feature validation"""
        # Create features with various issues
        features = pd.DataFrame(
            {
                "good_feature": [1.0, 2.0, 3.0],
                "empty_feature": [np.nan, np.nan, np.nan],
                "inf_feature": [1.0, np.inf, -np.inf],
                "text_feature": ["a", "b", "c"],
            }
        )

        result = self.calculator._validate_output_features(features)

        # Should handle all issues
        self.assertIsInstance(result, pd.DataFrame)

        # Empty feature should be filled
        self.assertFalse(result["empty_feature"].isna().all())

        # Infinite values should be handled
        self.assertFalse(np.isinf(result["inf_feature"]).any())

        # Text feature should be converted to numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(result["text_feature"]))

    def test_set_sector_data(self):
        """Test setting sector data"""
        sector_data = self.sample_sector_data.copy()
        self.calculator.set_sector_data("test_sector", sector_data)

        self.assertIn("test_sector", self.calculator.sector_data)
        pd.testing.assert_frame_equal(self.calculator.sector_data["test_sector"], sector_data)

    def test_set_market_index(self):
        """Test setting market index data"""
        market_data = self.sample_data.copy()
        self.calculator.set_market_index(market_data)

        pd.testing.assert_frame_equal(self.calculator.market_index, market_data)

    def test_set_industry_constituents(self):
        """Test setting industry constituents"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        self.calculator.set_industry_constituents("technology", symbols)

        self.assertIn("technology", self.calculator.industry_constituents)
        self.assertEqual(self.calculator.industry_constituents["technology"], symbols)

    def test_get_sector_for_industry(self):
        """Test industry to sector mapping"""
        # Test known mappings
        self.assertEqual(self.calculator._get_sector_for_industry("software"), "technology")
        self.assertEqual(self.calculator._get_sector_for_industry("banks"), "financials")

        # Test unknown industry
        self.assertIsNone(self.calculator._get_sector_for_industry("unknown_industry"))

        # Test case insensitive
        self.assertEqual(self.calculator._get_sector_for_industry("SOFTWARE"), "technology")

    def test_feature_method_error_handling(self):
        """Test that individual feature method errors don't crash calculation"""
        # Mock one of the feature methods to raise an exception
        self.calculator.set_sector_data("technology", self.sample_sector_data.copy())

        with patch.object(
            self.calculator, "_add_sector_performance", side_effect=Exception("Test error")
        ):
            result = self.calculator.calculate(self.sample_data)

            # Should still return a DataFrame (other methods should work)
            self.assertIsInstance(result, pd.DataFrame)
            # Should log the error but continue


class TestSectorAnalyticsIntegration(unittest.TestCase):
    """Integration tests for SectorAnalyticsCalculator"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.calculator = SectorAnalyticsCalculator()

        # Create more realistic test data
        dates = pd.date_range("2023-01-01", periods=252, freq="D")  # 1 year
        np.random.seed(42)

        # Main symbol data with trend
        price_trend = np.linspace(100, 110, 252) + np.random.randn(252) * 2
        self.symbol_data = pd.DataFrame(
            {
                "open": price_trend + np.random.randn(252) * 0.5,
                "high": price_trend + np.random.randn(252) * 0.5 + 1,
                "low": price_trend + np.random.randn(252) * 0.5 - 1,
                "close": price_trend,
                "volume": 1000000 + np.secure_randint(-200000, 200000, 252),
            },
            index=dates,
        )

        # Technology sector data (higher growth)
        tech_trend = np.linspace(200, 240, 252) + np.random.randn(252) * 3
        self.tech_data = pd.DataFrame(
            {
                "open": tech_trend + np.random.randn(252) * 0.5,
                "high": tech_trend + np.random.randn(252) * 0.5 + 2,
                "low": tech_trend + np.random.randn(252) * 0.5 - 2,
                "close": tech_trend,
                "volume": 5000000 + np.secure_randint(-500000, 500000, 252),
            },
            index=dates,
        )

        # Financial sector data (lower growth)
        fin_trend = np.linspace(80, 85, 252) + np.random.randn(252) * 1.5
        self.fin_data = pd.DataFrame(
            {
                "open": fin_trend + np.random.randn(252) * 0.3,
                "high": fin_trend + np.random.randn(252) * 0.3 + 0.5,
                "low": fin_trend + np.random.randn(252) * 0.3 - 0.5,
                "close": fin_trend,
                "volume": 3000000 + np.secure_randint(-300000, 300000, 252),
            },
            index=dates,
        )

        # Set up calculator with sector data
        self.calculator.set_sector_data("technology", self.tech_data)
        self.calculator.set_sector_data("financials", self.fin_data)
        self.calculator.set_market_index(self.symbol_data)  # Use symbol as market proxy

    def test_full_calculation_pipeline(self):
        """Test complete feature calculation pipeline"""
        result = self.calculator.calculate(self.symbol_data)

        # Basic validation
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.symbol_data))
        self.assertGreater(len(result.columns), 0)

        # Check for key feature categories
        columns = result.columns.tolist()

        # Correlation features
        corr_features = [col for col in columns if "corr_" in col]
        self.assertGreater(len(corr_features), 0)

        # Momentum features
        momentum_features = [col for col in columns if "momentum" in col]
        self.assertGreater(len(momentum_features), 0)

        # Rotation features
        rotation_features = [col for col in columns if "rotation" in col]
        self.assertGreater(len(rotation_features), 0)

        # Validate feature ranges
        for col in columns:
            values = result[col].dropna()
            if len(values) > 0:
                # No infinite values
                self.assertFalse(np.isinf(values).any(), f"Column {col} has infinite values")

                # Correlation features should be in [-1, 1]
                if "correlation" in col:
                    self.assertTrue(values.between(-1, 1).all(), f"Correlation {col} out of range")

                # Score features should be in [0, 1]
                if "score" in col:
                    self.assertTrue(values.between(0, 1).all(), f"Score {col} out of range")

    def test_feature_consistency(self):
        """Test that features are calculated consistently"""
        # Calculate features twice
        result1 = self.calculator.calculate(self.symbol_data)
        result2 = self.calculator.calculate(self.symbol_data)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_feature_completeness(self):
        """Test that all expected features are calculated"""
        result = self.calculator.calculate(self.symbol_data)
        expected_features = self.calculator.get_feature_names()

        # Should have most expected features (allowing for some that might not
        # be calculated due to insufficient data)
        calculated_features = set(result.columns)
        expected_set = set(expected_features)

        # At least 70% of expected features should be present
        overlap = len(calculated_features.intersection(expected_set))
        coverage = overlap / len(expected_set)
        self.assertGreater(coverage, 0.7, f"Only {coverage:.1%} of expected features calculated")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

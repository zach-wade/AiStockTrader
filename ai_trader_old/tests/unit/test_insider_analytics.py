"""
Unit tests for InsiderAnalyticsCalculator

Tests the insider trading feature calculation with various scenarios
including edge cases and data validation.
"""

# Standard library imports
import os
import sys
import unittest

# Third-party imports
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

# Local imports
from main.feature_pipeline.calculators.insider_analytics import (
    InsiderAnalyticsCalculator,
    InsiderConfig,
)


class TestInsiderAnalyticsCalculator(unittest.TestCase):
    """Test suite for InsiderAnalyticsCalculator"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "insider": {
                "min_transaction_value": 10000,
                "lookback_windows": [30, 90],
                "cluster_window": 7,
                "bullish_ratio": 2.0,
                "bearish_ratio": 0.5,
            }
        }

        self.calculator = InsiderAnalyticsCalculator(self.config)

        # Create sample market data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        self.market_data = pd.DataFrame(
            {
                "open": np.secure_uniform(100, 110, len(dates)),
                "high": np.secure_uniform(110, 120, len(dates)),
                "low": np.secure_uniform(90, 100, len(dates)),
                "close": np.secure_uniform(100, 110, len(dates)),
                "volume": np.secure_randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        # Create sample insider data
        self.insider_data = self._create_sample_insider_data(dates)

    def _create_sample_insider_data(self, dates):
        """Create sample insider transaction data"""
        transactions = []

        for i in range(50):  # 50 transactions
            date = np.secure_choice(dates)
            insider_name = f"Insider_{i % 10}"  # 10 different insiders
            title = np.secure_choice(["CEO", "CFO", "Director", "EVP"])
            transaction_type = np.secure_choice(["Buy", "Sell"])
            shares = np.secure_randint(100, 10000)
            price = np.secure_uniform(100, 110)
            value = shares * price

            transactions.append(
                {
                    "transaction_date": date,
                    "insider_name": insider_name,
                    "title": title,
                    "transaction_type": transaction_type,
                    "shares": shares,
                    "price": price,
                    "value": value,
                }
            )

        return pd.DataFrame(transactions)

    def test_initialization(self):
        """Test calculator initialization"""
        self.assertIsInstance(self.calculator.insider_config, InsiderConfig)
        self.assertEqual(self.calculator.insider_config.min_transaction_value, 10000)
        self.assertEqual(self.calculator.insider_config.lookback_windows, [30, 90])

    def test_validate_inputs(self):
        """Test input validation"""
        # Valid data
        self.assertTrue(self.calculator.validate_inputs(self.market_data))

        # Missing required columns
        invalid_data = self.market_data.drop("close", axis=1)
        self.assertFalse(self.calculator.validate_inputs(invalid_data))

        # Empty data
        empty_data = pd.DataFrame()
        self.assertFalse(self.calculator.validate_inputs(empty_data))

    def test_get_required_columns(self):
        """Test required columns specification"""
        required_cols = self.calculator.get_required_columns()
        expected_cols = ["open", "high", "low", "close", "volume"]
        self.assertEqual(set(required_cols), set(expected_cols))

    def test_get_feature_names(self):
        """Test feature names generation"""
        feature_names = self.calculator.get_feature_names()

        # Check some expected features
        self.assertIn("insider_sentiment_score", feature_names)
        self.assertIn("insider_buy_count_30d", feature_names)
        self.assertIn("insider_cluster_score", feature_names)
        self.assertIn("ceo_cfo_net_activity", feature_names)

        # Check that all features are strings
        self.assertTrue(all(isinstance(name, str) for name in feature_names))

        # Check for no duplicates
        self.assertEqual(len(feature_names), len(set(feature_names)))

    def test_set_insider_data(self):
        """Test setting insider data"""
        self.calculator.set_insider_data(self.insider_data)
        self.assertIsNotNone(self.calculator.insider_data)
        self.assertEqual(len(self.calculator.insider_data), len(self.insider_data))

    def test_validate_insider_data(self):
        """Test insider data validation"""
        # Valid data
        self.calculator.set_insider_data(self.insider_data)
        self.assertTrue(self.calculator._validate_insider_data())

        # Missing required columns
        invalid_insider = self.insider_data.drop("transaction_type", axis=1)
        self.calculator.set_insider_data(invalid_insider)
        self.assertFalse(self.calculator._validate_insider_data())

        # No valid transaction types
        invalid_types = self.insider_data.copy()
        invalid_types["transaction_type"] = "Invalid"
        self.calculator.set_insider_data(invalid_types)
        self.assertFalse(self.calculator._validate_insider_data())

        # No positive values
        invalid_values = self.insider_data.copy()
        invalid_values["value"] = -100
        self.calculator.set_insider_data(invalid_values)
        self.assertFalse(self.calculator._validate_insider_data())

    def test_calculate_with_insider_data(self):
        """Test feature calculation with insider data"""
        self.calculator.set_insider_data(self.insider_data)
        features = self.calculator.calculate(self.market_data.head(100))

        # Check that features are returned
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 100)
        self.assertGreater(len(features.columns), 0)

        # Check for expected feature columns
        self.assertIn("insider_sentiment_score", features.columns)
        self.assertIn("insider_buy_count_30d", features.columns)

        # Check data types
        for col in features.columns:
            if "count" in col or any(x in col for x in ["bullish", "bearish", "buying", "selling"]):
                self.assertTrue(features[col].dtype in [np.int64, np.int32, np.float64])

    def test_calculate_without_insider_data(self):
        """Test feature calculation without insider data"""
        features = self.calculator.calculate(self.market_data.head(100))

        # Should return empty features with default values
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 100)

        # All features should have default values (mostly 0)
        feature_names = self.calculator.get_feature_names()
        for feature_name in feature_names:
            if feature_name in features.columns:
                self.assertTrue(
                    (features[feature_name] == 0).all() or features[feature_name].isna().all()
                )

    def test_calculate_empty_data(self):
        """Test calculation with empty market data"""
        empty_data = pd.DataFrame()
        features = self.calculator.calculate(empty_data)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertTrue(features.empty)

    def test_preprocess_data(self):
        """Test data preprocessing"""
        self.calculator.set_insider_data(self.insider_data)

        # Add some NaN values to test handling
        data_with_nans = self.market_data.copy()
        data_with_nans.loc[data_with_nans.index[10], "close"] = np.nan

        processed_data = self.calculator.preprocess_data(data_with_nans)

        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(len(processed_data), len(data_with_nans))

        # Check that insider data was filtered to relevant time range
        self.assertIsNotNone(self.calculator.insider_data)

    def test_postprocess_features(self):
        """Test feature postprocessing"""
        # Create features with problematic values
        test_features = pd.DataFrame(
            {
                "insider_sentiment_score": [0.5, -2.0, 3.0, np.nan],
                "insider_bs_ratio_30d": [1.5, np.inf, -np.inf, 0.5],
                "insider_buy_count_30d": [1, 2, np.nan, 0],
                "insider_bullish_30d": [1, 0, np.nan, 1],
            }
        )

        processed_features = self.calculator.postprocess_features(test_features)

        # Check sentiment score clipping
        self.assertTrue((processed_features["insider_sentiment_score"] >= -1).all())
        self.assertTrue((processed_features["insider_sentiment_score"] <= 1).all())

        # Check infinite value handling
        self.assertFalse(np.isinf(processed_features["insider_bs_ratio_30d"]).any())

        # Check NaN handling for count features
        self.assertEqual(processed_features["insider_buy_count_30d"].fillna(0).sum(), 3)

        # Check binary features
        binary_values = processed_features["insider_bullish_30d"].dropna()
        self.assertTrue(binary_values.isin([0, 1]).all())

    def test_transaction_metrics(self):
        """Test transaction metrics calculation"""
        self.calculator.set_insider_data(self.insider_data)
        features = pd.DataFrame(index=self.market_data.index[:50])

        result = self.calculator._add_transaction_metrics(self.market_data.head(50), features)

        # Check that transaction metric columns are added
        expected_cols = ["insider_buy_count_30d", "insider_sell_count_30d", "insider_net_count_30d"]
        for col in expected_cols:
            self.assertIn(col, result.columns)

        # Check data types
        for col in expected_cols:
            self.assertTrue(result[col].dtype in [np.int64, np.int32, np.float64])

    def test_sentiment_calculation(self):
        """Test sentiment calculation"""
        # Create features with transaction data
        features = pd.DataFrame(
            {
                "insider_net_count_30d": [1, -1, 0, 2, -2],
                "insider_net_value_30d": [10000, -5000, 0, 20000, -15000],
            }
        )

        result = self.calculator._add_insider_sentiment(self.market_data.head(5), features)

        self.assertIn("insider_sentiment_score", result.columns)
        self.assertIn("insider_sentiment_strength", result.columns)

        # Check sentiment score range
        sentiment_scores = result["insider_sentiment_score"].dropna()
        if not sentiment_scores.empty:
            self.assertTrue((sentiment_scores >= -1).all())
            self.assertTrue((sentiment_scores <= 1).all())

    def test_moving_averages(self):
        """Test moving average calculation"""
        features = pd.DataFrame(
            {
                "insider_sentiment_score": np.secure_uniform(-1, 1, 100),
                "insider_net_count_30d": np.secure_randint(-5, 5, 100),
            },
            index=self.market_data.index[:100],
        )

        result = self.calculator._add_moving_averages(features)

        # Check that moving average columns are added
        expected_ma_cols = [
            "insider_sentiment_ma_30",
            "insider_sentiment_ma_90",
            "insider_activity_ma_30",
        ]
        for col in expected_ma_cols:
            self.assertIn(col, result.columns)

    def test_feature_interactions(self):
        """Test feature interaction calculation"""
        features = pd.DataFrame(
            {
                "insider_sentiment_score": [0.5, -0.3, 0.8, 0.0],
                "insider_conviction_ratio": [2.0, 1.5, 3.0, 1.0],
                "executive_net_trades_30d": [1, -1, 2, 0],
                "insider_cluster_score": [3, -2, 0, 1],
                "insider_performance_score": [0.1, -0.05, 0.15, 0.0],
            }
        )

        result = self.calculator._add_feature_interactions(features)

        # Check that interaction columns are added
        expected_interaction_cols = [
            "sentiment_conviction_product",
            "executive_sentiment_alignment",
            "cluster_performance_combo",
        ]
        for col in expected_interaction_cols:
            self.assertIn(col, result.columns)

    def test_error_handling(self):
        """Test error handling in calculation"""
        # Test with corrupted insider data
        corrupted_data = pd.DataFrame({"invalid": [1, 2, 3]})
        self.calculator.set_insider_data(corrupted_data)

        # Should not raise exception but return default features
        features = self.calculator.calculate(self.market_data.head(10))
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 10)

    def test_performance_monitoring(self):
        """Test that calculation completes within reasonable time"""
        # Standard library imports
        import time

        self.calculator.set_insider_data(self.insider_data)

        start_time = time.time()
        features = self.calculator.calculate(self.market_data.head(100))
        end_time = time.time()

        # Should complete within 10 seconds for 100 data points
        self.assertLess(end_time - start_time, 10.0)
        self.assertGreater(len(features.columns), 0)


if __name__ == "__main__":
    unittest.main()

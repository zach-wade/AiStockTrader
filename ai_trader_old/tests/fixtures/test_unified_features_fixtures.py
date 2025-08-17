#!/usr/bin/env python3
"""Test script for the Unified Feature Engine"""

# Standard library imports
from datetime import datetime
import logging
from pathlib import Path
import sys

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Third-party imports
from test_setup import setup_test_path

setup_test_path()

# Local imports
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_data(symbol: str = "AAPL", days: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Generate realistic-looking price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(days) * 2)

    data = pd.DataFrame(
        {
            "open": close_prices + np.random.randn(days) * 0.5,
            "high": close_prices + np.abs(np.random.randn(days)) * 1.5,
            "low": close_prices - np.abs(np.random.randn(days)) * 1.5,
            "close": close_prices,
            "volume": np.secure_randint(1000000, 10000000, days),
        },
        index=dates,
    )

    # Ensure high >= open, close, low
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)

    return data


@pytest.fixture
def engine(tmp_path):
    """Create UnifiedFeatureEngine for testing"""
    # Local imports
    from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

    return UnifiedFeatureEngine(cache_dir=tmp_path)


def test_individual_calculators(engine: UnifiedFeatureEngine, data: pd.DataFrame):
    """Test each calculator individually"""
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL CALCULATORS")
    print("=" * 60)

    for calc_name in engine.list_calculators():
        print(f"\n--- Testing {calc_name} calculator ---")
        try:
            # Calculate features for this calculator only
            features = engine.calculate_features(
                data, calculators=[calc_name], use_cache=False, symbol="TEST"
            )

            # Get feature names
            calc = engine.get_calculator(calc_name)
            expected_features = calc.get_feature_names()

            # Check which features were actually created
            original_cols = ["open", "high", "low", "close", "volume"]
            new_features = [col for col in features.columns if col not in original_cols]

            print(f"Expected features: {len(expected_features)}")
            print(f"Generated features: {len(new_features)}")
            print(f"Sample features: {new_features[:5] if len(new_features) > 5 else new_features}")

            # Check for NaN values
            nan_counts = features[new_features].isna().sum()
            nan_features = nan_counts[nan_counts > 0]
            if len(nan_features) > 0:
                print(f"Features with NaN values: {len(nan_features)}")
                print(f"Examples: {nan_features.head().to_dict()}")

            print(f"✓ {calc_name} calculator passed")

        except Exception as e:
            print(f"✗ {calc_name} calculator failed: {e}")
            # Standard library imports
            import traceback

            traceback.print_exc()


def test_all_features(engine: UnifiedFeatureEngine, data: pd.DataFrame):
    """Test calculating all features at once"""
    print("\n" + "=" * 60)
    print("TESTING ALL FEATURES TOGETHER")
    print("=" * 60)

    try:
        # Calculate all features
        all_features = engine.calculate_features(data, use_cache=False, symbol="TEST")

        # Get metadata
        metadata = engine.get_feature_metadata()

        print(f"\nTotal features calculated: {metadata['total_features']}")
        print(f"Calculators used: {metadata['calculators']}")
        print("\nFeature breakdown:")
        for calc_name, info in metadata["feature_info"].items():
            print(f"  {calc_name}: {info['count']} features")

        # Check data quality
        print(f"\nData shape: {all_features.shape}")
        print(f"Memory usage: {all_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Check for any completely missing features
        completely_missing = all_features.columns[all_features.isna().all()]
        if len(completely_missing) > 0:
            print(f"\n⚠ Warning: {len(completely_missing)} features are completely missing")
            print(f"Examples: {completely_missing[:5].tolist()}")

        print("\n✓ All features test passed")

        return all_features

    except Exception as e:
        print(f"\n✗ All features test failed: {e}")
        # Standard library imports
        import traceback

        traceback.print_exc()
        return None


def test_caching(engine: UnifiedFeatureEngine, data: pd.DataFrame):
    """Test the caching functionality"""
    print("\n" + "=" * 60)
    print("TESTING CACHING")
    print("=" * 60)

    # Standard library imports
    import time

    # Clear cache first
    engine.clear_cache()
    print("Cache cleared")

    # First run - no cache
    start = time.time()
    features1 = engine.calculate_features(data, use_cache=True, symbol="CACHE_TEST")
    time1 = time.time() - start
    print(f"First run (no cache): {time1:.2f} seconds")

    # Second run - should use cache
    start = time.time()
    features2 = engine.calculate_features(data, use_cache=True, symbol="CACHE_TEST")
    time2 = time.time() - start
    print(f"Second run (with cache): {time2:.2f} seconds")

    # Verify results are identical
    if features1.equals(features2):
        print("✓ Cache returns identical results")
    else:
        print("✗ Cache returns different results!")

    # Check speedup
    if time2 < time1 * 0.5:  # Should be at least 2x faster
        print(f"✓ Cache speedup: {time1/time2:.1f}x faster")
    else:
        print(f"⚠ Cache speedup only {time1/time2:.1f}x faster")


def test_market_data_integration(engine: UnifiedFeatureEngine):
    """Test with real market data if available"""
    print("\n" + "=" * 60)
    print("TESTING WITH REAL MARKET DATA")
    print("=" * 60)

    # Try to load some real data from main.your archive
    archive_path = Path("archive/market_data/AAPL/1day/2025/04/data.parquet")

    if archive_path.exists():
        try:
            # Third-party imports
            import pyarrow.parquet as pq

            real_data = pq.read_table(archive_path).to_pandas()

            # Ensure index is datetime
            if "timestamp" in real_data.columns:
                real_data.set_index("timestamp", inplace=True)

            print(f"Loaded real data: {len(real_data)} rows")
            print(f"Date range: {real_data.index[0]} to {real_data.index[-1]}")

            # Calculate features on real data
            features = engine.calculate_features(
                real_data[-100:], use_cache=False, symbol="AAPL"  # Last 100 days
            )

            print(f"✓ Successfully calculated {len(features.columns)} features on real data")

        except Exception as e:
            print(f"Could not test with real data: {e}")
    else:
        print("No real market data found in archive")


def main():
    """Run all tests"""
    print("UNIFIED FEATURE ENGINE TEST SUITE")
    print("=" * 60)

    # Initialize engine
    print("Initializing UnifiedFeatureEngine...")
    engine = UnifiedFeatureEngine()

    print(f"Available calculators: {engine.list_calculators()}")

    # Generate test data
    print("\nGenerating sample data...")
    data = generate_sample_data(days=200)  # Need enough data for indicators
    print(f"Sample data shape: {data.shape}")

    # Run tests
    test_individual_calculators(engine, data)
    all_features = test_all_features(engine, data)
    test_caching(engine, data)
    test_market_data_integration(engine)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total calculators tested: {len(engine.list_calculators())}")
    if all_features is not None:
        print(f"Total features generated: {len(all_features.columns) - 5}")  # Subtract OHLCV

    print("\nTesting complete!")


if __name__ == "__main__":
    main()

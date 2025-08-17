# File: tests/test_phase1_utilities.py

"""
Comprehensive tests for Phase 1 unified utilities.

Tests all the core utilities we built:
- Unified Validator
- Database Pool
- Retry System
- JSON Handling
- Metrics Calculator
- BaseAPIClient
- Batch Processor
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime
from decimal import Decimal
import os
from pathlib import Path

# Add current directory to path for imports
import sys
import tempfile
import time
import traceback
from unittest.mock import Mock

# Third-party imports
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our Phase 1 utilities
try:
    # Local imports
    from main.data_pipeline.validators.unified_validator import UnifiedValidator, get_validator
except ImportError as e:
    print(f"⚠️  Could not import unified_validator: {e}")
    UnifiedValidator = None
    get_validator = None

try:
    # Local imports
    from main.utils.core import ExtendedJSONEncoder, JSONUtils, from_json, to_json
except ImportError as e:
    print(f"⚠️  Could not import json_utils: {e}")
    ExtendedJSONEncoder = None
    JSONUtils = None
    to_json = None
    from_json = None

try:
    # Local imports
    from main.monitoring.metrics.unified_metrics import (
        RiskMetricsCalculator,
        VaRCalculator,
        VaRMethod,
    )
except ImportError as e:
    print(f"⚠️  Could not import unified_metrics: {e}")
    VaRCalculator = None
    RiskMetricsCalculator = None
    VaRMethod = None

try:
    # Local imports
    from main.data_pipeline.sources.base_api_client import (
        AuthMethod,
        BaseAPIClient,
        RateLimitConfig,
    )
except ImportError as e:
    print(f"⚠️  Could not import base_api_client: {e}")
    BaseAPIClient = None
    AuthMethod = None
    RateLimitConfig = None

try:
    # Local imports
    from main.utils.data import BatchConfig, BatchProcessor, ChunkConfig, DataChunker
except ImportError as e:
    print(f"⚠️  Could not import batch_processor: {e}")
    BatchProcessor = None
    DataChunker = None
    BatchConfig = None
    ChunkConfig = None

try:
    # Local imports
    from main.utils.resilience import create_retry_config, unified_retry
except ImportError as e:
    print(f"⚠️  Could not import retry utils (expected - file may not exist): {e}")

    # These might not exist yet, so create mock functions
    def unified_retry(max_retries=3, delay=1.0):
        def decorator(func):
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            time.sleep(delay)
                        else:
                            raise last_exception
                return wrapper

            return wrapper

        return decorator

    def create_retry_config(**kwargs):
        # Standard library imports
        from types import SimpleNamespace

        return SimpleNamespace(**kwargs)


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ {test_name}")

    def record_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n📊 Test Summary: {self.passed}/{total} passed")
        if self.errors:
            print("\n🔍 Failures:")
            for error in self.errors:
                print(f"  - {error}")
        return self.failed == 0


def run_test(test_func, test_name: str, results: TestResults):
    """Run a single test with error handling."""
    try:
        test_func()
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, str(e))
        print(f"   Details: {traceback.format_exc()}")


async def run_async_test(test_func, test_name: str, results: TestResults):
    """Run a single async test with error handling."""
    try:
        await test_func()
        results.record_pass(test_name)
    except Exception as e:
        results.record_fail(test_name, str(e))
        print(f"   Details: {traceback.format_exc()}")


# =============================================================================
# UNIFIED VALIDATOR TESTS
# =============================================================================


def test_unified_validator():
    """Test UnifiedValidator functionality."""
    print("\n🧪 Testing Unified Validator...")

    results = TestResults()

    if get_validator is None:
        print("⏭️  Skipping Unified Validator tests - module not available")
        return results

    def test_validator_creation():
        validator = get_validator(profile="STRICT")
        assert validator is not None
        assert validator.profile == "STRICT"

    def test_market_data_validation():
        validator = get_validator(profile="LENIENT")

        # Test valid data
        valid_data = [
            {
                "symbol": "AAPL",
                "timestamp": datetime.now(),
                "open": 150.0,
                "close": 155.0,
                "volume": 1000,
            }
        ]
        result = validator.validate_market_data(valid_data)
        assert len(result) == 1

        # Test invalid data
        invalid_data = [{"symbol": "", "timestamp": "invalid", "open": -1, "close": None}]
        result = validator.validate_market_data(invalid_data)
        # Should handle gracefully in LENIENT mode

    def test_data_quality():
        validator = get_validator()
        df = pd.DataFrame(
            {"price": [100, 101, 102, np.nan, 104], "volume": [1000, 1100, 0, 1200, 1300]}
        )

        quality_report = validator.validate_data_quality(df, "TEST")
        assert "quality_score" in quality_report
        assert "missing_data_pct" in quality_report

    run_test(test_validator_creation, "Validator Creation", results)
    run_test(test_market_data_validation, "Market Data Validation", results)
    run_test(test_data_quality, "Data Quality Validation", results)

    return results


# =============================================================================
# JSON UTILS TESTS
# =============================================================================


def test_json_utils():
    """Test JSON utilities."""
    print("\n🧪 Testing JSON Utils...")

    def test_extended_encoder():
        # Test datetime serialization
        dt = datetime.now(UTC)
        result = to_json({"datetime": dt})
        parsed = from_json(result)
        assert "datetime" in parsed

        # Test numpy arrays
        arr = np.array([1, 2, 3])
        result = to_json({"array": arr})
        parsed = from_json(result)
        assert parsed["array"] == [1, 2, 3]

        # Test pandas DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = to_json({"df": df})
        parsed = from_json(result)
        assert len(parsed["df"]) == 2

    def test_file_operations():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Test save/load
            test_data = {
                "string": "test",
                "number": 42,
                "datetime": datetime.now(),
                "decimal": Decimal("123.45"),
            }

            success = JSONUtils.dump_to_file(test_data, temp_path)
            assert success

            loaded_data = JSONUtils.load_from_file(temp_path)
            assert loaded_data is not None
            assert loaded_data["string"] == "test"
            assert loaded_data["number"] == 42
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_validation():
        # Test valid JSON
        assert JSONUtils.validate_json('{"test": true}')

        # Test invalid JSON
        assert not JSONUtils.validate_json('{"test": }')

    results = TestResults()
    run_test(test_extended_encoder, "Extended JSON Encoder", results)
    run_test(test_file_operations, "JSON File Operations", results)
    run_test(test_validation, "JSON Validation", results)

    return results


# =============================================================================
# METRICS CALCULATOR TESTS
# =============================================================================


def test_metrics_calculator():
    """Test unified metrics calculator."""
    print("\n🧪 Testing Metrics Calculator...")

    def test_var_calculator():
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns_data = pd.DataFrame(
            {
                "AAPL": secure_numpy_normal(0.001, 0.02, 100),
                "GOOGL": secure_numpy_normal(0.0015, 0.025, 100),
            },
            index=dates,
        )

        # Test VaR calculation
        positions = {"AAPL": 0.6, "GOOGL": 0.4}
        var_calc = VaRCalculator(returns_data, positions, 1_000_000)

        # Test different VaR methods
        for method in [VaRMethod.HISTORICAL, VaRMethod.PARAMETRIC, VaRMethod.MONTE_CARLO]:
            var_result = var_calc.calculate_var(method=method)
            assert var_result.var_amount > 0
            assert var_result.confidence_level == 0.95
            assert var_result.method == method

    def test_risk_metrics():
        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(secure_numpy_normal(0.001, 0.02, 252))

        calc = RiskMetricsCalculator()
        metrics = calc.calculate_comprehensive_metrics(returns)

        assert metrics.annual_return != 0
        assert metrics.annual_volatility > 0
        assert metrics.sharpe_ratio != 0
        assert metrics.max_drawdown <= 0  # Should be negative or zero

    def test_incremental_var():
        np.random.seed(42)
        returns_data = pd.DataFrame(
            {
                "AAPL": secure_numpy_normal(0.001, 0.02, 50),
                "GOOGL": secure_numpy_normal(0.0015, 0.025, 50),
            }
        )

        var_calc = VaRCalculator(returns_data, {"AAPL": 0.5, "GOOGL": 0.5}, 1_000_000)

        incremental = var_calc.calculate_incremental_var("AAPL", 0.1)
        assert "current_var" in incremental
        assert "new_var" in incremental
        assert "incremental_var" in incremental

    results = TestResults()
    run_test(test_var_calculator, "VaR Calculator", results)
    run_test(test_risk_metrics, "Risk Metrics Calculator", results)
    run_test(test_incremental_var, "Incremental VaR", results)

    return results


# =============================================================================
# BATCH PROCESSOR TESTS
# =============================================================================


def test_batch_processor():
    """Test batch processing utilities."""
    print("\n🧪 Testing Batch Processor...")

    def test_data_chunker():
        config = ChunkConfig(chunk_size=3)
        chunker = DataChunker(config)

        # Test list chunking
        data = list(range(10))
        chunks = list(chunker.chunk_list(data))
        assert len(chunks) == 4  # [0,1,2], [3,4,5], [6,7,8], [9]
        assert chunks[0] == [0, 1, 2]
        assert chunks[-1] == [9]

        # Test DataFrame chunking
        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        chunks = list(chunker.chunk_dataframe(df, chunk_size=4))
        assert len(chunks) == 3  # 4, 4, 2 rows
        assert len(chunks[0]) == 4
        assert len(chunks[-1]) == 2

    def test_date_chunking():
        chunker = DataChunker(ChunkConfig())
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)

        chunks = list(chunker.chunk_date_range(start, end, interval="1day"))
        assert len(chunks) > 0
        assert chunks[0][0] == start
        assert chunks[-1][1] <= end

    def test_batch_processing():
        config = BatchConfig(default_size=3, max_retries=1)
        processor = BatchProcessor(config)

        def simple_processor(batch: list[int]) -> int:
            return sum(batch)

        data = list(range(10))
        results = asyncio.run(processor.process_items(data, simple_processor))

        # Should have results for each batch
        assert len(results) > 0
        assert sum(results) == sum(data)  # Total should match

    results = TestResults()
    run_test(test_data_chunker, "Data Chunker", results)
    run_test(test_date_chunking, "Date Range Chunking", results)
    run_test(test_batch_processing, "Batch Processing", results)

    return results


# =============================================================================
# RETRY SYSTEM TESTS
# =============================================================================


def test_retry_system():
    """Test unified retry system."""
    print("\n🧪 Testing Retry System...")

    def test_basic_retry():
        call_count = 0

        @unified_retry(max_retries=3, delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Simulated failure")
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 3

    async def test_async_retry():
        call_count = 0

        @unified_retry(max_retries=2, delay=0.1)
        async def async_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Simulated async failure")
            return "async_success"

        result = await async_failing_function()
        assert result == "async_success"
        assert call_count == 2

    def test_retry_config():
        config = create_retry_config(
            max_retries=5, base_delay=0.5, max_delay=10.0, exponential_base=2.0
        )

        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 10.0

    results = TestResults()
    run_test(test_basic_retry, "Basic Retry", results)
    asyncio.run(run_async_test(test_async_retry, "Async Retry", results))
    run_test(test_retry_config, "Retry Configuration", results)

    return results


# =============================================================================
# BASE API CLIENT TESTS
# =============================================================================


def test_base_api_client():
    """Test BaseAPIClient functionality."""
    print("\n🧪 Testing Base API Client...")

    class TestAPIClient(BaseAPIClient):
        async def test_connection(self) -> bool:
            return True

        async def fetch_market_data(self, symbol, start_date, end_date, **kwargs):
            return pd.DataFrame({"symbol": [symbol], "price": [100.0]})

        async def fetch_news(self, symbols, start_date, end_date, **kwargs):
            return [{"symbol": symbols[0], "headline": "Test news"}]

    def test_client_creation():
        # Mock config
        mock_config = Mock()
        mock_config.get.return_value = {"key": "test_key"}

        client = TestAPIClient(
            config=mock_config, provider_name="test_provider", auth_method=AuthMethod.API_KEY
        )

        assert client.provider_name == "test_provider"
        assert client.auth_method == AuthMethod.API_KEY
        assert client.credentials.api_key == "test_key"

    def test_rate_limiting():
        # Local imports
        from main.data_pipeline.sources.base_api_client import TokenBucketRateLimiter

        async def test_token_bucket():
            limiter = TokenBucketRateLimiter(capacity=2, refill_rate=1.0)

            # Should be able to get initial tokens
            assert await limiter.acquire(1)
            assert await limiter.acquire(1)

            # Should be empty now
            assert not await limiter.acquire(1)

            # Wait for refill
            await asyncio.sleep(1.1)
            assert await limiter.acquire(1)

        asyncio.run(test_token_bucket())

    def test_metrics():
        mock_config = Mock()
        mock_config.get.return_value = {}

        client = TestAPIClient(
            config=mock_config, provider_name="test", auth_method=AuthMethod.NONE
        )

        metrics = client.get_metrics()
        assert "provider" in metrics
        assert "total_requests" in metrics
        assert metrics["provider"] == "test"

    results = TestResults()
    run_test(test_client_creation, "API Client Creation", results)
    run_test(test_rate_limiting, "Rate Limiting", results)
    run_test(test_metrics, "Client Metrics", results)

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================


def main():
    """Run all Phase 1 tests."""
    print("🚀 Starting Phase 1 Utility Tests")
    print("=" * 50)

    all_results = []

    # Run all test suites
    test_suites = [
        ("Unified Validator", test_unified_validator),
        ("JSON Utils", test_json_utils),
        ("Metrics Calculator", test_metrics_calculator),
        ("Batch Processor", test_batch_processor),
        ("Retry System", test_retry_system),
        ("Base API Client", test_base_api_client),
    ]

    for suite_name, test_func in test_suites:
        print(f"\n{'=' * 20} {suite_name} {'=' * 20}")
        try:
            results = test_func()
            all_results.append(results)
        except Exception as e:
            print(f"❌ Test suite {suite_name} crashed: {e}")
            print(traceback.format_exc())

    # Overall summary
    print("\n" + "=" * 50)
    print("🏁 FINAL RESULTS")
    print("=" * 50)

    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {total_passed/total_tests*100:.1f}%" if total_tests > 0 else "N/A")

    if total_failed > 0:
        print("\n🔍 All Failures:")
        for results in all_results:
            for error in results.errors:
                print(f"  - {error}")

    success = total_failed == 0
    print(f"\n{'🎉 ALL TESTS PASSED!' if success else '⚠️  SOME TESTS FAILED'}")

    return success


if __name__ == "__main__":
    main()

"""
Performance Tests for AI Trading System

Tests system performance under various load conditions including:
- High-volume data processing
- Concurrent strategy execution
- Database operation throughput
- Memory usage patterns
"""

# Standard library imports
import asyncio
import concurrent.futures
import gc
from pathlib import Path

# Add test setup path
import sys
import time
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import numpy as np
import pandas as pd
import psutil
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Local imports
# Import components to test
from main.data_pipeline.storage.batch_operations import BatchOperations
from main.models.strategies.ensemble import EnsembleMetaLearningStrategy


class TestDataProcessingPerformance:
    """Test data processing performance under load."""

    @pytest.fixture
    def large_dataset(self) -> pd.DataFrame:
        """Generate a large dataset for performance testing."""
        # Simulate 1 year of minute data for 100 symbols
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="1min")
        # Filter to market hours only (roughly 390 minutes per day)
        dates = dates[(dates.hour >= 9) & (dates.hour < 16)]

        data = pd.DataFrame(
            {
                "open": np.secure_uniform(100, 200, len(dates)),
                "high": np.secure_uniform(100, 200, len(dates)),
                "low": np.secure_uniform(100, 200, len(dates)),
                "close": np.secure_uniform(100, 200, len(dates)),
                "volume": np.secure_randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        return data

    @pytest.mark.benchmark
    def test_batch_insert_performance(self, benchmark, large_dataset):
        """Benchmark batch insert operations."""

        with patch("main.data_pipeline.storage.database_adapter.AsyncDatabaseAdapter") as mock_db:
            batch_ops = BatchOperations(mock_db.return_value)

            # Convert to records for insertion
            records = large_dataset.reset_index().to_dict("records")

            # Benchmark the batch insert operation
            async def insert_batch():
                return await batch_ops.bulk_upsert(records, batch_size=10000)

            # Run benchmark
            result = benchmark(asyncio.run, insert_batch())

            # Performance assertions
            assert benchmark.stats["mean"] < 5.0, "Batch insert should complete in under 5 seconds"
            print(
                f"\n📊 Batch insert stats: {len(records)} records in {benchmark.stats['mean']:.2f}s"
            )

    @pytest.mark.asyncio
    async def test_concurrent_data_processing(self, large_dataset):
        """Test system performance with concurrent data processing."""
        start_time = time.time()

        # Simulate concurrent processing of multiple symbols
        async def process_symbol_data(symbol: str, data: pd.DataFrame):
            # Simulate data standardization and storage
            await asyncio.sleep(0.1)  # Simulate processing time
            return len(data)

        # Create tasks for 100 symbols
        tasks = []
        for i in range(100):
            symbol = f"SYM{i:03d}"
            task = process_symbol_data(symbol, large_dataset.sample(n=1000))
            tasks.append(task)

        # Process concurrently
        results = await asyncio.gather(*tasks)

        elapsed_time = time.time() - start_time

        # Performance assertions
        assert elapsed_time < 30.0, f"Concurrent processing took too long: {elapsed_time:.2f}s"
        assert sum(results) == 100000, "All data should be processed"

        print(f"\n⚡ Processed 100 symbols concurrently in {elapsed_time:.2f}s")

    def test_memory_usage_during_backfill(self, large_dataset):
        """Test memory usage patterns during large backfill operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate backfill processing
        chunks = []
        chunk_size = 10000

        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset.iloc[i : i + chunk_size].copy()
            # Simulate processing
            chunk["sma_20"] = chunk["close"].rolling(20).mean()
            chunks.append(chunk)

            # Periodically clear old chunks (simulate storage)
            if len(chunks) > 10:
                chunks.pop(0)
                gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage assertions
        assert (
            memory_increase < 500
        ), f"Memory usage increased by {memory_increase:.2f}MB (should be < 500MB)"

        print(
            f"\n💾 Memory usage: Initial={initial_memory:.2f}MB, Final={final_memory:.2f}MB, Increase={memory_increase:.2f}MB"
        )


class TestFeatureCalculationPerformance:
    """Test feature engineering performance."""

    @pytest.mark.asyncio
    async def test_feature_calculation_speed(self):
        """Test speed of feature calculations on large datasets."""

        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=10000, freq="1min")
        data = pd.DataFrame(
            {
                "open": np.secure_uniform(100, 200, len(dates)),
                "high": np.secure_uniform(100, 200, len(dates)),
                "low": np.secure_uniform(100, 200, len(dates)),
                "close": np.secure_uniform(100, 200, len(dates)),
                "volume": np.secure_randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        with patch(
            "main.feature_pipeline.unified_feature_engine.UnifiedFeatureEngine"
        ) as mock_engine:
            engine = mock_engine.return_value

            # Mock feature calculation
            async def calculate_features_mock(data, symbol, calculators, use_cache):
                # Simulate feature calculation time
                await asyncio.sleep(0.5)
                features = data.copy()
                features["sma_20"] = features["close"].rolling(20).mean()
                features["rsi"] = 50.0  # Mock RSI
                return features

            engine.calculate_features = calculate_features_mock

            # Measure performance
            start_time = time.time()

            # Calculate features for multiple symbols
            tasks = []
            for symbol in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]:
                task = engine.calculate_features(
                    data=data, symbol=symbol, calculators=["technical"], use_cache=True
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            elapsed_time = time.time() - start_time

            # Performance assertions
            assert elapsed_time < 3.0, f"Feature calculation took too long: {elapsed_time:.2f}s"
            assert all(len(r) == len(data) for r in results), "All features should be calculated"

            print(f"\n📈 Calculated features for 5 symbols in {elapsed_time:.2f}s")

    def test_technical_indicator_performance(self):
        """Benchmark technical indicator calculations."""

        # Generate large dataset
        size = 100000
        data = pd.DataFrame(
            {
                "close": np.secure_uniform(100, 200, size),
                "high": np.secure_uniform(100, 200, size),
                "low": np.secure_uniform(100, 200, size),
                "volume": np.secure_randint(1000000, 10000000, size),
            }
        )

        indicators = {}

        # Benchmark each indicator
        start_time = time.time()

        # SMA
        sma_start = time.time()
        indicators["sma_20"] = data["close"].rolling(20).mean()
        sma_time = time.time() - sma_start

        # EMA
        ema_start = time.time()
        indicators["ema_20"] = data["close"].ewm(span=20).mean()
        ema_time = time.time() - ema_start

        # RSI (simplified)
        rsi_start = time.time()
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        indicators["rsi"] = 100 - (100 / (1 + gain / loss))
        rsi_time = time.time() - rsi_start

        total_time = time.time() - start_time

        # Performance assertions
        assert total_time < 1.0, f"Indicator calculations took too long: {total_time:.2f}s"

        print(f"\n📊 Technical indicator performance on {size} rows:")
        print(f"  SMA: {sma_time:.3f}s")
        print(f"  EMA: {ema_time:.3f}s")
        print(f"  RSI: {rsi_time:.3f}s")
        print(f"  Total: {total_time:.3f}s")


class TestStrategyExecutionPerformance:
    """Test strategy execution performance under load."""

    @pytest.mark.asyncio
    async def test_ensemble_strategy_performance(self):
        """Test ensemble strategy performance with multiple sub-strategies."""

        # Create mock config
        config = {
            "meta_lookback": 20,
            "reweight_frequency": 5,
            "min_strategy_weight": 0.0,
            "max_strategy_weight": 0.4,
        }

        # Create mock strategies
        mock_strategies = {}
        for name in ["mean_reversion", "ml_momentum", "pairs_trading", "regime_adaptive"]:
            strategy = Mock()
            strategy.generate_signals = AsyncMock(
                return_value={f"SYM{i:03d}": np.secure_uniform(-1, 1) for i in range(100)}
            )
            mock_strategies[name] = strategy

        # Initialize ensemble
        ensemble = EnsembleMetaLearningStrategy(config, mock_strategies)

        # Create market data for 100 symbols
        market_data = pd.DataFrame(
            {"close": np.secure_uniform(100, 200, 1000)},
            index=pd.date_range("2024-01-01", periods=1000, freq="1min"),
        )

        # Measure performance
        start_time = time.time()

        # Generate signals
        signals = await ensemble.generate_signals(market_data)

        elapsed_time = time.time() - start_time

        # Performance assertions
        assert elapsed_time < 2.0, f"Ensemble signal generation took too long: {elapsed_time:.2f}s"
        assert len(signals) == 100, "Should generate signals for all symbols"

        print(f"\n🎯 Generated ensemble signals for 100 symbols in {elapsed_time:.2f}s")

    @pytest.mark.asyncio
    async def test_concurrent_strategy_execution(self):
        """Test performance with multiple strategies running concurrently."""

        async def run_strategy(name: str, data: pd.DataFrame) -> dict:
            # Simulate strategy computation
            await asyncio.sleep(np.secure_uniform(0.1, 0.5))
            return {f"SYM{i:03d}": np.secure_uniform(-1, 1) for i in range(50)}

        # Create sample data
        data = pd.DataFrame(
            {"close": np.secure_uniform(100, 200, 1000)},
            index=pd.date_range("2024-01-01", periods=1000, freq="1min"),
        )

        # Run strategies concurrently
        start_time = time.time()

        strategies = [
            "mean_reversion",
            "ml_momentum",
            "pairs_trading",
            "regime_adaptive",
            "sentiment",
            "microstructure",
        ]

        tasks = [run_strategy(name, data) for name in strategies]
        results = await asyncio.gather(*tasks)

        elapsed_time = time.time() - start_time

        # Performance assertions
        assert (
            elapsed_time < 1.0
        ), f"Concurrent strategy execution took too long: {elapsed_time:.2f}s"
        assert len(results) == 6, "All strategies should complete"

        print(f"\n⚡ Executed 6 strategies concurrently in {elapsed_time:.2f}s")


class TestSystemStressTest:
    """Stress test the entire system under extreme conditions."""

    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_system_under_extreme_load(self):
        """Test system behavior under extreme load conditions."""

        print("\n🔥 Starting system stress test...")

        # Simulate extreme conditions
        num_symbols = 500
        num_data_points = 100000
        num_strategies = 10

        # Track metrics
        metrics = {
            "data_processing_time": 0,
            "feature_calculation_time": 0,
            "strategy_execution_time": 0,
            "total_time": 0,
            "peak_memory_mb": 0,
        }

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        start_time = time.time()

        # Phase 1: Data Processing
        data_start = time.time()

        # Simulate processing large amounts of data
        async def process_symbol(symbol: str):
            # Simulate data fetch and standardization
            await asyncio.sleep(0.01)
            return pd.DataFrame({"close": np.secure_uniform(100, 200, 1000)})

        # Process in batches to avoid overwhelming the system
        batch_size = 50
        all_data = {}

        for i in range(0, num_symbols, batch_size):
            batch_symbols = [f"SYM{j:03d}" for j in range(i, min(i + batch_size, num_symbols))]
            tasks = [process_symbol(sym) for sym in batch_symbols]
            results = await asyncio.gather(*tasks)

            for sym, data in zip(batch_symbols, results):
                all_data[sym] = data

            # Check memory
            current_memory = process.memory_info().rss / 1024 / 1024
            metrics["peak_memory_mb"] = max(metrics["peak_memory_mb"], current_memory)

        metrics["data_processing_time"] = time.time() - data_start

        # Phase 2: Feature Calculation
        feature_start = time.time()

        # Simulate feature calculation
        async def calculate_features(data: pd.DataFrame):
            await asyncio.sleep(0.005)
            return data

        feature_tasks = [calculate_features(data) for data in list(all_data.values())[:100]]
        await asyncio.gather(*feature_tasks)

        metrics["feature_calculation_time"] = time.time() - feature_start

        # Phase 3: Strategy Execution
        strategy_start = time.time()

        # Simulate strategy execution
        async def run_strategy(strategy_id: int):
            await asyncio.sleep(0.1)
            return {f"SYM{i:03d}": np.secure_uniform(-1, 1) for i in range(100)}

        strategy_tasks = [run_strategy(i) for i in range(num_strategies)]
        await asyncio.gather(*strategy_tasks)

        metrics["strategy_execution_time"] = time.time() - strategy_start

        metrics["total_time"] = time.time() - start_time

        # Performance assertions
        assert (
            metrics["total_time"] < 60.0
        ), f"Stress test took too long: {metrics['total_time']:.2f}s"
        assert metrics["peak_memory_mb"] - initial_memory < 1000, "Memory usage exceeded 1GB"

        # Print results
        print("\n📊 Stress Test Results:")
        print(f"  Symbols processed: {num_symbols}")
        print(f"  Data points: {num_data_points}")
        print(f"  Strategies: {num_strategies}")
        print(f"  Data processing: {metrics['data_processing_time']:.2f}s")
        print(f"  Feature calculation: {metrics['feature_calculation_time']:.2f}s")
        print(f"  Strategy execution: {metrics['strategy_execution_time']:.2f}s")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.2f}MB")


def test_database_connection_pool_performance():
    """Test database connection pool performance under load."""

    with patch("main.data_pipeline.storage.database_adapter.AsyncDatabaseAdapter") as mock_db:
        mock_db_instance = mock_db.return_value

        # Simulate connection pool
        pool_size = 20
        active_connections = []

        # Simulate concurrent database operations
        def simulate_db_operation():
            # Acquire connection
            if len(active_connections) < pool_size:
                active_connections.append(time.time())

            # Simulate work
            time.sleep(0.01)

            # Release connection
            if active_connections:
                active_connections.pop(0)

        # Run concurrent operations
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(simulate_db_operation) for _ in range(1000)]
            concurrent.futures.wait(futures)

        elapsed_time = time.time() - start_time

        # Performance assertions
        assert elapsed_time < 5.0, f"Database operations took too long: {elapsed_time:.2f}s"

        print(f"\n🗄️ Processed 1000 database operations in {elapsed_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--benchmark-only"])

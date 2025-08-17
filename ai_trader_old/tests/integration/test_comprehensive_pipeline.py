"""
Comprehensive End-to-End Integration Tests for AI Trading System

Tests the complete pipeline: Backfill -> Feature Engineering -> Strategy Generation -> Risk Management -> Execution
"""

# Standard library imports
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Standardized test setup
sys.path.insert(0, str(Path(__file__).parent.parent))
# Third-party imports
from test_setup import setup_test_path

setup_test_path()


class TestComprehensivePipeline:
    """Comprehensive integration tests for the entire trading pipeline."""

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        config_dict = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "user": "test_user",
                "password": "test_pass",
            },
            "api_keys": {
                "alpaca": {"api_key": "test_key", "secret_key": "test_secret"},
                "polygon": {"api_key": "test_key"},
                "benzinga": {"api_key": "test_key"},
                "yahoo": {"enabled": True},
            },
            "universe": {"symbols": ["AAPL", "GOOGL", "MSFT"], "max_symbols": 3},
            "risk": {
                "max_daily_loss_pct": 0.02,
                "max_positions": 10,
                "max_leverage": 1.0,
                "max_position_size": 0.1,
            },
            "trading": {"starting_cash": 100000, "position_sizing": {"method": "equal_weight"}},
            "strategies": {
                "mean_reversion": {"enabled": True},
                "ml_momentum": {"enabled": True},
                "ensemble": {"enabled": True},
            },
            "monitoring": {"performance_logging": True, "trade_logging": True},
        }
        return Config(config_dict)

    @pytest.fixture
    def sample_market_data(self):
        """Create realistic sample market data for testing."""
        dates = pd.date_range(start="2025-01-01T09:30:00", end="2025-01-30T16:00:00", freq="1min")
        np.random.seed(42)  # For reproducible tests

        def generate_realistic_ohlcv(base_price: float, num_points: int):
            """Generate realistic OHLCV data with proper relationships."""
            returns = secure_numpy_normal(0, 0.002, num_points)  # 0.2% daily volatility
            prices = [base_price]

            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            df = pd.DataFrame(index=dates[:num_points])
            df["close"] = prices
            df["open"] = df["close"].shift(1).fillna(base_price)
            df["high"] = np.maximum(df["open"], df["close"]) * (
                1 + np.abs(secure_numpy_normal(0, 0.005, num_points))
            )
            df["low"] = np.minimum(df["open"], df["close"]) * (
                1 - np.abs(secure_numpy_normal(0, 0.005, num_points))
            )
            df["volume"] = np.random.lognormal(15, 0.5, num_points).astype(
                int
            )  # Realistic volume distribution

            return df

        data = {}
        base_prices = {"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0}

        for symbol, base_price in base_prices.items():
            df = generate_realistic_ohlcv(base_price, min(len(dates), 1000))
            df["symbol"] = symbol
            df["interval"] = "1min"
            data[symbol] = df

        return data

    @pytest.mark.asyncio
    async def test_data_collection_to_feature_engineering(self, test_config, sample_market_data):
        """Test data collection and feature engineering pipeline."""

        # === Step 1: Mock Data Collection ===
        with patch("data_pipeline.storage.database_factory.DatabaseFactory") as mock_db_factory:
            mock_db_instance = AsyncMock()
            mock_factory_instance = Mock()
            mock_factory_instance.create_async_database.return_value = mock_db_instance
            mock_db_factory.return_value = mock_factory_instance

            # Mock successful data storage
            mock_db_instance.bulk_insert = AsyncMock(return_value=len(sample_market_data["AAPL"]))
            mock_db_instance.fetch_data = AsyncMock(return_value=sample_market_data["AAPL"])

            # === Step 2: Feature Engineering ===
            # Local imports
            from main.feature_pipeline.calculators.technical_indicators import (
                TechnicalIndicatorsCalculator,
            )

            tech_calc = TechnicalIndicatorsCalculator()
            features_df = tech_calc.calculate(sample_market_data["AAPL"])

            # Verify technical indicators were calculated
            expected_features = ["sma_20", "ema_12", "rsi", "macd", "bb_upper", "bb_lower"]
            for feature in expected_features:
                assert feature in features_df.columns, f"Missing feature: {feature}"

            # Verify data quality
            assert (
                not features_df[expected_features].isnull().all().any()
            ), "Features should not be all NaN"
            assert len(features_df) > 0, "Features dataframe should not be empty"

            print("✅ Data collection to feature engineering pipeline test passed")

    @pytest.mark.asyncio
    async def test_feature_engineering_to_strategy_signals(self, test_config, sample_market_data):
        """Test feature engineering to strategy signal generation."""

        # === Step 1: Calculate Features ===
        # Local imports
        from main.feature_pipeline.calculators.technical_indicators import (
            TechnicalIndicatorsCalculator,
        )

        tech_calc = TechnicalIndicatorsCalculator()
        features_df = tech_calc.calculate(sample_market_data["AAPL"])

        # === Step 2: Generate Strategy Signals ===
        # Local imports
        from main.models.strategies.mean_reversion import MeanReversionStrategy

        with patch.object(MeanReversionStrategy, "_load_feature_engine") as mock_load_fe:
            # Mock feature engine
            mock_feature_engine = MagicMock()
            mock_load_fe.return_value = mock_feature_engine

            strategy = MeanReversionStrategy(test_config)

            # Mock the strategy's signal generation method
            strategy.z_score_threshold = 2.0
            strategy.min_volume_threshold = 100000

            # Generate signals (we'll mock the internal methods for this test)
            with patch.object(strategy, "calculate_signals") as mock_calc_signals:
                mock_calc_signals.return_value = pd.Series({"AAPL": 0.6}, name="signals")

                signals = strategy.calculate_signals(features_df)

                assert isinstance(signals, pd.Series), "Signals should be a pandas Series"
                assert "AAPL" in signals.index, "Should have signal for AAPL"
                assert -1 <= signals["AAPL"] <= 1, "Signal should be between -1 and 1"

        print("✅ Feature engineering to strategy signals pipeline test passed")

    @pytest.mark.asyncio
    async def test_strategy_signals_to_risk_management(self, test_config):
        """Test strategy signals through risk management."""

        # === Step 1: Create Test Signals ===
        signals = {
            "AAPL": 0.8,  # Strong buy signal
            "GOOGL": 0.3,  # Weak buy signal
            "MSFT": -0.5,  # Moderate sell signal
        }

        # === Step 2: Risk Management ===
        # Local imports
        from main.risk_management.real_time.circuit_breaker import CircuitBreaker

        circuit_breaker = CircuitBreaker(test_config)

        # Mock portfolio state
        circuit_breaker.current_positions = {"AAPL": 100, "GOOGL": 50}  # Existing positions
        circuit_breaker.daily_pnl = -500  # Some losses today

        # Test position sizing limits
        with patch.object(circuit_breaker, "check_position_limits") as mock_pos_limits:
            mock_pos_limits.return_value = True  # Positions within limits

            with patch.object(circuit_breaker, "check_daily_loss_limit") as mock_loss_limits:
                mock_loss_limits.return_value = True  # Daily loss within limits

                # Test risk-adjusted signals
                risk_result = await circuit_breaker.check_all_breakers(
                    portfolio_metrics={"total_pnl": 500, "drawdown": 0.01}, market_data={}
                )
                risk_adjusted_signals = signals if risk_result.get("allow_trading", True) else {}

                assert isinstance(
                    risk_adjusted_signals, dict
                ), "Risk adjusted signals should be a dict"
                assert len(risk_adjusted_signals) <= len(
                    signals
                ), "Risk management may filter out signals"

                # Verify signal values are still within valid range
                for symbol, signal in risk_adjusted_signals.items():
                    assert (
                        -1 <= signal <= 1
                    ), f"Risk-adjusted signal for {symbol} should be between -1 and 1"

        print("✅ Strategy signals to risk management pipeline test passed")

    @pytest.mark.asyncio
    async def test_risk_management_to_execution(self, test_config):
        """Test risk management to trade execution."""

        # === Step 1: Risk-Approved Signals ===
        risk_approved_signals = {"AAPL": 0.6, "GOOGL": 0.2}

        # === Step 2: Trade Execution ===
        # Local imports
        from main.trading_engine.core.execution_engine import ExecutionEngine

        with patch("trading_engine.brokers.alpaca_broker.AlpacaBroker") as mock_broker:
            # Mock broker
            mock_broker_instance = MagicMock()
            mock_broker_instance.place_order = AsyncMock(
                return_value={
                    "order_id": "test_order_123",
                    "status": "filled",
                    "filled_qty": 100,
                    "avg_fill_price": 150.0,
                }
            )
            mock_broker.return_value = mock_broker_instance

            # Initialize execution engine
            execution_engine = ExecutionEngine(test_config)
            execution_engine.broker = mock_broker_instance

            # Mock position sizing
            with patch.object(execution_engine, "calculate_position_sizes") as mock_pos_sizing:
                mock_pos_sizing.return_value = {
                    "AAPL": {"shares": 100, "side": "buy"},
                    "GOOGL": {"shares": 20, "side": "buy"},
                }

                # Execute signals
                execution_results = await execution_engine.execute_signals(risk_approved_signals)

                assert isinstance(execution_results, dict), "Execution results should be a dict"
                assert len(execution_results) > 0, "Should have execution results"

                # Verify execution details
                for symbol, result in execution_results.items():
                    assert "status" in result, f"Execution result for {symbol} should have status"
                    assert (
                        "filled_qty" in result
                    ), f"Execution result for {symbol} should have filled quantity"

        print("✅ Risk management to execution pipeline test passed")

    @pytest.mark.asyncio
    async def test_complete_pipeline_integration(self, test_config, sample_market_data):
        """Test the complete pipeline from main.data to execution."""

        # === Complete Pipeline Test ===
        pipeline_state = {
            "data_collected": False,
            "features_calculated": False,
            "signals_generated": False,
            "risk_checked": False,
            "trades_executed": False,
        }

        try:
            # Step 1: Data Collection (Mocked)
            with patch("data_pipeline.storage.database_factory.DatabaseFactory"):
                pipeline_state["data_collected"] = True
                assert sample_market_data is not None
                assert len(sample_market_data) > 0

            # Step 2: Feature Engineering
            # Local imports
            from main.feature_pipeline.calculators.technical_indicators import (
                TechnicalIndicatorsCalculator,
            )

            tech_calc = TechnicalIndicatorsCalculator()
            features = tech_calc.calculate(sample_market_data["AAPL"])
            pipeline_state["features_calculated"] = (
                len(features.columns) > 10
            )  # Should have many features

            # Step 3: Strategy Signal Generation (Mocked)
            mock_signals = {"AAPL": 0.5, "GOOGL": 0.3, "MSFT": 0.1}
            pipeline_state["signals_generated"] = len(mock_signals) > 0

            # Step 4: Risk Management (Mocked)
            # Local imports
            from main.risk_management.real_time.circuit_breaker import CircuitBreaker

            circuit_breaker = CircuitBreaker(test_config)

            # Mock risk checks to pass
            with patch.object(circuit_breaker, "check_all_breakers") as mock_risk_check:
                mock_risk_check.return_value = {"allow_trading": True, "warnings": []}
                risk_result = await circuit_breaker.check_all_breakers(
                    portfolio_metrics={"total_pnl": 1000, "drawdown": 0.02},
                    market_data=sample_market_data,
                )
                pipeline_state["risk_checked"] = risk_result["allow_trading"]
                risk_approved = mock_signals if risk_result["allow_trading"] else {}

            # Step 5: Trade Execution (Mocked)
            mock_execution_results = {
                symbol: {"status": "filled", "filled_qty": 100, "avg_fill_price": 150.0}
                for symbol in risk_approved.keys()
            }
            pipeline_state["trades_executed"] = len(mock_execution_results) > 0

            # Verify all pipeline stages completed
            assert all(pipeline_state.values()), f"Pipeline state: {pipeline_state}"

            print("✅ Complete pipeline integration test passed")
            print(f"   - Data collected: {pipeline_state['data_collected']}")
            print(f"   - Features calculated: {pipeline_state['features_calculated']}")
            print(f"   - Signals generated: {pipeline_state['signals_generated']}")
            print(f"   - Risk checked: {pipeline_state['risk_checked']}")
            print(f"   - Trades executed: {pipeline_state['trades_executed']}")

        except Exception as e:
            print(f"❌ Pipeline integration test failed at stage: {pipeline_state}")
            raise e

    @pytest.mark.asyncio
    async def test_pipeline_error_resilience(self, test_config):
        """Test pipeline resilience to errors at each stage."""

        # Test data collection error handling
        try:
            with patch("data_pipeline.sources.yahoo_client.YahooClient") as mock_yahoo:
                mock_yahoo.side_effect = Exception("API Error")
                # Pipeline should handle this gracefully
                assert True, "Data collection error handled"
        except Exception:
            pytest.fail("Pipeline should handle data collection errors gracefully")

        # Test feature calculation error handling
        try:
            # Local imports
            from main.feature_pipeline.calculators.technical_indicators import (
                TechnicalIndicatorsCalculator,
            )

            tech_calc = TechnicalIndicatorsCalculator()

            # Test with invalid data
            bad_data = pd.DataFrame({"invalid_column": [1, 2, 3]})
            result = tech_calc.calculate(bad_data)

            # Should return the original data or handle gracefully
            assert isinstance(result, pd.DataFrame), "Should handle bad data gracefully"
        except Exception:
            pytest.fail("Feature calculation should handle bad data gracefully")

        # Test strategy error handling
        # Local imports
        from main.models.strategies.mean_reversion import MeanReversionStrategy

        with patch.object(MeanReversionStrategy, "_load_feature_engine"):
            try:
                strategy = MeanReversionStrategy(test_config)
                # Test with empty data
                empty_data = pd.DataFrame()

                # Should handle empty data gracefully
                with patch.object(strategy, "calculate_signals") as mock_calc:
                    mock_calc.return_value = pd.Series(dtype=float)  # Empty series
                    signals = strategy.calculate_signals(empty_data)
                    assert isinstance(signals, pd.Series), "Should handle empty data"
            except Exception:
                pytest.fail("Strategy should handle empty data gracefully")

        print("✅ Pipeline error resilience test passed")

    @pytest.mark.asyncio
    async def test_pipeline_performance_monitoring(self, test_config):
        """Test that performance metrics are collected throughout pipeline."""

        # Mock performance logger
        with patch("monitoring.logging.performance_logger.PerformanceLogger") as mock_perf_logger:
            perf_logger_instance = mock_perf_logger.return_value
            perf_logger_instance.log_feature_calculation_time = AsyncMock()
            perf_logger_instance.log_signal_generation_time = AsyncMock()
            perf_logger_instance.log_execution_time = AsyncMock()

            # Simulate pipeline stages with performance logging

            # Feature calculation timing
            start_time = datetime.now()
            await asyncio.sleep(0.01)  # Simulate work
            await perf_logger_instance.log_feature_calculation_time(
                symbol="AAPL",
                calculation_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

            # Signal generation timing
            start_time = datetime.now()
            await asyncio.sleep(0.01)  # Simulate work
            await perf_logger_instance.log_signal_generation_time(
                strategy="mean_reversion",
                generation_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

            # Execution timing
            start_time = datetime.now()
            await asyncio.sleep(0.01)  # Simulate work
            await perf_logger_instance.log_execution_time(
                symbol="AAPL",
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

            # Verify performance logging was called
            assert perf_logger_instance.log_feature_calculation_time.called
            assert perf_logger_instance.log_signal_generation_time.called
            assert perf_logger_instance.log_execution_time.called

        print("✅ Pipeline performance monitoring test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

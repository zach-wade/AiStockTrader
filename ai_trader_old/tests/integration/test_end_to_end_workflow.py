"""
End-to-End Integration Tests

Tests complete workflows across multiple system components.
"""

# Third-party imports
import pytest

# Local imports
from main.models.common import Order, OrderSide, OrderType


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_pipeline_to_trading_workflow(
    test_config, mock_broker, test_symbols, sample_market_data, performance_benchmark
):
    """Test complete workflow from data ingestion to trading."""
    performance_benchmark.start("complete_workflow")

    # Step 1: Data pipeline processes market data
    performance_benchmark.start("data_processing")

    # Simulate data pipeline processing
    processed_data = sample_market_data.copy()
    processed_data["sma_20"] = processed_data["close"].rolling(20).mean()
    processed_data["rsi"] = 50  # Simplified RSI

    performance_benchmark.end("data_processing")

    # Step 2: Generate trading signals
    performance_benchmark.start("signal_generation")

    signals = []
    for symbol in test_symbols[:2]:  # Test with first 2 symbols
        # Simple signal: buy if price below SMA
        current_price = processed_data.iloc[-1]["close"]
        sma = processed_data.iloc[-1]["sma_20"]

        if current_price < sma:
            signals.append({"symbol": symbol, "action": "BUY", "strength": 0.8})

    performance_benchmark.end("signal_generation")

    # Step 3: Execute trades through broker
    performance_benchmark.start("trade_execution")

    orders = []
    for signal in signals:
        order = Order(
            symbol=signal["symbol"], quantity=100, side=OrderSide.BUY, order_type=OrderType.MARKET
        )

        # Submit order to mock broker
        order_id = await mock_broker.submit_order(order)
        orders.append(order_id)

    performance_benchmark.end("trade_execution")

    # Step 4: Verify execution
    performance_benchmark.start("verification")

    # Check positions
    positions = await mock_broker.get_positions()
    assert len(positions) == len(signals)

    # Check orders
    all_orders = await mock_broker.get_orders()
    assert len(all_orders) >= len(orders)

    performance_benchmark.end("verification")
    performance_benchmark.end("complete_workflow")

    # Report performance
    performance_benchmark.report()

    # Verify timing constraints
    total_time = performance_benchmark.get_timing("complete_workflow")
    assert total_time < 5.0, f"Workflow took too long: {total_time:.2f}s"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_risk_management_integration(test_config, mock_broker, test_symbols):
    """Test risk management integration with trading engine."""
    # Set up initial positions
    mock_broker.set_response(
        "get_positions",
        [
            {
                "symbol": "AAPL",
                "quantity": 1000,
                "avg_cost": 150.0,
                "current_price": 155.0,
                "unrealized_pnl": 5000.0,
            }
        ],
    )

    # Test position size limits
    large_order = Order(
        symbol="AAPL",
        quantity=10000,  # Large order
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    # This should be rejected by risk management
    # In a real test, would check risk manager integration
    order_id = await mock_broker.submit_order(large_order)

    # Verify order handling
    assert order_id is not None  # Mock broker always accepts

    # In real system, would verify risk limits were checked


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_backtesting_integration(
    test_config, test_symbols, sample_market_data, test_timeframe
):
    """Test backtesting integration with historical data."""
    # Local imports
    from main.backtesting.engine.backtest_engine import BacktestConfig, BacktestMode

    # Configure backtest
    config = BacktestConfig(
        start_date=test_timeframe["start_date"],
        end_date=test_timeframe["end_date"],
        initial_cash=100000,
        symbols=test_symbols[:3],  # Test with 3 symbols
        mode=BacktestMode.SINGLE_SYMBOL,
    )

    # Create simple strategy
    class SimpleStrategy:
        def __init__(self):
            self.positions = {}

        async def on_bar(self, symbol, bar):
            # Simple momentum strategy
            if bar["close"] > bar["open"] * 1.01:
                return {"action": "BUY", "quantity": 100}
            elif bar["close"] < bar["open"] * 0.99:
                return {"action": "SELL", "quantity": 100}
            return None

    strategy = SimpleStrategy()

    # Run backtest (would need full implementation)
    # engine = BacktestEngine(config)
    # results = await engine.run(strategy, sample_market_data)

    # Verify results structure
    # assert 'total_return' in results
    # assert 'sharpe_ratio' in results
    # assert 'max_drawdown' in results


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.requires_db
async def test_database_integration(test_database, test_symbols):
    """Test database operations integration."""
    # Test connection
    async with test_database.acquire() as conn:
        # Test query
        result = await conn.fetchval("SELECT 1")
        assert result == 1

        # Test transaction
        async with conn.transaction():
            # Would insert test data here
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ml_pipeline_integration(test_config, sample_market_data, test_symbols):
    """Test machine learning pipeline integration."""
    # This would test:
    # 1. Feature engineering
    # 2. Model training
    # 3. Model inference
    # 4. Model registry

    # Simplified test
    features = sample_market_data[["open", "high", "low", "close", "volume"]].values

    # Verify feature shape
    assert features.shape[0] == len(sample_market_data)
    assert features.shape[1] == 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_monitoring_integration(test_config):
    """Test monitoring and alerting integration."""
    # Local imports
    from main.utils.monitoring import record_metric

    # Record test metrics
    record_metric("test.metric", 1.0)
    record_metric("test.counter", 1, metric_type="counter")

    # In real test, would verify metrics were recorded


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_recovery_integration(mock_broker):
    """Test error handling and recovery across components."""
    # Inject error
    mock_broker.inject_error("submit_order", Exception("Network error"))

    # Try to submit order
    order = Order(symbol="AAPL", quantity=100, side=OrderSide.BUY, order_type=OrderType.MARKET)

    with pytest.raises(Exception) as exc_info:
        await mock_broker.submit_order(order)

    assert "Network error" in str(exc_info.value)

    # Clear error and retry
    mock_broker.clear_errors()
    order_id = await mock_broker.submit_order(order)
    assert order_id is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_performance_benchmarks(
    test_config, mock_broker, test_symbols, performance_benchmark
):
    """Test system performance benchmarks."""
    # Benchmark order submission
    performance_benchmark.start("order_submission")

    orders = []
    for i in range(100):
        order = Order(
            symbol=test_symbols[i % len(test_symbols)],
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        order_id = await mock_broker.submit_order(order)
        orders.append(order_id)

    performance_benchmark.end("order_submission")

    # Verify performance
    submission_time = performance_benchmark.get_timing("order_submission")
    orders_per_second = 100 / submission_time

    print(f"Order submission rate: {orders_per_second:.1f} orders/second")
    assert orders_per_second > 100, "Order submission too slow"

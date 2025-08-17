"""
Trading Commands Module

Handles all trading-related CLI commands including live trading, paper trading,
backtesting, and trade monitoring.
"""

# Standard library imports
import asyncio
from datetime import datetime, timedelta

# Third-party imports
import click

# Local imports
from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.utils.core import get_logger

logger = get_logger(__name__)


@click.group()
def trading():
    """Trading system commands."""
    pass


@trading.command()
@click.option(
    "--mode",
    type=click.Choice(["live", "paper", "backtest", "dev"]),
    default="paper",
    help="Trading mode",
)
@click.option("--symbols", help="Comma-separated symbols to trade (default: from config)")
@click.option("--strategy", help="Trading strategy to use (default: from config)")
@click.option("--enable-ml/--disable-ml", default=True, help="Enable/disable ML predictions")
@click.option("--disable-monitoring", is_flag=True, help="Disable monitoring dashboard")
@click.option("--disable-streaming", is_flag=True, help="Disable real-time streaming")
@click.option("--dashboard-port", type=int, default=None, help="Port for monitoring dashboard")
@click.option("--risk-limit", type=float, default=None, help="Maximum risk per trade (percentage)")
@click.option(
    "--position-limit", type=int, default=None, help="Maximum number of concurrent positions"
)
@click.option("--capital", type=float, default=None, help="Trading capital to use")
@click.option("--log-trades", is_flag=True, help="Log all trades to database")
@click.option("--dry-run", is_flag=True, help="Run without placing actual orders")
@click.pass_context
def trade(
    ctx,
    mode: str,
    symbols: str | None,
    strategy: str | None,
    enable_ml: bool,
    disable_monitoring: bool,
    disable_streaming: bool,
    dashboard_port: int | None,
    risk_limit: float | None,
    position_limit: int | None,
    capital: float | None,
    log_trades: bool,
    dry_run: bool,
):
    """Run the trading system with specified configuration.

    Examples:
        # Paper trading with default settings
        python ai_trader.py trading trade --mode paper

        # Live trading with specific symbols
        python ai_trader.py trading trade --mode live --symbols AAPL,GOOGL,MSFT

        # Backtesting with ML disabled
        python ai_trader.py trading trade --mode backtest --disable-ml

        # Paper trading with risk limits
        python ai_trader.py trading trade --mode paper --risk-limit 2.0 --position-limit 5
    """
    # Local imports
    from main.models.inference.model_registry import ModelRegistry
    from main.monitoring.dashboards.v2.trading_dashboard_v2 import TradingDashboard
    from main.trading_engine.brokers.broker_factory import BrokerFactory
    from main.trading_engine.core.trading_system import TradingSystem

    logger.info(f"Starting trading system in {mode.upper()} mode")

    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    # Override config with CLI arguments
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        config.trading.symbols = symbol_list

    if strategy:
        config.trading.strategy = strategy

    if risk_limit is not None:
        config.trading.risk_management.max_risk_per_trade = risk_limit / 100

    if position_limit is not None:
        config.trading.risk_management.max_positions = position_limit

    if capital is not None:
        config.trading.capital = capital

    if dashboard_port:
        config.monitoring.dashboard.port = dashboard_port

    # Initialize components
    try:
        # Create database connection
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(config)

        # Initialize broker
        broker_factory = BrokerFactory()
        broker = broker_factory.create_broker(mode, config)

        # Initialize model registry if ML is enabled
        model_registry = None
        if enable_ml:
            model_registry = ModelRegistry(config)
            # Load models as needed

        # Initialize dashboard if monitoring is enabled
        dashboard = None
        if not disable_monitoring:
            dashboard = TradingDashboard(config)
            dashboard.start()
            logger.info(f"Dashboard running on port {config.monitoring.dashboard.port}")

        # Create and run trading system
        trading_system = TradingSystem(config=config, broker=broker, db_adapter=db_adapter)

        # Configure options
        if dry_run:
            trading_system.set_dry_run(True)

        # Run the trading system
        asyncio.run(trading_system.run())

    except KeyboardInterrupt:
        logger.info("Trading system stopped by user")
    except Exception as e:
        logger.error(f"Trading system error: {e}", exc_info=True)
        raise
    finally:
        if dashboard:
            dashboard.stop()
        if db_adapter:
            asyncio.run(db_adapter.close())


@trading.command()
@click.option("--model-path", help="Path to saved model (or use --model-type for latest)")
@click.option(
    "--model-type", type=click.Choice(["xgboost", "lstm", "ensemble"]), help="Type of model to use"
)
@click.option("--symbols", help="Comma-separated list of symbols to backtest")
@click.option("--start-date", help="Backtest start date (YYYY-MM-DD)")
@click.option("--end-date", help="Backtest end date (YYYY-MM-DD)")
@click.option("--initial-capital", type=float, default=100000, help="Initial capital for backtest")
@click.option("--commission", type=float, default=0.001, help="Commission rate (default: 0.1%)")
@click.option("--slippage", type=float, default=0.0005, help="Slippage rate (default: 0.05%)")
@click.option("--strategy", help="Trading strategy to use")
@click.option("--compare", is_flag=True, help="Compare multiple models")
@click.option("--output", help="Output file for results (CSV or JSON)")
@click.pass_context
def backtest(
    ctx,
    model_path: str | None,
    model_type: str | None,
    symbols: str | None,
    start_date: str | None,
    end_date: str | None,
    initial_capital: float,
    commission: float,
    slippage: float,
    strategy: str | None,
    compare: bool,
    output: str | None,
):
    """Run backtesting simulation with historical data.

    Examples:
        # Backtest with XGBoost model
        python ai_trader.py trading backtest --model-type xgboost --symbols AAPL,GOOGL

        # Backtest specific date range
        python ai_trader.py trading backtest --start-date 2024-01-01 --end-date 2024-12-31

        # Compare multiple models
        python ai_trader.py trading backtest --compare --output results.csv
    """
    # Local imports
    from main.backtesting.engine.backtest_engine import BacktestEngine

    logger.info("Starting backtest simulation")

    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    # Parse dates
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start = datetime.now() - timedelta(days=365)

    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end = datetime.now()

    # Parse symbols
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        symbol_list = config.trading.symbols

    try:
        # Initialize backtest engine
        engine = BacktestEngine(
            config=config, initial_capital=initial_capital, commission=commission, slippage=slippage
        )

        if compare:
            # Compare multiple models
            results = asyncio.run(
                engine.compare_models(
                    symbol_list, start, end, models=["xgboost", "lstm", "ensemble"]
                )
            )
            _print_comparison_results(results)
        else:
            # Run single backtest
            results = asyncio.run(
                engine.run_backtest(
                    symbols=symbol_list,
                    start_date=start,
                    end_date=end,
                    model_path=model_path,
                    model_type=model_type,
                    strategy=strategy,
                )
            )
            _print_backtest_results(results)

        # Save results if output specified
        if output:
            _save_backtest_results(results, output)
            logger.info(f"Results saved to {output}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise


@trading.command()
@click.option("--symbols", help="Comma-separated list of symbols")
@click.option(
    "--period",
    type=click.Choice(["1d", "1w", "1m", "3m", "ytd", "all"]),
    default="1d",
    help="Time period for positions",
)
@click.option(
    "--format", type=click.Choice(["table", "json", "csv"]), default="table", help="Output format"
)
@click.pass_context
def positions(ctx, symbols: str | None, period: str, format: str):
    """View current trading positions and P&L.

    Examples:
        # View all positions
        python ai_trader.py trading positions

        # View specific symbols
        python ai_trader.py trading positions --symbols AAPL,GOOGL

        # Export to CSV
        python ai_trader.py trading positions --format csv > positions.csv
    """
    # Third-party imports
    import pandas as pd

    # Local imports
    from main.trading_engine.core.position_manager import PositionManager

    logger.info("Fetching trading positions")

    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    # Parse symbols
    symbol_list = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]

    try:
        # Initialize position manager
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(config)

        position_manager = PositionManager(db_adapter)

        # Fetch positions
        positions_data = asyncio.run(
            position_manager.get_positions(symbols=symbol_list, period=period)
        )

        # Format and display
        if format == "table":
            df = pd.DataFrame(positions_data)
            print(df.to_string())
        elif format == "json":
            # Standard library imports
            import json

            print(json.dumps(positions_data, indent=2, default=str))
        elif format == "csv":
            df = pd.DataFrame(positions_data)
            print(df.to_csv(index=False))

    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}", exc_info=True)
        raise
    finally:
        if db_adapter:
            asyncio.run(db_adapter.close())


@trading.command()
@click.option(
    "--period",
    type=click.Choice(["1d", "1w", "1m", "3m", "ytd", "all"]),
    default="1d",
    help="Time period for performance",
)
@click.option("--detailed", is_flag=True, help="Show detailed metrics")
@click.pass_context
def performance(ctx, period: str, detailed: bool):
    """View trading performance metrics.

    Examples:
        # View today's performance
        python ai_trader.py trading performance

        # View monthly performance with details
        python ai_trader.py trading performance --period 1m --detailed
    """
    # Third-party imports
    import pandas as pd

    # Local imports
    from main.monitoring.performance.performance_tracker import PerformanceTracker

    logger.info(f"Calculating performance for period: {period}")

    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    try:
        # Initialize performance tracker
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(config)

        tracker = PerformanceTracker(db_adapter)

        # Calculate metrics
        metrics = asyncio.run(tracker.calculate_metrics(period=period))

        # Display results
        if detailed:
            # Show detailed metrics
            for key, value in metrics.items():
                print(f"{key}: {value}")
        else:
            # Show summary
            df = pd.DataFrame([metrics])
            print(df.to_string())

    except Exception as e:
        logger.error(f"Failed to calculate performance: {e}", exc_info=True)
        raise
    finally:
        if db_adapter:
            asyncio.run(db_adapter.close())


# Helper functions


def _print_backtest_results(results: dict):
    """Print formatted backtest results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    # Performance metrics
    print("\n📊 Performance Metrics:")
    print(f"  Total Return: {results.get('total_return', 0):.2%}")
    print(f"  Annual Return: {results.get('annual_return', 0):.2%}")
    print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"  Win Rate: {results.get('win_rate', 0):.2%}")

    # Trade statistics
    print("\n📈 Trade Statistics:")
    print(f"  Total Trades: {results.get('total_trades', 0)}")
    print(f"  Winning Trades: {results.get('winning_trades', 0)}")
    print(f"  Losing Trades: {results.get('losing_trades', 0)}")
    print(f"  Avg Win: ${results.get('avg_win', 0):.2f}")
    print(f"  Avg Loss: ${results.get('avg_loss', 0):.2f}")

    # Portfolio metrics
    print("\n💰 Portfolio Metrics:")
    print(f"  Starting Capital: ${results.get('starting_capital', 0):,.2f}")
    print(f"  Ending Capital: ${results.get('ending_capital', 0):,.2f}")
    print(f"  Peak Value: ${results.get('peak_value', 0):,.2f}")

    print("\n" + "=" * 60)


def _print_comparison_results(results: dict):
    """Print model comparison results."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    for model_name, metrics in results.items():
        print(f"\n📊 {model_name.upper()} Model:")
        print(f"  Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max DD: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1].get("sharpe_ratio", 0))
    print(f"\n🏆 Best Model: {best_model[0].upper()}")
    print("=" * 60)


def _save_backtest_results(results: dict, output_path: str):
    """Save backtest results to file."""
    # Standard library imports
    import json

    # Third-party imports
    import pandas as pd

    if output_path.endswith(".json"):
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
    elif output_path.endswith(".csv"):
        df = pd.DataFrame([results])
        df.to_csv(output_path, index=False)
    else:
        logger.warning(f"Unknown output format for {output_path}, saving as JSON")
        with open(output_path + ".json", "w") as f:
            json.dump(results, f, indent=2, default=str)

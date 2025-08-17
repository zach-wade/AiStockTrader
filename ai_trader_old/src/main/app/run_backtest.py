"""
Backtest runner for evaluating ML models on historical data.

This module provides functionality to run backtests using saved ML models,
generating comprehensive performance reports and metrics.
"""

# Standard library imports
from datetime import datetime, timedelta
import logging
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.backtesting.engine.backtest_engine import BacktestConfig, BacktestMode
from main.backtesting.factories import BacktestEngineFactory
from main.config.config_manager import get_config
from main.feature_pipeline.feature_store_compat import FeatureStore
from main.models.strategies.ml_model_strategy import MLModelStrategy
from main.models.utils.model_loader import ModelLoader, list_available_models

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Runner for ML model backtesting.

    Coordinates loading models, preparing data, running backtests,
    and generating performance reports.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize the backtest runner.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config()
        self.model_loader = ModelLoader()
        self.feature_store = FeatureStore(config=self.config)
        self.backtest_factory = BacktestEngineFactory()

        # Default backtest parameters
        self.default_lookback_days = self.config.get("backtesting", {}).get(
            "default_lookback_days", 365
        )
        self.default_initial_cash = self.config.get("backtesting", {}).get("initial_cash", 100000)

    async def run_model_backtest(
        self,
        model_path: str,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        initial_cash: float | None = None,
    ) -> dict[str, Any]:
        """
        Run backtest for a specific model.

        Args:
            model_path: Path to the saved model
            symbols: List of symbols to test
            start_date: Backtest start date
            end_date: Backtest end date
            initial_cash: Starting capital

        Returns:
            Dictionary containing backtest results and metrics
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.default_lookback_days)
        if initial_cash is None:
            initial_cash = self.default_initial_cash

        logger.info(f"Starting backtest for model: {model_path}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Initial cash: ${initial_cash:,.2f}")

        try:
            # Create ML model strategy
            strategy = MLModelStrategy(
                model_path=model_path,
                config=self.config,
                feature_engine=None,  # Will use feature store directly
            )

            # Get model info
            model_info = strategy.get_model_info()
            logger.info(
                f"Loaded {model_info['model_type']} model with {model_info['features_count']} features"
            )

            # Create backtest config
            backtest_config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                symbols=symbols,
                mode=BacktestMode.PORTFOLIO,
                use_adjusted_prices=True,
                include_dividends=True,
                include_splits=True,
            )

            # Create backtest engine
            backtest_engine = self.backtest_factory.create(
                config=backtest_config,
                strategy=strategy,
                data_source=self.feature_store,
                cost_model=None,  # Use default
            )

            # Run backtest
            logger.info("Running backtest...")
            result = await backtest_engine.run()

            # Process results
            results = self._process_backtest_results(result, model_info)

            logger.info("Backtest completed successfully")
            return results

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "model_path": model_path}

    async def run_multiple_backtests(
        self,
        model_paths: list[str],
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        initial_cash: float | None = None,
    ) -> dict[str, Any]:
        """
        Run backtests for multiple models and compare results.

        Args:
            model_paths: List of model paths to test
            symbols: List of symbols to test
            start_date: Backtest start date
            end_date: Backtest end date
            initial_cash: Starting capital

        Returns:
            Dictionary containing comparison results
        """
        results = {}

        for model_path in model_paths:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing model: {model_path}")
            logger.info(f"{'='*60}")

            result = await self.run_model_backtest(
                model_path=model_path,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
            )

            results[model_path] = result

        # Generate comparison
        comparison = self._compare_results(results)

        return {"individual_results": results, "comparison": comparison}

    def _process_backtest_results(self, result: Any, model_info: dict[str, Any]) -> dict[str, Any]:
        """Process raw backtest results into formatted output."""
        # BacktestResult has: config, portfolio_history, trades, metrics, equity_curve,
        # drawdown_curve, positions_history, events_processed, execution_time

        metrics = result.metrics if hasattr(result, "metrics") else {}

        # Calculate additional metrics if needed
        if not result.trades.empty:
            trades_df = result.trades

            # Win/loss analysis
            if "pnl" in trades_df.columns:
                winning_trades = trades_df[trades_df["pnl"] > 0]
                losing_trades = trades_df[trades_df["pnl"] < 0]

                metrics["winning_trades"] = len(winning_trades)
                metrics["losing_trades"] = len(losing_trades)
                metrics["avg_win"] = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
                metrics["avg_loss"] = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0

            # Trade frequency
            if "timestamp" in trades_df.columns:
                trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
                trading_days = (trades_df["timestamp"].max() - trades_df["timestamp"].min()).days
                metrics["trades_per_day"] = len(trades_df) / max(trading_days, 1)

        # Format results
        formatted_results = {
            "success": True,
            "model_info": model_info,
            "backtest_config": {
                "start_date": (
                    result.config.start_date.isoformat()
                    if hasattr(result.config.start_date, "isoformat")
                    else str(result.config.start_date)
                ),
                "end_date": (
                    result.config.end_date.isoformat()
                    if hasattr(result.config.end_date, "isoformat")
                    else str(result.config.end_date)
                ),
                "initial_cash": result.config.initial_cash,
                "symbols": result.config.symbols,
            },
            "performance_metrics": {
                "total_return": metrics.get("total_return", 0),
                "annual_return": metrics.get("annual_return", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "volatility": metrics.get("volatility", 0),
                "final_equity": metrics.get("final_equity", result.config.initial_cash),
            },
            "trade_metrics": {
                "total_trades": metrics.get(
                    "total_trades", len(result.trades) if hasattr(result, "trades") else 0
                ),
                "win_rate": metrics.get("win_rate", 0),
                "winning_trades": metrics.get("winning_trades", 0),
                "losing_trades": metrics.get("losing_trades", 0),
                "avg_win": metrics.get("avg_win", 0),
                "avg_loss": metrics.get("avg_loss", 0),
                "trades_per_day": metrics.get("trades_per_day", 0),
                "total_commission": metrics.get("total_commission", 0),
            },
            "execution_info": {
                "events_processed": (
                    result.events_processed if hasattr(result, "events_processed") else 0
                ),
                "execution_time": result.execution_time if hasattr(result, "execution_time") else 0,
            },
        }

        # Add equity curve data
        if not result.equity_curve.empty:
            formatted_results["equity_curve"] = {
                "dates": result.equity_curve.index.tolist(),
                "values": result.equity_curve.values.tolist(),
            }

        # Add drawdown data
        if not result.drawdown_curve.empty:
            formatted_results["drawdown_curve"] = {
                "dates": result.drawdown_curve.index.tolist(),
                "values": result.drawdown_curve.values.tolist(),
            }

        return formatted_results

    def _compare_results(self, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Compare results from multiple backtests."""
        comparison = {"summary": [], "best_by_metric": {}}

        # Metrics to compare
        metrics_to_compare = [
            ("total_return", "max"),
            ("sharpe_ratio", "max"),
            ("max_drawdown", "min"),
            ("win_rate", "max"),
            ("volatility", "min"),
        ]

        # Collect summary data
        for model_path, result in results.items():
            if not result.get("success", False):
                continue

            perf = result.get("performance_metrics", {})
            trade = result.get("trade_metrics", {})

            summary_row = {
                "model_path": model_path,
                "model_type": result.get("model_info", {}).get("model_type", "unknown"),
                "total_return": perf.get("total_return", 0),
                "sharpe_ratio": perf.get("sharpe_ratio", 0),
                "max_drawdown": perf.get("max_drawdown", 0),
                "win_rate": trade.get("win_rate", 0),
                "total_trades": trade.get("total_trades", 0),
                "volatility": perf.get("volatility", 0),
                "final_equity": perf.get("final_equity", 0),
            }

            comparison["summary"].append(summary_row)

        # Find best model for each metric
        if comparison["summary"]:
            summary_df = pd.DataFrame(comparison["summary"])

            for metric, optimization in metrics_to_compare:
                if metric in summary_df.columns:
                    if optimization == "max":
                        best_idx = summary_df[metric].idxmax()
                    else:
                        best_idx = summary_df[metric].idxmin()

                    best_model = summary_df.iloc[best_idx]
                    comparison["best_by_metric"][metric] = {
                        "model_path": best_model["model_path"],
                        "model_type": best_model["model_type"],
                        "value": best_model[metric],
                    }

        return comparison


async def run_backtest(
    model_path: str,
    symbols: list[str],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    initial_cash: float = 100000,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Convenience function to run a single model backtest.

    Args:
        model_path: Path to saved model
        symbols: List of symbols to test
        start_date: Backtest start date
        end_date: Backtest end date
        initial_cash: Starting capital
        config: Configuration dictionary

    Returns:
        Backtest results
    """
    runner = BacktestRunner(config)
    return await runner.run_model_backtest(
        model_path=model_path,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
    )


def find_and_list_models(models_dir: str = "models") -> None:
    """Print available models for backtesting."""
    models = list_available_models(models_dir)

    if not models:
        print("No models found")
        return

    print("\nAvailable models for backtesting:")
    print("=" * 80)

    for model_type, model_list in models.items():
        print(f"\n{model_type.upper()} Models:")
        print("-" * 40)

        for i, model in enumerate(model_list[:5]):  # Show top 5
            print(f"{i+1}. {model['timestamp']}")

            if "metrics" in model:
                metrics = model["metrics"]
                print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
                print(f"   F1 Score: {metrics.get('f1_score', 'N/A'):.3f}")

            if "symbols" in model:
                print(f"   Trained on: {', '.join(model['symbols'][:3])}")

            print(f"   Path: {model['path']}")

        if len(model_list) > 5:
            print(f"   ... and {len(model_list) - 5} more")

    print("\n" + "=" * 80)

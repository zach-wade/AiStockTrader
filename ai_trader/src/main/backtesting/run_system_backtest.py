"""
Full System Backtest Runner & Validation Orchestrator.

This script orchestrates a comprehensive, three-phase backtesting process.
Phase 1: Selects a high-quality universe of tradable symbols.
Phase 2: Runs all strategies across the curated universe to find the top performer.
Phase 3: Subjects the top performer to a rigorous walk-forward validation.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

# Import all necessary components from their correct locations in the project
from main.config.config_manager import get_config
from omegaconf import DictConfig
from main.data_pipeline.historical.manager import HistoricalManager
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine
from main.utils.core import setup_logging
from main.data_pipeline.ingestion.data_source_manager import DataSourceManager
from main.data_pipeline.historical.data_fetcher import DataFetcher

# Import the core backtesting components
from .engine.backtest_engine import BacktestEngine
from .analysis.performance_metrics import PerformanceAnalyzer
from .analysis.validation_suite import StrategyValidationSuite
from .analysis.symbol_selector import SymbolSelector

# Import all official strategies to be tested
from main.models.strategies.base_strategy import BaseStrategy
from main.models.strategies.mean_reversion import MeanReversionStrategy
from main.models.strategies.ml_momentum import MLMomentumStrategy
from main.models.strategies.breakout import BreakoutStrategy
from main.models.strategies.ensemble.main_ensemble import AdvancedStrategyEnsemble
from main.interfaces.database import IAsyncDatabase
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.ingestion.base_source import BaseSource
logger = logging.getLogger(__name__)


class SystemBacktestRunner:
    """
    Orchestrates a comprehensive backtesting and validation process.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initializes the runner and all its required components via dependency injection.
        """
        self.config = config
        
        db_factory = DatabaseFactory()
        self.db_adapter: IAsyncDatabase = db_factory.create_async_database(config.model_dump())
        self.data_source_manager = DataSourceManager(config)
        clients: Dict[str, BaseSource] = self.data_source_manager.clients
        self.data_fetcher = DataFetcher(
            config=config,
            db_adapter=self.db_adapter,
            clients=clients
        )
        self.data_provider = HistoricalManager(
            config=config,
            db_adapter=self.db_adapter,
            clients=clients,
            data_fetcher=self.data_fetcher
        )
        self.feature_engine = UnifiedFeatureEngine(config)
        self.backtest_engine = BacktestEngine(config)
        self.performance_analyzer = PerformanceAnalyzer()
        self.symbol_selector = SymbolSelector(config)
        self.validation_suite = StrategyValidationSuite(
            config=self.config,
            backtest_engine=self.backtest_engine,
            performance_analyzer=self.performance_analyzer
        )
        
        self.strategies: Dict[str, BaseStrategy] = self._initialize_strategies()
        logger.info(f"SystemBacktestRunner initialized with {len(self.strategies)} strategies.")

    def _initialize_strategies(self) -> Dict[str, BaseStrategy]:
        """Creates instances of all strategies to be included in the backtest."""
        # FIX: Correctly pass both the `Config` object and feature_engine instance.
        # This resolves the "Expected 1 positional argument" error for strategies.
        strategies = {
            "MeanReversion": MeanReversionStrategy(self.config, self.feature_engine),
            "MLMomentum": MLMomentumStrategy(self.config, self.feature_engine),
            "Breakout": BreakoutStrategy(self.config, self.feature_engine),
            "AdvancedEnsemble": AdvancedStrategyEnsemble(self.config, self.feature_engine)
        }
        return strategies

    async def run_all_backtests(self, broad_universe_symbols: List[str], start_date: datetime, end_date: datetime):
        """
        Runs the full backtest workflow, including universe selection and deep validation.
        """
        all_results = defaultdict(list)
        
        # --- PHASE 1: UNIVERSE SELECTION & BROAD SCAN ---
        logger.info("="*80)
        logger.info("PHASE 1: Starting Universe Selection and Quality Analysis")
        logger.info("="*80)

        logger.info(f"Fetching data for {len(broad_universe_symbols)} candidate symbols...")
        # FIX: Use the correct method name from the data provider.
        historical_data_map = await self.data_provider.get_bulk_daily_data(broad_universe_symbols, start_date, end_date)

        symbol_metrics = await self.symbol_selector.analyze_symbols(
            symbols=broad_universe_symbols,
            market_data=historical_data_map
        )

        tradable_symbols = self.symbol_selector.select_top_symbols(
            symbol_metrics=symbol_metrics,
            max_symbols=self.config.get('backtesting', {}).get('max_symbols_to_test', 50)
        )

        if not tradable_symbols:
            logger.error("Universe selection resulted in 0 tradable symbols. Halting backtest.")
            return

        logger.info(f"Universe selection complete. Proceeding to backtest {len(tradable_symbols)} high-quality symbols.")

        logger.info("="*80)
        logger.info("PHASE 2: Running Backtests on Curated Universe")
        logger.info("="*80)

        for symbol in tradable_symbols:
            symbol_data = historical_data_map.get(symbol)
            if symbol_data is None or symbol_data.empty:
                continue
            
            score_metric = symbol_metrics.get(symbol)
            # FIX: Safely access score, providing a default if not found.
            score = score_metric.overall_score if score_metric else 0.0
            logger.info(f"--- Processing Symbol: {symbol} (Overall Score: {score:.1f}) ---")
            
            # FIX: The feature engine's method is `calculate_features`.
            features = self.feature_engine.calculate_features(symbol, symbol_data)
            
            for name, strategy in self.strategies.items():
                try:
                    # FIX: The backtest engine now returns a dictionary with 'equity_curve' and 'trades'.
                    backtest_result = await self.backtest_engine.run(strategy, symbol, features)
                    
                    # FIX: Pass both required arguments to the performance analyzer's correct method name.
                    metrics = self.performance_analyzer.calculate_metrics(
                        equity_curve=backtest_result['equity_curve'],
                        trades=backtest_result['trades']
                    )
                    metrics['strategy'] = name
                    metrics['symbol'] = symbol
                    all_results[name].append(metrics)
                    
                    logger.info(f"✓ {name}: Sharpe={metrics.get('sharpe_ratio', 0):.2f}, Return={metrics.get('total_return_pct', 0):.2%}")
                
                except Exception as e:
                    logger.error(f"Backtest failed for strategy '{name}' on symbol '{symbol}': {e}", exc_info=True)
                    all_results[name].append({'error': str(e), 'strategy': name, 'symbol': symbol})
            
        # FIX: Correctly capture the returned DataFrame.
        summary_df = self._generate_summary_report(all_results)

        # --- PHASE 3: DEEP DIVE VALIDATION ---
        if summary_df is not None and not summary_df.empty:
            await self._run_deep_validation_on_best_strategy(summary_df, historical_data_map)
        else:
            logger.warning("No successful strategies to run deep validation on.")

    def _generate_summary_report(self, all_results: Dict[str, List[Dict]]) -> Optional[pd.DataFrame]:
        """Aggregates results, prints a summary, and returns the summary DataFrame."""
        logger.info("--- Generating Final Backtest Summary Report ---")
        summary_data = []
        for strategy_name, results_list in all_results.items():
            valid_results = [r for r in results_list if 'error' not in r]
            if not valid_results: continue
            
            df = pd.DataFrame(valid_results)
            avg_metrics = df.mean(numeric_only=True)
            summary_data.append({'Strategy': strategy_name, **avg_metrics})

        if not summary_data:
            logger.warning("No successful backtests to report.")
            return None

        summary_df = pd.DataFrame(summary_data).set_index('Strategy')
        summary_df = summary_df.sort_values(by='sharpe_ratio', ascending=False)
        
        print("\n" + "="*80)
        print("                        PHASE 1 & 2: BROAD SCAN SUMMARY")
        print("="*80)
        print(summary_df[['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate']].to_string(float_format="%.3f"))
        print("="*80)
        
        return summary_df

    async def _run_deep_validation_on_best_strategy(self, summary_df: pd.DataFrame, historical_data_map: Dict[str, pd.DataFrame]):
        """Identifies the best strategy and runs a walk-forward analysis on it."""
        best_strategy_name = summary_df['sharpe_ratio'].idxmax()
        best_strategy_obj = self.strategies[best_strategy_name]
        
        validation_symbol = self.config.get('backtesting', {}).get('validation_symbol', 'SPY')
        if validation_symbol not in historical_data_map:
            logger.error(f"Validation symbol '{validation_symbol}' not found. Cannot run deep validation.")
            return

        logger.info("\n" + "="*80)
        logger.info(f"        PHASE 3: DEEP DIVE VALIDATION on '{best_strategy_name}' using '{validation_symbol}'")
        logger.info("="*80)

        validation_features = self.feature_engine.calculate_features(
            validation_symbol, historical_data_map[validation_symbol]
        )

        await self.validation_suite.run_walk_forward_analysis(
            strategy=best_strategy_obj,
            symbol=validation_symbol,
            features=validation_features
        )

async def main():
    """Main entry point for the backtest runner script."""
    setup_logging()
    
    config = get_config()
    runner = SystemBacktestRunner(config)
    
    backtest_config = config.get('backtesting', {})
    end_date = datetime.now()
    start_date = end_date - timedelta(days=backtest_config.get('default_lookback_days', 365 * 2))
    
    broad_universe = backtest_config.get('broad_universe', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'V', 'SPY'])
    
    await runner.run_all_backtests(broad_universe, start_date, end_date)

if __name__ == "__main__":
    asyncio.run(main())

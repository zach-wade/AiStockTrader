"""
Advanced Strategy Validation Suite.

Provides a toolkit for rigorous, out-of-sample strategy validation, including
walk-forward analysis and Monte Carlo simulations to test robustness.
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dataclasses import dataclass

# Import our official, specialized components
from ..engine.backtest_engine import BacktestEngine
from .performance_metrics import PerformanceAnalyzer
from main.models.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Stores the results from a single walk-forward period."""
    train_period: Tuple[datetime, datetime]
    test_period: Tuple[datetime, datetime]
    in_sample_metrics: Dict[str, float]
    out_of_sample_metrics: Dict[str, float]


class StrategyValidationSuite:
    """
    A toolkit for advanced, out-of-sample validation of trading strategies.
    This class uses a BacktestEngine to run simulations.
    """
    
    def __init__(self, config: Dict, backtest_engine: BacktestEngine, performance_analyzer: PerformanceAnalyzer):
        """
        Initializes the validation suite with its required tools.

        Args:
            config: The global application configuration.
            backtest_engine: The core engine used to run simulations.
            performance_analyzer: The utility for calculating performance metrics.
        """
        self.config = config
        self.validation_config = config.get('validation', {})
        self.backtest_engine = backtest_engine
        self.performance_analyzer = performance_analyzer
        logger.info("StrategyValidationSuite initialized.")

    async def run_walk_forward_analysis(
        self, 
        strategy: BaseStrategy, 
        symbol: str,
        features: pd.DataFrame
    ) -> List[WalkForwardResult]:
        """
        Runs a full walk-forward analysis for a single strategy and symbol.

        Args:
            strategy: The strategy instance to test.
            symbol: The symbol to test on.
            features: A DataFrame containing the full history of features for the symbol.

        Returns:
            A list of WalkForwardResult objects, one for each period.
        """
        logger.info(f"--- Starting Walk-Forward Analysis for {strategy.name} on {symbol} ---")
        
        periods = self._generate_walk_forward_periods(features.index)
        logger.info(f"Generated {len(periods)} walk-forward periods to analyze.")
        
        all_period_results = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            logger.info(f"Processing Period {i+1}/{len(periods)}: Test from {test_start.date()} to {test_end.date()}")
            
            # In a true walk-forward, the strategy would be re-trained here on the train_set
            # For now, we simulate by running the same strategy on two different data slices.
            
            # Run on in-sample (training) data
            train_features = features.loc[train_start:train_end]
            in_sample_equity = await self.backtest_engine.run(strategy, symbol, train_features)
            in_sample_metrics = self.performance_analyzer.calculate_all_metrics(in_sample_equity)

            # Run on out-of-sample (testing) data
            test_features = features.loc[test_start:test_end]
            out_of_sample_equity = await self.backtest_engine.run(strategy, symbol, test_features)
            out_of_sample_metrics = self.performance_analyzer.calculate_all_metrics(out_of_sample_equity)
            
            all_period_results.append(WalkForwardResult(
                train_period=(train_start, train_end),
                test_period=(test_start, test_end),
                in_sample_metrics=in_sample_metrics,
                out_of_sample_metrics=out_of_sample_metrics
            ))
            logger.info(f"  In-Sample Sharpe: {in_sample_metrics.get('sharpe_ratio', 0):.2f} -> "
                        f"Out-of-Sample Sharpe: {out_of_sample_metrics.get('sharpe_ratio', 0):.2f}")

        self._log_walk_forward_summary(all_period_results)
        return all_period_results

    def _generate_walk_forward_periods(self, date_index: pd.DatetimeIndex) -> List[tuple]:
        """Generates rolling training and testing periods."""
        train_months = self.validation_config.get('walk_forward_train_months', 12)
        test_months = self.validation_config.get('walk_forward_test_months', 3)
        step_months = self.validation_config.get('walk_forward_step_months', 3)
        
        periods = []
        start_date = date_index.min()
        end_date = date_index.max()
        
        current_date = start_date
        while current_date + pd.DateOffset(months=train_months + test_months) <= end_date:
            train_start = current_date
            train_end = current_date + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            
            periods.append((train_start, train_end, test_start, test_end))
            current_date += pd.DateOffset(months=step_months)
            
        return periods

    def _log_walk_forward_summary(self, results: List[WalkForwardResult]):
        """Analyzes and logs the summary of a walk-forward test."""
        if not results:
            logger.warning("No walk-forward periods were successfully analyzed.")
            return

        oos_sharpes = [r.out_of_sample_metrics.get('sharpe_ratio', 0) for r in results]
        is_sharpes = [r.in_sample_metrics.get('sharpe_ratio', 0) for r in results]
        
        degradation = [(is_sharpe - oos_sharpe) / abs(is_sharpe) if abs(is_sharpe) > 0.1 else 0 
                       for is_sharpe, oos_sharpe in zip(is_sharpes, oos_sharpes)]

        summary = {
            "Total Periods": len(results),
            "Avg Out-of-Sample Sharpe": np.mean(oos_sharpes),
            "Out-of-Sample Sharpe Std Dev": np.std(oos_sharpes),
            "Avg Degradation": np.mean(degradation),
            "Periods with Positive Sharpe": sum(1 for s in oos_sharpes if s > 0),
        }
        
        logger.info("--- Walk-Forward Analysis Summary ---")
        for key, value in summary.items():
            logger.info(f"  {key}: {value:.3f}")
        logger.info("------------------------------------")
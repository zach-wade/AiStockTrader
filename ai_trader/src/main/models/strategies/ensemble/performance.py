"""
Encapsulates performance tracking for all strategies within the ensemble.

This enhanced version calculates sophisticated performance metrics like Sharpe Ratio
and Maximum Drawdown, providing the necessary inputs for the advanced
WeightAllocator.
"""
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Deque
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """
    ENHANCED: A more detailed data structure for tracking the comprehensive
    performance metrics required by advanced allocation models.
    """
    strategy_name: str
    total_signals: int = 0
    winning_signals: int = 0
    # Core metrics used by the allocator
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    # Data needed for calculations
    recent_returns: Deque[float] = field(default_factory=lambda: deque(maxlen=252)) # Store ~1 year of daily returns
    cumulative_returns: Deque[float] = field(default_factory=lambda: deque(maxlen=252))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceTracker:
    """
    Manages and calculates detailed performance metrics for all component strategies,
    creating a feedback loop for the WeightAllocator.
    """
    def __init__(self, strategy_names: list[str]):
        self.strategy_performance: Dict[str, StrategyPerformance] = {
            name: StrategyPerformance(strategy_name=name)
            for name in strategy_names
        }
        # Initialize cumulative returns with a starting value of 1
        for perf in self.strategy_performance.values():
            perf.cumulative_returns.append(1.0)
            
        logger.info(f"PerformanceTracker initialized for {len(strategy_names)} strategies.")

    def update(self, trade_result: Dict[str, Any]):
        """
        ENHANCED: Updates performance for contributing strategies and recalculates
        all advanced metrics on every new trade result.
        """
        ensemble_data = trade_result.get('metadata', {}).get('ensemble_data', {})
        contributing_strategies = ensemble_data.get('contributing_strategies', [])
        
        if not contributing_strategies:
            return

        for strategy_name in contributing_strategies:
            if strategy_name not in self.strategy_performance:
                logger.warning(f"Performance update for unknown strategy '{strategy_name}'. Skipping.")
                continue

            perf = self.strategy_performance[strategy_name]
            # We assume the return provided in trade_result is the weighted return for this strategy
            trade_return = trade_result.get('return', 0.0)

            # --- Step 1: Record the new return ---
            perf.recent_returns.append(trade_return)
            perf.total_signals += 1
            if trade_return > 0:
                perf.winning_signals += 1
            
            # --- Step 2: Recalculate all metrics with the new data point ---
            perf.win_rate = self._calculate_win_rate(perf)
            perf.sharpe_ratio = self._calculate_sharpe_ratio(perf)
            perf.max_drawdown = self._calculate_max_drawdown(perf, trade_return)
            perf.last_updated = datetime.now(timezone.utc)
            
            logger.debug(f"Updated performance for {strategy_name}: Sharpe={perf.sharpe_ratio:.2f}, Drawdown={perf.max_drawdown:.2%}")

    def get_performance_data(self) -> Dict[str, StrategyPerformance]:
        """Returns the current performance data for all strategies."""
        return self.strategy_performance

    # --- NEW: Private helper methods for calculating metrics ---

    def _calculate_win_rate(self, perf: StrategyPerformance) -> float:
        """Calculates the win rate."""
        if perf.total_signals == 0:
            return 0.0
        return perf.winning_signals / perf.total_signals

    def _calculate_sharpe_ratio(self, perf: StrategyPerformance, risk_free_rate: float = 0.0) -> float:
        """Calculates the annualized Sharpe ratio."""
        returns = np.array(perf.recent_returns)
        if len(returns) < 10 or np.std(returns) == 0:
            return 0.0
            
        # Assuming daily returns, annualize by sqrt(252)
        daily_excess_return = np.mean(returns) - (risk_free_rate / 252)
        annualized_excess_return = daily_excess_return * 252
        annualized_volatility = np.std(returns) * np.sqrt(252)
        
        return annualized_excess_return / annualized_volatility

    def _calculate_max_drawdown(self, perf: StrategyPerformance, latest_return: float) -> float:
        """
        Efficiently updates the max drawdown calculation.
        """
        # Update cumulative return series
        last_cumulative = perf.cumulative_returns[-1]
        perf.cumulative_returns.append(last_cumulative * (1 + latest_return))
        
        # Recalculate drawdown from the stored cumulative series
        cumulative_series = np.array(perf.cumulative_returns)
        running_max = np.maximum.accumulate(cumulative_series)
        drawdowns = (cumulative_series - running_max) / running_max
        
        # The max drawdown is the minimum value in the drawdowns series
        return np.min(drawdowns) if len(drawdowns) > 0 else 0.0
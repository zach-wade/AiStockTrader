"""
Backtesting factory implementations.

This module provides concrete implementations of backtesting interfaces
to support dependency injection and clean architecture.
"""

from typing import Any, Optional
from main.interfaces.backtesting import (
    IBacktestEngine, IBacktestEngineFactory, BacktestConfig
)
from .engine.backtest_engine import BacktestEngine
from .engine.cost_model import create_default_cost_model


class BacktestEngineFactory:
    """Factory for creating BacktestEngine instances."""
    
    def create(
        self,
        config: BacktestConfig,
        strategy: Any,
        data_source: Any = None,
        cost_model: Any = None,
        **kwargs
    ) -> IBacktestEngine:
        """
        Create a BacktestEngine instance.
        
        Args:
            config: Backtest configuration
            strategy: Trading strategy to test
            data_source: Data source for market data
            cost_model: Cost model for transaction costs
            **kwargs: Additional parameters
            
        Returns:
            Configured BacktestEngine instance
        """
        # Use default cost model if none provided
        if cost_model is None:
            cost_model = create_default_cost_model()
        
        return BacktestEngine(
            config=config,
            strategy=strategy,
            data_source=data_source,
            cost_model=cost_model,
            **kwargs
        )


# Default factory instance for convenience
default_backtest_factory = BacktestEngineFactory()


def get_backtest_factory() -> IBacktestEngineFactory:
    """Get the default backtest factory instance."""
    return default_backtest_factory


__all__ = [
    'BacktestEngineFactory',
    'default_backtest_factory',
    'get_backtest_factory'
]
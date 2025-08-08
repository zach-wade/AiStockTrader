"""
Backtesting interfaces and contracts.

This module defines the interfaces for backtesting components to ensure
clean dependency inversion and avoid circular imports between models
and backtesting modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, Any, Dict, List, Optional, runtime_checkable


class BacktestMode(Enum):
    """Backtesting execution modes."""
    SINGLE_SYMBOL = "single_symbol"
    MULTI_SYMBOL = "multi_symbol"
    PORTFOLIO = "portfolio"
    WALK_FORWARD = "walk_forward"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_cash: float = 100000.0
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1day"
    mode: BacktestMode = BacktestMode.PORTFOLIO
    use_adjusted_prices: bool = True
    include_dividends: bool = True
    include_splits: bool = True
    commission_per_trade: float = 1.0
    slippage_percentage: float = 0.001


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    metrics: Dict[str, Any]
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    performance_stats: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IBacktestEngine(Protocol):
    """Interface for backtest engines."""
    
    async def run(self) -> BacktestResult:
        """
        Run the backtest and return results.
        
        Returns:
            BacktestResult containing metrics and performance data
        """
        ...
    
    def validate_config(self) -> bool:
        """
        Validate the backtest configuration.
        
        Returns:
            True if configuration is valid
        """
        ...


@runtime_checkable
class IBacktestEngineFactory(Protocol):
    """Interface for creating backtest engines."""
    
    def create(
        self,
        config: BacktestConfig,
        strategy: Any,
        data_source: Any = None,
        cost_model: Any = None,
        **kwargs
    ) -> IBacktestEngine:
        """
        Create a backtest engine instance.
        
        Args:
            config: Backtest configuration
            strategy: Trading strategy to test
            data_source: Data source for market data
            cost_model: Cost model for transaction costs
            **kwargs: Additional parameters
            
        Returns:
            Configured backtest engine
        """
        ...


@runtime_checkable
class IStrategy(Protocol):
    """Interface for trading strategies used in backtesting."""
    
    def generate_signals(self, data: Any) -> List[Dict[str, Any]]:
        """
        Generate trading signals from market data.
        
        Args:
            data: Market data
            
        Returns:
            List of trading signals
        """
        ...
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            parameters: New parameter values
        """
        ...


class IPerformanceMetrics(Protocol):
    """Interface for calculating performance metrics."""
    
    def calculate_metrics(self, equity_curve: List[float], trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate performance metrics from equity curve and trades.
        
        Args:
            equity_curve: Time series of portfolio values
            trades: List of executed trades
            
        Returns:
            Dictionary of performance metrics
        """
        ...


__all__ = [
    'BacktestMode',
    'BacktestConfig', 
    'BacktestResult',
    'IBacktestEngine',
    'IBacktestEngineFactory',
    'IStrategy',
    'IPerformanceMetrics'
]
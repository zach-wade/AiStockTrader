"""
Ensemble portfolio allocation module.

This module provides sophisticated allocation algorithms for combining
signals from multiple strategies into optimal portfolio weights.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import defaultdict

from main.models.strategies.base_strategy import Signal
from main.utils.core import create_event_tracker
from main.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class AllocationConstraints:
    """Constraints for portfolio allocation."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_position_size: float = 0.2  # Max 20% per position
    min_position_size: float = 0.01  # Min 1% per position
    max_leverage: float = 1.0  # No leverage by default
    max_concentration: float = 0.5  # Max 50% in any sector
    long_only: bool = True  # Long-only constraint
    
    
@dataclass
class AllocationConfig:
    """Configuration for allocation algorithms."""
    method: str = "risk_parity"  # risk_parity, mean_variance, equal_weight, signal_strength
    lookback_days: int = 60
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    transaction_cost: float = 0.001  # 10 bps
    risk_free_rate: float = 0.02  # 2% annual
    target_volatility: Optional[float] = 0.15  # 15% annual
    constraints: AllocationConstraints = field(default_factory=AllocationConstraints)
    
    
@dataclass
class AllocationResult:
    """Result of portfolio allocation."""
    weights: Dict[str, float]  # Symbol -> weight
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_n: float  # Effective number of positions
    metadata: Dict[str, Any] = field(default_factory=dict)
    

class EnsembleAllocator:
    """
    Sophisticated portfolio allocation for ensemble strategies.
    
    Combines signals from multiple strategies and determines optimal
    portfolio weights considering risk, return, and constraints.
    """
    
    def __init__(
        self,
        config: AllocationConfig,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize ensemble allocator.
        
        Args:
            config: Allocation configuration
            metrics_collector: Optional metrics collector
        """
        self.config = config
        self.metrics = metrics_collector
        self.event_tracker = create_event_tracker("ensemble_allocator")
        
        # Historical data cache
        self._returns_cache: Dict[str, pd.Series] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._last_rebalance: Optional[datetime] = None
        
    def allocate(
        self,
        signals: List[Signal],
        current_weights: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> AllocationResult:
        """
        Allocate portfolio based on signals and market conditions.
        
        Args:
            signals: List of signals from strategies
            current_weights: Current portfolio weights
            market_data: Historical market data by symbol
            timestamp: Current timestamp
            
        Returns:
            Allocation result with target weights
        """
        # Group signals by symbol
        symbol_signals = self._group_signals_by_symbol(signals)
        
        # Check if rebalancing is needed
        if not self._should_rebalance(timestamp):
            return self._create_result_from_current(current_weights, market_data)
            
        # Update market data cache
        self._update_market_data_cache(market_data)
        
        # Apply allocation method
        if self.config.method == "risk_parity":
            result = self._risk_parity_allocation(symbol_signals, market_data)
        elif self.config.method == "mean_variance":
            result = self._mean_variance_allocation(symbol_signals, market_data)
        elif self.config.method == "signal_strength":
            result = self._signal_strength_allocation(symbol_signals)
        else:  # equal_weight
            result = self._equal_weight_allocation(symbol_signals)
            
        # Apply constraints
        result = self._apply_constraints(result, current_weights)
        
        # Calculate portfolio metrics
        result = self._calculate_portfolio_metrics(result, market_data)
        
        # Track allocation
        self._track_allocation(result, len(signals))
        
        self._last_rebalance = timestamp
        
        return result
        
    def _group_signals_by_symbol(self, signals: List[Signal]) -> Dict[str, List[Signal]]:
        """Group signals by symbol."""
        symbol_signals = defaultdict(list)
        for signal in signals:
            symbol_signals[signal.symbol].append(signal)
        return dict(symbol_signals)
        
    def _should_rebalance(self, timestamp: datetime) -> bool:
        """Check if portfolio should be rebalanced."""
        if self._last_rebalance is None:
            return True
            
        if self.config.rebalance_frequency == "daily":
            return timestamp.date() > self._last_rebalance.date()
        elif self.config.rebalance_frequency == "weekly":
            return (timestamp - self._last_rebalance).days >= 7
        elif self.config.rebalance_frequency == "monthly":
            return timestamp.month != self._last_rebalance.month
            
        return True
        
    def _update_market_data_cache(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update cached returns and correlations."""
        returns = {}
        
        for symbol, data in market_data.items():
            if len(data) > self.config.lookback_days:
                # Calculate returns
                prices = data['close'].iloc[-self.config.lookback_days:]
                returns[symbol] = prices.pct_change().dropna()
                
        self._returns_cache = returns
        
        # Update correlation matrix
        if returns:
            returns_df = pd.DataFrame(returns)
            self._correlation_matrix = returns_df.corr()
            
    def _risk_parity_allocation(
        self,
        symbol_signals: Dict[str, List[Signal]],
        market_data: Dict[str, pd.DataFrame]
    ) -> AllocationResult:
        """
        Risk parity allocation - equal risk contribution.
        """
        symbols = list(symbol_signals.keys())
        n = len(symbols)
        
        if n == 0:
            return AllocationResult(
                weights={},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                diversification_ratio=1,
                effective_n=0
            )
            
        # Get covariance matrix
        returns_df = pd.DataFrame(self._returns_cache)[symbols]
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        # Risk parity optimization
        def risk_contribution(weights, cov_matrix):
            """Calculate risk contribution of each asset."""
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
            
        def objective(weights, cov_matrix):
            """Minimize difference in risk contributions."""
            contrib = risk_contribution(weights, cov_matrix.values)
            # Equal risk contribution target
            target = np.ones(len(weights)) / len(weights)
            return np.sum((contrib - target) ** 2)
            
        # Initial guess - equal weight
        x0 = np.ones(n) / n
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum to 1
        ]
        
        # Bounds
        bounds = [(0, self.config.constraints.max_position_size) for _ in range(n)]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Create weights dictionary
        weights = {symbol: float(w) for symbol, w in zip(symbols, result.x)}
        
        # Adjust weights based on signal strength
        weights = self._adjust_weights_by_signals(weights, symbol_signals)
        
        return AllocationResult(
            weights=weights,
            expected_return=0,  # Will be calculated later
            expected_volatility=0,
            sharpe_ratio=0,
            diversification_ratio=1,
            effective_n=self._calculate_effective_n(weights)
        )
        
    def _mean_variance_allocation(
        self,
        symbol_signals: Dict[str, List[Signal]],
        market_data: Dict[str, pd.DataFrame]
    ) -> AllocationResult:
        """
        Mean-variance optimization (Markowitz).
        """
        symbols = list(symbol_signals.keys())
        n = len(symbols)
        
        if n == 0:
            return AllocationResult(
                weights={},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                diversification_ratio=1,
                effective_n=0
            )
            
        # Calculate expected returns based on signals
        expected_returns = self._calculate_expected_returns(symbol_signals, market_data)
        
        # Get covariance matrix
        returns_df = pd.DataFrame(self._returns_cache)[symbols]
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        # Objective: maximize Sharpe ratio
        def objective(weights):
            """Negative Sharpe ratio for minimization."""
            portfolio_return = weights @ expected_returns
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol
            return -sharpe
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum to 1
        ]
        
        # Add target volatility constraint if specified
        if self.config.target_volatility:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(x @ cov_matrix.values @ x) - self.config.target_volatility
            })
            
        # Bounds
        bounds = [(0, self.config.constraints.max_position_size) for _ in range(n)]
        
        # Initial guess
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Create weights dictionary
        weights = {symbol: float(w) for symbol, w in zip(symbols, result.x)}
        
        return AllocationResult(
            weights=weights,
            expected_return=0,
            expected_volatility=0,
            sharpe_ratio=0,
            diversification_ratio=1,
            effective_n=self._calculate_effective_n(weights)
        )
        
    def _signal_strength_allocation(
        self,
        symbol_signals: Dict[str, List[Signal]]
    ) -> AllocationResult:
        """
        Allocate based on signal strength/confidence.
        """
        # Calculate average signal strength per symbol
        signal_strengths = {}
        
        for symbol, signals in symbol_signals.items():
            # Average confidence across signals for this symbol
            avg_confidence = np.mean([s.confidence for s in signals])
            # Consider only buy signals for long-only portfolio
            buy_signals = [s for s in signals if s.direction == 'buy']
            if buy_signals:
                signal_strengths[symbol] = avg_confidence
                
        # Normalize to weights
        total_strength = sum(signal_strengths.values())
        if total_strength > 0:
            weights = {
                symbol: strength / total_strength
                for symbol, strength in signal_strengths.items()
            }
        else:
            weights = {}
            
        return AllocationResult(
            weights=weights,
            expected_return=0,
            expected_volatility=0,
            sharpe_ratio=0,
            diversification_ratio=1,
            effective_n=self._calculate_effective_n(weights)
        )
        
    def _equal_weight_allocation(
        self,
        symbol_signals: Dict[str, List[Signal]]
    ) -> AllocationResult:
        """
        Simple equal weight allocation.
        """
        # Filter for buy signals only
        buy_symbols = []
        for symbol, signals in symbol_signals.items():
            if any(s.direction == 'buy' for s in signals):
                buy_symbols.append(symbol)
                
        n = len(buy_symbols)
        if n > 0:
            weight = 1.0 / n
            weights = {symbol: weight for symbol in buy_symbols}
        else:
            weights = {}
            
        return AllocationResult(
            weights=weights,
            expected_return=0,
            expected_volatility=0,
            sharpe_ratio=0,
            diversification_ratio=1,
            effective_n=self._calculate_effective_n(weights)
        )
        
    def _adjust_weights_by_signals(
        self,
        weights: Dict[str, float],
        symbol_signals: Dict[str, List[Signal]]
    ) -> Dict[str, float]:
        """Adjust weights based on signal strength."""
        adjusted_weights = {}
        
        for symbol, weight in weights.items():
            if symbol in symbol_signals:
                signals = symbol_signals[symbol]
                # Average signal confidence
                avg_confidence = np.mean([s.confidence for s in signals])
                # Adjust weight by confidence (0.5 to 1.5x)
                adjustment = 0.5 + avg_confidence
                adjusted_weights[symbol] = weight * adjustment
            else:
                adjusted_weights[symbol] = weight
                
        # Renormalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {s: w/total for s, w in adjusted_weights.items()}
            
        return adjusted_weights
        
    def _calculate_expected_returns(
        self,
        symbol_signals: Dict[str, List[Signal]],
        market_data: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """Calculate expected returns based on signals and historical data."""
        symbols = list(symbol_signals.keys())
        expected_returns = []
        
        for symbol in symbols:
            # Historical return
            if symbol in self._returns_cache:
                hist_return = self._returns_cache[symbol].mean() * 252
            else:
                hist_return = 0.05  # Default 5%
                
            # Adjust by signal strength
            signals = symbol_signals[symbol]
            avg_confidence = np.mean([s.confidence for s in signals])
            
            # Expected return = base return * (0.5 + signal confidence)
            expected_return = hist_return * (0.5 + avg_confidence)
            expected_returns.append(expected_return)
            
        return np.array(expected_returns)
        
    def _apply_constraints(
        self,
        result: AllocationResult,
        current_weights: Dict[str, float]
    ) -> AllocationResult:
        """Apply portfolio constraints."""
        weights = result.weights.copy()
        
        # Apply position size constraints
        for symbol, weight in list(weights.items()):
            if weight < self.config.constraints.min_position_size:
                del weights[symbol]
            elif weight > self.config.constraints.max_position_size:
                weights[symbol] = self.config.constraints.max_position_size
                
        # Renormalize after constraints
        total = sum(weights.values())
        if total > 0:
            weights = {s: w/total for s, w in weights.items()}
            
        # Apply transaction cost penalty
        if current_weights:
            for symbol, new_weight in weights.items():
                old_weight = current_weights.get(symbol, 0)
                turnover = abs(new_weight - old_weight)
                # Reduce allocation by transaction cost
                weights[symbol] = new_weight * (1 - self.config.transaction_cost * turnover)
                
        result.weights = weights
        return result
        
    def _calculate_portfolio_metrics(
        self,
        result: AllocationResult,
        market_data: Dict[str, pd.DataFrame]
    ) -> AllocationResult:
        """Calculate portfolio risk and return metrics."""
        weights = result.weights
        if not weights:
            return result
            
        symbols = list(weights.keys())
        weight_array = np.array([weights[s] for s in symbols])
        
        # Get returns
        returns_df = pd.DataFrame({
            s: self._returns_cache.get(s, pd.Series())
            for s in symbols
        })
        
        # Portfolio return
        mean_returns = returns_df.mean() * 252
        result.expected_return = float(weight_array @ mean_returns.values)
        
        # Portfolio volatility
        cov_matrix = returns_df.cov() * 252
        result.expected_volatility = float(np.sqrt(weight_array @ cov_matrix.values @ weight_array))
        
        # Sharpe ratio
        if result.expected_volatility > 0:
            result.sharpe_ratio = (
                (result.expected_return - self.config.risk_free_rate) /
                result.expected_volatility
            )
            
        # Diversification ratio
        individual_vols = returns_df.std() * np.sqrt(252)
        weighted_avg_vol = float(weight_array @ individual_vols.values)
        if result.expected_volatility > 0:
            result.diversification_ratio = weighted_avg_vol / result.expected_volatility
            
        return result
        
    def _calculate_effective_n(self, weights: Dict[str, float]) -> float:
        """Calculate effective number of positions (inverse HHI)."""
        if not weights:
            return 0
            
        weight_array = np.array(list(weights.values()))
        hhi = np.sum(weight_array ** 2)
        
        return 1 / hhi if hhi > 0 else 0
        
    def _create_result_from_current(
        self,
        current_weights: Dict[str, float],
        market_data: Dict[str, pd.DataFrame]
    ) -> AllocationResult:
        """Create result maintaining current weights."""
        result = AllocationResult(
            weights=current_weights.copy(),
            expected_return=0,
            expected_volatility=0,
            sharpe_ratio=0,
            diversification_ratio=1,
            effective_n=self._calculate_effective_n(current_weights)
        )
        
        return self._calculate_portfolio_metrics(result, market_data)
        
    def _track_allocation(self, result: AllocationResult, signal_count: int) -> None:
        """Track allocation metrics."""
        self.event_tracker.track("allocation", {
            "method": self.config.method,
            "num_positions": len(result.weights),
            "signal_count": signal_count,
            "expected_return": result.expected_return,
            "expected_volatility": result.expected_volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "effective_n": result.effective_n
        })
        
        if self.metrics:
            self.metrics.gauge(
                "allocator.portfolio_sharpe",
                result.sharpe_ratio
            )
            self.metrics.gauge(
                "allocator.effective_positions",
                result.effective_n
            )


class WeightAllocator(EnsembleAllocator):
    """
    Weight allocator for ensemble strategies.
    
    This is an alias for EnsembleAllocator to maintain backward compatibility
    with code that imports WeightAllocator.
    """
    pass
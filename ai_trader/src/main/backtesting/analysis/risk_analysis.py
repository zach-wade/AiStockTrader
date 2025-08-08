# File: backtesting/analysis/risk_analysis.py
"""
Risk metrics and analysis
Created: 2025-06-16
"""

"""
Advanced risk analysis including VaR, stress testing, and scenario analysis.

Provides comprehensive risk metrics and stress testing capabilities
for strategy evaluation and risk management.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

from main.config.config_manager import get_config
from main.feature_pipeline.calculators.helpers import safe_divide

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Comprehensive risk analysis for trading strategies."""
    
    def __init__(self, config: Any = None):
        """Initialize risk analyzer."""
        if config is None:
            config = get_config()
        self.config = config
        
        # Risk parameters
        self.var_confidence_levels = config.get('risk.var_confidence_levels', [0.95, 0.99])
        self.cvar_confidence_level = config.get('risk.cvar_confidence_level', 0.95)
        self.lookback_days = config.get('risk.lookback_days', 252)
        
        # Stress test scenarios
        self.stress_scenarios = self._load_stress_scenarios()
        
        # Risk-free rate for Sharpe calculations
        self.risk_free_rate = config.get('risk.risk_free_rate', 0.02)
    
    def _load_stress_scenarios(self) -> Dict[str, Dict]:
        """Load predefined stress test scenarios."""
        return {
            'market_crash': {
                'name': 'Market Crash (2008-like)',
                'equity_shock': -0.40,
                'volatility_multiplier': 3.0,
                'correlation_increase': 0.3,
                'duration_days': 60
            },
            'flash_crash': {
                'name': 'Flash Crash',
                'equity_shock': -0.10,
                'volatility_multiplier': 5.0,
                'correlation_increase': 0.5,
                'duration_days': 1
            },
            'rate_shock': {
                'name': 'Interest Rate Shock',
                'equity_shock': -0.15,
                'volatility_multiplier': 2.0,
                'correlation_increase': 0.2,
                'duration_days': 30
            },
            'black_swan': {
                'name': 'Black Swan Event',
                'equity_shock': -0.25,
                'volatility_multiplier': 4.0,
                'correlation_increase': 0.6,
                'duration_days': 10
            },
            'sector_crisis': {
                'name': 'Sector-Specific Crisis',
                'equity_shock': -0.30,
                'volatility_multiplier': 2.5,
                'correlation_increase': 0.1,
                'duration_days': 90
            }
        }
    
    def calculate_var(self, returns: pd.Series, 
                     confidence_levels: Optional[List[float]] = None,
                     method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) using various methods.
        
        Args:
            returns: Series of returns
            confidence_levels: Confidence levels for VaR
            method: 'historical', 'parametric', or 'cornish_fisher'
            
        Returns:
            Dictionary of VaR values
        """
        if confidence_levels is None:
            confidence_levels = self.var_confidence_levels
        
        results = {}
        
        for confidence in confidence_levels:
            if method == 'historical':
                var = self._historical_var(returns, confidence)
            elif method == 'parametric':
                var = self._parametric_var(returns, confidence)
            elif method == 'cornish_fisher':
                var = self._cornish_fisher_var(returns, confidence)
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            results[f'var_{int(confidence*100)}'] = var
        
        return results
    
    def _historical_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate historical VaR."""
        return -np.percentile(returns, (1 - confidence) * 100)
    
    def _parametric_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        return -(mean + z_score * std)
    
    def _cornish_fisher_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Cornish-Fisher VaR adjusting for skewness and kurtosis."""
        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        z = stats.norm.ppf(1 - confidence)
        
        # Cornish-Fisher expansion
        cf_z = (z + 
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        return -(mean + cf_z * std)
    
    def calculate_cvar(self, returns: pd.Series,
                      confidence: Optional[float] = None) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Series of returns
            confidence: Confidence level
            
        Returns:
            CVaR value
        """
        if confidence is None:
            confidence = self.cvar_confidence_level
        
        var = self._historical_var(returns, confidence)
        conditional_returns = returns[returns <= -var]
        
        if len(conditional_returns) == 0:
            return var
        
        return -conditional_returns.mean()
    
    def calculate_risk_metrics(self, returns: pd.Series,
                             benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns for relative metrics
            
        Returns:
            Dictionary of risk metrics
        """
        # Basic statistics
        metrics = {
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'max_drawdown_duration': self._calculate_max_drawdown_duration(returns)
        }
        
        # VaR and CVaR
        var_results = self.calculate_var(returns)
        metrics.update(var_results)
        metrics['cvar_95'] = self.calculate_cvar(returns, 0.95)
        
        # Risk-adjusted returns
        sharpe = safe_divide(metrics['annual_return'] - self.risk_free_rate, metrics['annual_volatility'], default_value=0.0)
        metrics['sharpe_ratio'] = sharpe
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = safe_divide(metrics['annual_return'] - self.risk_free_rate, downside_std, default_value=0.0)
        metrics['sortino_ratio'] = sortino
        
        # Calmar ratio
        calmar = safe_divide(metrics['annual_return'], abs(metrics['max_drawdown']), default_value=0.0)
        metrics['calmar_ratio'] = calmar
        
        # Information ratio (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = safe_divide(excess_returns.mean() * 252, tracking_error, default_value=0.0)
            metrics['information_ratio'] = information_ratio
            metrics['tracking_error'] = tracking_error
            
            # Beta
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = benchmark_returns.var()
            beta = safe_divide(covariance, benchmark_variance, default_value=0.0)
            metrics['beta'] = beta
            
            # Alpha (Jensen's alpha)
            alpha = metrics['annual_return'] - (self.risk_free_rate + 
                                               beta * (benchmark_returns.mean() * 252 - self.risk_free_rate))
            metrics['alpha'] = alpha
        
        # Tail ratio
        metrics['tail_ratio'] = abs(safe_divide(np.percentile(returns, 95), np.percentile(returns, 5), default_value=0.0))
        
        # Win rate and profit factor
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        metrics['win_rate'] = safe_divide(len(winning_returns), len(returns), default_value=0.0)
        metrics['profit_factor'] = safe_divide(winning_returns.sum(), abs(losing_returns.sum()), default_value=np.inf)
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = safe_divide(cumulative - running_max, running_max, default_value=0.0)
        return drawdown.min()
    
    def _calculate_max_drawdown_duration(self, returns: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        
        # Find drawdown periods
        is_drawdown = cumulative < running_max
        
        # Calculate duration of each drawdown period
        drawdown_periods = []
        current_duration = 0
        
        for in_drawdown in is_drawdown:
            if in_drawdown:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            drawdown_periods.append(current_duration)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def stress_test(self, portfolio_returns: pd.Series,
                   portfolio_positions: pd.DataFrame,
                   scenario: str) -> Dict[str, any]:
        """
        Run stress test on portfolio.
        
        Args:
            portfolio_returns: Historical portfolio returns
            portfolio_positions: Current positions with weights
            scenario: Stress scenario name
            
        Returns:
            Stress test results
        """
        if scenario not in self.stress_scenarios:
            raise ValueError(f"Unknown stress scenario: {scenario}")
        
        scenario_params = self.stress_scenarios[scenario]
        
        # Calculate current portfolio metrics
        current_volatility = portfolio_returns.std() * np.sqrt(252)
        current_value = portfolio_positions['value'].sum()
        
        # Apply stress scenario
        stressed_returns = []
        
        # Simulate stressed returns
        for _ in range(scenario_params['duration_days']):
            # Apply equity shock
            base_return = safe_divide(scenario_params['equity_shock'], scenario_params['duration_days'], default_value=0.0)
            
            # Add increased volatility
            vol_shock = secure_numpy_normal(0, safe_divide(current_volatility * scenario_params['volatility_multiplier'], np.sqrt(252), default_value=0.0))
            
            stressed_return = base_return + vol_shock
            stressed_returns.append(stressed_return)
        
        stressed_returns = pd.Series(stressed_returns)
        
        # Calculate impact
        cumulative_return = (1 + stressed_returns).prod() - 1
        stressed_value = current_value * (1 + cumulative_return)
        value_loss = current_value - stressed_value
        
        # Calculate stressed risk metrics
        stressed_var_95 = self._historical_var(stressed_returns, 0.95) * current_value
        stressed_var_99 = self._historical_var(stressed_returns, 0.99) * current_value
        stressed_cvar_95 = self.calculate_cvar(stressed_returns, 0.95) * current_value
        
        return {
            'scenario': scenario_params['name'],
            'current_value': current_value,
            'stressed_value': stressed_value,
            'value_loss': value_loss,
            'percentage_loss': cumulative_return,
            'stressed_var_95': stressed_var_95,
            'stressed_var_99': stressed_var_99,
            'stressed_cvar_95': stressed_cvar_95,
            'max_daily_loss': stressed_returns.min() * current_value,
            'probability_of_loss': (stressed_returns < 0).mean()
        }
    
    def run_all_stress_tests(self, portfolio_returns: pd.Series,
                           portfolio_positions: pd.DataFrame) -> Dict[str, Dict]:
        """Run all stress test scenarios."""
        results = {}
        
        for scenario_name in self.stress_scenarios:
            try:
                results[scenario_name] = self.stress_test(
                    portfolio_returns, portfolio_positions, scenario_name
                )
            except Exception as e:
                logger.error(f"Stress test failed for {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}
        
        return results
    
    def monte_carlo_var(self, returns: pd.DataFrame,
                       positions: pd.DataFrame,
                       n_simulations: int = 10000,
                       time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            returns: Historical returns for each asset
            positions: Current positions
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
            
        Returns:
            Monte Carlo VaR results
        """
        # Calculate return statistics
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Use Ledoit-Wolf shrinkage for more stable covariance
        lw = LedoitWolf()
        lw.fit(returns)
        cov_matrix = pd.DataFrame(
            lw.covariance_,
            index=returns.columns,
            columns=returns.columns
        )
        
        # Portfolio weights
        portfolio_value = positions['value'].sum()
        weights = safe_divide(positions['value'], portfolio_value, default_value=0.0)
        
        # Run simulations
        portfolio_returns = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.multivariate_normal(
                mean_returns * time_horizon,
                cov_matrix * time_horizon,
                size=1
            )[0]
            
            # Calculate portfolio return
            portfolio_return = np.dot(weights, random_returns)
            portfolio_returns.append(portfolio_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate VaR and CVaR
        results = {}
        for confidence in self.var_confidence_levels:
            var = -np.percentile(portfolio_returns, (1 - confidence) * 100) * portfolio_value
            results[f'mc_var_{int(confidence*100)}'] = var
            
            # CVaR
            threshold = np.percentile(portfolio_returns, (1 - confidence) * 100)
            cvar = -portfolio_returns[portfolio_returns <= threshold].mean() * portfolio_value
            results[f'mc_cvar_{int(confidence*100)}'] = cvar
        
        return results
    
    def calculate_risk_attribution(self, returns: pd.DataFrame,
                                 positions: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate risk attribution by position.
        
        Args:
            returns: Returns for each asset
            positions: Current positions
            
        Returns:
            Risk attribution analysis
        """
        # Portfolio statistics
        portfolio_value = positions['value'].sum()
        weights = safe_divide(positions['value'], portfolio_value, default_value=0.0)
        
        # Covariance matrix
        cov_matrix = returns.cov() * 252  # Annualized
        
        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Marginal VaR (derivative of portfolio VaR with respect to position)
        marginal_var = {}
        component_var = {}
        
        for asset in positions.index:
            if asset in returns.columns:
                # Marginal VaR
                marginal_var[asset] = safe_divide(
                    np.dot(cov_matrix[asset], weights), portfolio_volatility, default_value=0.0
                )
                
                # Component VaR
                component_var[asset] = marginal_var[asset] * weights[asset]
        
        # Risk contribution percentage
        total_risk = sum(component_var.values())
        risk_contribution = {
            asset: safe_divide(comp_var, total_risk, default_value=0.0) * 100
            for asset, comp_var in component_var.items()
        }
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'marginal_var': marginal_var,
            'component_var': component_var,
            'risk_contribution_pct': risk_contribution
        }
    
    def calculate_correlation_risk(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation-related risk metrics."""
        correlation_matrix = returns.corr()
        
        # Average correlation
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        avg_correlation = correlation_matrix.where(mask).stack().mean()
        
        # Maximum correlation
        max_correlation = correlation_matrix.where(mask).max().max()
        
        # Correlation risk (how much correlations can increase in stress)
        stressed_correlation = correlation_matrix + 0.3  # Stress scenario
        stressed_correlation = stressed_correlation.clip(-1, 1)
        
        # Calculate increase in portfolio risk
        n_assets = len(returns.columns)
        equal_weights = safe_divide(np.ones(n_assets), n_assets, default_value=0.0)
        
        normal_risk = np.sqrt(np.dot(equal_weights, np.dot(returns.cov(), equal_weights))) * np.sqrt(252)
        
        stressed_cov = returns.std() * stressed_correlation * returns.std().values.reshape(-1, 1)
        stressed_risk = np.sqrt(np.dot(equal_weights, np.dot(stressed_cov, equal_weights))) * np.sqrt(252)
        
        correlation_risk_increase = safe_divide(stressed_risk - normal_risk, normal_risk, default_value=0.0)
        
        return {
            'average_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'correlation_risk_increase': correlation_risk_increase,
            'diversification_ratio': safe_divide(1, np.sqrt(n_assets), default_value=0.0)
        }
# Alias for backward compatibility
RiskAnalysis = RiskAnalyzer

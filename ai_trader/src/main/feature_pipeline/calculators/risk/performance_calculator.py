"""
Performance Calculator

Specialized calculator for risk-adjusted performance metrics including:
- Sharpe ratio and variations
- Sortino ratio and downside risk
- Information ratio and tracking error
- Treynor ratio and systematic risk
- Alpha and beta analysis
- Risk-adjusted return metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats

from .base_risk import BaseRiskCalculator
from ..helpers import safe_divide

from main.utils.core import get_logger

logger = get_logger(__name__)


class PerformanceCalculator(BaseRiskCalculator):
    """Calculator for risk-adjusted performance metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance calculator."""
        super().__init__(config)
        
        # Performance-specific configuration
        self.performance_config = self.risk_config.get_performance_config()
        self.lookback_window = self.performance_config['lookback_window']
        self.min_observations = self.performance_config['min_observations']
        self.risk_free_rate = self.performance_config['risk_free_rate']
        self.benchmark_return = self.performance_config['benchmark_return']
        self.sharpe_annualization = self.performance_config['sharpe_annualization']
        
        logger.debug(f"Initialized PerformanceCalculator with {self.lookback_window}d lookback")
    
    def get_required_columns(self) -> List[str]:
        """Get list of required input columns."""
        return ['close']
    
    def get_feature_names(self) -> List[str]:
        """Return list of performance feature names."""
        feature_names = [
            # Basic performance metrics
            'total_return',
            'annualized_return',
            'volatility',
            'annualized_volatility',
            'excess_return',
            'active_return',
            
            # Sharpe ratio variations
            'sharpe_ratio',
            'sharpe_ratio_annualized',
            'modified_sharpe_ratio',
            'adjusted_sharpe_ratio',
            'probabilistic_sharpe_ratio',
            'deflated_sharpe_ratio',
            
            # Sortino ratio and downside risk
            'sortino_ratio',
            'sortino_ratio_annualized',
            'downside_volatility',
            'downside_deviation',
            'upside_volatility',
            'upside_deviation',
            'upside_downside_ratio',
            
            # Information ratio and tracking
            'information_ratio',
            'tracking_error',
            'tracking_error_annualized',
            'active_share',
            'r_squared',
            'correlation_with_benchmark',
            
            # Treynor ratio and systematic risk
            'treynor_ratio',
            'treynor_ratio_annualized',
            'systematic_risk',
            'unsystematic_risk',
            'beta',
            'alpha',
            'jensen_alpha',
            
            # Risk-adjusted returns
            'risk_adjusted_return',
            'return_over_var',
            'return_over_drawdown',
            'omega_ratio',
            'kappa_ratio',
            'gain_loss_ratio',
            
            # Performance consistency
            'hit_ratio',
            'loss_ratio',
            'win_loss_ratio',
            'profit_factor',
            'expectancy',
            'consistency_ratio',
            
            # Rolling performance metrics
            'rolling_sharpe_mean',
            'rolling_sharpe_std',
            'rolling_sortino_mean',
            'rolling_sortino_std',
            'performance_stability',
            'performance_trend',
            
            # Advanced performance metrics
            'conditional_sharpe_ratio',
            'skewness_adjusted_sharpe',
            'kurtosis_adjusted_sharpe',
            'var_adjusted_return',
            'tail_ratio',
            'capture_ratio_up',
            'capture_ratio_down',
            'capture_ratio_total',
            
            # Performance attribution
            'return_attribution_skill',
            'return_attribution_luck',
            'skill_luck_ratio',
            'performance_persistence',
            'performance_momentum',
        ]
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with performance features
        """
        try:
            # Validate input data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)
            
            # Calculate returns
            returns = self.calculate_returns(data)
            
            if len(returns) < self.min_observations:
                logger.warning("Insufficient data for performance calculation")
                return self.create_empty_features(data.index)
            
            # Create benchmark returns (simplified)
            benchmark_returns = self._create_benchmark_returns(returns)
            
            # Create features DataFrame
            features = self.create_empty_features(data.index)
            
            # Calculate performance metrics
            features = self._calculate_basic_performance(returns, features)
            features = self._calculate_sharpe_metrics(returns, features)
            features = self._calculate_sortino_metrics(returns, features)
            features = self._calculate_information_metrics(returns, benchmark_returns, features)
            features = self._calculate_treynor_metrics(returns, benchmark_returns, features)
            features = self._calculate_risk_adjusted_returns(returns, features)
            features = self._calculate_performance_consistency(returns, features)
            features = self._calculate_rolling_performance(returns, features)
            features = self._calculate_advanced_performance(returns, features)
            features = self._calculate_performance_attribution(returns, benchmark_returns, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating performance features: {e}")
            return self.create_empty_features(data.index)
    
    def _create_benchmark_returns(self, returns: pd.Series) -> pd.Series:
        """Create benchmark returns for comparison."""
        try:
            # Simple benchmark: market return with similar volatility
            daily_benchmark_return = self.benchmark_return / 252
            benchmark_volatility = returns.std()
            
            # Create benchmark with similar volatility but steady return
            np.random.seed(42)  # For consistency
            benchmark_returns = pd.Series(
                secure_numpy_normal(daily_benchmark_return, benchmark_volatility * 0.5, len(returns)),
                index=returns.index
            )
            
            return benchmark_returns
            
        except Exception as e:
            logger.warning(f"Error creating benchmark returns: {e}")
            return pd.Series(0, index=returns.index)
    
    def _calculate_basic_performance(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic performance metrics."""
        try:
            # Total return
            total_return = (1 + returns).prod() - 1
            features['total_return'] = total_return
            
            # Annualized return
            years = len(returns) / 252
            if years > 0:
                annualized_return = (1 + total_return) ** (1/years) - 1
                features['annualized_return'] = annualized_return
            
            # Volatility
            volatility = returns.std()
            features['volatility'] = volatility
            
            # Annualized volatility
            annualized_volatility = volatility * np.sqrt(252)
            features['annualized_volatility'] = annualized_volatility
            
            # Excess return
            daily_risk_free = self.risk_free_rate / 252
            excess_return = returns.mean() - daily_risk_free
            features['excess_return'] = excess_return
            
            # Active return (vs benchmark)
            active_return = returns.mean() - self.benchmark_return / 252
            features['active_return'] = active_return
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating basic performance: {e}")
            return features
    
    def _calculate_sharpe_metrics(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sharpe ratio and variations."""
        try:
            daily_risk_free = self.risk_free_rate / 252
            excess_returns = returns - daily_risk_free
            
            # Basic Sharpe ratio
            if returns.std() > 0:
                sharpe_ratio = safe_divide(excess_returns.mean(), returns.std())
                features['sharpe_ratio'] = sharpe_ratio
                
                # Annualized Sharpe ratio
                annualized_sharpe = sharpe_ratio * np.sqrt(252)
                features['sharpe_ratio_annualized'] = annualized_sharpe
            
            # Modified Sharpe ratio (adjusted for skewness and kurtosis)
            if len(returns) > 30:
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
                
                if returns.std() > 0:
                    # Cornish-Fisher adjustment
                    cf_adjustment = (1 + (skewness / 6) * sharpe_ratio + 
                                   (kurtosis - 3) / 24 * sharpe_ratio**2)
                    modified_sharpe = sharpe_ratio * cf_adjustment
                    features['modified_sharpe_ratio'] = modified_sharpe
                
                # Adjusted Sharpe ratio
                adjusted_sharpe = sharpe_ratio * (1 + (skewness / 6) * sharpe_ratio - 
                                                (kurtosis - 3) / 24 * sharpe_ratio**2)
                features['adjusted_sharpe_ratio'] = adjusted_sharpe
            
            # Probabilistic Sharpe ratio
            if len(returns) > 30 and returns.std() > 0:
                # Simplified calculation
                sr_std = np.sqrt(safe_divide((1 + 0.5 * sharpe_ratio**2), len(returns)))
                prob_sharpe = stats.norm.cdf(safe_divide(sharpe_ratio, sr_std))
                features['probabilistic_sharpe_ratio'] = prob_sharpe
            
            # Deflated Sharpe ratio (simplified)
            if features.get('sharpe_ratio', 0) != 0:
                deflated_sharpe = features['sharpe_ratio'] * 0.9  # Simplified adjustment
                features['deflated_sharpe_ratio'] = deflated_sharpe
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating Sharpe metrics: {e}")
            return features
    
    def _calculate_sortino_metrics(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sortino ratio and downside risk metrics."""
        try:
            target_return = self.performance_config['sortino_target_return'] / 252
            
            # Downside deviation
            downside_returns = returns[returns < target_return]
            if len(downside_returns) > 0:
                downside_deviation = np.sqrt(((downside_returns - target_return)**2).mean())
                features['downside_deviation'] = downside_deviation
                features['downside_volatility'] = downside_returns.std()
                
                # Sortino ratio
                excess_return = returns.mean() - target_return
                if downside_deviation > 0:
                    sortino_ratio = safe_divide(excess_return, downside_deviation)
                    features['sortino_ratio'] = sortino_ratio
                    
                    # Annualized Sortino ratio
                    annualized_sortino = sortino_ratio * np.sqrt(252)
                    features['sortino_ratio_annualized'] = annualized_sortino
            
            # Upside deviation
            upside_returns = returns[returns > target_return]
            if len(upside_returns) > 0:
                upside_deviation = np.sqrt(((upside_returns - target_return)**2).mean())
                features['upside_deviation'] = upside_deviation
                features['upside_volatility'] = upside_returns.std()
                
                # Upside/downside ratio
                if features.get('downside_deviation', 0) > 0:
                    upside_downside_ratio = safe_divide(upside_deviation, features['downside_deviation'])
                    features['upside_downside_ratio'] = upside_downside_ratio
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating Sortino metrics: {e}")
            return features
    
    def _calculate_information_metrics(self, returns: pd.Series, benchmark_returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate information ratio and tracking metrics."""
        try:
            # Active returns
            active_returns = returns - benchmark_returns
            
            # Tracking error
            tracking_error = active_returns.std()
            features['tracking_error'] = tracking_error
            features['tracking_error_annualized'] = tracking_error * np.sqrt(252)
            
            # Information ratio
            if tracking_error > 0:
                information_ratio = safe_divide(active_returns.mean(), tracking_error)
                features['information_ratio'] = information_ratio
            
            # Correlation with benchmark
            correlation = returns.corr(benchmark_returns)
            features['correlation_with_benchmark'] = correlation if not np.isnan(correlation) else 0
            
            # R-squared
            if not np.isnan(correlation):
                r_squared = correlation**2
                features['r_squared'] = r_squared
            
            # Active share (simplified)
            if len(returns) > 0:
                active_share = np.abs(active_returns).mean()
                features['active_share'] = active_share
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating information metrics: {e}")
            return features
    
    def _calculate_treynor_metrics(self, returns: pd.Series, benchmark_returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate Treynor ratio and systematic risk metrics."""
        try:
            # Calculate beta
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            
            if benchmark_variance > 0:
                beta = safe_divide(covariance, benchmark_variance)
                features['beta'] = beta
                
                # Treynor ratio
                daily_risk_free = self.risk_free_rate / 252
                excess_return = returns.mean() - daily_risk_free
                
                if beta != 0:
                    treynor_ratio = safe_divide(excess_return, beta)
                    features['treynor_ratio'] = treynor_ratio
                    features['treynor_ratio_annualized'] = treynor_ratio * 252
                
                # Alpha (Jensen's alpha)
                benchmark_excess = benchmark_returns.mean() - daily_risk_free
                alpha = excess_return - beta * benchmark_excess
                features['alpha'] = alpha
                features['jensen_alpha'] = alpha * 252  # Annualized
                
                # Systematic and unsystematic risk
                total_variance = returns.var()
                systematic_variance = beta**2 * benchmark_variance
                unsystematic_variance = total_variance - systematic_variance
                
                features['systematic_risk'] = np.sqrt(max(0, systematic_variance))
                features['unsystematic_risk'] = np.sqrt(max(0, unsystematic_variance))
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating Treynor metrics: {e}")
            return features
    
    def _calculate_risk_adjusted_returns(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk-adjusted return metrics."""
        try:
            # Risk-adjusted return (return per unit of risk)
            volatility = returns.std()
            if volatility > 0:
                risk_adjusted_return = safe_divide(returns.mean(), volatility)
                features['risk_adjusted_return'] = risk_adjusted_return
            
            # Return over VaR
            if len(returns) > 30:
                var_95 = returns.quantile(0.05)  # 5% VaR
                if var_95 < 0:
                    return_over_var = safe_divide(returns.mean(), abs(var_95))
                    features['return_over_var'] = return_over_var
            
            # Return over maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = safe_divide((cumulative_returns - running_max), running_max)
            max_drawdown = abs(drawdown.min())
            
            if max_drawdown > 0:
                return_over_drawdown = safe_divide(returns.mean(), max_drawdown)
                features['return_over_drawdown'] = return_over_drawdown
            
            # Omega ratio
            threshold = 0  # Target return
            gains = returns[returns > threshold].sum()
            losses = abs(returns[returns <= threshold].sum())
            
            if losses > 0:
                omega_ratio = safe_divide(gains, losses)
                features['omega_ratio'] = omega_ratio
            
            # Kappa ratio (simplified)
            if len(returns) > 20:
                downside_risk = returns[returns < 0].std()
                if downside_risk > 0:
                    kappa_ratio = safe_divide(returns.mean(), downside_risk)
                    features['kappa_ratio'] = kappa_ratio
            
            # Gain/loss ratio
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                avg_gain = positive_returns.mean()
                avg_loss = abs(negative_returns.mean())
                
                if avg_loss > 0:
                    gain_loss_ratio = safe_divide(avg_gain, avg_loss)
                    features['gain_loss_ratio'] = gain_loss_ratio
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating risk-adjusted returns: {e}")
            return features
    
    def _calculate_performance_consistency(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance consistency metrics."""
        try:
            # Hit ratio (percentage of positive returns)
            positive_returns = (returns > 0).sum()
            hit_ratio = safe_divide(positive_returns, len(returns))
            features['hit_ratio'] = hit_ratio
            
            # Loss ratio
            loss_ratio = 1 - hit_ratio
            features['loss_ratio'] = loss_ratio
            
            # Win/loss ratio
            if loss_ratio > 0:
                win_loss_ratio = safe_divide(hit_ratio, loss_ratio)
                features['win_loss_ratio'] = win_loss_ratio
            
            # Profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            
            if gross_loss > 0:
                profit_factor = safe_divide(gross_profit, gross_loss)
                features['profit_factor'] = profit_factor
            
            # Expectancy
            if len(returns) > 0:
                expectancy = returns.mean()
                features['expectancy'] = expectancy
            
            # Consistency ratio (based on rolling returns)
            if len(returns) >= 60:
                rolling_returns = returns.rolling(window=20).mean()
                consistency_ratio = safe_divide(len(rolling_returns[rolling_returns > 0]), len(rolling_returns.dropna()))
                features['consistency_ratio'] = consistency_ratio
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating performance consistency: {e}")
            return features
    
    def _calculate_rolling_performance(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        try:
            if len(returns) < 60:
                return features
            
            # Rolling Sharpe ratio
            daily_risk_free = self.risk_free_rate / 252
            window = 60
            
            rolling_sharpe = []
            for i in range(window, len(returns)):
                window_returns = returns.iloc[i-window:i]
                excess_returns = window_returns - daily_risk_free
                
                if window_returns.std() > 0:
                    sharpe = safe_divide(excess_returns.mean(), window_returns.std())
                    rolling_sharpe.append(sharpe)
            
            if rolling_sharpe:
                features['rolling_sharpe_mean'] = np.mean(rolling_sharpe)
                features['rolling_sharpe_std'] = np.std(rolling_sharpe)
            
            # Rolling Sortino ratio
            rolling_sortino = []
            for i in range(window, len(returns)):
                window_returns = returns.iloc[i-window:i]
                downside_returns = window_returns[window_returns < 0]
                
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std()
                    if downside_std > 0:
                        sortino = safe_divide(window_returns.mean(), downside_std)
                        rolling_sortino.append(sortino)
            
            if rolling_sortino:
                features['rolling_sortino_mean'] = np.mean(rolling_sortino)
                features['rolling_sortino_std'] = np.std(rolling_sortino)
            
            # Performance stability
            if rolling_sharpe:
                stability = 1 - (np.std(rolling_sharpe) / (np.mean(rolling_sharpe) + 1e-6))
                features['performance_stability'] = max(0, stability)
            
            # Performance trend
            if len(rolling_sharpe) > 10:
                x = np.arange(len(rolling_sharpe))
                try:
                    trend = np.polyfit(x, rolling_sharpe, 1)[0]
                    features['performance_trend'] = trend
                except (ValueError, TypeError, np.linalg.LinAlgError):
                    features['performance_trend'] = 0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating rolling performance: {e}")
            return features
    
    def _calculate_advanced_performance(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced performance metrics."""
        try:
            # Conditional Sharpe ratio
            if len(returns) > 30:
                # Sharpe ratio conditional on positive returns
                positive_returns = returns[returns > 0]
                if len(positive_returns) > 5:
                    daily_risk_free = self.risk_free_rate / 252
                    excess_positive = positive_returns - daily_risk_free
                    
                    if positive_returns.std() > 0:
                        conditional_sharpe = safe_divide(excess_positive.mean(), positive_returns.std())
                        features['conditional_sharpe_ratio'] = conditional_sharpe
            
            # Skewness and kurtosis adjusted Sharpe
            if len(returns) > 30:
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
                sharpe = features.get('sharpe_ratio', 0)
                
                # Skewness adjustment
                skew_adjusted_sharpe = sharpe * (1 + (skewness / 6) * sharpe)
                features['skewness_adjusted_sharpe'] = skew_adjusted_sharpe
                
                # Kurtosis adjustment
                kurt_adjusted_sharpe = sharpe * (1 - (kurtosis - 3) / 24 * sharpe**2)
                features['kurtosis_adjusted_sharpe'] = kurt_adjusted_sharpe
            
            # VaR adjusted return
            if len(returns) > 30:
                var_95 = returns.quantile(0.05)
                if var_95 < 0:
                    var_adjusted_return = safe_divide(returns.mean(), abs(var_95))
                    features['var_adjusted_return'] = var_adjusted_return
            
            # Tail ratio
            if len(returns) > 50:
                tail_95 = returns.quantile(0.95)
                tail_05 = returns.quantile(0.05)
                
                if tail_05 < 0:
                    tail_ratio = safe_divide(tail_95, abs(tail_05))
                    features['tail_ratio'] = tail_ratio
            
            # Capture ratios (simplified)
            if len(returns) > 30:
                positive_returns = returns[returns > 0]
                negative_returns = returns[returns < 0]
                
                if len(positive_returns) > 0:
                    capture_up = safe_divide(positive_returns.mean(), returns.mean()) if returns.mean() > 0 else 0
                    features['capture_ratio_up'] = capture_up
                
                if len(negative_returns) > 0:
                    capture_down = safe_divide(negative_returns.mean(), returns.mean()) if returns.mean() < 0 else 0
                    features['capture_ratio_down'] = abs(capture_down)
                
                # Total capture ratio
                if features.get('capture_ratio_down', 0) > 0:
                    total_capture = safe_divide(features.get('capture_ratio_up', 0), features['capture_ratio_down'])
                    features['capture_ratio_total'] = total_capture
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating advanced performance: {e}")
            return features
    
    def _calculate_performance_attribution(self, returns: pd.Series, benchmark_returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance attribution metrics."""
        try:
            # Simple skill vs luck attribution
            correlation = returns.corr(benchmark_returns)
            
            if not np.isnan(correlation):
                # Skill component (correlated with benchmark)
                skill_component = correlation * returns.std()
                features['return_attribution_skill'] = skill_component
                
                # Luck component (uncorrelated residual)
                luck_component = np.sqrt(max(0, returns.var() - skill_component**2))
                features['return_attribution_luck'] = luck_component
                
                # Skill/luck ratio
                if luck_component > 0:
                    skill_luck_ratio = safe_divide(skill_component, luck_component)
                    features['skill_luck_ratio'] = skill_luck_ratio
            
            # Performance persistence
            if len(returns) > 60:
                # Split into two halves
                mid_point = len(returns) // 2
                first_half = returns.iloc[:mid_point]
                second_half = returns.iloc[mid_point:]
                
                # Correlation between first and second half performance
                first_half_cumulative = (1 + first_half).cumprod()
                second_half_cumulative = (1 + second_half).cumprod()
                
                if len(first_half_cumulative) > 0 and len(second_half_cumulative) > 0:
                    # Simple persistence measure
                    first_return = first_half_cumulative.iloc[-1] - 1
                    second_return = second_half_cumulative.iloc[-1] - 1
                    
                    # Persistence indicator
                    persistence = 1 if (first_return > 0 and second_return > 0) or (first_return < 0 and second_return < 0) else 0
                    features['performance_persistence'] = persistence
            
            # Performance momentum
            if len(returns) > 30:
                # Recent vs historical performance
                recent_returns = returns.iloc[-30:]
                historical_returns = returns.iloc[:-30]
                
                if len(historical_returns) > 0:
                    recent_mean = recent_returns.mean()
                    historical_mean = historical_returns.mean()
                    
                    # Momentum indicator
                    momentum = recent_mean - historical_mean
                    features['performance_momentum'] = momentum
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating performance attribution: {e}")
            return features
"""
Beta Analysis Calculator

Specialized calculator for dynamic beta analysis including:
- Rolling beta coefficients across multiple time windows
- Regime-dependent beta (bull/bear market betas)
- Beta stability and dynamics analysis
- Beta asymmetry and risk decomposition
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from .base_correlation import BaseCorrelationCalculator

logger = logging.getLogger(__name__)


class BetaAnalysisCalculator(BaseCorrelationCalculator):
    """Calculator for dynamic beta analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize beta analysis calculator."""
        super().__init__(config)
        
        # Beta-specific parameters
        self.beta_config = self.correlation_config.get_window_config('beta')
        self.beta_lookback_periods = self.beta_config.get('beta_lookback_periods', [20, 60, 120, 252])
        self.regime_threshold = self.beta_config.get('regime_threshold', 0.02)
        
        logger.debug(f"Initialized BetaAnalysisCalculator with {len(self.beta_lookback_periods)} lookback periods")
    
    def get_feature_names(self) -> List[str]:
        """Return list of beta analysis feature names."""
        feature_names = []
        
        # Dynamic beta features for each window
        for window in self.beta_lookback_periods:
            feature_names.extend([
                f'beta_{window}d',
                f'beta_{window}d_std',
                f'beta_{window}d_change'
            ])
        
        # Regime-dependent betas
        feature_names.extend([
            'beta_bull_market',
            'beta_bear_market', 
            'beta_asymmetry',
            'beta_regime_sensitivity'
        ])
        
        # Beta stability and dynamics
        feature_names.extend([
            'beta_stability_score',
            'beta_trend_20d',
            'beta_momentum_20d',
            'beta_mean_reversion',
            'beta_volatility_20d'
        ])
        
        # Risk decomposition
        feature_names.extend([
            'systematic_risk_ratio',
            'idiosyncratic_risk_ratio',
            'total_risk_20d',
            'beta_adjusted_alpha_20d'
        ])
        
        # Beta percentile features
        feature_names.extend([
            'beta_percentile_1y',
            'beta_relative_strength',
            'beta_shock_sensitivity',
            'downside_beta_20d'
        ])
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate beta analysis features.
        
        Args:
            data: DataFrame with symbol, timestamp, close columns
            
        Returns:
            DataFrame with beta analysis features
        """
        try:
            # Validate and preprocess data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)
            
            processed_data = self.preprocess_data(data)
            
            if processed_data.empty:
                logger.warning("No data available after preprocessing")
                return self.create_empty_features(data.index)
            
            # Create features DataFrame
            unique_timestamps = processed_data['timestamp'].unique()
            features = self.create_empty_features(pd.Index(unique_timestamps))
            
            # Get market proxy for beta calculation
            market_returns = self.get_market_proxy(processed_data)
            
            if market_returns.empty:
                logger.warning("No market proxy available for beta calculation")
                return features
            
            # Calculate rolling betas
            features = self._calculate_rolling_betas(processed_data, features, market_returns)
            
            # Calculate regime-dependent betas
            features = self._calculate_regime_betas(processed_data, features, market_returns)
            
            # Calculate beta stability and dynamics
            features = self._calculate_beta_dynamics(features)
            
            # Calculate risk decomposition
            features = self._calculate_risk_decomposition(processed_data, features, market_returns)
            
            # Calculate beta percentile features
            features = self._calculate_beta_percentiles(features)
            
            # Align features with original data
            if len(features) != len(data):
                features = self._align_features_with_data(features, data)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating beta analysis features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_rolling_betas(self, data: pd.DataFrame, features: pd.DataFrame, 
                               market_returns: pd.Series) -> pd.DataFrame:
        """Calculate rolling beta coefficients."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty:
                return features
            
            # Calculate betas for each symbol and window
            all_symbol_betas = {}
            
            for symbol in returns_pivot.columns:
                if symbol in market_returns.index:  # Skip if symbol is market proxy
                    continue
                
                symbol_returns = returns_pivot[symbol]
                symbol_betas = {}
                
                for window in self.beta_lookback_periods:
                    # Calculate rolling beta
                    rolling_beta = self.calculate_beta(symbol_returns, market_returns, window)
                    
                    # Calculate beta standard deviation
                    beta_std = rolling_beta.rolling(window=window//2, min_periods=5).std()
                    
                    # Calculate beta change
                    beta_change = rolling_beta.diff(periods=5)
                    
                    symbol_betas[f'beta_{window}d'] = rolling_beta
                    symbol_betas[f'beta_{window}d_std'] = beta_std
                    symbol_betas[f'beta_{window}d_change'] = beta_change
                
                all_symbol_betas[symbol] = symbol_betas
            
            # Aggregate betas across symbols (using median)
            for window in self.beta_lookback_periods:
                beta_values = []
                beta_std_values = []
                beta_change_values = []
                
                for symbol_betas in all_symbol_betas.values():
                    beta_values.append(symbol_betas[f'beta_{window}d'])
                    beta_std_values.append(symbol_betas[f'beta_{window}d_std'])
                    beta_change_values.append(symbol_betas[f'beta_{window}d_change'])
                
                if beta_values:
                    # Calculate median betas across symbols
                    median_beta = pd.concat(beta_values, axis=1).median(axis=1)
                    median_beta_std = pd.concat(beta_std_values, axis=1).median(axis=1)
                    median_beta_change = pd.concat(beta_change_values, axis=1).median(axis=1)
                    
                    # Store in features
                    for timestamp in median_beta.index:
                        if timestamp in features.index:
                            features.loc[timestamp, f'beta_{window}d'] = median_beta.loc[timestamp]
                            features.loc[timestamp, f'beta_{window}d_std'] = median_beta_std.loc[timestamp]
                            features.loc[timestamp, f'beta_{window}d_change'] = median_beta_change.loc[timestamp]
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating rolling betas: {e}")
            return features
    
    def _calculate_regime_betas(self, data: pd.DataFrame, features: pd.DataFrame,
                              market_returns: pd.Series) -> pd.DataFrame:
        """Calculate regime-dependent beta coefficients."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty:
                return features
            
            # Define market regimes based on volatility and returns
            market_volatility = market_returns.rolling(window=20).std()
            high_vol_threshold = market_volatility.quantile(0.67)
            
            # Bull market: positive returns and low volatility
            bull_market_mask = (market_returns > 0) & (market_volatility < high_vol_threshold)
            
            # Bear market: negative returns or high volatility
            bear_market_mask = (market_returns < 0) | (market_volatility > high_vol_threshold)
            
            # Calculate regime betas for each symbol
            bull_betas = []
            bear_betas = []
            
            for symbol in returns_pivot.columns:
                if symbol in market_returns.index:  # Skip market proxy
                    continue
                
                symbol_returns = returns_pivot[symbol]
                
                # Bull market beta
                bull_returns = symbol_returns[bull_market_mask]
                bull_market_ret = market_returns[bull_market_mask]
                
                if len(bull_returns) > 10:
                    bull_beta = self.calculate_beta(bull_returns, bull_market_ret)
                    bull_betas.append(bull_beta)
                
                # Bear market beta
                bear_returns = symbol_returns[bear_market_mask]
                bear_market_ret = market_returns[bear_market_mask]
                
                if len(bear_returns) > 10:
                    bear_beta = self.calculate_beta(bear_returns, bear_market_ret)
                    bear_betas.append(bear_beta)
            
            # Calculate median regime betas
            if bull_betas:
                median_bull_beta = np.median(bull_betas)
                features['beta_bull_market'] = median_bull_beta
            
            if bear_betas:
                median_bear_beta = np.median(bear_betas)
                features['beta_bear_market'] = median_bear_beta
            
            # Beta asymmetry
            if bull_betas and bear_betas:
                beta_asymmetry = median_bear_beta - median_bull_beta
                features['beta_asymmetry'] = beta_asymmetry
                
                # Regime sensitivity
                regime_sensitivity = abs(beta_asymmetry)
                features['beta_regime_sensitivity'] = regime_sensitivity
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating regime betas: {e}")
            return features
    
    def _calculate_beta_dynamics(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate beta stability and dynamics features."""
        try:
            # Use 20-day beta for dynamics analysis
            if 'beta_20d' not in features.columns:
                return features
            
            beta_series = features['beta_20d']
            
            # Beta stability (inverse of volatility)
            beta_volatility = beta_series.rolling(window=60, min_periods=20).std()
            beta_stability = 1.0 / (1.0 + beta_volatility)
            features['beta_stability_score'] = beta_stability
            features['beta_volatility_20d'] = beta_volatility
            
            # Beta trend
            def trend_function(values):
                if len(values) < 5:
                    return 0.0
                try:
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    return slope
                except (ValueError, TypeError, np.linalg.LinAlgError):
                    return 0.0
            
            beta_trend = beta_series.rolling(window=20, min_periods=5).apply(trend_function)
            features['beta_trend_20d'] = beta_trend
            
            # Beta momentum
            beta_momentum = beta_series.diff(periods=5)
            features['beta_momentum_20d'] = beta_momentum
            
            # Mean reversion
            rolling_mean = beta_series.rolling(window=60, min_periods=20).mean()
            mean_reversion = (beta_series - rolling_mean) / (rolling_mean + self.numerical_tolerance)
            features['beta_mean_reversion'] = mean_reversion
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating beta dynamics: {e}")
            return features
    
    def _calculate_risk_decomposition(self, data: pd.DataFrame, features: pd.DataFrame,
                                    market_returns: pd.Series) -> pd.DataFrame:
        """Calculate risk decomposition using beta."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty or 'beta_20d' not in features.columns:
                return features
            
            # Calculate total risk for each symbol
            symbol_risks = []
            systematic_risks = []
            idiosyncratic_risks = []
            alphas = []
            
            for symbol in returns_pivot.columns:
                if symbol in market_returns.index:  # Skip market proxy
                    continue
                
                symbol_returns = returns_pivot[symbol]
                
                # Calculate 20-day rolling metrics
                for i in range(20, len(symbol_returns)):
                    window_returns = symbol_returns.iloc[i-20:i]
                    window_market = market_returns.iloc[i-20:i]
                    
                    if len(window_returns.dropna()) < 10:
                        continue
                    
                    # Total risk (volatility)
                    total_risk = window_returns.std()
                    
                    # Beta for this window
                    beta = self.calculate_beta(window_returns, window_market)
                    
                    if not np.isnan(beta):
                        # Systematic risk
                        market_vol = window_market.std()
                        systematic_risk = abs(beta) * market_vol
                        
                        # Idiosyncratic risk (residual volatility)
                        predicted_returns = beta * window_market
                        residuals = window_returns - predicted_returns
                        idiosyncratic_risk = residuals.std()
                        
                        # Alpha (excess return)
                        avg_return = window_returns.mean()
                        avg_market_return = window_market.mean()
                        alpha = avg_return - beta * avg_market_return
                        
                        symbol_risks.append(total_risk)
                        systematic_risks.append(systematic_risk)
                        idiosyncratic_risks.append(idiosyncratic_risk)
                        alphas.append(alpha)
            
            # Calculate median risk metrics
            if symbol_risks:
                total_risk_median = np.median(symbol_risks)
                systematic_risk_median = np.median(systematic_risks)
                idiosyncratic_risk_median = np.median(idiosyncratic_risks)
                alpha_median = np.median(alphas)
                
                # Risk ratios
                systematic_ratio = systematic_risk_median / (total_risk_median + self.numerical_tolerance)
                idiosyncratic_ratio = idiosyncratic_risk_median / (total_risk_median + self.numerical_tolerance)
                
                features['systematic_risk_ratio'] = systematic_ratio
                features['idiosyncratic_risk_ratio'] = idiosyncratic_ratio
                features['total_risk_20d'] = total_risk_median
                features['beta_adjusted_alpha_20d'] = alpha_median
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating risk decomposition: {e}")
            return features
    
    def _calculate_beta_percentiles(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate beta percentile and relative features."""
        try:
            # Use 60-day beta for percentile calculations
            if 'beta_60d' not in features.columns:
                return features
            
            beta_series = features['beta_60d']
            
            # Beta percentile over 1 year
            rolling_percentile = beta_series.rolling(window=252, min_periods=60).rank(pct=True)
            features['beta_percentile_1y'] = rolling_percentile
            
            # Beta relative strength (current vs median)
            rolling_median = beta_series.rolling(window=252, min_periods=60).median()
            relative_strength = beta_series / (rolling_median + self.numerical_tolerance)
            features['beta_relative_strength'] = relative_strength
            
            # Beta shock sensitivity (response to large market moves)
            if 'beta_20d' in features.columns and 'beta_60d' in features.columns:
                beta_short = features['beta_20d']
                beta_long = features['beta_60d']
                shock_sensitivity = abs(beta_short - beta_long)
                features['beta_shock_sensitivity'] = shock_sensitivity
            
            # Downside beta (beta during market declines)
            # This is a simplified version - would need market returns for proper calculation
            downside_proxy = np.where(beta_series.diff() < 0, beta_series, np.nan)
            downside_beta = pd.Series(downside_proxy, index=beta_series.index).rolling(window=20).mean()
            features['downside_beta_20d'] = downside_beta
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating beta percentiles: {e}")
            return features
    
    def _align_features_with_data(self, features: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Align features DataFrame with original data structure."""
        try:
            # Create a mapping from timestamp to features
            if 'timestamp' in original_data.columns:
                # Expand features to match all rows in original data
                expanded_features = original_data[['timestamp']].merge(
                    features.reset_index().rename(columns={'index': 'timestamp'}),
                    on='timestamp',
                    how='left'
                )
                
                # Set index to match original data
                expanded_features = expanded_features.drop('timestamp', axis=1)
                expanded_features.index = original_data.index
                
                # Fill NaN values with 0
                expanded_features = expanded_features.fillna(0.0)
                
                return expanded_features
            else:
                # If no timestamp column, try to align by index
                return features.reindex(original_data.index, fill_value=0.0)
                
        except Exception as e:
            logger.warning(f"Error aligning features with data: {e}")
            return features
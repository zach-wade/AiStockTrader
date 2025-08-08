"""
Volatility Calculator

Specialized calculator for volatility modeling and analysis including:
- Historical volatility (simple, exponentially weighted)
- Realized volatility (Parkinson, Garman-Klass estimators)
- GARCH volatility modeling
- Volatility clustering and persistence
- Volatility forecasting and term structure
- Volatility risk metrics and ratios
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
import warnings

from .base_risk import BaseRiskCalculator
from .risk_config import VolatilityMethod

logger = logging.getLogger(__name__)


class VolatilityCalculator(BaseRiskCalculator):
    """Calculator for volatility metrics and modeling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize volatility calculator."""
        super().__init__(config)
        
        # Volatility-specific configuration
        self.vol_config = self.risk_config.get_volatility_config()
        self.lookback_window = self.vol_config['lookback_window']
        self.min_observations = self.vol_config['min_observations']
        self.annualization_factor = self.vol_config['annualization_factor']
        self.ewma_decay = self.vol_config['ewma_decay_factor']
        
        logger.debug(f"Initialized VolatilityCalculator with {self.lookback_window}d lookback")
    
    def get_feature_names(self) -> List[str]:
        """Return list of volatility feature names."""
        feature_names = []
        
        # Basic volatility metrics
        feature_names.extend([
            'historical_volatility',
            'annualized_volatility',
            'realized_volatility',
            'intraday_volatility',
            'overnight_volatility',
            'volatility_ratio',
        ])
        
        # EWMA volatility
        feature_names.extend([
            'ewma_volatility',
            'ewma_variance',
            'ewma_vol_ratio',
            'ewma_persistence',
        ])
        
        # Realized volatility estimators
        feature_names.extend([
            'parkinson_volatility',
            'garman_klass_volatility',
            'rogers_satchell_volatility',
            'yang_zhang_volatility',
        ])
        
        # Rolling volatility metrics
        feature_names.extend([
            'volatility_20d',
            'volatility_60d',
            'volatility_252d',
            'volatility_ratio_20_60',
            'volatility_ratio_60_252',
            'volatility_trend_20d',
            'volatility_trend_60d',
        ])
        
        # Volatility clustering and persistence
        feature_names.extend([
            'volatility_clustering',
            'volatility_persistence',
            'volatility_mean_reversion',
            'volatility_autocorr_1',
            'volatility_autocorr_5',
            'volatility_half_life',
        ])
        
        # Volatility distribution characteristics
        feature_names.extend([
            'volatility_skewness',
            'volatility_kurtosis',
            'volatility_min',
            'volatility_max',
            'volatility_range',
            'volatility_percentile_25',
            'volatility_percentile_50',
            'volatility_percentile_75',
            'volatility_percentile_95',
        ])
        
        # Volatility forecasting and term structure
        feature_names.extend([
            'volatility_forecast_1d',
            'volatility_forecast_5d',
            'volatility_forecast_21d',
            'volatility_term_structure_slope',
            'volatility_term_structure_level',
        ])
        
        # GARCH volatility (if available)
        feature_names.extend([
            'garch_volatility',
            'garch_conditional_vol',
            'garch_unconditional_vol',
            'garch_persistence',
            'garch_aic',
            'garch_bic',
        ])
        
        # Volatility risk metrics
        feature_names.extend([
            'volatility_of_volatility',
            'volatility_shock_sensitivity',
            'volatility_regime_indicator',
            'volatility_stress_level',
        ])
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features
        """
        try:
            # Validate input data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)
            
            # Calculate returns
            returns = self.calculate_returns(data)
            
            if len(returns) < self.min_observations:
                logger.warning("Insufficient data for volatility calculation")
                return self.create_empty_features(data.index)
            
            # Create features DataFrame
            features = self.create_empty_features(data.index)
            
            # Calculate volatility metrics
            features = self._calculate_basic_volatility(returns, features)
            features = self._calculate_ewma_volatility(returns, features)
            features = self._calculate_realized_volatility(data, features)
            features = self._calculate_rolling_volatility(returns, features)
            features = self._calculate_volatility_clustering(returns, features)
            features = self._calculate_volatility_distribution(returns, features)
            features = self._calculate_volatility_forecasting(returns, features)
            features = self._calculate_garch_volatility(returns, features)
            features = self._calculate_volatility_risk_metrics(returns, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating volatility features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_basic_volatility(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic volatility metrics."""
        try:
            # Historical volatility
            hist_vol = returns.std()
            features['historical_volatility'] = hist_vol
            
            # Annualized volatility
            annualized_vol = hist_vol * np.sqrt(self.annualization_factor)
            features['annualized_volatility'] = annualized_vol
            
            # Simple realized volatility
            realized_vol = np.sqrt(np.sum(returns**2))
            features['realized_volatility'] = realized_vol
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating basic volatility: {e}")
            return features
    
    def _calculate_ewma_volatility(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate EWMA (Exponentially Weighted Moving Average) volatility."""
        try:
            # EWMA variance
            ewma_var = returns.ewm(alpha=1-self.ewma_decay, adjust=False).var()
            ewma_vol = np.sqrt(ewma_var)
            
            # Current EWMA volatility
            features['ewma_volatility'] = ewma_vol.iloc[-1] if len(ewma_vol) > 0 else 0.0
            features['ewma_variance'] = ewma_var.iloc[-1] if len(ewma_var) > 0 else 0.0
            
            # EWMA to historical volatility ratio
            hist_vol = returns.std()
            if hist_vol != 0:
                features['ewma_vol_ratio'] = features['ewma_volatility'] / hist_vol
            
            # EWMA persistence (autocorrelation of EWMA volatility)
            if len(ewma_vol) > 10:
                try:
                    ewma_persistence = ewma_vol.autocorr(lag=1)
                    features['ewma_persistence'] = ewma_persistence if not np.isnan(ewma_persistence) else 0.0
                except Exception as e:
                    logger.debug(f"Error calculating EWMA persistence: {e}")
                    features['ewma_persistence'] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating EWMA volatility: {e}")
            return features
    
    def _calculate_realized_volatility(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate realized volatility using high-frequency estimators."""
        try:
            # Check if we have OHLC data
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                logger.debug("OHLC data not available for realized volatility")
                return features
            
            # Get latest values
            open_price = data['open'].iloc[-1]
            high_price = data['high'].iloc[-1]
            low_price = data['low'].iloc[-1]
            close_price = data['close'].iloc[-1]
            prev_close = data['close'].iloc[-2] if len(data) > 1 else close_price
            
            # Parkinson estimator
            if high_price > low_price:
                parkinson_vol = np.sqrt(
                    (1 / (4 * np.log(2))) * (np.log(high_price / low_price))**2
                )
                features['parkinson_volatility'] = parkinson_vol
            
            # Garman-Klass estimator
            if high_price > low_price and close_price > 0:
                gk_vol = np.sqrt(
                    0.5 * (np.log(high_price / low_price))**2 - 
                    (2 * np.log(2) - 1) * (np.log(close_price / open_price))**2
                )
                features['garman_klass_volatility'] = gk_vol
            
            # Rogers-Satchell estimator
            if all(p > 0 for p in [open_price, high_price, low_price, close_price]):
                rs_vol = np.sqrt(
                    np.log(high_price / close_price) * np.log(high_price / open_price) +
                    np.log(low_price / close_price) * np.log(low_price / open_price)
                )
                features['rogers_satchell_volatility'] = rs_vol
            
            # Yang-Zhang estimator (simplified)
            if prev_close > 0:
                overnight_return = np.log(open_price / prev_close)
                close_to_close_return = np.log(close_price / prev_close)
                
                features['overnight_volatility'] = abs(overnight_return)
                features['intraday_volatility'] = features.get('garman_klass_volatility', 0.0)
                
                # Volatility ratio (intraday vs overnight)
                if features['overnight_volatility'] != 0:
                    features['volatility_ratio'] = features['intraday_volatility'] / features['overnight_volatility']
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating realized volatility: {e}")
            return features
    
    def _calculate_rolling_volatility(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling volatility metrics."""
        try:
            # Different lookback windows
            windows = [20, 60, 252]
            
            for window in windows:
                if len(returns) >= window:
                    rolling_vol = returns.rolling(window=window).std()
                    current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0.0
                    features[f'volatility_{window}d'] = current_vol
            
            # Volatility ratios
            if 'volatility_20d' in features.columns and 'volatility_60d' in features.columns:
                if features['volatility_60d'] != 0:
                    features['volatility_ratio_20_60'] = features['volatility_20d'] / features['volatility_60d']
            
            if 'volatility_60d' in features.columns and 'volatility_252d' in features.columns:
                if features['volatility_252d'] != 0:
                    features['volatility_ratio_60_252'] = features['volatility_60d'] / features['volatility_252d']
            
            # Volatility trends
            for window in [20, 60]:
                if len(returns) >= window * 2:
                    rolling_vol = returns.rolling(window=window).std()
                    if len(rolling_vol) > 10:
                        # Calculate trend (slope of volatility over time)
                        recent_vol = rolling_vol.iloc[-10:]
                        if len(recent_vol) > 1:
                            x = np.arange(len(recent_vol))
                            try:
                                slope = np.polyfit(x, recent_vol, 1)[0]
                                features[f'volatility_trend_{window}d'] = slope
                            except Exception as e:
                                logger.debug(f"Error calculating volatility trend for {window}d: {e}")
                                features[f'volatility_trend_{window}d'] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating rolling volatility: {e}")
            return features
    
    def _calculate_volatility_clustering(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility clustering and persistence metrics."""
        try:
            # Calculate squared returns (proxy for volatility)
            squared_returns = returns**2
            
            # Volatility clustering (autocorrelation of squared returns)
            if len(squared_returns) > 10:
                try:
                    clustering = squared_returns.autocorr(lag=1)
                    features['volatility_clustering'] = clustering if not np.isnan(clustering) else 0.0
                except Exception as e:
                    logger.debug(f"Error calculating volatility clustering: {e}")
                    features['volatility_clustering'] = 0.0
            
            # Volatility persistence (higher order autocorrelations)
            autocorr_lags = [1, 5]
            for lag in autocorr_lags:
                if len(squared_returns) > lag + 5:
                    try:
                        autocorr = squared_returns.autocorr(lag=lag)
                        features[f'volatility_autocorr_{lag}'] = autocorr if not np.isnan(autocorr) else 0.0
                    except Exception as e:
                        logger.debug(f"Error calculating volatility autocorr for lag {lag}: {e}")
                        features[f'volatility_autocorr_{lag}'] = 0.0
            
            # Volatility persistence (weighted average of autocorrelations)
            autocorr_values = []
            for lag in range(1, 6):
                if len(squared_returns) > lag + 5:
                    try:
                        autocorr = squared_returns.autocorr(lag=lag)
                        if not np.isnan(autocorr):
                            autocorr_values.append(autocorr)
                    except Exception as e:
                        logger.debug(f"Error in autocorr calculation: {e}")
                        continue
            
            if autocorr_values:
                # Weighted average with exponential decay
                weights = np.exp(-np.arange(len(autocorr_values)) * 0.5)
                persistence = np.average(autocorr_values, weights=weights)
                features['volatility_persistence'] = persistence
                
                # Half-life of volatility shocks
                if persistence > 0 and persistence < 1:
                    half_life = np.log(0.5) / np.log(persistence)
                    features['volatility_half_life'] = half_life
            
            # Mean reversion in volatility
            if len(returns) > 60:
                rolling_vol = returns.rolling(window=20).std()
                if len(rolling_vol) > 40:
                    vol_mean = rolling_vol.mean()
                    vol_deviations = rolling_vol - vol_mean
                    
                    # Calculate mean reversion speed
                    try:
                        mean_reversion = -vol_deviations.autocorr(lag=1)
                        features['volatility_mean_reversion'] = mean_reversion if not np.isnan(mean_reversion) else 0.0
                    except Exception as e:
                        logger.debug(f"Error calculating volatility mean reversion: {e}")
                        features['volatility_mean_reversion'] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating volatility clustering: {e}")
            return features
    
    def _calculate_volatility_distribution(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility distribution characteristics."""
        try:
            # Calculate rolling volatility
            if len(returns) >= 60:
                rolling_vol = returns.rolling(window=20).std()
                vol_data = rolling_vol.dropna()
                
                if len(vol_data) > 10:
                    # Distribution characteristics
                    features['volatility_skewness'] = vol_data.skew()
                    features['volatility_kurtosis'] = vol_data.kurtosis()
                    features['volatility_min'] = vol_data.min()
                    features['volatility_max'] = vol_data.max()
                    features['volatility_range'] = vol_data.max() - vol_data.min()
                    
                    # Percentiles
                    percentiles = [25, 50, 75, 95]
                    for p in percentiles:
                        features[f'volatility_percentile_{p}'] = vol_data.quantile(p/100)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating volatility distribution: {e}")
            return features
    
    def _calculate_volatility_forecasting(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility forecasting metrics."""
        try:
            # Simple volatility forecasting using EWMA
            if len(returns) >= 30:
                ewma_var = returns.ewm(alpha=1-self.ewma_decay, adjust=False).var()
                current_vol = np.sqrt(ewma_var.iloc[-1]) if len(ewma_var) > 0 else 0.0
                
                # Simple persistence-based forecasting
                persistence = features.get('volatility_persistence', 0.8)
                
                # Forecast volatility for different horizons
                horizons = [1, 5, 21]
                for horizon in horizons:
                    # Simple mean reversion model
                    long_term_vol = returns.std()
                    forecast_vol = long_term_vol + (current_vol - long_term_vol) * (persistence ** horizon)
                    features[f'volatility_forecast_{horizon}d'] = forecast_vol
            
            # Volatility term structure
            if all(f'volatility_forecast_{h}d' in features.columns for h in [1, 5, 21]):
                vol_1d = features['volatility_forecast_1d']
                vol_21d = features['volatility_forecast_21d']
                
                # Term structure slope
                if vol_1d != 0:
                    features['volatility_term_structure_slope'] = (vol_21d - vol_1d) / vol_1d
                
                # Term structure level
                features['volatility_term_structure_level'] = (vol_1d + vol_21d) / 2
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating volatility forecasting: {e}")
            return features
    
    def _calculate_garch_volatility(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate GARCH volatility metrics."""
        try:
            # Check if arch package is available
            try:
                import arch
                has_arch = True
            except ImportError:
                has_arch = False
                logger.debug("ARCH package not available for GARCH modeling")
            
            if has_arch and len(returns) >= 100:
                try:
                    # Scale returns for numerical stability
                    scaled_returns = returns * 100
                    
                    # Fit GARCH(1,1) model
                    model = arch.arch_model(scaled_returns, vol='Garch', p=1, q=1)
                    fitted_model = model.fit(disp='off')
                    
                    # Extract volatility
                    conditional_vol = fitted_model.conditional_volatility / 100  # Scale back
                    
                    # Current conditional volatility
                    features['garch_conditional_vol'] = conditional_vol.iloc[-1]
                    
                    # Unconditional volatility
                    features['garch_unconditional_vol'] = np.sqrt(fitted_model.params['omega'] / 
                                                                (1 - fitted_model.params['alpha[1]'] - 
                                                                 fitted_model.params['beta[1]'])) / 100
                    
                    # GARCH persistence
                    garch_persistence = fitted_model.params['alpha[1]'] + fitted_model.params['beta[1]']
                    features['garch_persistence'] = garch_persistence
                    
                    # Model fit statistics
                    features['garch_aic'] = fitted_model.aic
                    features['garch_bic'] = fitted_model.bic
                    
                    # Use GARCH volatility as main volatility measure
                    features['garch_volatility'] = conditional_vol.iloc[-1]
                    
                except Exception as e:
                    logger.debug(f"Error fitting GARCH model: {e}")
                    # Use EWMA as fallback
                    features['garch_volatility'] = features.get('ewma_volatility', 0.0)
            else:
                # Use EWMA as fallback
                features['garch_volatility'] = features.get('ewma_volatility', 0.0)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating GARCH volatility: {e}")
            return features
    
    def _calculate_volatility_risk_metrics(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility risk metrics."""
        try:
            # Volatility of volatility
            if len(returns) >= 60:
                rolling_vol = returns.rolling(window=20).std()
                if len(rolling_vol) > 20:
                    vol_of_vol = rolling_vol.std()
                    features['volatility_of_volatility'] = vol_of_vol
            
            # Volatility shock sensitivity
            if len(returns) >= 30:
                # Calculate how much volatility changes after large moves
                large_moves = returns[abs(returns) > returns.std() * 2]
                if len(large_moves) > 5:
                    # Average volatility change after large moves
                    post_shock_vol = []
                    for idx in large_moves.index:
                        try:
                            pos = returns.index.get_loc(idx)
                            if pos < len(returns) - 10:
                                pre_vol = returns.iloc[max(0, pos-10):pos].std()
                                post_vol = returns.iloc[pos:pos+10].std()
                                if pre_vol > 0:
                                    post_shock_vol.append(post_vol / pre_vol)
                        except Exception as e:
                            logger.debug(f"Error in shock response calculation: {e}")
                            continue
                    
                    if post_shock_vol:
                        features['volatility_shock_sensitivity'] = np.mean(post_shock_vol)
            
            # Volatility regime indicator
            current_vol = features.get('historical_volatility', 0.0)
            if len(returns) >= 252:
                long_term_vol = returns.std()
                if long_term_vol > 0:
                    vol_regime = current_vol / long_term_vol
                    features['volatility_regime_indicator'] = vol_regime
            
            # Volatility stress level
            if 'volatility_percentile_95' in features.columns:
                vol_95 = features['volatility_percentile_95']
                if vol_95 > 0:
                    stress_level = min(current_vol / vol_95, 2.0)  # Cap at 2x
                    features['volatility_stress_level'] = stress_level
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating volatility risk metrics: {e}")
            return features
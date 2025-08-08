"""
Statistical Moments Calculator

Specialized calculator for statistical moments and distribution properties including:
- Higher order moments (mean, std, skew, kurtosis, higher moments)
- Distribution tests (normality, Anderson-Darling, Kolmogorov-Smirnov)
- Tail analysis and shape parameters
- Cross-moment features
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from scipy import stats
from scipy.stats import jarque_bera, normaltest, anderson

from .base_statistical import BaseStatisticalCalculator

logger = logging.getLogger(__name__)


class MomentsCalculator(BaseStatisticalCalculator):
    """Calculator for statistical moments and distribution properties."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize moments calculator."""
        super().__init__(config)
        
        # Distribution test parameters
        self.distribution_test_windows = self.config.get('distribution_test_windows', [60, 120])
        self.cross_moment_window = self.config.get('cross_moment_window', 60)
    
    def get_feature_names(self) -> List[str]:
        """Return list of moment feature names."""
        feature_names = []
        
        # Higher moments for each window
        for window in self.stat_config.moment_windows:
            feature_names.extend([
                f'mean_{window}', f'std_{window}', f'skew_{window}', f'kurt_{window}',
                f'jarque_bera_{window}', f'tail_ratio_{window}'
            ])
            
            # Higher order moments
            for order in range(5, self.stat_config.max_moment_order + 1):
                feature_names.append(f'moment_{order}_{window}')
        
        # Distribution tests
        for window in self.distribution_test_windows:
            feature_names.extend([
                f'normal_pvalue_{window}', f'anderson_stat_{window}', f'ks_stat_{window}'
            ])
        
        # Cross-moment features
        feature_names.extend([
            'hl_spread_skew', 'hl_spread_kurt', 'return_volume_corr',
            'price_range_consistency', 'volatility_clustering', 'tail_asymmetry'
        ])
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistical moments and distribution features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with moment features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_empty_features(data.index)
            
            # Calculate returns for moment analysis
            returns = self.calculate_returns(data['close'])
            
            # Calculate higher order moments
            features = self._calculate_higher_moments(returns, features)
            
            # Calculate distribution tests
            features = self._calculate_distribution_tests(returns, features)
            
            # Calculate cross-moment features
            features = self._calculate_cross_moments(data, features)
            
            # Calculate derived moment features
            features = self._calculate_derived_moments(data, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating moment features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_higher_moments(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate higher order statistical moments."""
        for window in self.stat_config.moment_windows:
            # Standard moments
            features[f'mean_{window}'] = returns.rolling(window).mean()
            features[f'std_{window}'] = returns.rolling(window).std()
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()
            
            # Higher moments (5th, 6th, etc.)
            for order in range(5, self.stat_config.max_moment_order + 1):
                def moment_func(x):
                    if len(x) < window // 2:
                        return np.nan
                    return stats.moment(x, moment=order)
                
                features[f'moment_{order}_{window}'] = self.rolling_apply_safe(
                    returns, window, moment_func
                )
            
            # Jarque-Bera test statistic for normality
            def jarque_bera_func(x):
                if len(x) < 8:  # Minimum for Jarque-Bera test
                    return np.nan
                try:
                    return jarque_bera(x)[0]
                except (ValueError, RuntimeWarning):
                    return np.nan
            
            features[f'jarque_bera_{window}'] = self.rolling_apply_safe(
                returns, window, jarque_bera_func
            )
            
            # Tail ratio (measure of extreme values)
            def tail_ratio_func(x):
                if len(x) < 20:  # Need sufficient data for percentiles
                    return np.nan
                try:
                    p95 = np.percentile(x, 95)
                    p5 = np.percentile(x, 5)
                    if abs(p5) < self.numerical_tolerance:
                        return np.nan
                    return p95 / p5
                except (ValueError, ZeroDivisionError):
                    return np.nan
            
            features[f'tail_ratio_{window}'] = self.rolling_apply_safe(
                returns, window, tail_ratio_func
            )
        
        return features
    
    def _calculate_distribution_tests(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical distribution tests."""
        for window in self.distribution_test_windows:
            # Normality test (D'Agostino and Pearson)
            def normal_test_func(x):
                if len(x) < 20:  # Minimum for reliable normality test
                    return np.nan
                try:
                    return normaltest(x)[1]  # p-value
                except (ValueError, RuntimeWarning):
                    return np.nan
            
            features[f'normal_pvalue_{window}'] = self.rolling_apply_safe(
                returns, window, normal_test_func
            )
            
            # Anderson-Darling test statistic
            def anderson_func(x):
                if len(x) < 20:
                    return np.nan
                try:
                    return anderson(x, dist='norm')[0]
                except (ValueError, RuntimeWarning):
                    return np.nan
            
            features[f'anderson_stat_{window}'] = self.rolling_apply_safe(
                returns, window, anderson_func
            )
            
            # Kolmogorov-Smirnov test against normal distribution
            def ks_test_func(x):
                if len(x) < 20:
                    return np.nan
                try:
                    mean_x, std_x = x.mean(), x.std()
                    if std_x < self.numerical_tolerance:
                        return np.nan
                    return stats.kstest(x, 'norm', args=(mean_x, std_x))[0]
                except (ValueError, RuntimeWarning):
                    return np.nan
            
            features[f'ks_stat_{window}'] = self.rolling_apply_safe(
                returns, window, ks_test_func
            )
        
        return features
    
    def _calculate_cross_moments(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-moment features between different variables."""
        # High-Low spread moments
        if all(col in data.columns for col in ['high', 'low', 'close']):
            hl_spread = (data['high'] - data['low']) / data['close']
            features['hl_spread_skew'] = hl_spread.rolling(self.cross_moment_window).skew()
            features['hl_spread_kurt'] = hl_spread.rolling(self.cross_moment_window).kurt()
        
        # Return-Volume correlation
        if 'volume' in data.columns:
            returns = self.calculate_returns(data['close'])
            volume_change = data['volume'].pct_change()
            
            def correlation_func(ret_vol_pair):
                ret_data = ret_vol_pair[:len(ret_vol_pair)//2]
                vol_data = ret_vol_pair[len(ret_vol_pair)//2:]
                
                if len(ret_data) < 10 or len(vol_data) < 10:
                    return np.nan
                try:
                    return np.corrcoef(ret_data, vol_data)[0, 1]
                except (ValueError, RuntimeWarning):
                    return np.nan
            
            # Combine returns and volume changes for rolling correlation
            combined_series = pd.concat([returns, volume_change], axis=1).apply(
                lambda row: np.concatenate([row.iloc[0:1].values, row.iloc[1:2].values]), axis=1
            )
            
            features['return_volume_corr'] = returns.rolling(30).corr(volume_change)
        
        return features
    
    def _calculate_derived_moments(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived moment-based features."""
        returns = self.calculate_returns(data['close'])
        
        # Price range consistency (measure of OHLC relationship stability)
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            price_range = data['high'] - data['low']
            
            # Consistency of price range relative to typical price
            range_ratio = price_range / typical_price
            features['price_range_consistency'] = 1.0 / (1.0 + range_ratio.rolling(30).std())
        
        # Volatility clustering (correlation in squared returns)
        squared_returns = returns ** 2
        features['volatility_clustering'] = squared_returns.rolling(20).corr(squared_returns.shift(1))
        
        # Tail asymmetry (difference between left and right tail behavior)
        def tail_asymmetry_func(x):
            if len(x) < 30:
                return np.nan
            try:
                left_tail = np.percentile(x, 5)
                right_tail = np.percentile(x, 95)
                median_val = np.median(x)
                
                left_distance = abs(median_val - left_tail)
                right_distance = abs(right_tail - median_val)
                
                if (left_distance + right_distance) < self.numerical_tolerance:
                    return 0.0
                
                return (right_distance - left_distance) / (right_distance + left_distance)
            except (ValueError, RuntimeWarning):
                return np.nan
        
        features['tail_asymmetry'] = self.rolling_apply_safe(
            returns, 60, tail_asymmetry_func
        )
        
        return features
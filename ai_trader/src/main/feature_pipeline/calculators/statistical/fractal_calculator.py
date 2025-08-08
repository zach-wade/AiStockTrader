"""
Fractal Calculator

Specialized calculator for fractal analysis and self-similarity measures including:
- Hurst exponent (multiple methods)
- Detrended Fluctuation Analysis (DFA)
- Multifractal spectrum analysis
- Self-similarity tests
- Fractal dimension estimation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from scipy import stats

from .base_statistical import BaseStatisticalCalculator

logger = logging.getLogger(__name__)


class FractalCalculator(BaseStatisticalCalculator):
    """Calculator for fractal analysis and self-similarity measures."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fractal calculator."""
        super().__init__(config)
        
        # Fractal analysis parameters
        self.hurst_min_lags = self.config.get('hurst_min_lags', 2)
        self.hurst_max_lags = self.config.get('hurst_max_lags', 20)
        self.dfa_scales = self.config.get('dfa_scales', [4, 8, 16, 32])
        self.multifractal_q_values = self.config.get('multifractal_q_values', [-5, -3, -1, 0, 1, 3, 5])
        
        # Minimum data requirements for reliable fractal analysis
        self.min_data_for_fractals = self.config.get('min_data_for_fractals', 50)
    
    def get_feature_names(self) -> List[str]:
        """Return list of fractal feature names."""
        feature_names = []
        
        # Hurst exponent using different methods
        for method in self.stat_config.hurst_methods:
            feature_names.append(f'hurst_{method}')
        
        # Other fractal measures
        feature_names.extend([
            'dfa_alpha', 'multifractal_width', 'self_similarity_test',
            'fractal_dimension', 'persistence_strength', 'antipersistence_strength',
            'trend_strength', 'cyclical_strength', 'roughness_index'
        ])
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fractal analysis features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with fractal features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_empty_features(data.index)
            
            # Calculate returns for fractal analysis
            returns = self.calculate_returns(data['close'])
            
            # Use rolling windows for fractal analysis
            fractal_window = max(100, self.min_data_for_fractals)
            
            # Calculate Hurst exponents using different methods
            features = self._calculate_hurst_exponents(returns, features, fractal_window)
            
            # Calculate DFA
            features = self._calculate_dfa_features(returns, features, fractal_window)
            
            # Calculate multifractal analysis
            features = self._calculate_multifractal_features(returns, features, fractal_window)
            
            # Calculate self-similarity and derived measures
            features = self._calculate_similarity_measures(returns, features, fractal_window)
            
            # Calculate additional fractal-based features
            features = self._calculate_derived_fractal_features(data, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating fractal features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_hurst_exponents(self, returns: pd.Series, features: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate Hurst exponents using different methods."""
        
        # R/S Analysis method
        if 'rs' in self.stat_config.hurst_methods:
            def hurst_rs_func(x):
                return self._calculate_hurst_rs(x)
            
            features['hurst_rs'] = self.rolling_apply_safe(
                returns, window, hurst_rs_func
            )
        
        # DMA (Detrended Moving Average) method
        if 'dma' in self.stat_config.hurst_methods:
            def hurst_dma_func(x):
                return self._calculate_hurst_dma(x)
            
            features['hurst_dma'] = self.rolling_apply_safe(
                returns, window, hurst_dma_func
            )
        
        # Peng method (simplified DFA)
        if 'peng' in self.stat_config.hurst_methods:
            def hurst_peng_func(x):
                return self._calculate_hurst_peng(x)
            
            features['hurst_peng'] = self.rolling_apply_safe(
                returns, window, hurst_peng_func
            )
        
        return features
    
    def _calculate_dfa_features(self, returns: pd.Series, features: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate Detrended Fluctuation Analysis features."""
        def dfa_func(x):
            return self._calculate_dfa(x)
        
        features['dfa_alpha'] = self.rolling_apply_safe(
            returns, window, dfa_func
        )
        
        return features
    
    def _calculate_multifractal_features(self, returns: pd.Series, features: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate multifractal analysis features."""
        def multifractal_func(x):
            return self._calculate_multifractal_width(x)
        
        features['multifractal_width'] = self.rolling_apply_safe(
            returns, window, multifractal_func
        )
        
        return features
    
    def _calculate_similarity_measures(self, returns: pd.Series, features: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate self-similarity and related measures."""
        def self_similarity_func(x):
            return self._test_self_similarity(x)
        
        features['self_similarity_test'] = self.rolling_apply_safe(
            returns, window, self_similarity_func
        )
        
        # Fractal dimension (simplified box-counting approximation)
        def fractal_dimension_func(x):
            return self._estimate_fractal_dimension(x)
        
        features['fractal_dimension'] = self.rolling_apply_safe(
            returns, window, fractal_dimension_func
        )
        
        return features
    
    def _calculate_derived_fractal_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features based on fractal analysis."""
        # Persistence and antipersistence strength
        if 'hurst_rs' in features.columns:
            hurst_values = features['hurst_rs']
            
            # Persistence strength (how much H > 0.5)
            features['persistence_strength'] = (hurst_values - 0.5).clip(lower=0)
            
            # Antipersistence strength (how much H < 0.5)
            features['antipersistence_strength'] = (0.5 - hurst_values).clip(lower=0)
        
        # Trend strength based on DFA alpha
        if 'dfa_alpha' in features.columns:
            # Strong trend when alpha > 1, mean reversion when alpha < 0.5
            features['trend_strength'] = features['dfa_alpha'].apply(
                lambda x: max(0, x - 1) if not np.isnan(x) else np.nan
            )
        
        # Cyclical strength (deviation from pure random walk)
        returns = self.calculate_returns(data['close'])
        def cyclical_strength_func(x):
            if len(x) < 30:
                return np.nan
            
            # Measure regularity in autocorrelation structure
            autocorr = self.calculate_autocorrelation(pd.Series(x), max_lags=10)
            if len(autocorr) < 5:
                return np.nan
            
            # Look for cyclical patterns in autocorrelation
            positive_autocorr = np.sum(autocorr > 0.1)
            return positive_autocorr / len(autocorr)
        
        features['cyclical_strength'] = self.rolling_apply_safe(
            returns, 60, cyclical_strength_func
        )
        
        # Roughness index (measure of path irregularity)
        def roughness_func(x):
            if len(x) < 20:
                return np.nan
            
            # Calculate first differences
            diffs = np.diff(x)
            
            # Roughness as normalized sum of absolute second differences
            if len(diffs) < 2:
                return np.nan
            
            second_diffs = np.abs(np.diff(diffs))
            path_length = np.sum(np.abs(diffs))
            
            if path_length < self.numerical_tolerance:
                return np.nan
            
            return np.sum(second_diffs) / path_length
        
        features['roughness_index'] = self.rolling_apply_safe(
            returns, 50, roughness_func
        )
        
        return features
    
    def _calculate_hurst_rs(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        if len(data) < 20:
            return np.nan
        
        try:
            # R/S Analysis
            lags = range(self.hurst_min_lags, min(self.hurst_max_lags, len(data) // 3))
            rs_values = []
            
            for lag in lags:
                if lag >= len(data):
                    continue
                
                # Calculate cumulative deviations
                mean_val = np.mean(data[:lag])
                cumdev = np.cumsum(data[:lag] - mean_val)
                
                # Range
                R = np.max(cumdev) - np.min(cumdev)
                
                # Standard deviation
                S = np.std(data[:lag])
                
                if S > self.numerical_tolerance:
                    rs_values.append(R / S)
                else:
                    rs_values.append(np.nan)
            
            # Filter out invalid values
            valid_rs = [(lag, rs) for lag, rs in zip(lags, rs_values) if not np.isnan(rs) and rs > 0]
            
            if len(valid_rs) < 3:
                return np.nan
            
            # Linear fit in log-log space
            log_lags = [np.log(lag) for lag, rs in valid_rs]
            log_rs = [np.log(rs) for lag, rs in valid_rs]
            
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            
            return hurst
            
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan
    
    def _calculate_hurst_dma(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using Detrended Moving Average method."""
        if len(data) < 20:
            return np.nan
        
        try:
            # DMA method
            lags = range(2, min(20, len(data) // 4))
            fluctuations = []
            
            for lag in lags:
                # Moving average detrending
                ma = pd.Series(data).rolling(window=lag, center=True).mean()
                detrended = data - ma.fillna(method='bfill').fillna(method='ffill')
                
                # Calculate fluctuation
                fluct = np.sqrt(np.mean(detrended**2))
                fluctuations.append(fluct)
            
            # Filter valid fluctuations
            valid_data = [(lag, fluct) for lag, fluct in zip(lags, fluctuations) 
                         if not np.isnan(fluct) and fluct > 0]
            
            if len(valid_data) < 3:
                return np.nan
            
            # Power law fit
            log_lags = [np.log(lag) for lag, fluct in valid_data]
            log_fluct = [np.log(fluct) for lag, fluct in valid_data]
            
            hurst = np.polyfit(log_lags, log_fluct, 1)[0]
            
            return hurst
            
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan
    
    def _calculate_hurst_peng(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using Peng method (simplified DFA)."""
        return self._calculate_dfa(data)
    
    def _calculate_dfa(self, data: np.ndarray) -> float:
        """Calculate Detrended Fluctuation Analysis exponent."""
        if len(data) < 16:
            return np.nan
        
        try:
            # Integrate the series
            y = np.cumsum(data - np.mean(data))
            
            # Calculate fluctuation for different scales
            scales = [s for s in self.dfa_scales if s <= len(y) // 4]
            if len(scales) < 2:
                return np.nan
            
            fluctuations = []
            
            for scale in scales:
                # Divide into segments
                n_segments = len(y) // scale
                if n_segments < 2:
                    continue
                
                segment_fluctuations = []
                
                for i in range(n_segments):
                    segment = y[i*scale:(i+1)*scale]
                    x = np.arange(len(segment))
                    
                    # Linear detrending
                    coeffs = np.polyfit(x, segment, 1)
                    fit = np.polyval(coeffs, x)
                    
                    # Calculate fluctuation
                    segment_fluctuations.append(np.sqrt(np.mean((segment - fit)**2)))
                
                if segment_fluctuations:
                    fluctuations.append(np.mean(segment_fluctuations))
            
            # Power law fit
            if len(fluctuations) >= 2:
                scales_used = scales[:len(fluctuations)]
                log_scales = np.log(scales_used)
                log_fluct = np.log([f for f in fluctuations if f > 0])
                
                if len(log_fluct) >= 2:
                    alpha = np.polyfit(log_scales[:len(log_fluct)], log_fluct, 1)[0]
                    return alpha
            
            return np.nan
            
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan
    
    def _calculate_multifractal_width(self, data: np.ndarray) -> float:
        """Calculate width of multifractal spectrum."""
        if len(data) < 50:
            return np.nan
        
        try:
            h_q_values = []
            
            for q in self.multifractal_q_values:
                if q == 0:
                    # Special case for q=0 (log-based calculation)
                    log_data = np.log(np.abs(data) + 1e-10)
                    h = self._calculate_hurst_rs(log_data)
                else:
                    # Calculate q-th order structure function
                    scales = range(2, min(15, len(data) // 6))
                    structure_funcs = []
                    
                    for scale in scales:
                        if scale >= len(data):
                            continue
                        
                        increments = data[scale:] - data[:-scale]
                        
                        if q > 0:
                            # Positive q
                            moment = np.mean(np.abs(increments)**q)
                            if moment > 0:
                                structure_funcs.append(moment**(1/q))
                        else:
                            # Negative q
                            abs_increments = np.abs(increments)
                            # Avoid division by zero
                            valid_increments = abs_increments[abs_increments > 1e-10]
                            if len(valid_increments) > 0:
                                moment = np.mean(valid_increments**q)
                                if moment > 0:
                                    structure_funcs.append(moment**(1/q))
                    
                    # Fit power law
                    if len(structure_funcs) >= 2:
                        valid_scales = scales[:len(structure_funcs)]
                        log_scales = np.log(valid_scales)
                        log_struct = np.log([s for s in structure_funcs if s > 0])
                        
                        if len(log_struct) >= 2:
                            h = np.polyfit(log_scales[:len(log_struct)], log_struct, 1)[0]
                        else:
                            h = 0.5
                    else:
                        h = 0.5
                
                if not np.isnan(h):
                    h_q_values.append(h)
            
            # Width of spectrum
            if len(h_q_values) >= 2:
                return np.max(h_q_values) - np.min(h_q_values)
            else:
                return np.nan
                
        except (ValueError, RuntimeWarning):
            return np.nan
    
    def _test_self_similarity(self, data: np.ndarray) -> float:
        """Test for self-similarity using scale comparison."""
        if len(data) < 30:
            return np.nan
        
        try:
            # Compare distributions at different scales
            scale1 = data[::2]  # Every 2nd point
            scale2 = data[::4]  # Every 4th point
            
            if len(scale1) >= 10 and len(scale2) >= 10:
                # KS test between scaled distributions
                ks_stat, p_value = stats.ks_2samp(scale1, scale2)
                return p_value  # High p-value indicates self-similarity
            else:
                return np.nan
                
        except (ValueError, RuntimeWarning):
            return np.nan
    
    def _estimate_fractal_dimension(self, data: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting method approximation."""
        if len(data) < 20:
            return np.nan
        
        try:
            # Simplified box-counting approach
            # Normalize data to [0, 1]
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
            
            # Different box sizes
            box_sizes = [2**i for i in range(1, int(np.log2(len(data)/4)) + 1)]
            counts = []
            
            for box_size in box_sizes:
                if box_size >= len(data):
                    continue
                
                # Count boxes needed to cover the curve
                n_boxes = 0
                for i in range(0, len(data), box_size):
                    segment = data_norm[i:i+box_size]
                    if len(segment) > 0:
                        y_range = np.max(segment) - np.min(segment)
                        # Number of boxes in y direction needed
                        y_boxes = max(1, int(np.ceil(y_range * box_size)))
                        n_boxes += y_boxes
                
                counts.append(n_boxes)
            
            # Fit power law: N(r) ~ r^(-D)
            if len(counts) >= 2:
                valid_data = [(1/bs, count) for bs, count in zip(box_sizes, counts) if count > 0]
                
                if len(valid_data) >= 2:
                    log_scales = [np.log(scale) for scale, count in valid_data]
                    log_counts = [np.log(count) for scale, count in valid_data]
                    
                    # Fractal dimension is negative slope
                    dimension = -np.polyfit(log_scales, log_counts, 1)[0]
                    return dimension
            
            return np.nan
            
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
            return np.nan
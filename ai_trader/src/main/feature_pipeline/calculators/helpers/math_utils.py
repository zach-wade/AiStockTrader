"""
Mathematical Utilities for Feature Calculators

Provides safe mathematical operations and statistical calculations
used across all feature calculators to ensure consistency and reliability.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any, List
from scipy import stats
from scipy.stats import entropy as scipy_entropy
import warnings

from main.utils.core import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# Core mathematical utilities have been moved to main.utils.math_utils to avoid circular dependencies
# Importing here for backward compatibility
from main.utils.math_utils import safe_divide, safe_log, safe_sqrt


def calculate_moving_average(
    series: pd.Series,
    window: int,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate simple moving average.
    
    Args:
        series: Input series
        window: Window size
        min_periods: Minimum periods required
        
    Returns:
        Moving average series
    """
    min_periods = min_periods or max(1, window // 2)
    return series.rolling(window=window, min_periods=min_periods).mean()


def calculate_exponential_average(
    series: pd.Series,
    span: int,
    adjust: bool = False
) -> pd.Series:
    """
    Calculate exponential moving average.
    
    Args:
        series: Input series
        span: Span for EMA
        adjust: Use adjusted EMA calculation
        
    Returns:
        EMA series
    """
    return series.ewm(span=span, adjust=adjust).mean()


def calculate_weighted_average(
    series: pd.Series,
    weights: Optional[np.ndarray] = None,
    window: Optional[int] = None
) -> Union[float, pd.Series]:
    """
    Calculate weighted average.
    
    Args:
        series: Input series
        weights: Weights array (None for linear weights)
        window: Window size for rolling calculation
        
    Returns:
        Weighted average value or series
    """
    if window is not None:
        def weighted_mean(x):
            if weights is None:
                w = np.arange(1, len(x) + 1)
            else:
                w = weights[:len(x)]
            return np.average(x, weights=w)
            
        return series.rolling(window=window).apply(weighted_mean, raw=True)
    else:
        if weights is None:
            weights = np.arange(1, len(series) + 1)
        return np.average(series, weights=weights)


def calculate_rolling_std(
    series: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1
) -> pd.Series:
    """
    Calculate rolling standard deviation.
    
    Args:
        series: Input series
        window: Window size
        min_periods: Minimum periods required
        ddof: Delta degrees of freedom
        
    Returns:
        Rolling standard deviation series
    """
    min_periods = min_periods or max(1, window // 2)
    return series.rolling(window=window, min_periods=min_periods).std(ddof=ddof)


def calculate_rolling_quantile(
    series: pd.Series,
    window: int,
    quantile: float,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate rolling quantile.
    
    Args:
        series: Input series
        window: Window size
        quantile: Quantile value (0-1)
        min_periods: Minimum periods required
        
    Returns:
        Rolling quantile series
    """
    min_periods = min_periods or max(1, window // 2)
    return series.rolling(window=window, min_periods=min_periods).quantile(quantile)


def remove_outliers(
    series: pd.Series,
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.Series:
    """
    Remove outliers from series.
    
    Args:
        series: Input series
        method: Method to use ('zscore', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        
    Returns:
        Series with outliers removed
    """
    try:
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            mask = z_scores < threshold
        elif method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            mask = (series >= Q1 - threshold * IQR) & (series <= Q3 + threshold * IQR)
        elif method == 'mad':
            median = series.median()
            mad = np.median(np.abs(series - median))
            mask = np.abs(series - median) <= threshold * mad
        else:
            return series
            
        return series[mask]
        
    except Exception as e:
        logger.warning(f"Error removing outliers: {e}")
        return series


def fit_distribution(
    data: pd.Series,
    distribution: str = 'normal'
) -> Dict[str, Any]:
    """
    Fit statistical distribution to data.
    
    Args:
        data: Input data
        distribution: Distribution type ('normal', 't', 'lognormal')
        
    Returns:
        Dictionary with distribution parameters and fit statistics
    """
    try:
        clean_data = data.dropna()
        
        if len(clean_data) < 10:
            return {'error': 'Insufficient data'}
            
        if distribution == 'normal':
            params = stats.norm.fit(clean_data)
            ks_stat, p_value = stats.kstest(clean_data, 'norm', args=params)
            
            return {
                'distribution': 'normal',
                'params': {'loc': params[0], 'scale': params[1]},
                'ks_statistic': ks_stat,
                'p_value': p_value
            }
            
        elif distribution == 't':
            params = stats.t.fit(clean_data)
            ks_stat, p_value = stats.kstest(clean_data, 't', args=params)
            
            return {
                'distribution': 't',
                'params': {'df': params[0], 'loc': params[1], 'scale': params[2]},
                'ks_statistic': ks_stat,
                'p_value': p_value
            }
            
        elif distribution == 'lognormal':
            # Ensure positive values
            positive_data = clean_data[clean_data > 0]
            if len(positive_data) < 10:
                return {'error': 'Insufficient positive data'}
                
            params = stats.lognorm.fit(positive_data)
            ks_stat, p_value = stats.kstest(positive_data, 'lognorm', args=params)
            
            return {
                'distribution': 'lognormal',
                'params': {'s': params[0], 'loc': params[1], 'scale': params[2]},
                'ks_statistic': ks_stat,
                'p_value': p_value
            }
            
        else:
            return {'error': f'Unknown distribution: {distribution}'}
            
    except Exception as e:
        logger.error(f"Error fitting distribution: {e}")
        return {'error': str(e)}


def calculate_correlation(
    series1: pd.Series,
    series2: pd.Series,
    method: str = 'pearson',
    min_periods: int = 20
) -> float:
    """
    Calculate correlation between two series.
    
    Args:
        series1: First series
        series2: Second series
        method: Correlation method ('pearson', 'spearman', 'kendall')
        min_periods: Minimum periods required
        
    Returns:
        Correlation coefficient
    """
    try:
        # Align series
        aligned = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        
        if len(aligned) < min_periods:
            return np.nan
            
        if method == 'pearson':
            return aligned['s1'].corr(aligned['s2'], method='pearson')
        elif method == 'spearman':
            return aligned['s1'].corr(aligned['s2'], method='spearman')
        elif method == 'kendall':
            return aligned['s1'].corr(aligned['s2'], method='kendall')
        else:
            return np.nan
            
    except Exception as e:
        logger.warning(f"Error calculating correlation: {e}")
        return np.nan


def calculate_covariance(
    series1: pd.Series,
    series2: pd.Series,
    min_periods: int = 20
) -> float:
    """
    Calculate covariance between two series.
    
    Args:
        series1: First series
        series2: Second series
        min_periods: Minimum periods required
        
    Returns:
        Covariance value
    """
    try:
        # Align series
        aligned = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        
        if len(aligned) < min_periods:
            return np.nan
            
        return aligned['s1'].cov(aligned['s2'])
        
    except Exception as e:
        logger.warning(f"Error calculating covariance: {e}")
        return np.nan


def normalize_series(
    series: pd.Series,
    method: str = 'minmax',
    feature_range: Tuple[float, float] = (0, 1)
) -> pd.Series:
    """
    Normalize series to specified range.
    
    Args:
        series: Input series
        method: Normalization method ('minmax', 'zscore')
        feature_range: Target range for minmax normalization
        
    Returns:
        Normalized series
    """
    try:
        if method == 'minmax':
            min_val = series.min()
            max_val = series.max()
            
            if max_val == min_val:
                return pd.Series(feature_range[0], index=series.index)
                
            normalized = (series - min_val) / (max_val - min_val)
            scaled = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
            return scaled
            
        elif method == 'zscore':
            mean = series.mean()
            std = series.std()
            
            if std == 0:
                return pd.Series(0, index=series.index)
                
            return (series - mean) / std
            
        else:
            return series
            
    except Exception as e:
        logger.warning(f"Error normalizing series: {e}")
        return series


def standardize_series(series: pd.Series) -> pd.Series:
    """
    Standardize series to zero mean and unit variance.
    
    Args:
        series: Input series
        
    Returns:
        Standardized series
    """
    return normalize_series(series, method='zscore')


def winsorize_series(
    series: pd.Series,
    limits: Tuple[float, float] = (0.05, 0.05)
) -> pd.Series:
    """
    Winsorize series by capping extreme values.
    
    Args:
        series: Input series
        limits: Lower and upper percentile limits
        
    Returns:
        Winsorized series
    """
    try:
        lower_percentile = series.quantile(limits[0])
        upper_percentile = series.quantile(1 - limits[1])
        
        return series.clip(lower=lower_percentile, upper=upper_percentile)
        
    except Exception as e:
        logger.warning(f"Error winsorizing series: {e}")
        return series


def calculate_entropy(
    series: pd.Series,
    bins: int = 10,
    method: str = 'shannon'
) -> float:
    """
    Calculate entropy of a series.
    
    Args:
        series: Input series
        bins: Number of bins for discretization
        method: Entropy method ('shannon', 'normalized')
        
    Returns:
        Entropy value
    """
    try:
        # Discretize the series
        discretized = pd.cut(series.dropna(), bins=bins, labels=False)
        
        # Calculate probability distribution
        value_counts = discretized.value_counts(normalize=True)
        probabilities = value_counts.values
        
        # Calculate entropy
        entropy_value = scipy_entropy(probabilities)
        
        if method == 'normalized':
            # Normalize by maximum possible entropy
            max_entropy = np.log(bins)
            entropy_value = entropy_value / max_entropy if max_entropy > 0 else 0
            
        return entropy_value
        
    except Exception as e:
        logger.warning(f"Error calculating entropy: {e}")
        return 0.0


def calculate_hurst_exponent(
    series: pd.Series,
    lags: Optional[List[int]] = None
) -> float:
    """
    Calculate Hurst exponent to measure long-term memory.
    
    Args:
        series: Input series
        lags: List of lag values to use
        
    Returns:
        Hurst exponent (0.5 = random walk, <0.5 = mean reverting, >0.5 = trending)
    """
    try:
        if lags is None:
            lags = range(2, min(100, len(series) // 2))
            
        # Calculate R/S statistic for each lag
        rs_values = []
        
        for lag in lags:
            # Divide series into non-overlapping segments
            segments = [series[i:i+lag] for i in range(0, len(series) - lag + 1, lag)]
            
            rs_segment_values = []
            for segment in segments:
                if len(segment) < lag:
                    continue
                    
                # Calculate mean-adjusted series
                mean_adj = segment - segment.mean()
                
                # Calculate cumulative sum
                cumsum = mean_adj.cumsum()
                
                # Calculate range
                R = cumsum.max() - cumsum.min()
                
                # Calculate standard deviation
                S = segment.std()
                
                if S > 0:
                    rs_segment_values.append(R / S)
                    
            if rs_segment_values:
                rs_values.append(np.mean(rs_segment_values))
                
        if len(rs_values) < 2:
            return 0.5  # Default to random walk
            
        # Fit log(R/S) = log(c) + H * log(lag)
        log_lags = np.log(list(lags)[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Linear regression
        slope, _ = np.polyfit(log_lags, log_rs, 1)
        
        return float(slope)
        
    except Exception as e:
        logger.warning(f"Error calculating Hurst exponent: {e}")
        return 0.5
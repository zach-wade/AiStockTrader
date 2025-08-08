"""
Data Validation Utilities for Feature Calculators

Provides comprehensive data validation and preprocessing functions
to ensure data quality and consistency across all feature calculators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta

from main.utils.core import get_logger, ensure_utc

logger = get_logger(__name__)


def validate_ohlcv_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 10
) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data for technical analysis.
    
    Args:
        data: DataFrame to validate
        required_columns: Required columns (default: OHLCV columns)
        min_rows: Minimum number of rows required
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Check if DataFrame is empty
    if data is None or data.empty:
        errors.append("Data is empty or None")
        return False, errors
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check minimum rows
    if len(data) < min_rows:
        errors.append(f"Insufficient data: {len(data)} rows < {min_rows} required")
    
    # Validate OHLC relationships if all price columns exist
    price_cols = ['open', 'high', 'low', 'close']
    if all(col in data.columns for col in price_cols):
        # Check for invalid OHLC relationships
        invalid_high = (data['high'] < data['low']).sum()
        if invalid_high > 0:
            errors.append(f"Found {invalid_high} rows where high < low")
        
        invalid_high_open = (data['high'] < data['open']).sum()
        if invalid_high_open > 0:
            errors.append(f"Found {invalid_high_open} rows where high < open")
        
        invalid_high_close = (data['high'] < data['close']).sum()
        if invalid_high_close > 0:
            errors.append(f"Found {invalid_high_close} rows where high < close")
        
        invalid_low_open = (data['low'] > data['open']).sum()
        if invalid_low_open > 0:
            errors.append(f"Found {invalid_low_open} rows where low > open")
        
        invalid_low_close = (data['low'] > data['close']).sum()
        if invalid_low_close > 0:
            errors.append(f"Found {invalid_low_close} rows where low > close")
        
        # Check for non-positive prices
        for col in price_cols:
            if col in data.columns:
                non_positive = (data[col] <= 0).sum()
                if non_positive > 0:
                    errors.append(f"Found {non_positive} non-positive values in {col}")
    
    # Check volume
    if 'volume' in data.columns:
        negative_volume = (data['volume'] < 0).sum()
        if negative_volume > 0:
            errors.append(f"Found {negative_volume} negative volume values")
    
    # Check for excessive missing values
    for col in required_columns:
        if col in data.columns:
            missing_pct = data[col].isna().sum() / len(data) * 100
            if missing_pct > 10:
                errors.append(f"Column {col} has {missing_pct:.1f}% missing values")
    
    return len(errors) == 0, errors


def validate_price_data(
    data: pd.DataFrame,
    price_column: str = 'close',
    min_rows: int = 10
) -> Tuple[bool, List[str]]:
    """
    Validate price data for calculations.
    
    Args:
        data: DataFrame to validate
        price_column: Name of price column
        min_rows: Minimum number of rows required
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Basic checks
    if data is None or data.empty:
        errors.append("Data is empty or None")
        return False, errors
    
    if price_column not in data.columns:
        errors.append(f"Price column '{price_column}' not found")
        return False, errors
    
    if len(data) < min_rows:
        errors.append(f"Insufficient data: {len(data)} rows < {min_rows} required")
    
    # Check price data quality
    prices = data[price_column]
    
    # Non-positive prices
    non_positive = (prices <= 0).sum()
    if non_positive > 0:
        errors.append(f"Found {non_positive} non-positive price values")
    
    # Missing values
    missing = prices.isna().sum()
    if missing > 0:
        missing_pct = missing / len(prices) * 100
        errors.append(f"Found {missing} ({missing_pct:.1f}%) missing price values")
    
    # Check for extreme price changes (potential data errors)
    if len(prices) > 1:
        returns = prices.pct_change().dropna()
        extreme_returns = (returns.abs() > 0.5).sum()  # >50% change
        if extreme_returns > 0:
            errors.append(f"Found {extreme_returns} extreme price changes (>50%)")
    
    return len(errors) == 0, errors


def validate_volume_data(
    data: pd.DataFrame,
    volume_column: str = 'volume',
    min_rows: int = 10
) -> Tuple[bool, List[str]]:
    """
    Validate volume data.
    
    Args:
        data: DataFrame to validate
        volume_column: Name of volume column
        min_rows: Minimum number of rows required
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Basic checks
    if data is None or data.empty:
        errors.append("Data is empty or None")
        return False, errors
    
    if volume_column not in data.columns:
        errors.append(f"Volume column '{volume_column}' not found")
        return False, errors
    
    if len(data) < min_rows:
        errors.append(f"Insufficient data: {len(data)} rows < {min_rows} required")
    
    # Check volume data quality
    volume = data[volume_column]
    
    # Negative volume
    negative = (volume < 0).sum()
    if negative > 0:
        errors.append(f"Found {negative} negative volume values")
    
    # Missing values
    missing = volume.isna().sum()
    if missing > 0:
        missing_pct = missing / len(volume) * 100
        errors.append(f"Found {missing} ({missing_pct:.1f}%) missing volume values")
    
    # Check if all volumes are zero
    if (volume == 0).all():
        errors.append("All volume values are zero")
    
    return len(errors) == 0, errors


def validate_news_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> Tuple[bool, List[str]]:
    """
    Validate news data for sentiment analysis.
    
    Args:
        data: DataFrame to validate
        required_columns: Required columns
        min_rows: Minimum number of rows required
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if required_columns is None:
        required_columns = ['title', 'published_at']
    
    # Basic checks
    if data is None or data.empty:
        errors.append("Data is empty or None")
        return False, errors
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    if len(data) < min_rows:
        errors.append(f"Insufficient data: {len(data)} rows < {min_rows} required")
    
    # Check text columns
    text_columns = ['title', 'content', 'summary']
    for col in text_columns:
        if col in data.columns:
            # Check for empty strings
            empty_count = (data[col].fillna('') == '').sum()
            if empty_count > len(data) * 0.5:
                errors.append(f"Column {col} has {empty_count} empty values")
    
    # Check timestamp column
    if 'published_at' in data.columns:
        try:
            # Try to convert to datetime
            pd.to_datetime(data['published_at'])
        except Exception as e:
            errors.append(f"Invalid datetime format in published_at: {str(e)}")
    
    return len(errors) == 0, errors


def validate_options_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 10
) -> Tuple[bool, List[str]]:
    """
    Validate options data.
    
    Args:
        data: DataFrame to validate
        required_columns: Required columns
        min_rows: Minimum number of rows required
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if required_columns is None:
        required_columns = ['strike', 'expiration', 'bid', 'ask', 'volume', 'open_interest']
    
    # Basic checks
    if data is None or data.empty:
        errors.append("Data is empty or None")
        return False, errors
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    if len(data) < min_rows:
        errors.append(f"Insufficient data: {len(data)} rows < {min_rows} required")
    
    # Validate strike prices
    if 'strike' in data.columns:
        invalid_strikes = (data['strike'] <= 0).sum()
        if invalid_strikes > 0:
            errors.append(f"Found {invalid_strikes} non-positive strike prices")
    
    # Validate bid-ask spread
    if 'bid' in data.columns and 'ask' in data.columns:
        invalid_spread = (data['bid'] > data['ask']).sum()
        if invalid_spread > 0:
            errors.append(f"Found {invalid_spread} rows where bid > ask")
    
    # Validate volume and open interest
    for col in ['volume', 'open_interest']:
        if col in data.columns:
            negative = (data[col] < 0).sum()
            if negative > 0:
                errors.append(f"Found {negative} negative values in {col}")
    
    return len(errors) == 0, errors


def preprocess_data(
    data: pd.DataFrame,
    handle_missing: str = 'forward_fill',
    remove_outliers: bool = False,
    outlier_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Preprocess data with common cleaning operations.
    
    Args:
        data: Input DataFrame
        handle_missing: Method to handle missing values
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outliers
        
    Returns:
        Preprocessed DataFrame
    """
    if data is None or data.empty:
        return data
    
    processed = data.copy()
    
    # Handle missing values
    processed = handle_missing_values(processed, method=handle_missing)
    
    # Remove outliers if requested
    if remove_outliers:
        numeric_columns = processed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in processed.columns:
                z_scores = np.abs((processed[col] - processed[col].mean()) / processed[col].std())
                processed = processed[z_scores < outlier_threshold]
    
    return processed


def handle_missing_values(
    data: pd.DataFrame,
    method: str = 'forward_fill',
    numeric_fill: Optional[float] = None,
    categorical_fill: Optional[str] = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        data: Input DataFrame
        method: Method to handle missing values
        numeric_fill: Value to fill numeric columns
        categorical_fill: Value to fill categorical columns
        
    Returns:
        DataFrame with missing values handled
    """
    if data is None or data.empty:
        return data
    
    processed = data.copy()
    
    if method == 'forward_fill':
        processed = processed.ffill()
    elif method == 'backward_fill':
        processed = processed.bfill()
    elif method == 'interpolate':
        numeric_columns = processed.select_dtypes(include=[np.number]).columns
        processed[numeric_columns] = processed[numeric_columns].interpolate(method='linear')
    elif method == 'drop':
        processed = processed.dropna()
    elif method == 'fill':
        # Fill with specified values
        if numeric_fill is not None:
            numeric_columns = processed.select_dtypes(include=[np.number]).columns
            processed[numeric_columns] = processed[numeric_columns].fillna(numeric_fill)
        if categorical_fill is not None:
            categorical_columns = processed.select_dtypes(exclude=[np.number]).columns
            processed[categorical_columns] = processed[categorical_columns].fillna(categorical_fill)
    
    return processed


def check_data_quality(
    data: pd.DataFrame,
    max_missing_pct: float = 0.1,
    max_duplicate_pct: float = 0.05
) -> Dict[str, Any]:
    """
    Comprehensive data quality check.
    
    Args:
        data: DataFrame to check
        max_missing_pct: Maximum acceptable missing percentage
        max_duplicate_pct: Maximum acceptable duplicate percentage
        
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'column_types': {},
        'missing_values': {},
        'duplicates': {},
        'numeric_stats': {},
        'warnings': []
    }
    
    # Column types
    quality_report['column_types'] = data.dtypes.astype(str).to_dict()
    
    # Missing values
    for col in data.columns:
        missing_count = data[col].isna().sum()
        missing_pct = missing_count / len(data)
        quality_report['missing_values'][col] = {
            'count': missing_count,
            'percentage': missing_pct
        }
        
        if missing_pct > max_missing_pct:
            quality_report['warnings'].append(
                f"Column {col} has {missing_pct:.1%} missing values"
            )
    
    # Duplicates
    duplicate_count = data.duplicated().sum()
    duplicate_pct = duplicate_count / len(data)
    quality_report['duplicates'] = {
        'count': duplicate_count,
        'percentage': duplicate_pct
    }
    
    if duplicate_pct > max_duplicate_pct:
        quality_report['warnings'].append(
            f"Found {duplicate_pct:.1%} duplicate rows"
        )
    
    # Numeric column statistics
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            quality_report['numeric_stats'][col] = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'zeros': (col_data == 0).sum(),
                'negative': (col_data < 0).sum()
            }
    
    return quality_report


def ensure_datetime_index(
    data: pd.DataFrame,
    datetime_column: Optional[str] = None,
    freq: Optional[str] = None
) -> pd.DataFrame:
    """
    Ensure DataFrame has a datetime index.
    
    Args:
        data: Input DataFrame
        datetime_column: Column to use as datetime index
        freq: Frequency to resample to
        
    Returns:
        DataFrame with datetime index
    """
    if data is None or data.empty:
        return data
    
    processed = data.copy()
    
    # If datetime_column is specified, use it
    if datetime_column and datetime_column in processed.columns:
        processed[datetime_column] = pd.to_datetime(processed[datetime_column])
        processed = processed.set_index(datetime_column)
    
    # Ensure index is datetime
    if not isinstance(processed.index, pd.DatetimeIndex):
        # Try to convert existing index
        try:
            processed.index = pd.to_datetime(processed.index)
        except Exception:
            # If conversion fails, look for a timestamp column
            timestamp_columns = ['timestamp', 'date', 'datetime', 'time']
            for col in timestamp_columns:
                if col in processed.columns:
                    processed[col] = pd.to_datetime(processed[col])
                    processed = processed.set_index(col)
                    break
    
    # Apply UTC timezone
    if isinstance(processed.index, pd.DatetimeIndex) and processed.index.tz is None:
        processed.index = processed.index.tz_localize('UTC')
    
    # Resample if frequency is specified
    if freq and isinstance(processed.index, pd.DatetimeIndex):
        processed = processed.resample(freq).last()
    
    return processed


def align_time_series(
    *dataframes: pd.DataFrame,
    join: str = 'inner',
    fill_method: Optional[str] = None
) -> List[pd.DataFrame]:
    """
    Align multiple time series DataFrames.
    
    Args:
        dataframes: Variable number of DataFrames to align
        join: Join method ('inner', 'outer', 'left', 'right')
        fill_method: Method to fill missing values after alignment
        
    Returns:
        List of aligned DataFrames
    """
    if not dataframes:
        return []
    
    # Find common index
    if join == 'inner':
        common_index = dataframes[0].index
        for df in dataframes[1:]:
            common_index = common_index.intersection(df.index)
    elif join == 'outer':
        common_index = dataframes[0].index
        for df in dataframes[1:]:
            common_index = common_index.union(df.index)
    else:
        common_index = dataframes[0].index
    
    # Align all dataframes
    aligned = []
    for df in dataframes:
        aligned_df = df.reindex(common_index)
        
        # Fill missing values if specified
        if fill_method:
            aligned_df = handle_missing_values(aligned_df, method=fill_method)
        
        aligned.append(aligned_df)
    
    return aligned


def resample_data(
    data: pd.DataFrame,
    target_freq: str,
    aggregation: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Resample time series data to target frequency.
    
    Args:
        data: Input DataFrame with datetime index
        target_freq: Target frequency (e.g., '1D', '1H', '5T')
        aggregation: Column-specific aggregation methods
        
    Returns:
        Resampled DataFrame
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Default aggregation methods
    if aggregation is None:
        aggregation = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    # Build aggregation dict for existing columns
    agg_dict = {}
    for col in data.columns:
        if col in aggregation:
            agg_dict[col] = aggregation[col]
        elif data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'last'
    
    # Resample
    resampled = data.resample(target_freq).agg(agg_dict)
    
    return resampled
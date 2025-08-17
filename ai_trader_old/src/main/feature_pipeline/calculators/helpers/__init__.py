"""
Feature Calculator Helpers Module

Provides shared utilities for all feature calculators to promote DRY principles
and code reuse across the feature pipeline.

Components:
- math_utils: Statistical calculations and safe numeric operations
- validation: Data validation and preprocessing utilities
- feature_utils: Feature naming, postprocessing, and aggregation
- time_utils: Market-aware time calculations and windowing
"""

# Math utilities (core functions now imported from shared utils)
from .math_utils import (
    calculate_correlation,
    calculate_covariance,
    calculate_entropy,
    calculate_exponential_average,
    calculate_hurst_exponent,
    calculate_moving_average,
    calculate_rolling_quantile,
    calculate_rolling_std,
    calculate_weighted_average,
    fit_distribution,
    normalize_series,
    remove_outliers,
    safe_divide,
    safe_log,
    safe_sqrt,
    standardize_series,
    winsorize_series,
)

# Create alias for backward compatibility
calculate_rolling_mean = calculate_moving_average

# Validation utilities
# Feature utilities
from .feature_utils import (
    aggregate_features,
    apply_feature_engineering,
    clip_features,
    combine_features,
    create_feature_dataframe,
    create_interaction_features,
    create_lag_features,
    create_rolling_features,
    generate_feature_name,
    normalize_features,
    postprocess_features,
)

# Time utilities
from .time_utils import (
    align_to_market_time,
    calculate_business_days,
    calculate_time_decay,
    calculate_time_weighted_average,
    create_temporal_features,
    create_time_windows,
    get_market_sessions,
    get_time_of_day_weights,
    get_trading_days,
    is_market_hours,
)
from .validation import (
    align_time_series,
    check_data_quality,
    ensure_datetime_index,
    handle_missing_values,
    preprocess_data,
    resample_data,
    validate_news_data,
    validate_ohlcv_data,
    validate_options_data,
    validate_price_data,
    validate_volume_data,
)

__all__ = [
    # Math utilities
    "safe_divide",
    "safe_log",
    "safe_sqrt",
    "calculate_moving_average",
    "calculate_rolling_mean",  # Alias for calculate_moving_average
    "calculate_exponential_average",
    "calculate_weighted_average",
    "calculate_rolling_std",
    "calculate_rolling_quantile",
    "remove_outliers",
    "fit_distribution",
    "calculate_correlation",
    "calculate_covariance",
    "normalize_series",
    "standardize_series",
    "winsorize_series",
    "calculate_entropy",
    "calculate_hurst_exponent",
    # Validation utilities
    "validate_ohlcv_data",
    "validate_price_data",
    "validate_volume_data",
    "validate_news_data",
    "validate_options_data",
    "preprocess_data",
    "handle_missing_values",
    "check_data_quality",
    "ensure_datetime_index",
    "align_time_series",
    "resample_data",
    # Feature utilities
    "generate_feature_name",
    "create_feature_dataframe",
    "postprocess_features",
    "clip_features",
    "normalize_features",
    "aggregate_features",
    "combine_features",
    "create_interaction_features",
    "create_lag_features",
    "create_rolling_features",
    "apply_feature_engineering",
    # Time utilities
    "get_market_sessions",
    "align_to_market_time",
    "calculate_time_decay",
    "create_time_windows",
    "get_trading_days",
    "calculate_business_days",
    "is_market_hours",
    "get_time_of_day_weights",
    "calculate_time_weighted_average",
    "create_temporal_features",
]

"""
Feature Engineering Utilities for Feature Calculators

Provides utilities for feature naming, transformation, aggregation,
and post-processing to ensure consistency across all calculators.
"""

# Standard library imports
from collections.abc import Callable
from itertools import combinations
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import get_logger

logger = get_logger(__name__)


def generate_feature_name(
    base_name: str,
    symbol: str | None = None,
    suffix: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Generate standardized feature names.

    Args:
        base_name: Base feature name
        symbol: Optional symbol to include
        suffix: Optional suffix
        params: Optional parameters to include in name

    Returns:
        Standardized feature name
    """
    parts = []

    # Add symbol if provided
    if symbol:
        parts.append(symbol)

    # Add base name
    parts.append(base_name)

    # Add parameters if provided
    if params:
        for key, value in sorted(params.items()):
            if isinstance(value, float):
                parts.append(f"{key}_{value:.2f}")
            else:
                parts.append(f"{key}_{value}")

    # Add suffix if provided
    if suffix:
        parts.append(suffix)

    return "_".join(parts).lower()


def create_feature_dataframe(
    index: pd.Index, features: dict[str, pd.Series] | None = None, fill_value: float = np.nan
) -> pd.DataFrame:
    """
    Create a feature DataFrame with proper index.

    Args:
        index: Index for the DataFrame
        features: Dictionary of feature name to Series
        fill_value: Value to fill missing data

    Returns:
        Feature DataFrame
    """
    if features:
        # Create DataFrame from features dict
        df = pd.DataFrame(features, index=index)

        # Fill any missing values
        if not np.isnan(fill_value):
            df = df.fillna(fill_value)
    else:
        # Create empty DataFrame
        df = pd.DataFrame(index=index)

    return df


def postprocess_features(
    features: pd.DataFrame,
    clip_quantiles: tuple[float, float] | None = None,
    normalize: bool = False,
    standardize: bool = False,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Post-process features with common transformations.

    Args:
        features: Input features DataFrame
        clip_quantiles: Quantiles for clipping (e.g., (0.01, 0.99))
        normalize: Whether to normalize features to [0, 1]
        standardize: Whether to standardize features (z-score)
        fill_value: Value to fill remaining NaN

    Returns:
        Post-processed features
    """
    if features.empty:
        return features

    processed = features.copy()

    # Handle infinite values
    processed = processed.replace([np.inf, -np.inf], np.nan)

    # Clip outliers if specified
    if clip_quantiles:
        processed = clip_features(processed, clip_quantiles[0], clip_quantiles[1])

    # Normalize or standardize
    if normalize:
        processed = normalize_features(processed)
    elif standardize:
        processed = standardize_features(processed)

    # Fill remaining NaN values
    processed = processed.fillna(value=fill_value)

    return processed


def clip_features(
    features: pd.DataFrame, lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> pd.DataFrame:
    """
    Clip features to specified quantiles.

    Args:
        features: Input features
        lower_quantile: Lower quantile for clipping
        upper_quantile: Upper quantile for clipping

    Returns:
        Clipped features
    """
    clipped = features.copy()

    for col in clipped.columns:
        if clipped[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            lower = clipped[col].quantile(lower_quantile)
            upper = clipped[col].quantile(upper_quantile)
            clipped[col] = clipped[col].clip(lower=lower, upper=upper)

    return clipped


def normalize_features(
    features: pd.DataFrame, feature_range: tuple[float, float] = (0, 1)
) -> pd.DataFrame:
    """
    Normalize features to specified range.

    Args:
        features: Input features
        feature_range: Target range for normalization

    Returns:
        Normalized features
    """
    normalized = features.copy()

    for col in normalized.columns:
        if normalized[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            min_val = normalized[col].min()
            max_val = normalized[col].max()

            if max_val > min_val:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
                normalized[col] = (
                    normalized[col] * (feature_range[1] - feature_range[0]) + feature_range[0]
                )
            else:
                normalized[col] = feature_range[0]

    return normalized


def standardize_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize features to zero mean and unit variance.

    Args:
        features: Input features

    Returns:
        Standardized features
    """
    standardized = features.copy()

    for col in standardized.columns:
        if standardized[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            mean = standardized[col].mean()
            std = standardized[col].std()

            if std > 0:
                standardized[col] = (standardized[col] - mean) / std
            else:
                standardized[col] = 0

    return standardized


def aggregate_features(
    features: pd.DataFrame,
    aggregations: dict[str, str | list[str] | Callable],
    group_by: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate features with specified methods.

    Args:
        features: Input features
        aggregations: Dictionary of column to aggregation method(s)
        group_by: Optional grouping column(s)

    Returns:
        Aggregated features
    """
    try:
        if group_by:
            # Group and aggregate
            grouped = features.groupby(group_by)
            aggregated = grouped.agg(aggregations)

            # Flatten column names if multi-level
            if isinstance(aggregated.columns, pd.MultiIndex):
                aggregated.columns = ["_".join(col).strip() for col in aggregated.columns.values]
        else:
            # Simple aggregation
            aggregated = features.agg(aggregations)

            if isinstance(aggregated, pd.Series):
                aggregated = aggregated.to_frame().T

        return aggregated

    except Exception as e:
        logger.error(f"Error aggregating features: {e}")
        return pd.DataFrame()


def combine_features(
    *feature_dfs: pd.DataFrame, join: str = "outer", suffix_format: str = "_{}"
) -> pd.DataFrame:
    """
    Combine multiple feature DataFrames.

    Args:
        feature_dfs: Variable number of feature DataFrames
        join: Join method ('inner', 'outer')
        suffix_format: Format for suffix when columns overlap

    Returns:
        Combined feature DataFrame
    """
    if not feature_dfs:
        return pd.DataFrame()

    if len(feature_dfs) == 1:
        return feature_dfs[0]

    # Start with first DataFrame
    combined = feature_dfs[0].copy()

    # Join remaining DataFrames
    for i, df in enumerate(feature_dfs[1:], 1):
        # Add suffix to overlapping columns
        overlapping = set(combined.columns) & set(df.columns)
        if overlapping:
            df = df.copy()
            for col in overlapping:
                df.rename(columns={col: col + suffix_format.format(i)}, inplace=True)

        # Join DataFrames
        combined = combined.join(df, how=join)

    return combined


def create_interaction_features(
    features: pd.DataFrame,
    columns: list[str] | None = None,
    operations: list[str] = ["multiply", "divide", "add", "subtract"],
    max_interactions: int = 10,
) -> pd.DataFrame:
    """
    Create interaction features between columns.

    Args:
        features: Input features
        columns: Columns to create interactions for (None = all numeric)
        operations: List of operations to apply
        max_interactions: Maximum number of interactions to create

    Returns:
        DataFrame with interaction features
    """
    if features.empty:
        return features

    # Select numeric columns if not specified
    if columns is None:
        columns = features.select_dtypes(include=[np.number]).columns.tolist()

    # Limit columns if too many
    if len(columns) > 10:
        # Select most variable columns
        variances = features[columns].var()
        columns = variances.nlargest(10).index.tolist()

    interaction_features = features.copy()
    interaction_count = 0

    # Create interactions
    for col1, col2 in combinations(columns, 2):
        if interaction_count >= max_interactions:
            break

        if "multiply" in operations:
            interaction_features[f"{col1}_x_{col2}"] = features[col1] * features[col2]
            interaction_count += 1

        if "divide" in operations and interaction_count < max_interactions:
            # Safe division
            denominator = features[col2].replace(0, np.nan)
            interaction_features[f"{col1}_div_{col2}"] = features[col1] / denominator
            interaction_count += 1

        if "add" in operations and interaction_count < max_interactions:
            interaction_features[f"{col1}_plus_{col2}"] = features[col1] + features[col2]
            interaction_count += 1

        if "subtract" in operations and interaction_count < max_interactions:
            interaction_features[f"{col1}_minus_{col2}"] = features[col1] - features[col2]
            interaction_count += 1

    return interaction_features


def create_lag_features(
    features: pd.DataFrame,
    columns: list[str] | None = None,
    lags: list[int] = [1, 2, 3, 5, 10],
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Create lagged features.

    Args:
        features: Input features
        columns: Columns to lag (None = all)
        lags: List of lag periods
        group_by: Optional grouping column (e.g., 'symbol')

    Returns:
        DataFrame with lagged features
    """
    if features.empty:
        return features

    # Select columns to lag
    if columns is None:
        columns = features.columns.tolist()
        if group_by and group_by in columns:
            columns.remove(group_by)

    lagged_features = features.copy()

    # Create lags
    for lag in lags:
        for col in columns:
            if group_by and group_by in features.columns:
                # Group-wise lagging
                lagged_features[f"{col}_lag_{lag}"] = features.groupby(group_by)[col].shift(lag)
            else:
                # Simple lagging
                lagged_features[f"{col}_lag_{lag}"] = features[col].shift(lag)

    return lagged_features


def create_rolling_features(
    features: pd.DataFrame,
    columns: list[str] | None = None,
    windows: list[int] = [5, 10, 20],
    operations: list[str] = ["mean", "std", "min", "max"],
    group_by: str | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Create rolling window features.

    Args:
        features: Input features
        columns: Columns to calculate rolling features for
        windows: List of window sizes
        operations: List of operations to apply
        group_by: Optional grouping column
        min_periods: Minimum periods required

    Returns:
        DataFrame with rolling features
    """
    if features.empty:
        return features

    # Select columns
    if columns is None:
        columns = features.select_dtypes(include=[np.number]).columns.tolist()
        if group_by and group_by in columns:
            columns.remove(group_by)

    rolling_features = features.copy()

    # Create rolling features
    for window in windows:
        min_per = min_periods or max(1, window // 2)

        for col in columns:
            if group_by and group_by in features.columns:
                # Group-wise rolling
                grouped = features.groupby(group_by)[col]

                if "mean" in operations:
                    rolling_features[f"{col}_rolling_mean_{window}"] = (
                        grouped.rolling(window=window, min_periods=min_per)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )

                if "std" in operations:
                    rolling_features[f"{col}_rolling_std_{window}"] = (
                        grouped.rolling(window=window, min_periods=min_per)
                        .std()
                        .reset_index(level=0, drop=True)
                    )

                if "min" in operations:
                    rolling_features[f"{col}_rolling_min_{window}"] = (
                        grouped.rolling(window=window, min_periods=min_per)
                        .min()
                        .reset_index(level=0, drop=True)
                    )

                if "max" in operations:
                    rolling_features[f"{col}_rolling_max_{window}"] = (
                        grouped.rolling(window=window, min_periods=min_per)
                        .max()
                        .reset_index(level=0, drop=True)
                    )
            else:
                # Simple rolling
                if "mean" in operations:
                    rolling_features[f"{col}_rolling_mean_{window}"] = (
                        features[col].rolling(window=window, min_periods=min_per).mean()
                    )

                if "std" in operations:
                    rolling_features[f"{col}_rolling_std_{window}"] = (
                        features[col].rolling(window=window, min_periods=min_per).std()
                    )

                if "min" in operations:
                    rolling_features[f"{col}_rolling_min_{window}"] = (
                        features[col].rolling(window=window, min_periods=min_per).min()
                    )

                if "max" in operations:
                    rolling_features[f"{col}_rolling_max_{window}"] = (
                        features[col].rolling(window=window, min_periods=min_per).max()
                    )

    return rolling_features


def apply_feature_engineering(
    features: pd.DataFrame, engineering_config: dict[str, Any]
) -> pd.DataFrame:
    """
    Apply comprehensive feature engineering based on configuration.

    Args:
        features: Input features
        engineering_config: Configuration for feature engineering

    Returns:
        Engineered features
    """
    engineered = features.copy()

    # Apply lag features
    if engineering_config.get("create_lags", False):
        lag_config = engineering_config.get("lag_config", {})
        engineered = create_lag_features(
            engineered,
            columns=lag_config.get("columns"),
            lags=lag_config.get("lags", [1, 2, 3, 5, 10]),
            group_by=lag_config.get("group_by"),
        )

    # Apply rolling features
    if engineering_config.get("create_rolling", False):
        rolling_config = engineering_config.get("rolling_config", {})
        engineered = create_rolling_features(
            engineered,
            columns=rolling_config.get("columns"),
            windows=rolling_config.get("windows", [5, 10, 20]),
            operations=rolling_config.get("operations", ["mean", "std"]),
            group_by=rolling_config.get("group_by"),
        )

    # Apply interaction features
    if engineering_config.get("create_interactions", False):
        interaction_config = engineering_config.get("interaction_config", {})
        engineered = create_interaction_features(
            engineered,
            columns=interaction_config.get("columns"),
            operations=interaction_config.get("operations", ["multiply"]),
            max_interactions=interaction_config.get("max_interactions", 10),
        )

    # Apply post-processing
    if engineering_config.get("postprocess", False):
        postprocess_config = engineering_config.get("postprocess_config", {})
        engineered = postprocess_features(
            engineered,
            clip_quantiles=postprocess_config.get("clip_quantiles"),
            normalize=postprocess_config.get("normalize", False),
            standardize=postprocess_config.get("standardize", False),
            fill_value=postprocess_config.get("fill_value", 0.0),
        )

    return engineered

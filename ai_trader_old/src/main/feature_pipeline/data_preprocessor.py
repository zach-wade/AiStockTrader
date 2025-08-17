"""
Data Preprocessor

Comprehensive data cleaning, normalization, and preprocessing utilities for the feature pipeline.
Handles missing data, outliers, scaling, and data quality validation across different data types.
"""

# Standard library imports
from dataclasses import dataclass, field
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorConfig:
    """Configuration for data preprocessing operations."""

    # Missing data handling
    missing_threshold: float = 0.5  # Drop columns with >50% missing data
    imputation_method: str = (
        "forward_fill"  # 'forward_fill', 'backward_fill', 'mean', 'median', 'knn'
    )
    knn_neighbors: int = 5  # For KNN imputation

    # Outlier detection and handling
    outlier_method: str = "iqr"  # 'iqr', 'zscore', 'isolation_forest', 'none'
    outlier_threshold: float = 3.0  # Z-score threshold or IQR multiplier
    outlier_action: str = "clip"  # 'clip', 'remove', 'flag'

    # Scaling and normalization
    scaling_method: str = "robust"  # 'standard', 'minmax', 'robust', 'none'
    scale_features: bool = True

    # Data validation
    min_data_points: int = 50  # Minimum data points required
    max_gap_days: int = 7  # Maximum allowed gap in time series

    # Feature-specific settings
    price_columns: list[str] = field(default_factory=lambda: ["open", "high", "low", "close"])
    volume_columns: list[str] = field(default_factory=lambda: ["volume"])
    return_columns: list[str] = field(default_factory=lambda: ["returns"])

    # Quality thresholds
    min_volume_threshold: float = 1000  # Minimum daily volume
    price_change_threshold: float = 0.5  # Maximum single-day price change (50%)


class DataPreprocessor:
    """
    Comprehensive data preprocessing for financial time series data.

    Handles data cleaning, normalization, outlier detection, and quality validation
    for market data, alternative data, and derived features.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize data preprocessor.

        Args:
            config: Configuration dictionary for preprocessing parameters
        """
        self.config = PreprocessorConfig(**config) if config else PreprocessorConfig()
        self.scalers = {}  # Store fitted scalers for each feature type
        self.imputers = {}  # Store fitted imputers
        self.outlier_stats = {}  # Store outlier detection parameters

        logger.info(f"DataPreprocessor initialized with config: {self.config}")

    def preprocess_market_data(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Preprocess market data (OHLCV) with financial data specific cleaning.

        Args:
            data: DataFrame with OHLCV data
            symbol: Optional symbol for logging

        Returns:
            Cleaned and preprocessed DataFrame
        """
        logger.debug(f"Preprocessing market data for {symbol or 'unknown symbol'}")

        if data.empty:
            logger.warning("Empty data provided for market data preprocessing")
            return data

        # Make copy to avoid modifying original
        processed_data = data.copy()

        # Basic data validation
        processed_data = self._validate_market_data_structure(processed_data)

        # Handle missing data
        processed_data = self._handle_missing_data(processed_data, data_type="market")

        # Clean price data
        processed_data = self._clean_price_data(processed_data)

        # Clean volume data
        processed_data = self._clean_volume_data(processed_data)

        # Detect and handle outliers
        processed_data = self._handle_outliers(processed_data, data_type="market")

        # Validate data quality
        processed_data = self._validate_data_quality(processed_data, symbol)

        # Generate basic derived features
        processed_data = self._add_basic_features(processed_data)

        logger.debug(f"Market data preprocessing complete. Shape: {processed_data.shape}")
        return processed_data

    def preprocess_feature_data(
        self, data: pd.DataFrame, feature_type: str = "general"
    ) -> pd.DataFrame:
        """
        Preprocess calculated features with scaling and normalization.

        Args:
            data: DataFrame with calculated features
            feature_type: Type of features for specialized processing

        Returns:
            Scaled and normalized feature DataFrame
        """
        logger.debug(f"Preprocessing {feature_type} features")

        if data.empty:
            return data

        processed_data = data.copy()

        # Handle missing data in features
        processed_data = self._handle_missing_data(processed_data, data_type="features")

        # Handle infinite values
        processed_data = self._handle_infinite_values(processed_data)

        # Detect and handle outliers
        processed_data = self._handle_outliers(processed_data, data_type="features")

        # Apply scaling if requested
        if self.config.scale_features:
            processed_data = self._scale_features(processed_data, feature_type)

        logger.debug(f"Feature preprocessing complete. Shape: {processed_data.shape}")
        return processed_data

    def preprocess_alternative_data(self, data: pd.DataFrame, data_source: str) -> pd.DataFrame:
        """
        Preprocess alternative data (news, social media, etc.).

        Args:
            data: DataFrame with alternative data
            data_source: Source type ('news', 'social', 'economic', etc.)

        Returns:
            Cleaned alternative data DataFrame
        """
        logger.debug(f"Preprocessing {data_source} alternative data")

        if data.empty:
            return data

        processed_data = data.copy()

        # Source-specific preprocessing
        if data_source == "news":
            processed_data = self._preprocess_news_data(processed_data)
        elif data_source == "social":
            processed_data = self._preprocess_social_data(processed_data)
        elif data_source == "economic":
            processed_data = self._preprocess_economic_data(processed_data)
        else:
            # Generic alternative data processing
            processed_data = self._handle_missing_data(processed_data, data_type="alternative")
            processed_data = self._handle_outliers(processed_data, data_type="alternative")

        return processed_data

    def _validate_market_data_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix basic market data structure."""
        # Ensure required columns exist
        required_columns = ["timestamp"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Sort by timestamp
        data = data.sort_values("timestamp").reset_index(drop=True)

        # Remove duplicate timestamps
        if data["timestamp"].duplicated().any():
            logger.warning("Duplicate timestamps found, keeping last occurrence")
            data = data.drop_duplicates(subset=["timestamp"], keep="last")

        return data

    def _handle_missing_data(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Handle missing data based on configuration and data type."""
        # Check missing data percentage
        missing_pct = data.isnull().sum() / len(data)
        columns_to_drop = missing_pct[missing_pct > self.config.missing_threshold].index.tolist()

        if columns_to_drop:
            logger.warning(
                f"Dropping columns with >{self.config.missing_threshold*100}% missing data: {columns_to_drop}"
            )
            data = data.drop(columns=columns_to_drop)

        # Apply imputation strategy
        if self.config.imputation_method == "forward_fill":
            data = data.fillna(method="ffill")
            data = data.fillna(method="bfill")  # Backfill remaining
        elif self.config.imputation_method == "backward_fill":
            data = data.fillna(method="bfill")
            data = data.fillna(method="ffill")  # Forward fill remaining
        elif self.config.imputation_method == "mean":
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
        elif self.config.imputation_method == "median":
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        elif self.config.imputation_method == "knn":
            data = self._apply_knn_imputation(data, data_type)

        return data

    def _apply_knn_imputation(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Apply KNN imputation for missing values."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) == 0:
            return data

        # Use or create imputer for this data type
        imputer_key = f"{data_type}_knn"
        if imputer_key not in self.imputers:
            self.imputers[imputer_key] = KNNImputer(n_neighbors=self.config.knn_neighbors)

        try:
            # Fit and transform numeric columns
            data[numeric_columns] = self.imputers[imputer_key].fit_transform(data[numeric_columns])
        except Exception as e:
            logger.warning(f"KNN imputation failed: {e}. Falling back to forward fill.")
            data[numeric_columns] = (
                data[numeric_columns].fillna(method="ffill").fillna(method="bfill")
            )

        return data

    def _clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean price data for obvious errors."""
        price_cols = [col for col in self.config.price_columns if col in data.columns]

        for col in price_cols:
            if col in data.columns:
                # Remove negative prices
                data.loc[data[col] < 0, col] = np.nan

                # Remove zero prices (except for some valid cases)
                if col != "low":  # Low can legitimately be very small
                    data.loc[data[col] == 0, col] = np.nan

        # Validate OHLC relationship
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            data = self._validate_ohlc_consistency(data)

        return data

    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLC data consistency."""
        # High should be >= max(open, close) and low should be <= min(open, close)
        for idx in data.index:
            o, h, l, c = data.loc[idx, ["open", "high", "low", "close"]]

            if pd.notna([o, h, l, c]).all():
                # Fix high if it's too low
                if h < max(o, c):
                    logger.debug(f"Fixing high price at index {idx}: {h} -> {max(o, c)}")
                    data.loc[idx, "high"] = max(o, c)

                # Fix low if it's too high
                if l > min(o, c):
                    logger.debug(f"Fixing low price at index {idx}: {l} -> {min(o, c)}")
                    data.loc[idx, "low"] = min(o, c)

        return data

    def _clean_volume_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean volume data."""
        volume_cols = [col for col in self.config.volume_columns if col in data.columns]

        for col in volume_cols:
            if col in data.columns:
                # Remove negative volume
                data.loc[data[col] < 0, col] = np.nan

                # Flag suspiciously low volume
                low_volume_mask = data[col] < self.config.min_volume_threshold
                if low_volume_mask.any():
                    logger.debug(
                        f"Found {low_volume_mask.sum()} rows with volume < {self.config.min_volume_threshold}"
                    )

        return data

    def _handle_outliers(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Detect and handle outliers based on configuration."""
        if self.config.outlier_method == "none":
            return data

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.DataFrame(False, index=data.index, columns=numeric_columns)

        for col in numeric_columns:
            if col in data.columns:
                if self.config.outlier_method == "zscore":
                    z_scores = np.abs(stats.zscore(data[col].dropna()))
                    outlier_mask[col] = z_scores > self.config.outlier_threshold

                elif self.config.outlier_method == "iqr":
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.outlier_threshold * IQR
                    upper_bound = Q3 + self.config.outlier_threshold * IQR
                    outlier_mask[col] = (data[col] < lower_bound) | (data[col] > upper_bound)

        # Apply outlier action
        if self.config.outlier_action == "remove":
            # Remove rows with any outliers
            data = data[~outlier_mask.any(axis=1)]
        elif self.config.outlier_action == "clip":
            # Clip outliers to boundaries
            for col in numeric_columns:
                if outlier_mask[col].any():
                    if self.config.outlier_method == "iqr":
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - self.config.outlier_threshold * IQR
                        upper_bound = Q3 + self.config.outlier_threshold * IQR
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        elif self.config.outlier_action == "flag":
            # Add outlier flag columns
            for col in numeric_columns:
                if outlier_mask[col].any():
                    data[f"{col}_outlier_flag"] = outlier_mask[col]

        return data

    def _handle_infinite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite values in data."""
        # Replace infinite values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)

        # Log if infinite values were found
        inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.debug(f"Replaced {inf_count} infinite values with NaN")

        return data

    def _validate_data_quality(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Validate overall data quality."""
        # Check minimum data points
        if len(data) < self.config.min_data_points:
            logger.warning(
                f"Insufficient data points for {symbol}: {len(data)} < {self.config.min_data_points}"
            )

        # Check for large gaps in time series
        if "timestamp" in data.columns:
            time_diffs = data["timestamp"].diff().dt.days
            large_gaps = time_diffs > self.config.max_gap_days
            if large_gaps.any():
                gap_count = large_gaps.sum()
                logger.warning(
                    f"Found {gap_count} gaps > {self.config.max_gap_days} days in {symbol}"
                )

        # Check for extreme price movements
        if "close" in data.columns:
            returns = data["close"].pct_change()
            extreme_moves = np.abs(returns) > self.config.price_change_threshold
            if extreme_moves.any():
                extreme_count = extreme_moves.sum()
                logger.warning(
                    f"Found {extreme_count} extreme price movements (>{self.config.price_change_threshold*100}%) in {symbol}"
                )

        return data

    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features if not present."""
        # Add returns if not present
        if "close" in data.columns and "returns" not in data.columns:
            data["returns"] = data["close"].pct_change()

        # Add log returns
        if "close" in data.columns and "log_returns" not in data.columns:
            data["log_returns"] = np.log(data["close"] / data["close"].shift(1))

        # Add typical price
        if (
            all(col in data.columns for col in ["high", "low", "close"])
            and "typical_price" not in data.columns
        ):
            data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3

        return data

    def _scale_features(self, data: pd.DataFrame, feature_type: str) -> pd.DataFrame:
        """Apply feature scaling based on configuration."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ["timestamp"]  # Never scale timestamp-like columns
        scale_columns = [col for col in numeric_columns if col not in exclude_columns]

        if not scale_columns:
            return data

        # Choose scaler based on configuration
        scaler_key = f"{feature_type}_{self.config.scaling_method}"

        if scaler_key not in self.scalers:
            if self.config.scaling_method == "standard":
                self.scalers[scaler_key] = StandardScaler()
            elif self.config.scaling_method == "minmax":
                self.scalers[scaler_key] = MinMaxScaler()
            elif self.config.scaling_method == "robust":
                self.scalers[scaler_key] = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.config.scaling_method}")
                return data

        try:
            # Fit and transform the features
            data[scale_columns] = self.scalers[scaler_key].fit_transform(data[scale_columns])
            logger.debug(
                f"Applied {self.config.scaling_method} scaling to {len(scale_columns)} features"
            )
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")

        return data

    def _preprocess_news_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess news data."""
        # Handle text data cleaning
        if "headline" in data.columns:
            # Remove null headlines
            data = data.dropna(subset=["headline"])

            # Basic text cleaning
            data["headline"] = data["headline"].str.strip()
            data = data[data["headline"].str.len() > 5]  # Remove very short headlines

        # Handle publication dates
        if "published_at" in data.columns:
            data["published_at"] = pd.to_datetime(data["published_at"])

        return data

    def _preprocess_social_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess social media data."""
        # Remove duplicate posts
        if "text" in data.columns:
            data = data.drop_duplicates(subset=["text"])

            # Remove very short posts
            data = data[data["text"].str.len() > 10]

        # Handle engagement metrics
        engagement_cols = ["likes", "shares", "comments"]
        for col in engagement_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                data[col] = data[col].fillna(0)

        return data

    def _preprocess_economic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess economic indicator data."""
        # Handle economic data specific cleaning
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        # Economic data often has seasonal patterns, handle appropriately
        for col in numeric_columns:
            if col in data.columns:
                # Handle economic data outliers more conservatively
                data[col] = data[col].fillna(method="ffill")

        return data

    def get_preprocessing_summary(self) -> dict[str, Any]:
        """Get summary of preprocessing operations performed."""
        return {
            "config": self.config.__dict__,
            "fitted_scalers": list(self.scalers.keys()),
            "fitted_imputers": list(self.imputers.keys()),
            "outlier_stats_available": list(self.outlier_stats.keys()),
        }

    def reset_fitted_components(self):
        """Reset all fitted scalers and imputers."""
        self.scalers.clear()
        self.imputers.clear()
        self.outlier_stats.clear()
        logger.info("All fitted preprocessing components have been reset")

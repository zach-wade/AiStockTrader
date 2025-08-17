"""
Base Feature Calculator

Abstract base class for all feature calculators in the AI Trader system.
Provides common interface and utilities for feature calculation.
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.utils.core import AsyncCircuitBreaker, ErrorHandlingMixin, get_logger

from .helpers import (
    create_feature_dataframe,
    postprocess_features,
    preprocess_data,
    validate_ohlcv_data,
)

logger = get_logger(__name__)


class BaseFeatureCalculator(ErrorHandlingMixin, ABC):
    """
    Abstract base class for all feature calculators.

    Provides:
    - Common interface for feature calculation
    - Data validation and preprocessing
    - Error handling with circuit breaker pattern
    - Feature caching and performance optimization
    - Standardized logging and monitoring
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize base feature calculator.

        Args:
            config: Configuration dictionary for the calculator
        """
        super().__init__()

        # Configuration
        self.config = config or {}
        self.name = self.__class__.__name__

        # Feature metadata
        self._feature_names: list[str] | None = None
        self._feature_count: int | None = None

        # Performance settings
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour
        self._feature_cache: dict[str, pd.DataFrame] = {}

        # Validation settings
        self.validate_inputs = self.config.get("validate_inputs", True)
        self.min_required_rows = self.config.get("min_required_rows", 10)

        # Circuit breaker for resilience
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=self.config.get("circuit_breaker_threshold", 5),
            recovery_timeout=self.config.get("circuit_breaker_timeout", 60),
        )

        logger.info(f"Initialized {self.name} calculator")

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features from input data.

        Args:
            data: Input DataFrame with required columns

        Returns:
            DataFrame with calculated features
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Get list of feature names this calculator produces.

        Returns:
            List of feature names
        """
        pass

    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """
        Get list of required input columns.

        Returns:
            List of required column names
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate input data.

        Args:
            data: Input DataFrame to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check if data is None or empty
        if data is None:
            errors.append("Input data is None")
            return False, errors

        if data.empty:
            errors.append("Input data is empty")
            return False, errors

        # Check minimum rows
        if len(data) < self.min_required_rows:
            errors.append(
                f"Insufficient data: {len(data)} rows < {self.min_required_rows} required"
            )

        # Check required columns
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Additional validation based on data type
        if "open" in required_columns and "high" in required_columns:
            # OHLCV data validation
            is_valid, ohlcv_errors = validate_ohlcv_data(
                data, required_columns=required_columns, min_rows=self.min_required_rows
            )
            errors.extend(ohlcv_errors)

        return len(errors) == 0, errors

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before feature calculation.

        Args:
            data: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        # Apply standard preprocessing
        preprocessed = preprocess_data(
            data,
            handle_missing=self.config.get("handle_missing", "forward_fill"),
            remove_outliers=self.config.get("remove_outliers", False),
            outlier_threshold=self.config.get("outlier_threshold", 3.0),
        )

        # Sort by timestamp if available
        if "timestamp" in preprocessed.columns:
            preprocessed = preprocessed.sort_values("timestamp")

        # Set index if specified
        index_column = self.config.get("index_column", "timestamp")
        if index_column in preprocessed.columns:
            preprocessed = preprocessed.set_index(index_column)

        return preprocessed

    def postprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Postprocess calculated features.

        Args:
            features: Calculated features DataFrame

        Returns:
            Postprocessed features
        """
        # Apply standard postprocessing
        postprocessed = postprocess_features(
            features,
            clip_quantiles=self.config.get("clip_quantiles"),
            normalize=self.config.get("normalize", False),
            standardize=self.config.get("standardize", False),
            fill_value=self.config.get("fill_value", 0.0),
        )

        # Add feature name prefix if specified
        prefix = self.config.get("feature_prefix")
        if prefix:
            postprocessed.columns = [f"{prefix}_{col}" for col in postprocessed.columns]

        return postprocessed

    def calculate_with_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features with validation and error handling.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with calculated features
        """
        try:
            # Validate inputs if enabled
            if self.validate_inputs:
                is_valid, errors = self.validate_data(data)
                if not is_valid:
                    logger.error(f"{self.name} validation failed: {errors}")
                    return self._create_empty_features(data.index)

            # Check cache if enabled
            if self.enable_caching:
                cache_key = self._generate_cache_key(data)
                if cache_key in self._feature_cache:
                    logger.debug(f"Returning cached features for {self.name}")
                    return self._feature_cache[cache_key]

            # Preprocess data
            preprocessed = self.preprocess(data)

            # Calculate features
            features = self.calculate(preprocessed)

            # Postprocess features
            features = self.postprocess(features)

            # Cache results if enabled
            if self.enable_caching and cache_key:
                self._feature_cache[cache_key] = features

            # Log successful calculation
            logger.debug(
                f"{self.name} calculated {len(features.columns)} features "
                f"for {len(features)} observations"
            )

            return features

        except Exception as e:
            logger.error(f"Error in {self.name}.calculate_with_validation: {e}")
            return self._create_empty_features(data.index if hasattr(data, "index") else None)

    def get_feature_count(self) -> int:
        """
        Get number of features this calculator produces.

        Returns:
            Number of features
        """
        if self._feature_count is None:
            self._feature_count = len(self.get_feature_names())
        return self._feature_count

    def get_feature_info(self) -> dict[str, Any]:
        """
        Get information about this calculator and its features.

        Returns:
            Dictionary with calculator information
        """
        return {
            "name": self.name,
            "feature_count": self.get_feature_count(),
            "feature_names": self.get_feature_names(),
            "required_columns": self.get_required_columns(),
            "config": self.config,
            "cache_enabled": self.enable_caching,
            "validation_enabled": self.validate_inputs,
        }

    def _create_empty_features(self, index: pd.Index | None = None) -> pd.DataFrame:
        """
        Create empty features DataFrame with proper structure.

        Args:
            index: Index for the DataFrame

        Returns:
            Empty features DataFrame
        """
        feature_names = self.get_feature_names()

        if index is not None:
            return create_feature_dataframe(
                index=index,
                features={name: pd.Series(np.nan, index=index) for name in feature_names},
            )
        else:
            return pd.DataFrame(columns=feature_names)

    def _generate_cache_key(self, data: pd.DataFrame) -> str | None:
        """
        Generate cache key for input data.

        Args:
            data: Input DataFrame

        Returns:
            Cache key or None if caching not applicable
        """
        try:
            # Use data shape and first/last timestamps as cache key
            if "timestamp" in data.columns:
                first_ts = data["timestamp"].iloc[0]
                last_ts = data["timestamp"].iloc[-1]
                key_parts = [str(data.shape), str(first_ts), str(last_ts), str(self.config)]
            else:
                # Use data hash for non-time series data
                key_parts = [
                    str(data.shape),
                    str(pd.util.hash_pandas_object(data.head()).sum()),
                    str(self.config),
                ]

            return "_".join(key_parts)

        except Exception as e:
            logger.warning(f"Could not generate cache key: {e}")
            return None

    def clear_cache(self):
        """Clear feature cache."""
        self._feature_cache.clear()
        logger.info(f"Cleared cache for {self.name}")

    def __repr__(self) -> str:
        """String representation of calculator."""
        return (
            f"{self.name}("
            f"features={self.get_feature_count()}, "
            f"required_columns={len(self.get_required_columns())}, "
            f"cache_enabled={self.enable_caching})"
        )

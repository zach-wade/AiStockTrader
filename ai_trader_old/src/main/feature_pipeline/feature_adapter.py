"""
Feature adapter for bridging feature pipeline with trading strategies.

This module provides an adapter interface between the feature calculation
pipeline and trading strategies, handling feature selection, transformation,
and caching for optimal performance.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.feature_pipeline.feature_config import FeatureConfig
from main.feature_pipeline.feature_store import FeatureStoreRepository
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine
from main.utils.core import ErrorHandlingMixin, get_logger, timer
from main.utils.monitoring import record_metric

logger = get_logger(__name__)


class FeatureValidationError(Exception):
    """Raised when feature validation fails."""

    pass


@dataclass
class FeatureRequirement:
    """Requirements for features needed by a strategy."""

    name: str
    features: list[str]
    lookback_periods: list[int] = field(default_factory=lambda: [20, 60])
    update_frequency: str = "1min"
    required: bool = True


@dataclass
class StrategyFeatureMapping:
    """Maps strategies to their feature requirements."""

    strategy_name: str
    requirements: list[FeatureRequirement]
    priority: int = 1
    cache_ttl: int = 300  # seconds


@dataclass
class FeatureSet:
    """Container for a set of features."""

    features: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return list(self.features.columns)

    @property
    def symbols(self) -> list[str]:
        """Get list of symbols in the feature set."""
        if "symbol" in self.features.index.names:
            return list(self.features.index.get_level_values("symbol").unique())
        elif "symbol" in self.features.columns:
            return list(self.features["symbol"].unique())
        return []

    def filter_by_symbol(self, symbol: str) -> pd.DataFrame:
        """Filter features for a specific symbol."""
        if "symbol" in self.features.index.names:
            return self.features.xs(symbol, level="symbol")
        elif "symbol" in self.features.columns:
            return self.features[self.features["symbol"] == symbol]
        return self.features


@dataclass
class FeatureRequest:
    """Request for features from the adapter."""

    symbols: list[str]
    feature_groups: list[str]
    start_date: datetime
    end_date: datetime
    frequency: str = "1d"
    filters: dict[str, Any] = field(default_factory=dict)
    transformations: list[str] = field(default_factory=list)
    cache_key: str | None = None


class FeatureAdapter(ErrorHandlingMixin):
    """
    Adapts feature pipeline for use by trading strategies.

    Features:
    - Feature selection and filtering
    - On-demand calculation
    - Caching for performance
    - Feature transformation
    - Multi-symbol support
    - Async computation
    """

    def __init__(
        self,
        feature_engine: UnifiedFeatureEngine,
        feature_store: FeatureStoreRepository,
        config: FeatureConfig | None = None,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize feature adapter.

        Args:
            feature_engine: Unified feature calculation engine
            feature_store: Feature storage repository
            config: Feature configuration
            cache_ttl_seconds: Cache time-to-live
        """
        super().__init__()
        self.feature_engine = feature_engine
        self.feature_store = feature_store
        self.config = config or FeatureConfig()
        self.cache_ttl_seconds = cache_ttl_seconds

        # Feature cache
        self._cache: dict[str, FeatureSet] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        # Feature metadata
        self._feature_metadata = self._load_feature_metadata()

        # Performance tracking
        self._request_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

        logger.debug("FeatureAdapter initialized")

    @timer
    async def get_features(self, request: FeatureRequest) -> FeatureSet:
        """
        Get features based on request.

        Args:
            request: Feature request specification

        Returns:
            FeatureSet containing requested features
        """
        with self._handle_error("getting features"):
            self._request_count += 1

            # Generate cache key if not provided
            if not request.cache_key:
                request.cache_key = self._generate_cache_key(request)

            # Check cache
            cached = self._get_cached_features(request.cache_key)
            if cached is not None:
                self._cache_hits += 1
                logger.debug(f"Cache hit for key: {request.cache_key}")
                return cached

            self._cache_misses += 1
            logger.debug(f"Cache miss for key: {request.cache_key}")

            # Calculate features
            features = await self._calculate_features(request)

            # Apply transformations
            if request.transformations:
                features = self._apply_transformations(features, request.transformations)

            # Create feature set
            feature_set = FeatureSet(
                features=features,
                metadata={
                    "request": request,
                    "calculation_time": datetime.utcnow(),
                    "feature_count": len(features.columns),
                    "symbol_count": len(request.symbols),
                },
            )

            # Cache result
            self._cache_features(request.cache_key, feature_set)

            # Record metrics
            record_metric(
                "feature_adapter.features_retrieved",
                len(features.columns),
                tags={"symbols": len(request.symbols), "cache_hit": False},
            )

            return feature_set

    async def get_latest_features(
        self, symbols: list[str], feature_names: list[str], lookback_periods: int = 1
    ) -> pd.DataFrame:
        """
        Get latest features for symbols.

        Args:
            symbols: List of symbols
            feature_names: List of feature names
            lookback_periods: Number of periods to retrieve

        Returns:
            DataFrame with latest features
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_periods)

        # Create request
        request = FeatureRequest(
            symbols=symbols,
            feature_groups=self._get_feature_groups(feature_names),
            start_date=start_date,
            end_date=end_date,
        )

        # Get features
        feature_set = await self.get_features(request)

        # Filter to requested features
        available_features = [f for f in feature_names if f in feature_set.features.columns]

        return feature_set.features[available_features]

    def get_feature_info(self, feature_name: str) -> dict[str, Any]:
        """
        Get information about a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Feature metadata dictionary
        """
        return self._feature_metadata.get(
            feature_name,
            {
                "name": feature_name,
                "description": "Unknown feature",
                "group": "unknown",
                "dependencies": [],
            },
        )

    def list_available_features(self, group: str | None = None) -> list[str]:
        """
        List available features.

        Args:
            group: Optional feature group filter

        Returns:
            List of feature names
        """
        features = list(self._feature_metadata.keys())

        if group:
            features = [f for f in features if self._feature_metadata[f].get("group") == group]

        return sorted(features)

    def get_feature_dependencies(self, feature_names: list[str]) -> set[str]:
        """
        Get all dependencies for features.

        Args:
            feature_names: List of feature names

        Returns:
            Set of all required features including dependencies
        """
        required = set(feature_names)

        # Add dependencies recursively
        def add_deps(feature: str):
            if feature in self._feature_metadata:
                deps = self._feature_metadata[feature].get("dependencies", [])
                for dep in deps:
                    if dep not in required:
                        required.add(dep)
                        add_deps(dep)

        for feature in feature_names:
            add_deps(feature)

        return required

    def validate_features(
        self, features: pd.DataFrame, expected_features: list[str]
    ) -> tuple[bool, list[str]]:
        """
        Validate that features contain expected columns.

        Args:
            features: Feature DataFrame
            expected_features: Expected feature names

        Returns:
            Tuple of (is_valid, missing_features)
        """
        missing = [f for f in expected_features if f not in features.columns]

        return len(missing) == 0, missing

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self._cache_hits / self._request_count if self._request_count > 0 else 0

        return {
            "cache_size": len(self._cache),
            "request_count": self._request_count,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cached_symbols": self._get_cached_symbols(),
        }

    def clear_cache(self, symbol: str | None = None) -> int:
        """
        Clear feature cache.

        Args:
            symbol: Optional symbol to clear (clears all if None)

        Returns:
            Number of entries cleared
        """
        if symbol:
            # Clear entries for specific symbol
            to_clear = [key for key in self._cache if symbol in key]
        else:
            # Clear all
            to_clear = list(self._cache.keys())

        for key in to_clear:
            del self._cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]

        logger.info(f"Cleared {len(to_clear)} cache entries")
        return len(to_clear)

    async def _calculate_features(self, request: FeatureRequest) -> pd.DataFrame:
        """Calculate features based on request."""
        # Prepare calculation parameters
        calc_params = {
            "symbols": request.symbols,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "frequency": request.frequency,
        }

        # Calculate features by group
        all_features = []

        for group in request.feature_groups:
            logger.debug(f"Calculating features for group: {group}")

            # Get calculator for group
            calculator = self.feature_engine.get_calculator(group)
            if not calculator:
                logger.warning(f"No calculator found for group: {group}")
                continue

            # Calculate features
            group_features = await calculator.calculate_async(**calc_params)
            all_features.append(group_features)

        # Combine all features
        if not all_features:
            return pd.DataFrame()

        combined = pd.concat(all_features, axis=1)

        # Apply filters
        if request.filters:
            combined = self._apply_filters(combined, request.filters)

        return combined

    def _apply_transformations(
        self, features: pd.DataFrame, transformations: list[str]
    ) -> pd.DataFrame:
        """Apply transformations to features."""
        result = features.copy()

        for transform in transformations:
            if transform == "normalize":
                # Z-score normalization
                result = (result - result.mean()) / result.std()

            elif transform == "rank":
                # Rank transformation
                result = result.rank(pct=True)

            elif transform == "diff":
                # First difference
                result = result.diff()

            elif transform == "log":
                # Log transformation (with safety for negative values)
                result = np.sign(result) * np.log1p(np.abs(result))

            elif transform.startswith("lag_"):
                # Lag transformation
                periods = int(transform.split("_")[1])
                result = result.shift(periods)

            elif transform.startswith("ma_"):
                # Moving average
                window = int(transform.split("_")[1])
                result = result.rolling(window=window).mean()

            else:
                logger.warning(f"Unknown transformation: {transform}")

        return result

    def _apply_filters(self, features: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
        """Apply filters to features."""
        result = features

        for key, value in filters.items():
            if key == "min_date" and "date" in result.index.names:
                result = result[result.index.get_level_values("date") >= value]

            elif key == "max_date" and "date" in result.index.names:
                result = result[result.index.get_level_values("date") <= value]

            elif key == "symbols" and "symbol" in result.index.names:
                result = result[result.index.get_level_values("symbol").isin(value)]

            elif key in result.columns:
                # Column-based filter
                if isinstance(value, dict):
                    if "min" in value:
                        result = result[result[key] >= value["min"]]
                    if "max" in value:
                        result = result[result[key] <= value["max"]]
                else:
                    result = result[result[key] == value]

        return result

    def _generate_cache_key(self, request: FeatureRequest) -> str:
        """Generate cache key for request."""
        # Standard library imports
        import hashlib

        # Create key components
        components = [
            str(sorted(request.symbols)),
            str(sorted(request.feature_groups)),
            str(request.start_date),
            str(request.end_date),
            request.frequency,
            str(sorted(request.filters.items())),
            str(sorted(request.transformations)),
        ]

        # Generate hash
        key_string = "|".join(components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _get_cached_features(self, cache_key: str) -> FeatureSet | None:
        """Get features from cache if valid."""
        if cache_key not in self._cache:
            return None

        # Check if expired
        timestamp = self._cache_timestamps.get(cache_key)
        if timestamp:
            age = (datetime.utcnow() - timestamp).total_seconds()
            if age > self.cache_ttl_seconds:
                # Expired
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]
                return None

        return self._cache[cache_key]

    def _cache_features(self, cache_key: str, feature_set: FeatureSet) -> None:
        """Cache feature set."""
        self._cache[cache_key] = feature_set
        self._cache_timestamps[cache_key] = datetime.utcnow()

        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k]
            )[:100]

            for key in oldest_keys:
                del self._cache[key]
                del self._cache_timestamps[key]

    def _get_feature_groups(self, feature_names: list[str]) -> list[str]:
        """Get feature groups for feature names."""
        groups = set()

        for feature in feature_names:
            if feature in self._feature_metadata:
                group = self._feature_metadata[feature].get("group")
                if group:
                    groups.add(group)

        return list(groups)

    def _get_cached_symbols(self) -> list[str]:
        """Get list of symbols in cache."""
        symbols = set()

        for feature_set in self._cache.values():
            symbols.update(feature_set.symbols)

        return sorted(symbols)

    def _load_feature_metadata(self) -> dict[str, dict[str, Any]]:
        """Load feature metadata."""
        # This would typically load from configuration or database
        # For now, return a basic structure
        return {
            # Price features
            "returns": {
                "group": "price",
                "description": "Simple returns",
                "dependencies": ["close"],
            },
            "log_returns": {
                "group": "price",
                "description": "Log returns",
                "dependencies": ["close"],
            },
            # Volume features
            "volume_ratio": {
                "group": "volume",
                "description": "Volume relative to average",
                "dependencies": ["volume", "volume_ma"],
            },
            # Technical features
            "rsi": {
                "group": "technical",
                "description": "Relative Strength Index",
                "dependencies": ["close"],
            },
            "macd": {
                "group": "technical",
                "description": "MACD indicator",
                "dependencies": ["close"],
            },
            # Add more feature metadata as needed
        }


def create_feature_adapter(config: dict[str, Any] | None = None) -> FeatureAdapter:
    """Factory function to create a FeatureAdapter instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        FeatureAdapter: Configured feature adapter instance
    """
    if config is None:
        config = {}

    return FeatureAdapter(config)

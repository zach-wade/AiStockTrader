# File: src/ai_trader/models/inference/feature_pipeline.py

"""
Real-Time Feature Pipeline for Inference.

This class orchestrates the generation of real-time features for trading models,
composing specialized helper components for data buffering, feature calculation,
and caching.
"""

# Standard library imports
import asyncio  # Not directly used for orchestration loop here, but for async operations if any
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd  # Used by helper components, not directly here as orchestrator

# Local imports
from main.feature_pipeline.calculators.market_regime import (  # Kept here for clarity if direct init is desired
    MarketRegimeCalculator,
)
from main.feature_pipeline.calculators.microstructure import (  # Kept here for clarity if direct init is desired
    MicrostructureCalculator,
)
from main.feature_pipeline.calculators.technical import (
    UnifiedTechnicalIndicatorsFacade as UnifiedTechnicalIndicators,
)

# Corrected absolute imports for helpers and calculators
from main.feature_pipeline.calculators.technical import BaseTechnicalCalculator
from main.models.inference.feature_pipeline_helpers.feature_calculator_integrator import (
    FeatureCalculatorIntegrator,
)
from main.models.inference.feature_pipeline_helpers.inference_feature_cache import (
    InferenceFeatureCache,
)

# Import the new feature pipeline helper classes
from main.models.inference.feature_pipeline_helpers.realtime_data_buffer import RealtimeDataBuffer

logger = logging.getLogger(__name__)


class RealTimeFeaturePipeline:
    """
    Optimized feature pipeline for real-time trading inference.
    Orchestrates data buffering, feature computation, and caching by composing
    specialized helper components.
    """

    def __init__(
        self,
        lookback_periods: Optional[Dict[str, int]] = None,
        buffer_max_size: int = 500,
        cache_ttl_seconds: int = 5,
    ):
        """
        Initializes the real-time feature pipeline and its composing helper components.

        Args:
            lookback_periods: Dictionary of lookback periods for different features (e.g., RSI, MACD).
            buffer_max_size: Maximum number of data points to keep in the internal data buffer.
            cache_ttl_seconds: Time-to-live for cached feature results in seconds.
        """
        self.lookback_periods = lookback_periods or {
            "rsi": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb": 20,
            "atr": 14,
            "volume_ma": 20,
        }

        # Initialize helper components
        self._data_buffer = RealtimeDataBuffer(max_buffer_size=buffer_max_size)
        self._feature_cache = InferenceFeatureCache(cache_ttl_seconds=cache_ttl_seconds)
        self._feature_calculator_integrator = FeatureCalculatorIntegrator(
            lookback_periods=self.lookback_periods
        )

        # Feature importance (placeholder - typically loaded from a model)
        self._feature_importance_scores: Dict[str, float] = {}

        logger.info("RealTimeFeaturePipeline initialized with helper components.")

    async def update_and_calculate_features(
        self,
        symbol: str,  # Added symbol for cache key and context
        timestamp: datetime,
        latest_ohlcv: Dict[str, float],  # OHLCV for the latest bar
        news_sentiment: Optional[Dict] = None,
        order_flow: Optional[pd.DataFrame] = None,
        interval: str = "1min",  # Added interval for cache key
    ) -> Dict[str, float]:
        """
        Updates internal data buffers with the latest data point and then calculates
        all relevant features for real-time inference. Uses caching for efficiency.

        Args:
            symbol: The stock symbol.
            timestamp: The timestamp of the latest OHLCV bar.
            latest_ohlcv: Dictionary of OHLCV values for the latest bar.
                          Expected keys: 'open', 'high', 'low', 'close', 'volume'.
            news_sentiment: Optional dictionary of news sentiment scores.
            order_flow: Optional DataFrame containing real-time order flow data.
            interval: The time interval of the market data (e.g., '1min', '1hour').

        Returns:
            A dictionary of feature names and their latest float values.
        """
        features: Dict[str, float] = {}

        try:
            # Update internal buffers with the latest data point
            # Assuming latest_ohlcv contains 'close' and 'volume'
            self._data_buffer.update_buffers(
                timestamp=timestamp,
                price=latest_ohlcv.get("close", np.nan),
                volume=int(latest_ohlcv.get("volume", 0)),
            )

            # Retrieve buffered data as a DataFrame for lookback calculations
            # Include 'open', 'high', 'low', 'close', 'volume' for the calculators
            buffered_df = self._data_buffer.get_buffered_dataframe(
                columns=["open", "high", "low", "close", "volume"]  # Need all for OHLCV
            )

            if buffered_df is None or buffered_df.empty:
                logger.warning(
                    f"Insufficient buffered data for {symbol} at {timestamp}. Cannot calculate features."
                )
                return {}

            # Generate a cache key for the requested features for this symbol/interval
            # The list of features for the key can be generalized to all potential features.
            all_potential_features = list(
                self._feature_calculator_integrator.feature_set_def.extract_and_flatten_features(
                    pd.DataFrame(),
                    pd.DataFrame(),
                    {},
                    None,
                    None,  # Pass empty dataframes to get all keys
                ).keys()
            )  # Hack to get all feature names for cache key if not known

            cache_key = self._feature_cache.get_cache_key(symbol, all_potential_features, interval)

            # Try to retrieve from cache
            cached_features = await self._feature_cache.get_cached_features(cache_key)
            if cached_features:
                logger.debug(
                    f"Features for {symbol} ({interval}) at {timestamp} retrieved from cache."
                )
                return cached_features

            # If not in cache, compute all features using the integrator
            features = await self._feature_calculator_integrator.compute_all_features(
                market_data=buffered_df, news_sentiment=news_sentiment, order_flow=order_flow
            )

            # Cache the newly computed features
            if features:
                await self._feature_cache.set_cached_features(cache_key, features)

            return features

        except Exception as e:
            logger.error(
                f"Error calculating real-time features for {symbol} at {timestamp}: {e}",
                exc_info=True,
            )
            return {}

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Returns feature importance scores.
        This would typically be loaded from a trained model during initialization or runtime.
        """
        if not self._feature_importance_scores:
            logger.warning("Feature importance scores not loaded. Returning placeholder.")
            return {
                "rsi": 0.15,
                "macd_histogram": 0.12,
                "bb_percent": 0.10,
                "volume_ratio": 0.08,
                "returns_1d": 0.07,
                "atr": 0.06,
                "regime_volatility": 0.05,
                "news_sentiment": 0.05,
            }
        return self._feature_importance_scores

    def set_feature_importance(self, importance_scores: Dict[str, float]):
        """Sets feature importance scores, typically from a loaded model."""
        self._feature_importance_scores = importance_scores
        logger.info("Feature importance scores updated.")

    def get_feature_cache_size(self) -> int:
        """Returns the current size of the inference feature cache."""
        return self._feature_cache.get_cache_size()

# File: src/ai_trader/models/inference/prediction_engine_helpers/feature_data_manager.py

import logging
import json
import hashlib
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

# Corrected absolute import for FeatureStore
from main.feature_pipeline.feature_store import FeatureStore
from main.utils.cache import MemoryBackend
from main.utils.cache import CacheType

logger = logging.getLogger(__name__)

class FeatureDataManager:
    """
    Manages the retrieval and caching of features for model prediction.
    Interacts with the FeatureStore and optionally uses Redis for a distributed cache.
    """

    def __init__(self, config: Dict[str, Any], feature_store: FeatureStore):
        """
        Initializes the FeatureDataManager.

        Args:
            config: Application configuration for Redis settings and cache TTL.
            feature_store: An initialized FeatureStore instance.
        """
        self.config = config
        self.feature_store = feature_store
        
        # Use global cache for caching
        self.cache = get_global_cache()
        self.use_cache = config.get('prediction', {}).get('use_cache', True)
        
        # Cache TTL settings
        self._cache_ttl_seconds = config.get('prediction', {}).get('cache_ttl_seconds', 60)
        
        logger.debug("FeatureDataManager initialized.")

    def _get_cache_key(self, symbol: str, model_name: str, feature_names: List[str]) -> str:
        """
        Generates a consistent cache key for a set of features for a symbol and model.
        Includes a hash of feature names for uniqueness.
        """
        # Feature names are part of the cache key because different models might need different features
        sorted_feature_names = sorted(feature_names)
        features_hash = hashlib.md5(json.dumps(sorted_feature_names).encode('utf-8')).hexdigest()
        return f"features:{symbol}:{model_name}:{features_hash}"

    async def get_features_for_prediction(self, 
                                          symbol: str, 
                                          model_name: str, 
                                          required_feature_names: List[str], # Explicitly pass required features
                                          lookback_period: int, # Explicitly pass lookback for FeatureStore
                                          use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Retrieves features for prediction, prioritizing cache lookup (Redis then in-memory),
        then fetching from FeatureStore.

        Args:
            symbol: The stock symbol.
            model_name: The name of the model (part of cache key).
            required_feature_names: The exact list of feature names the model expects.
            lookback_period: The lookback period needed for feature computation.
            use_cache: If True, attempts to use cache.

        Returns:
            A Pandas DataFrame of features, or None if not available.
        """
        cache_key = self._get_cache_key(symbol, model_name, required_feature_names)
        
        if use_cache and self.use_cache:
            # Try cache first
            try:
                cached_data = await self.cache.get(CacheType.FEATURES, cache_key)
                if cached_data:
                    if isinstance(cached_data, str):
                        # Deserialize JSON string
                        features_dict_list = json.loads(cached_data)
                        df = pd.DataFrame(features_dict_list)
                    else:
                        # Direct DataFrame (from serialized format)
                        df = pd.DataFrame(cached_data)
                    
                    if not df.empty and 'timestamp' in df.columns:
                        df = df.set_index('timestamp')
                        df.index = pd.to_datetime(df.index, utc=True)
                    
                    logger.debug(f"Features for {symbol}/{model_name} loaded from cache.")
                    return df
            except Exception as e:
                logger.error(f"Error retrieving features from cache for {cache_key}: {e}", exc_info=True)

        # 3. Fetch from FeatureStore if not in cache or cache is disabled/expired
        try:
            # FeatureStore's `get_latest_features` typically computes/retrieves based on config.
            # We assume it returns a DataFrame with the features.
            features_df = await self.feature_store.get_latest_features(
                symbol=symbol,
                lookback_periods=lookback_period,
                feature_names=required_feature_names # Request specific features if FeatureStore supports this
            )
            
            if features_df is None or features_df.empty:
                logger.warning(f"FeatureStore returned no data for {symbol} with model {model_name}.")
                return None
            
            # Cache the fetched features
            if use_cache and self.use_cache:
                try:
                    # Convert DataFrame to list of dicts for caching
                    features_dict_list = features_df.to_dict(orient='records')
                    await self.cache.set(
                        CacheType.FEATURES,
                        cache_key,
                        features_dict_list,
                        self._cache_ttl_seconds
                    )
                    logger.debug(f"Features for {symbol}/{model_name} cached.")
                except Exception as e:
                    logger.error(f"Error caching features for {cache_key}: {e}", exc_info=True)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to get features for {symbol} from FeatureStore for model {model_name}: {e}", exc_info=True)
            return None

    async def clear_feature_cache(self, model_name: Optional[str] = None):
        """
        Clears the feature caches.

        Args:
            model_name: Optional. If provided, clears cache only for this model.
                        Otherwise, clears all feature caches.
        """
        try:
            if model_name:
                # Clear cache for specific model using pattern
                pattern = f"features:*:{model_name}:*"
                await self.cache.delete_pattern(CacheType.FEATURES, pattern)
                logger.info(f"Feature cache cleared for model: {model_name}.")
            else:
                # Clear all feature cache
                await self.cache.clear_type(CacheType.FEATURES)
                logger.info("All feature cache cleared.")
        except Exception as e:
            logger.error(f"Failed to clear feature cache: {e}", exc_info=True)

    async def get_cache_size(self) -> int:
        """Returns the current number of items in the cache."""
        try:
            return await self.cache.get_size(CacheType.FEATURES)
        except Exception as e:
            logger.error(f"Failed to get cache size: {e}")
            return 0
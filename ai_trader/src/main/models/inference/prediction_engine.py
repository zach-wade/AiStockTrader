"""
The Core Prediction Engine for Live Trading.

This class is the single, authoritative component for generating predictions
from trained models. It is designed for low-latency and is orchestrated by
higher-level services for operational tasks.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from main.config.config_manager import get_config
from main.feature_pipeline.feature_store import FeatureStore
from .prediction_engine_helpers.model_loader_cache import ModelLoaderCache
from .prediction_engine_helpers.feature_data_manager import FeatureDataManager
from .model_registry_helpers.core_predictor import CorePredictor

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Low-latency prediction orchestrator. Composes specialized helpers for
    model loading, feature management, and prediction calculation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the PredictionEngine and its core components."""
        self.config = config or get_config()
        
        # Instantiate helper components that manage specific responsibilities
        self.model_loader = ModelLoaderCache(
            models_base_dir=Path(self.config.get('paths', {}).get('models', 'models/trained'))
        )
        
        feature_store_path = Path(self.config.get('paths', {}).get('feature_store', 'data/features'))
        self.feature_manager = FeatureDataManager(
            config=self.config, 
            feature_store=FeatureStore(str(feature_store_path))
        )
        
        # The CorePredictor encapsulates the logic for a single prediction run
        self.core_predictor = CorePredictor(
            model_loader=self.model_loader,
            feature_manager=self.feature_manager
        )
        
        logger.info("PredictionEngine initialized.")

    async def predict(self, symbol: str, model_id: str, version: str, features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generates a prediction for a single symbol. This is the primary public method.
        It delegates the complex logic to the CorePredictor.
        """
        return await self.core_predictor.predict_single(symbol, model_id, version, features)

    async def predict_batch(self, prediction_requests: List[Dict]) -> List[Dict]:
        """
        Generates predictions for a batch of requests in parallel.
        
        Args:
            prediction_requests: A list of dicts, each with 'symbol', 'model_id', 'version'.
        """
        tasks = [
            self.predict(req['symbol'], req['model_id'], req['version'])
            for req in prediction_requests
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, logging any errors
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                req = prediction_requests[i]
                logger.error(f"Batch prediction failed for {req['symbol']}/{req['model_id']}: {result}", exc_info=True)
                final_results.append({'error': str(result), **req})
            else:
                final_results.append(result)
        
        return final_results
# File: src/ai_trader/models/inference/prediction_engine_helpers/core_predictor.py

import logging
import time
import json # For caching prediction results
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

# Corrected absolute imports for helper components
from main.models.inference.prediction_engine_helpers.model_loader_cache import ModelLoaderCache
from main.models.inference.prediction_engine_helpers.feature_data_manager import FeatureDataManager
from main.models.inference.prediction_engine_helpers.prediction_calculator import PredictionCalculator
from main.models.inference.prediction_engine_helpers.prediction_performance_monitor import PredictionPerformanceMonitor

logger = logging.getLogger(__name__)

class CorePredictor:
    """
    Encapsulates the core logic for making a single model prediction.
    Orchestrates interactions with model loading, feature management,
    prediction calculation, and performance monitoring.
    """

    def __init__(self, 
                 model_loader_cache: ModelLoaderCache,
                 feature_data_manager: FeatureDataManager,
                 prediction_calculator: PredictionCalculator,
                 performance_monitor: PredictionPerformanceMonitor):
        """
        Initializes the CorePredictor with its necessary helper dependencies.

        Args:
            model_loader_cache: Instance of ModelLoaderCache.
            feature_data_manager: Instance of FeatureDataManager.
            prediction_calculator: Instance of PredictionCalculator.
            performance_monitor: Instance of PredictionPerformanceMonitor.
        """
        self._model_loader_cache = model_loader_cache
        self._feature_data_manager = feature_data_manager
        self._prediction_calculator = prediction_calculator
        self._performance_monitor = performance_monitor
        logger.debug("CorePredictor initialized.")

    def predict_single(self, 
                       symbol: str, 
                       model_name: str, 
                       features_df: Optional[pd.DataFrame] = None, 
                       use_cache: bool = True) -> Dict[str, Any]:
        """
        Makes a single prediction for a symbol using the specified model.
        Manages the sequence of model loading, feature retrieval, prediction,
        performance tracking, and result caching.

        Args:
            symbol: The stock symbol to predict for.
            model_name: The name of the model to use.
            features_df: Optional. Pre-computed features DataFrame. If provided,
                         feature retrieval from FeatureStore is skipped.
            use_cache: If True, attempts to use features cache and caches the prediction result.

        Returns:
            A dictionary containing the prediction, probability, confidence,
            and relevant metadata (including timing and potential errors).
        """
        prediction_start_time = time.time()
        feature_retrieval_time = 0.0
        
        try:
            # 1. Load Model
            model_obj, model_metadata = self._model_loader_cache.load_model_object(model_name)
            
            # 2. Get Features
            if features_df is None:
                feature_retrieval_start = time.time()
                # Need specific feature names and lookback period from model_metadata
                required_features = model_metadata.get('features', []) 
                lookback_period = model_metadata.get('lookback_period', 20) 
                
                features_df = self._feature_data_manager.get_features_for_prediction(
                    symbol=symbol,
                    model_name=model_name,
                    required_feature_names=required_features,
                    lookback_period=lookback_period,
                    use_cache=use_cache
                )
                feature_retrieval_time = time.time() - feature_retrieval_start
                self._performance_monitor.record_feature_latency(feature_retrieval_time)
            
            if features_df is None or features_df.empty:
                logger.warning(f"No features available for prediction for {symbol} with model {model_name}.")
                raise ValueError("No features available for prediction.")

            # 3. Calculate Prediction
            prediction_output = self._prediction_calculator.calculate_prediction(
                model_obj=model_obj,
                features_df=features_df,
                model_metadata=model_metadata
            )
            
            # 4. Record Performance & Augment Result
            prediction_total_time = time.time() - prediction_start_time
            self._performance_monitor.record_prediction_latency(prediction_total_time)
            
            result = {
                'symbol': symbol,
                'model': model_name,
                'prediction_time': prediction_total_time,
                'feature_time': feature_retrieval_time,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'features_used_count': len(features_df.columns),
                **prediction_output # Unpack prediction, probability, confidence, error
            }
            
            # 5. Cache Prediction Result (if using Redis)
            if use_cache and self._feature_data_manager.use_redis and self._feature_data_manager.redis_client:
                cache_key = f"prediction_result:{symbol}:{model_name}"
                try:
                    self._feature_data_manager.redis_client.setex(
                        cache_key,
                        self._feature_data_manager._cache_ttl_seconds, 
                        json.dumps(result)
                    )
                except Exception as e:
                    logger.error(f"Failed to cache prediction result to Redis for {symbol}/{model_name}: {e}", exc_info=True)

            return result
            
        except FileNotFoundError as e:
            logger.error(f"Model file for {model_name} not found: {e}", exc_info=True)
            return self._create_error_result(symbol, model_name, prediction_start_time, f"Model not found: {str(e)}", feature_retrieval_time)
        except ValueError as e: # Catch "No features available" or other value issues
            logger.error(f"Data issue during prediction for {symbol} with model {model_name}: {e}", exc_info=True)
            return self._create_error_result(symbol, model_name, prediction_start_time, f"Data issue: {str(e)}", feature_retrieval_time)
        except Exception as e:
            logger.error(f"Prediction failed for {symbol} with model {model_name}: {e}", exc_info=True)
            return self._create_error_result(symbol, model_name, prediction_start_time, str(e), feature_retrieval_time)

    def _create_error_result(self, symbol: str, model_name: str, start_time: float, error_msg: str, feature_time: float) -> Dict[str, Any]:
        """Helper to create a standardized error result dictionary."""
        return {
            'symbol': symbol,
            'prediction': None,
            'probability': None,
            'confidence': 0.0,
            'error': error_msg,
            'model': model_name,
            'features_used_count': 0,
            'prediction_time': time.time() - start_time,
            'feature_time': feature_time,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
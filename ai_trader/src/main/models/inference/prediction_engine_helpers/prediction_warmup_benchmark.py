# File: src/ai_trader/models/inference/prediction_engine_helpers/prediction_warmup_benchmark.py

import logging
import time
import asyncio # For async sleep if needed in benchmark loops
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

class PredictionWarmupBenchmark:
    """
    Provides utilities for warming up the prediction engine's caches and
    benchmarking its real-time prediction performance.
    """

    def __init__(self, 
                 get_features_func: Callable[[str, str, List[str], int, bool], Awaitable[Optional[pd.DataFrame]]],
                 predict_func: Callable[[str, str, Optional[pd.DataFrame], bool], Dict[str, Any]]):
        """
        Initializes the PredictionWarmupBenchmark.

        Args:
            get_features_func: A callable async function from FeatureDataManager to get features.
                               Signature: (symbol, model_name, feature_names, lookback, use_cache) -> DataFrame
            predict_func: A callable function from PredictionEngine to make a single prediction.
                          Signature: (symbol, model_name, features_df, use_cache) -> Dict[str, Any]
        """
        self._get_features_func = get_features_func
        self._predict_func = predict_func
        logger.debug("PredictionWarmupBenchmark initialized.")

    async def warmup_engine(self, symbols: List[str], model_names: Optional[List[str]] = None, 
                           required_feature_names_map: Dict[str, List[str]] = {},
                           lookback_period_map: Dict[str, int] = {}):
        """
        Warms up the prediction engine by pre-loading features into caches.
        This reduces latency for initial predictions.

        Args:
            symbols: List of symbols to pre-load features for.
            model_names: Optional. List of model names to warmup. If None, uses all loaded models.
            required_feature_names_map: Dict {model_name: list_of_feature_names}.
            lookback_period_map: Dict {model_name: lookback_period}.
        """
        if not model_names:
            logger.warning("No model names provided for warmup. Skipping warmup.")
            return

        logger.info(f"Warming up prediction engine for {len(symbols)} symbols and {len(model_names)} models...")
        
        # Iterate through models and symbols to trigger feature loading
        warmup_tasks = []
        for model_name in model_names:
            # Get model-specific feature requirements and lookback, or fallback
            required_features = required_feature_names_map.get(model_name, ['close', 'volume']) # Fallback features
            lookback = lookback_period_map.get(model_name, 20) # Fallback lookback
            
            for symbol in symbols:
                # Call get_features_func without using cache (to force a fetch)
                # This puts the features into the cache.
                warmup_tasks.append(
                    self._get_features_func(symbol, model_name, required_features, lookback, False)
                )
        
        # Run all warmup tasks concurrently
        if warmup_tasks:
            results = await asyncio.gather(*warmup_tasks, return_exceptions=True)
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.warning(f"Warmup task {i} failed: {res}", exc_info=False) # Log errors but don't stop warmup
        
        logger.info("Warmup complete.")

    async def benchmark_engine(self, n_predictions: int = 100, 
                               symbols: Optional[List[str]] = None,
                               model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Benchmarks the prediction engine's performance.

        Args:
            n_predictions: Total number of predictions to run for the benchmark.
            symbols: Optional. List of symbols to use. Defaults to common stocks if None.
            model_name: Optional. The model name to benchmark. Defaults to the first loaded model.

        Returns:
            A dictionary containing benchmark results (latencies, throughput).
        """
        default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        benchmark_symbols = symbols if symbols else default_symbols
        
        if not benchmark_symbols:
            logger.error("No symbols available for benchmarking.")
            return {'error': 'No symbols for benchmark'}

        if not model_name:
            logger.warning("No specific model name provided for benchmark. Attempting to use a default loaded model.")
            # This class doesn't manage models, so it needs info from PredictionEngine
            # The PredictionEngine will pass it the model name when orchestrating.
            # For standalone testing, you'd need to mock or pass a specific model.
            return {'error': 'Model name not specified for benchmark.'}

        logger.info(f"Running benchmark: {n_predictions} predictions for model '{model_name}'.")
        
        # IMPORTANT: Clearing caches is typically done before a benchmark for clean results.
        # This should be done by the calling PredictionEngine, not by the benchmark helper itself.
        # It's not the responsibility of the benchmark to clear another component's cache.

        # Run predictions concurrently if possible (for true throughput)
        benchmark_tasks = []
        for i in range(n_predictions):
            symbol = benchmark_symbols[i % len(benchmark_symbols)] # Cycle through symbols
            # Pass features=None to force feature retrieval
            benchmark_tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(self._predict_func, symbol, model_name, None, True)
                    # _predict_func is synchronous, so run in a thread pool (asyncio.to_thread)
                )
            )

        start_time = time.time()
        results = await asyncio.gather(*benchmark_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        successful_predictions = [r for r in results if not isinstance(r, Exception) and r.get('prediction') is not None]
        
        # Extract latencies from individual prediction results
        prediction_times = [r['prediction_time'] for r in successful_predictions if 'prediction_time' in r]
        feature_times = [r['feature_time'] for r in successful_predictions if 'feature_time' in r]
        
        def calculate_stats(times: List[float]) -> Dict[str, float]:
            if not times:
                return {'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0}
            times_array = np.array(times)
            return {
                'mean': float(np.mean(times_array)),
                'median': float(np.median(times_array)),
                'p95': float(np.percentile(times_array, 95)),
                'p99': float(np.percentile(times_array, 99))
            }

        benchmark_results = {
            'total_predictions_attempted': n_predictions,
            'successful_predictions_completed': len(successful_predictions),
            'total_benchmark_time_seconds': total_time,
            'avg_time_per_prediction_seconds': total_time / n_predictions if n_predictions > 0 else 0.0,
            'predictions_per_second': n_predictions / total_time if total_time > 0 else 0.0,
            'prediction_latency_stats_seconds': calculate_stats(prediction_times),
            'feature_retrieval_latency_stats_seconds': calculate_stats(feature_times),
            'failed_predictions_count': n_predictions - len(successful_predictions)
        }
        
        logger.info(f"Benchmark complete. Avg time: {benchmark_results['avg_time_per_prediction_seconds']:.4f}s/pred.")
        return benchmark_results
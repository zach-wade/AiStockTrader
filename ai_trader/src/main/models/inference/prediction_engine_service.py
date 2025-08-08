"""
Service layer for managing and monitoring the Prediction Engine.

Provides an external API for operational tasks like warming up caches,
benchmarking performance, and retrieving system status.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional

from .prediction_engine import PredictionEngine
from main.config.config_manager import get_config

logger = logging.getLogger(__name__)


class PredictionEngineService:
    """
    Provides a service layer for operational and monitoring tasks related to the
    Prediction Engine. It orchestrates calls to the core PredictionEngine instance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the service and the core PredictionEngine it manages."""
        self.config = config or get_config()
        self.prediction_engine = PredictionEngine(self.config)
        logger.info("PredictionEngineService initialized.")

    async def warmup_feature_cache(self, symbols: List[str], lookback_days: int = 90):
        """
        Warms up the prediction engine's feature caches by pre-loading features.
        
        Args:
            symbols: List of symbols to pre-load features for.
            lookback_days: How many days of historical features to load.
        """
        logger.info(f"Initiating feature cache warmup for {len(symbols)} symbols...")
        await self.prediction_engine.feature_manager.warmup_cache(symbols, lookback_days)
        logger.info("Feature cache warmup complete.")

    async def benchmark_prediction_latency(self, symbol: str, model_id: str, version: str, n_runs: int = 100) -> Dict[str, float]:
        """
        Benchmarks the end-to-end prediction latency for a specific model.
        
        Args:
            symbol: A symbol to use for the benchmark.
            model_id: The model to benchmark.
            version: The model version to benchmark.
            n_runs: The number of prediction iterations to run.
            
        Returns:
            A dictionary with latency statistics (mean, median, p95, etc.).
        """
        logger.info(f"Starting prediction latency benchmark for {model_id}/{version}...")
        latencies = []
        for _ in range(n_runs):
            start_time = asyncio.get_event_loop().time()
            await self.prediction_engine.predict(symbol, model_id, version)
            end_time = asyncio.get_event_loop().time()
            latencies.append((end_time - start_time) * 1000) # milliseconds

        results = {
            "model_id": model_id,
            "version": version,
            "runs": n_runs,
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "max_latency_ms": np.max(latencies),
        }
        logger.info(f"Benchmark complete: {results}")
        return results
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Retrieves the current status of the prediction engine and its components."""
        return {
            "loaded_models_count": self.prediction_engine.model_loader.get_loaded_model_count(),
            "feature_cache_size_mb": self.prediction_engine.feature_manager.get_cache_size_mb(),
            "last_prediction_timestamp": self.prediction_engine.core_predictor.last_prediction_time,
        }

    def clear_all_caches(self):
        """Clears all internal caches (models and features)."""
        self.prediction_engine.model_loader.clear_cache()
        self.prediction_engine.feature_manager.clear_cache()
        logger.info("All PredictionEngine caches have been cleared.")
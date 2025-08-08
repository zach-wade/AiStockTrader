"""
Latency Optimizer for Prediction Engine

Provides optimization techniques to reduce prediction latency including:
- Model quantization and optimization
- Batch processing optimization
- Caching strategies
- Parallel inference
- Hardware acceleration support
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for latency reduction."""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class LatencyProfile:
    """Profile of latency measurements."""
    feature_extraction_ms: float
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    total_ms: float
    
    @property
    def breakdown(self) -> Dict[str, float]:
        """Get percentage breakdown of latency."""
        return {
            'feature_extraction': self.feature_extraction_ms / self.total_ms * 100,
            'preprocessing': self.preprocessing_ms / self.total_ms * 100,
            'inference': self.inference_ms / self.total_ms * 100,
            'postprocessing': self.postprocessing_ms / self.total_ms * 100
        }


class LatencyOptimizer:
    """Optimizes prediction engine latency."""
    
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
                 enable_profiling: bool = True,
                 max_batch_size: int = 32,
                 max_workers: int = 4):
        """
        Initialize latency optimizer.
        
        Args:
            optimization_level: Level of optimization to apply
            enable_profiling: Whether to enable latency profiling
            max_batch_size: Maximum batch size for processing
            max_workers: Maximum number of workers for parallel processing
        """
        self.optimization_level = optimization_level
        self.enable_profiling = enable_profiling
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        
        # Thread pool for parallel processing
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Profiling data
        self._profiles: List[LatencyProfile] = []
        self._profile_lock = threading.Lock()
        
        # Batch processing queue
        self._batch_queue = queue.Queue()
        self._batch_results = {}
        
        # Optimization settings based on level
        self._settings = self._get_optimization_settings()
    
    def optimize_model(self, model: Any) -> Any:
        """
        Optimize model for inference.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        if self.optimization_level == OptimizationLevel.NONE:
            return model
        
        try:
            # Apply quantization if available
            if self._settings['quantization'] and hasattr(model, 'quantize'):
                logger.info("Applying model quantization")
                model = model.quantize()
            
            # Apply optimization passes if available
            if self._settings['optimization_passes'] and hasattr(model, 'optimize'):
                logger.info("Applying optimization passes")
                model = model.optimize()
            
            # Enable inference mode if available
            if hasattr(model, 'eval'):
                model.eval()
            
            return model
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
            return model
    
    def optimize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize feature DataFrame for processing.
        
        Args:
            features: Features to optimize
            
        Returns:
            Optimized features
        """
        if self.optimization_level == OptimizationLevel.NONE:
            return features
        
        try:
            # Convert to optimal dtypes
            if self._settings['dtype_optimization']:
                features = self._optimize_dtypes(features)
            
            # Remove unnecessary columns
            if self._settings['feature_selection']:
                features = self._select_important_features(features)
            
            # Apply caching if enabled
            if self._settings['feature_caching']:
                features = self._cache_computed_features(features)
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature optimization failed: {e}")
            return features
    
    def batch_predict(self, predict_func: Callable, inputs: List[Any]) -> List[Any]:
        """
        Perform batched prediction with optimization.
        
        Args:
            predict_func: Prediction function
            inputs: List of inputs
            
        Returns:
            List of predictions
        """
        if len(inputs) <= 1 or not self._settings['batch_processing']:
            # Single prediction
            return [predict_func(inp) for inp in inputs]
        
        # Process in optimized batches
        results = []
        batch_size = min(self.max_batch_size, self._settings['optimal_batch_size'])
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            if self._settings['parallel_inference'] and len(batch) > 1:
                # Parallel processing
                batch_results = list(self._thread_pool.map(predict_func, batch))
            else:
                # Sequential processing
                batch_results = [predict_func(inp) for inp in batch]
            
            results.extend(batch_results)
        
        return results
    
    def profile(self, func: Callable, *args, **kwargs) -> Any:
        """
        Profile a function execution.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if not self.enable_profiling:
            return func(*args, **kwargs)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Log latency
        logger.debug(f"{func.__name__} latency: {latency_ms:.2f}ms")
        
        return result
    
    def profile_prediction(self, 
                          feature_extraction_func: Callable,
                          preprocessing_func: Callable,
                          inference_func: Callable,
                          postprocessing_func: Callable,
                          *args, **kwargs) -> Any:
        """
        Profile complete prediction pipeline.
        
        Args:
            feature_extraction_func: Feature extraction function
            preprocessing_func: Preprocessing function
            inference_func: Inference function
            postprocessing_func: Postprocessing function
            *args: Arguments for the functions
            **kwargs: Keyword arguments
            
        Returns:
            Prediction result
        """
        # Feature extraction
        start = time.time()
        features = feature_extraction_func(*args, **kwargs)
        feature_time = (time.time() - start) * 1000
        
        # Preprocessing
        start = time.time()
        processed = preprocessing_func(features)
        preprocess_time = (time.time() - start) * 1000
        
        # Inference
        start = time.time()
        prediction = inference_func(processed)
        inference_time = (time.time() - start) * 1000
        
        # Postprocessing
        start = time.time()
        result = postprocessing_func(prediction)
        postprocess_time = (time.time() - start) * 1000
        
        # Record profile
        profile = LatencyProfile(
            feature_extraction_ms=feature_time,
            preprocessing_ms=preprocess_time,
            inference_ms=inference_time,
            postprocessing_ms=postprocess_time,
            total_ms=feature_time + preprocess_time + inference_time + postprocess_time
        )
        
        with self._profile_lock:
            self._profiles.append(profile)
        
        return result
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get optimization recommendations based on profiling.
        
        Returns:
            Dictionary of recommendations
        """
        if not self._profiles:
            return {"message": "No profiling data available"}
        
        # Analyze profiles
        avg_profile = self._calculate_average_profile()
        bottlenecks = self._identify_bottlenecks(avg_profile)
        
        recommendations = {
            'average_latency_ms': avg_profile.total_ms,
            'latency_breakdown': avg_profile.breakdown,
            'bottlenecks': bottlenecks,
            'recommendations': []
        }
        
        # Generate recommendations based on bottlenecks
        if 'feature_extraction' in bottlenecks:
            recommendations['recommendations'].extend([
                "Consider caching computed features",
                "Reduce feature dimensionality",
                "Use more efficient feature extractors"
            ])
        
        if 'preprocessing' in bottlenecks:
            recommendations['recommendations'].extend([
                "Optimize data type conversions",
                "Use vectorized operations",
                "Consider GPU acceleration for preprocessing"
            ])
        
        if 'inference' in bottlenecks:
            recommendations['recommendations'].extend([
                "Use model quantization",
                "Consider model pruning",
                "Enable GPU inference if available",
                "Use ONNX runtime for optimization"
            ])
        
        if 'postprocessing' in bottlenecks:
            recommendations['recommendations'].extend([
                "Simplify postprocessing logic",
                "Use parallel processing for postprocessing",
                "Cache postprocessing results"
            ])
        
        return recommendations
    
    def _get_optimization_settings(self) -> Dict[str, Any]:
        """Get optimization settings based on level."""
        settings = {
            OptimizationLevel.NONE: {
                'quantization': False,
                'optimization_passes': False,
                'dtype_optimization': False,
                'feature_selection': False,
                'feature_caching': False,
                'batch_processing': False,
                'parallel_inference': False,
                'optimal_batch_size': 1
            },
            OptimizationLevel.BASIC: {
                'quantization': False,
                'optimization_passes': True,
                'dtype_optimization': True,
                'feature_selection': False,
                'feature_caching': False,
                'batch_processing': True,
                'parallel_inference': False,
                'optimal_batch_size': 8
            },
            OptimizationLevel.STANDARD: {
                'quantization': True,
                'optimization_passes': True,
                'dtype_optimization': True,
                'feature_selection': True,
                'feature_caching': True,
                'batch_processing': True,
                'parallel_inference': True,
                'optimal_batch_size': 16
            },
            OptimizationLevel.AGGRESSIVE: {
                'quantization': True,
                'optimization_passes': True,
                'dtype_optimization': True,
                'feature_selection': True,
                'feature_caching': True,
                'batch_processing': True,
                'parallel_inference': True,
                'optimal_batch_size': 32
            },
            OptimizationLevel.EXTREME: {
                'quantization': True,
                'optimization_passes': True,
                'dtype_optimization': True,
                'feature_selection': True,
                'feature_caching': True,
                'batch_processing': True,
                'parallel_inference': True,
                'optimal_batch_size': 64
            }
        }
        
        return settings.get(self.optimization_level, settings[OptimizationLevel.STANDARD])
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory and speed."""
        # Convert float64 to float32 where possible
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            if df[col].abs().max() < 3.4e38:  # float32 max
                df[col] = df[col].astype('float32')
        
        # Convert int64 to smaller ints where possible
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            max_val = df[col].max()
            min_val = df[col].min()
            
            if min_val >= 0 and max_val <= 255:
                df[col] = df[col].astype('uint8')
            elif min_val >= -128 and max_val <= 127:
                df[col] = df[col].astype('int8')
            elif min_val >= 0 and max_val <= 65535:
                df[col] = df[col].astype('uint16')
            elif min_val >= -32768 and max_val <= 32767:
                df[col] = df[col].astype('int16')
            elif min_val >= 0 and max_val <= 4294967295:
                df[col] = df[col].astype('uint32')
            elif min_val >= -2147483648 and max_val <= 2147483647:
                df[col] = df[col].astype('int32')
        
        return df
    
    def _select_important_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only important features (placeholder)."""
        # This would normally use feature importance from model
        # For now, just return as-is
        return df
    
    def _cache_computed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cache computed features (placeholder)."""
        # This would normally implement feature caching
        # For now, just return as-is
        return df
    
    def _calculate_average_profile(self) -> LatencyProfile:
        """Calculate average latency profile."""
        with self._profile_lock:
            if not self._profiles:
                return LatencyProfile(0, 0, 0, 0, 0)
            
            avg_feature = np.mean([p.feature_extraction_ms for p in self._profiles])
            avg_preprocess = np.mean([p.preprocessing_ms for p in self._profiles])
            avg_inference = np.mean([p.inference_ms for p in self._profiles])
            avg_postprocess = np.mean([p.postprocessing_ms for p in self._profiles])
            
            return LatencyProfile(
                feature_extraction_ms=avg_feature,
                preprocessing_ms=avg_preprocess,
                inference_ms=avg_inference,
                postprocessing_ms=avg_postprocess,
                total_ms=avg_feature + avg_preprocess + avg_inference + avg_postprocess
            )
    
    def _identify_bottlenecks(self, profile: LatencyProfile) -> List[str]:
        """Identify bottlenecks in the pipeline."""
        bottlenecks = []
        breakdown = profile.breakdown
        
        # Any stage taking more than 40% is a bottleneck
        threshold = 40.0
        
        for stage, percentage in breakdown.items():
            if percentage > threshold:
                bottlenecks.append(stage)
        
        return bottlenecks
    
    def cleanup(self):
        """Cleanup resources."""
        self._thread_pool.shutdown(wait=True)


# Global optimizer instance
_optimizer = None


def get_latency_optimizer(**kwargs) -> LatencyOptimizer:
    """Get global latency optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = LatencyOptimizer(**kwargs)
    return _optimizer


def optimize_for_latency(optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
    """
    Decorator to optimize function for latency.
    
    Args:
        optimization_level: Level of optimization
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_latency_optimizer(optimization_level=optimization_level)
            return optimizer.profile(func, *args, **kwargs)
        return wrapper
    return decorator
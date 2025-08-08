"""
Function Performance Tracking

Function execution timing and performance monitoring.
"""

import asyncio
import time
import logging
from typing import Dict, Callable, Optional, Any
from functools import wraps
from contextlib import asynccontextmanager
from collections import defaultdict

from .types import FunctionMetrics, PerformanceMetric, MetricType

logger = logging.getLogger(__name__)


class FunctionTracker:
    """Tracks function execution performance."""
    
    def __init__(self):
        """Initialize function tracker."""
        self.function_metrics: Dict[str, FunctionMetrics] = defaultdict(FunctionMetrics)
        self.metric_recorder = None  # Will be set by monitor
        
        logger.debug("Function tracker initialized")
    
    def set_metric_recorder(self, recorder: Callable):
        """Set the metric recorder function."""
        self.metric_recorder = recorder
    
    def time_function(self, func: Callable) -> Callable:
        """
        Decorator to time function execution.
        
        Args:
            func: Function to time
            
        Returns:
            Decorated function
        """
        func_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                self._record_function_metrics(func_name, duration, success)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                self._record_function_metrics(func_name, duration, success)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    def _record_function_metrics(self, func_name: str, duration: float, success: bool):
        """Record function execution metrics."""
        metrics = self.function_metrics[func_name]
        
        # Update function metrics
        metrics.total_calls += 1
        metrics.total_duration += duration
        
        if success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1
        
        metrics.min_duration = min(metrics.min_duration, duration)
        metrics.max_duration = max(metrics.max_duration, duration)
        metrics.recent_durations.append(duration)
        
        # Record as performance metric if recorder is available
        if self.metric_recorder:
            self.metric_recorder(
                name=f"function.{func_name}.duration",
                value=duration,
                metric_type=MetricType.TIMER,
                tags={'function': func_name, 'success': str(success)}
            )
        
        logger.debug(f"Recorded function metrics for {func_name}: {duration:.4f}s (success: {success})")
    
    @asynccontextmanager
    async def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing code blocks.
        
        Args:
            name: Timer name
            tags: Optional tags
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            # Record as performance metric if recorder is available
            if self.metric_recorder:
                self.metric_recorder(
                    name=f"{name}_duration",
                    value=duration,
                    metric_type=MetricType.TIMER,
                    tags=tags or {}
                )
            
            logger.debug(f"Timer '{name}' completed in {duration:.4f}s")
    
    def get_function_summary(self) -> Dict[str, Any]:
        """Get function performance summary."""
        summary = {}
        
        for func_name, metrics in self.function_metrics.items():
            summary[func_name] = {
                'total_calls': metrics.total_calls,
                'successful_calls': metrics.successful_calls,
                'failed_calls': metrics.failed_calls,
                'success_rate': round(metrics.success_rate, 2),
                'avg_duration': round(metrics.avg_duration, 4),
                'min_duration': round(metrics.min_duration if metrics.min_duration != float('inf') else 0, 4),
                'max_duration': round(metrics.max_duration, 4),
                'recent_avg_duration': round(metrics.recent_avg_duration, 4),
                'total_duration': round(metrics.total_duration, 4)
            }
        
        return summary
    
    def get_function_metrics(self, func_name: str) -> Optional[FunctionMetrics]:
        """Get metrics for a specific function."""
        return self.function_metrics.get(func_name)
    
    def get_top_functions_by_calls(self, limit: int = 10) -> Dict[str, Any]:
        """Get top functions by call count."""
        sorted_functions = sorted(
            self.function_metrics.items(),
            key=lambda x: x[1].total_calls,
            reverse=True
        )
        
        return {
            func_name: {
                'total_calls': metrics.total_calls,
                'avg_duration': round(metrics.avg_duration, 4),
                'success_rate': round(metrics.success_rate, 2)
            }
            for func_name, metrics in sorted_functions[:limit]
        }
    
    def get_top_functions_by_duration(self, limit: int = 10) -> Dict[str, Any]:
        """Get top functions by total duration."""
        sorted_functions = sorted(
            self.function_metrics.items(),
            key=lambda x: x[1].total_duration,
            reverse=True
        )
        
        return {
            func_name: {
                'total_duration': round(metrics.total_duration, 4),
                'avg_duration': round(metrics.avg_duration, 4),
                'total_calls': metrics.total_calls
            }
            for func_name, metrics in sorted_functions[:limit]
        }
    
    def get_slowest_functions(self, limit: int = 10) -> Dict[str, Any]:
        """Get slowest functions by average duration."""
        sorted_functions = sorted(
            self.function_metrics.items(),
            key=lambda x: x[1].avg_duration,
            reverse=True
        )
        
        return {
            func_name: {
                'avg_duration': round(metrics.avg_duration, 4),
                'max_duration': round(metrics.max_duration, 4),
                'total_calls': metrics.total_calls,
                'success_rate': round(metrics.success_rate, 2)
            }
            for func_name, metrics in sorted_functions[:limit]
        }
    
    def get_functions_with_errors(self) -> Dict[str, Any]:
        """Get functions that have failed calls."""
        return {
            func_name: {
                'failed_calls': metrics.failed_calls,
                'total_calls': metrics.total_calls,
                'success_rate': round(metrics.success_rate, 2),
                'avg_duration': round(metrics.avg_duration, 4)
            }
            for func_name, metrics in self.function_metrics.items()
            if metrics.failed_calls > 0
        }
    
    def reset_function_metrics(self, func_name: str = None):
        """Reset function metrics."""
        if func_name:
            if func_name in self.function_metrics:
                del self.function_metrics[func_name]
                logger.info(f"Reset metrics for function: {func_name}")
        else:
            self.function_metrics.clear()
            logger.info("Reset all function metrics")
    
    def get_function_count(self) -> int:
        """Get count of tracked functions."""
        return len(self.function_metrics)
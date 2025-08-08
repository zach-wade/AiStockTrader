"""
Memory Monitoring and Management System for AI Trading System

This module provides comprehensive memory monitoring, alerting, and optimization
utilities specifically designed for high-frequency trading operations.
"""

import gc
import os
import psutil
import time
import logging
import threading
import functools
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from contextlib import contextmanager
import weakref

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single point-in-time memory usage snapshot"""
    timestamp: datetime
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory percentage
    available_mb: float  # Available system memory in MB
    swap_percent: float  # Swap usage percentage
    
    # Process-specific metrics
    num_fds: int = 0  # Number of file descriptors
    num_threads: int = 0  # Number of threads
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MemoryThresholds:
    """Memory usage thresholds for alerting"""
    warning_mb: float = 1500  # 1.5GB warning
    critical_mb: float = 2000  # 2GB critical
    warning_percent: float = 75.0  # 75% of system memory
    critical_percent: float = 85.0  # 85% of system memory
    
    # Growth rate thresholds (MB per minute)
    growth_warning: float = 50.0  # 50MB/min
    growth_critical: float = 100.0  # 100MB/min


class MemoryAlert:
    """Memory usage alert"""
    
    def __init__(self, alert_type: str, severity: str, message: str, 
                 current_usage: float, threshold: float, timestamp: datetime = None):
        self.alert_type = alert_type
        self.severity = severity  # 'warning', 'critical'
        self.message = message
        self.current_usage = current_usage
        self.threshold = threshold
        self.timestamp = timestamp or datetime.now()
    
    def __str__(self):
        return f"{self.severity.upper()}: {self.message} (Current: {self.current_usage:.1f}, Threshold: {self.threshold:.1f})"


class MemoryTracker:
    """Track memory usage patterns and detect anomalies"""
    
    def __init__(self, max_history: int = 1000):
        self.snapshots = deque(maxlen=max_history)
        self.function_memory = defaultdict(list)  # Track memory usage per function
        self.object_references = weakref.WeakSet()  # Track large objects
        self.alerts = deque(maxlen=100)
        self.thresholds = MemoryThresholds()
        
        # Performance tracking
        self.gc_stats = {
            'collections': [0, 0, 0],  # Collections per generation
            'last_collection_time': 0,
            'total_collection_time': 0
        }
        
        self._lock = threading.Lock()
    
    def take_snapshot(self, custom_metrics: Dict[str, float] = None) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        process = psutil.Process()
        system_memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = process.memory_info()
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=process.memory_percent(),
            available_mb=system_memory.available / 1024 / 1024,
            swap_percent=swap.percent,
            num_fds=process.num_fds() if hasattr(process, 'num_fds') else 0,
            num_threads=process.num_threads(),
            custom_metrics=custom_metrics or {}
        )
        
        with self._lock:
            self.snapshots.append(snapshot)
            self._check_thresholds(snapshot)
        
        return snapshot
    
    def _check_thresholds(self, snapshot: MemorySnapshot):
        """Check if current usage exceeds thresholds"""
        alerts = []
        
        # Check absolute memory usage
        if snapshot.rss_mb > self.thresholds.critical_mb:
            alerts.append(MemoryAlert(
                'absolute_usage', 'critical',
                f"Memory usage critically high: {snapshot.rss_mb:.1f}MB",
                snapshot.rss_mb, self.thresholds.critical_mb
            ))
        elif snapshot.rss_mb > self.thresholds.warning_mb:
            alerts.append(MemoryAlert(
                'absolute_usage', 'warning',
                f"Memory usage high: {snapshot.rss_mb:.1f}MB",
                snapshot.rss_mb, self.thresholds.warning_mb
            ))
        
        # Check percentage usage
        if snapshot.percent > self.thresholds.critical_percent:
            alerts.append(MemoryAlert(
                'percentage_usage', 'critical',
                f"Memory percentage critically high: {snapshot.percent:.1f}%",
                snapshot.percent, self.thresholds.critical_percent
            ))
        elif snapshot.percent > self.thresholds.warning_percent:
            alerts.append(MemoryAlert(
                'percentage_usage', 'warning',
                f"Memory percentage high: {snapshot.percent:.1f}%",
                snapshot.percent, self.thresholds.warning_percent
            ))
        
        # Check growth rate (if we have enough history)
        if len(self.snapshots) >= 2:
            growth_rate = self._calculate_growth_rate()
            if growth_rate > self.thresholds.growth_critical:
                alerts.append(MemoryAlert(
                    'growth_rate', 'critical',
                    f"Memory growth rate critically high: {growth_rate:.1f}MB/min",
                    growth_rate, self.thresholds.growth_critical
                ))
            elif growth_rate > self.thresholds.growth_warning:
                alerts.append(MemoryAlert(
                    'growth_rate', 'warning',
                    f"Memory growth rate high: {growth_rate:.1f}MB/min",
                    growth_rate, self.thresholds.growth_warning
                ))
        
        # Add alerts and log them
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(str(alert))
    
    def _calculate_growth_rate(self) -> float:
        """Calculate memory growth rate in MB per minute"""
        if len(self.snapshots) < 2:
            return 0.0
        
        # Use last 5 snapshots for growth calculation
        recent_snapshots = list(self.snapshots)[-5:]
        if len(recent_snapshots) < 2:
            return 0.0
        
        oldest = recent_snapshots[0]
        newest = recent_snapshots[-1]
        
        memory_diff = newest.rss_mb - oldest.rss_mb
        time_diff = (newest.timestamp - oldest.timestamp).total_seconds() / 60  # minutes
        
        return memory_diff / max(time_diff, 0.1)  # Avoid division by zero
    
    def record_function_memory(self, function_name: str, memory_usage: float):
        """Record memory usage for a specific function"""
        with self._lock:
            self.function_memory[function_name].append({
                'timestamp': datetime.now(),
                'memory_mb': memory_usage
            })
            
            # Keep only last 50 records per function
            if len(self.function_memory[function_name]) > 50:
                self.function_memory[function_name] = self.function_memory[function_name][-50:]
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary"""
        if not self.snapshots:
            return {'error': 'No memory snapshots available'}
        
        current = self.snapshots[-1]
        
        # Calculate statistics from recent snapshots
        recent_snapshots = list(self.snapshots)[-60:]  # Last 60 snapshots
        rss_values = [s.rss_mb for s in recent_snapshots]
        
        summary = {
            'current': {
                'rss_mb': current.rss_mb,
                'vms_mb': current.vms_mb,
                'percent': current.percent,
                'available_mb': current.available_mb,
                'swap_percent': current.swap_percent,
                'num_fds': current.num_fds,
                'num_threads': current.num_threads
            },
            'statistics': {
                'min_rss_mb': min(rss_values),
                'max_rss_mb': max(rss_values),
                'avg_rss_mb': sum(rss_values) / len(rss_values),
                'growth_rate_mb_per_min': self._calculate_growth_rate()
            },
            'alerts': {
                'active_warnings': len([a for a in self.alerts if a.severity == 'warning']),
                'active_critical': len([a for a in self.alerts if a.severity == 'critical']),
                'recent_alerts': [str(a) for a in list(self.alerts)[-5:]]
            },
            'function_memory': {
                func: {
                    'count': len(records),
                    'avg_mb': sum(r['memory_mb'] for r in records) / len(records),
                    'max_mb': max(r['memory_mb'] for r in records)
                }
                for func, records in self.function_memory.items()
                if records
            },
            'gc_stats': self.gc_stats
        }
        
        return summary
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics"""
        start_time = time.time()
        
        # Record pre-GC state
        collected = [gc.collect(generation) for generation in range(3)]
        
        collection_time = time.time() - start_time
        
        # Update GC stats
        for i, count in enumerate(collected):
            self.gc_stats['collections'][i] += 1
        
        self.gc_stats['last_collection_time'] = collection_time
        self.gc_stats['total_collection_time'] += collection_time
        
        logger.info(f"Garbage collection completed: {collected} objects collected in {collection_time:.3f}s")
        
        return {
            'collected_objects': collected,
            'collection_time': collection_time,
            'total_objects_collected': sum(collected)
        }


class MemoryMonitor:
    """Main memory monitoring and management system"""
    
    def __init__(self, monitoring_interval: float = 30.0, auto_gc_threshold: float = 1500.0):
        self.monitoring_interval = monitoring_interval
        self.auto_gc_threshold = auto_gc_threshold  # MB
        self.tracker = MemoryTracker()
        
        self._monitoring = False
        self._monitor_thread = None
        self._last_gc_time = 0
        
        # Auto-GC settings
        self.auto_gc_enabled = True
        self.gc_cooldown = 300  # 5 minutes between auto-GC
    
    def start_monitoring(self):
        """Start continuous memory monitoring"""
        if self._monitoring:
            logger.warning("Memory monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Memory monitoring started (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                snapshot = self.tracker.take_snapshot()
                
                # Check if auto-GC should be triggered
                if (self.auto_gc_enabled and 
                    snapshot.rss_mb > self.auto_gc_threshold and
                    time.time() - self._last_gc_time > self.gc_cooldown):
                    
                    logger.info(f"Auto-triggering garbage collection (memory: {snapshot.rss_mb:.1f}MB)")
                    self.tracker.force_garbage_collection()
                    self._last_gc_time = time.time()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def memory_profile(self, include_gc: bool = False):
        """Decorator for profiling memory usage of functions"""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Take snapshot before execution
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024
                
                if include_gc:
                    # Force GC before measurement for accurate baseline
                    gc.collect()
                    memory_before = process.memory_info().rss / 1024 / 1024
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    # Take snapshot after execution
                    execution_time = time.time() - start_time
                    memory_after = process.memory_info().rss / 1024 / 1024
                    memory_delta = memory_after - memory_before
                    
                    # Record function memory usage
                    self.tracker.record_function_memory(func.__name__, memory_delta)
                    
                    # Log significant memory usage
                    if abs(memory_delta) > 10:  # 10MB threshold
                        logger.info(f"Function {func.__name__} memory impact: {memory_delta:+.1f}MB "
                                  f"(execution time: {execution_time:.3f}s)")
                
                return result
            
            return wrapper
        return decorator
    
    @contextmanager
    def memory_context(self, operation_name: str, gc_before: bool = False, gc_after: bool = False):
        """Context manager for monitoring memory usage of code blocks"""
        if gc_before:
            gc.collect()
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        logger.debug(f"Starting memory monitoring for: {operation_name}")
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            
            if gc_after:
                gc_stats = self.tracker.force_garbage_collection()
                logger.info(f"Post-operation GC for {operation_name}: {gc_stats['total_objects_collected']} objects")
            
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_delta = memory_after - memory_before
            
            logger.info(f"Memory context {operation_name}: {memory_delta:+.1f}MB "
                       f"(execution time: {execution_time:.3f}s)")
            
            # Record in tracker
            self.tracker.record_function_memory(f"CONTEXT:{operation_name}", memory_delta)
    
    def optimize_dataframe_memory(self, df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        if df.empty:
            return df
        
        original_memory = df.memory_usage(deep=deep).sum() / 1024 / 1024
        
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            # Optimize numeric columns
            if col_type != 'object' and col_type.name != 'category':
                if 'int' in str(col_type):
                    # Downcast integers
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                elif 'float' in str(col_type):
                    # Downcast floats
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
            
            # Convert strings to categories if beneficial
            elif col_type == 'object':
                unique_ratio = len(optimized_df[col].unique()) / len(optimized_df[col])
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
        
        optimized_memory = optimized_df.memory_usage(deep=deep).sum() / 1024 / 1024
        memory_saved = original_memory - optimized_memory
        
        if memory_saved > 0.1:  # Log if significant savings
            logger.info(f"DataFrame memory optimized: {memory_saved:.1f}MB saved "
                       f"({original_memory:.1f}MB -> {optimized_memory:.1f}MB)")
        
        return optimized_df
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report"""
        return self.tracker.get_memory_summary()
    
    def set_thresholds(self, **kwargs):
        """Update memory thresholds"""
        for key, value in kwargs.items():
            if hasattr(self.tracker.thresholds, key):
                setattr(self.tracker.thresholds, key, value)
                logger.info(f"Updated memory threshold {key}: {value}")
    
    def clear_alerts(self):
        """Clear all memory alerts"""
        self.tracker.alerts.clear()
        logger.info("Memory alerts cleared")


# Global memory monitor instance
_memory_monitor = None

def get_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor instance"""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor


# Convenience decorators
def memory_profiled(include_gc: bool = False):
    """Decorator for memory profiling functions"""
    return get_memory_monitor().memory_profile(include_gc=include_gc)


def memory_optimized(func: Callable):
    """Decorator for functions that should trigger GC after execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Trigger garbage collection for memory-intensive operations
            gc.collect()
    
    return wrapper


# Memory management utilities
def optimize_memory_usage():
    """Perform comprehensive memory optimization"""
    monitor = get_memory_monitor()
    
    logger.info("Starting comprehensive memory optimization...")
    
    # Force garbage collection
    gc_stats = monitor.tracker.force_garbage_collection()
    
    # Take snapshot after optimization
    snapshot = monitor.tracker.take_snapshot()
    
    logger.info(f"Memory optimization completed: {gc_stats['total_objects_collected']} objects collected, "
               f"current usage: {snapshot.rss_mb:.1f}MB")
    
    return {
        'gc_stats': gc_stats,
        'current_memory': snapshot.rss_mb,
        'timestamp': datetime.now()
    }
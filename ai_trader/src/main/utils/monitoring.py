"""
Monitoring utilities for the AI Trader system.

This module provides a unified interface to all monitoring utilities including:
- System metrics collection (CPU, memory, disk, network)
- Performance monitoring and function tracking
- Alert management and notification systems
- Memory profiling and leak detection
- Global monitoring coordination
- Metrics export and visualization support

This is the main interface module that imports all monitoring utilities from the 
monitoring/ subdirectory for easy access throughout the system.
"""

# Import and explicitly re-export monitoring utilities
from .monitoring import (
    # Data types
    MetricType,
    AlertLevel,
    PerformanceMetric,
    SystemResources,
    FunctionMetrics,
    Alert,
    
    # Core components
    SystemMetricsCollector,
    AlertManager,
    FunctionTracker,
    PerformanceMonitor,
    
    # Global monitoring interface
    get_global_monitor,
    set_global_monitor,
    reset_global_monitor,
    is_global_monitor_initialized,
    
    # Convenience functions
    record_metric,
    time_function,
    timer,
    start_monitoring,
    stop_monitoring,
    get_system_summary,
    get_function_summary,
    get_alerts_summary,
    set_default_thresholds,
    clear_metrics,
    export_metrics,
    
    # Memory monitoring
    MemoryMonitor,
    MemorySnapshot,
    MemoryThresholds,
    get_memory_monitor,
    memory_profiled
)

# Convenience aliases for common patterns
from .monitoring import record_metric as log_metric
from .monitoring import time_function as profile_function
from .monitoring import get_system_summary as system_status
from .monitoring import get_memory_monitor as memory_status

# Version info
__version__ = "2.0.0"
__author__ = "AI Trader Team"

# Default monitoring configuration
DEFAULT_COLLECTION_INTERVAL = 60  # seconds
DEFAULT_RETENTION_PERIOD = 24 * 60 * 60  # 24 hours in seconds
DEFAULT_ALERT_THRESHOLD_CPU = 80.0  # percent
DEFAULT_ALERT_THRESHOLD_MEMORY = 85.0  # percent
DEFAULT_ALERT_THRESHOLD_DISK = 90.0  # percent
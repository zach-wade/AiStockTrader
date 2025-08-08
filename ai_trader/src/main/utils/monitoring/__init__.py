"""
Monitoring utilities package.

This is the public API for the monitoring module.
All external code should import from here, not from internal modules.
"""

# Import types from the appropriate modules
from .types import (
    AlertLevel,
    PerformanceMetric,
    SystemResources,
    FunctionMetrics,
    Alert
)

# Import metric types from the unified metrics module
from .metrics import (
    MetricType,
    MetricRecord,
    ValidationMetric,
    PipelineStatus
)

from .collectors import SystemMetricsCollector

# Import MetricsCollector from main monitoring module if available
try:
    from main.monitoring.metrics.collector import MetricsCollector
except ImportError:
    # Create a full implementation if not available
    from datetime import datetime, timedelta
    from collections import defaultdict, deque
    from typing import Dict, List, Optional, Any, Tuple
    import threading
    import time
    import json
    
    class MetricsCollector:
        """
        Comprehensive metrics collection system.
        
        Collects, aggregates, and exports various types of metrics
        for monitoring system performance and business logic.
        """
        
        def __init__(self, storage_backend=None, retention_hours=24, 
                     aggregation_intervals=None):
            """
            Initialize metrics collector.
            
            Args:
                storage_backend: Optional backend for persistent storage
                retention_hours: Hours to retain metrics in memory
                aggregation_intervals: List of aggregation intervals in seconds
            """
            self.storage_backend = storage_backend
            self.retention_hours = retention_hours
            self.aggregation_intervals = aggregation_intervals or [60, 300, 3600]
            
            # Metrics storage
            self._metrics = defaultdict(lambda: deque(maxlen=10000))
            self._gauges = {}
            self._counters = defaultdict(int)
            self._histograms = defaultdict(list)
            
            # Thread safety
            self._lock = threading.RLock()
            
            # Background aggregation
            self._aggregation_thread = None
            self._running = False
            
            # Start background tasks
            self._start_background_tasks()
        
        def record_metric(self, name, value, tags=None, timestamp=None):
            """
            Record a metric value.
            
            Args:
                name: Metric name
                value: Metric value
                tags: Optional dict of tags
                timestamp: Optional timestamp (defaults to now)
            """
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            with self._lock:
                metric_key = self._create_metric_key(name, tags)
                self._metrics[metric_key].append({
                    'value': value,
                    'timestamp': timestamp,
                    'tags': tags or {}
                })
                
                # Update storage backend if available
                if self.storage_backend:
                    try:
                        self.storage_backend.store_metric(
                            name, value, tags, timestamp
                        )
                    except Exception as e:
                        # Log error but don't fail
                        pass
        
        def record_gauge(self, name, value, tags=None):
            """Record a gauge metric (current value)."""
            with self._lock:
                metric_key = self._create_metric_key(name, tags)
                self._gauges[metric_key] = {
                    'value': value,
                    'timestamp': datetime.utcnow(),
                    'tags': tags or {}
                }
        
        def increment_counter(self, name, value=1, tags=None):
            """Increment a counter metric."""
            with self._lock:
                metric_key = self._create_metric_key(name, tags)
                self._counters[metric_key] += value
        
        def record_histogram(self, name, value, tags=None):
            """Record a value for histogram calculation."""
            with self._lock:
                metric_key = self._create_metric_key(name, tags)
                self._histograms[metric_key].append(value)
                
                # Keep only recent values
                if len(self._histograms[metric_key]) > 10000:
                    self._histograms[metric_key] = self._histograms[metric_key][-10000:]
        
        def collect_metrics(self, time_range=None, metric_names=None, tags=None):
            """
            Collect metrics based on criteria.
            
            Args:
                time_range: Tuple of (start_time, end_time)
                metric_names: List of metric names to collect
                tags: Dict of tags to filter by
                
            Returns:
                Dict of metrics
            """
            with self._lock:
                collected = {
                    'metrics': {},
                    'gauges': {},
                    'counters': {},
                    'histograms': {}
                }
                
                # Collect time series metrics
                for key, values in self._metrics.items():
                    name, metric_tags = self._parse_metric_key(key)
                    
                    # Apply filters
                    if metric_names and name not in metric_names:
                        continue
                    if tags and not self._tags_match(metric_tags, tags):
                        continue
                    
                    # Filter by time range
                    if time_range:
                        start_time, end_time = time_range
                        filtered_values = [
                            v for v in values
                            if start_time <= v['timestamp'] <= end_time
                        ]
                    else:
                        filtered_values = list(values)
                    
                    if filtered_values:
                        collected['metrics'][key] = filtered_values
                
                # Collect gauges
                for key, gauge in self._gauges.items():
                    name, metric_tags = self._parse_metric_key(key)
                    
                    if metric_names and name not in metric_names:
                        continue
                    if tags and not self._tags_match(metric_tags, tags):
                        continue
                    
                    collected['gauges'][key] = gauge
                
                # Collect counters
                for key, count in self._counters.items():
                    name, metric_tags = self._parse_metric_key(key)
                    
                    if metric_names and name not in metric_names:
                        continue
                    if tags and not self._tags_match(metric_tags, tags):
                        continue
                    
                    collected['counters'][key] = count
                
                # Collect and calculate histogram stats
                for key, values in self._histograms.items():
                    name, metric_tags = self._parse_metric_key(key)
                    
                    if metric_names and name not in metric_names:
                        continue
                    if tags and not self._tags_match(metric_tags, tags):
                        continue
                    
                    if values:
                        import numpy as np
                        collected['histograms'][key] = {
                            'count': len(values),
                            'min': min(values),
                            'max': max(values),
                            'mean': np.mean(values),
                            'median': np.median(values),
                            'p95': np.percentile(values, 95),
                            'p99': np.percentile(values, 99),
                            'std': np.std(values)
                        }
                
                return collected
        
        def get_metric_stats(self, metric_name, tags=None, hours=1):
            """Get statistics for a specific metric."""
            metric_key = self._create_metric_key(metric_name, tags)
            
            with self._lock:
                if metric_key not in self._metrics:
                    return None
                
                # Get recent values
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                recent_values = [
                    v['value'] for v in self._metrics[metric_key]
                    if v['timestamp'] >= cutoff_time
                ]
                
                if not recent_values:
                    return None
                
                import numpy as np
                return {
                    'count': len(recent_values),
                    'min': min(recent_values),
                    'max': max(recent_values),
                    'mean': np.mean(recent_values),
                    'median': np.median(recent_values),
                    'std': np.std(recent_values),
                    'latest': recent_values[-1]
                }
        
        def get_metric_history(self, metric_name, tags=None, hours=1):
            """Get historical values for a metric."""
            metric_key = self._create_metric_key(metric_name, tags)
            
            with self._lock:
                if metric_key not in self._metrics:
                    return []
                
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                return [
                    v for v in self._metrics[metric_key]
                    if v['timestamp'] >= cutoff_time
                ]
        
        def aggregate_metrics(self, metric_name, interval_seconds=60, 
                            aggregation='mean', tags=None):
            """Aggregate metrics over time intervals."""
            metric_key = self._create_metric_key(metric_name, tags)
            
            with self._lock:
                if metric_key not in self._metrics:
                    return []
                
                # Group by time interval
                from collections import defaultdict
                import numpy as np
                
                buckets = defaultdict(list)
                for value in self._metrics[metric_key]:
                    bucket_time = value['timestamp'].replace(
                        second=0, microsecond=0
                    )
                    bucket_time = bucket_time.replace(
                        minute=bucket_time.minute // (interval_seconds // 60) * (interval_seconds // 60)
                    )
                    buckets[bucket_time].append(value['value'])
                
                # Aggregate each bucket
                aggregated = []
                for bucket_time, values in sorted(buckets.items()):
                    if values:
                        if aggregation == 'mean':
                            agg_value = np.mean(values)
                        elif aggregation == 'sum':
                            agg_value = sum(values)
                        elif aggregation == 'max':
                            agg_value = max(values)
                        elif aggregation == 'min':
                            agg_value = min(values)
                        elif aggregation == 'count':
                            agg_value = len(values)
                        else:
                            agg_value = np.mean(values)
                        
                        aggregated.append({
                            'timestamp': bucket_time,
                            'value': agg_value,
                            'count': len(values)
                        })
                
                return aggregated
        
        def export_metrics(self, format='json', metric_names=None):
            """Export metrics in specified format."""
            collected = self.collect_metrics(metric_names=metric_names)
            
            if format == 'json':
                # Convert datetime objects to strings
                def serialize(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: serialize(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [serialize(v) for v in obj]
                    return obj
                
                return json.dumps(serialize(collected), indent=2)
            
            elif format == 'prometheus':
                # Prometheus text format
                lines = []
                
                # Gauges
                for key, gauge in collected['gauges'].items():
                    name, tags = self._parse_metric_key(key)
                    tag_str = self._format_prometheus_tags(tags)
                    lines.append(f'{name}{tag_str} {gauge["value"]}')
                
                # Counters
                for key, count in collected['counters'].items():
                    name, tags = self._parse_metric_key(key)
                    tag_str = self._format_prometheus_tags(tags)
                    lines.append(f'{name}_total{tag_str} {count}')
                
                # Histograms
                for key, stats in collected['histograms'].items():
                    name, tags = self._parse_metric_key(key)
                    tag_str = self._format_prometheus_tags(tags)
                    lines.append(f'{name}_count{tag_str} {stats["count"]}')
                    lines.append(f'{name}_sum{tag_str} {stats["count"] * stats["mean"]}')
                    
                    for quantile in [0.5, 0.95, 0.99]:
                        value = stats.get(f'p{int(quantile*100)}', stats['median'])
                        quantile_tags = tags.copy() if tags else {}
                        quantile_tags['quantile'] = str(quantile)
                        qtag_str = self._format_prometheus_tags(quantile_tags)
                        lines.append(f'{name}{qtag_str} {value}')
                
                return '\n'.join(lines)
            
            else:
                return str(collected)
        
        def clear_old_metrics(self, hours=None):
            """Clear metrics older than specified hours."""
            if hours is None:
                hours = self.retention_hours
            
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            cleared_count = 0
            
            with self._lock:
                # Clear old time series data
                for key in list(self._metrics.keys()):
                    original_len = len(self._metrics[key])
                    self._metrics[key] = deque(
                        (v for v in self._metrics[key] if v['timestamp'] >= cutoff_time),
                        maxlen=10000
                    )
                    cleared_count += original_len - len(self._metrics[key])
                
                # Clear old gauges
                for key in list(self._gauges.keys()):
                    if self._gauges[key]['timestamp'] < cutoff_time:
                        del self._gauges[key]
                        cleared_count += 1
            
            return cleared_count
        
        def reset_counters(self):
            """Reset all counters to zero."""
            with self._lock:
                self._counters.clear()
        
        def shutdown(self):
            """Shutdown the metrics collector."""
            self._running = False
            if self._aggregation_thread:
                self._aggregation_thread.join(timeout=5)
        
        # Helper methods
        def _create_metric_key(self, name, tags):
            """Create a unique key for a metric."""
            if not tags:
                return name
            tag_str = ','.join(f'{k}={v}' for k, v in sorted(tags.items()))
            return f'{name},{tag_str}'
        
        def _parse_metric_key(self, key):
            """Parse metric key into name and tags."""
            parts = key.split(',', 1)
            name = parts[0]
            tags = {}
            
            if len(parts) > 1:
                for tag_pair in parts[1].split(','):
                    if '=' in tag_pair:
                        k, v = tag_pair.split('=', 1)
                        tags[k] = v
            
            return name, tags
        
        def _tags_match(self, metric_tags, filter_tags):
            """Check if metric tags match filter tags."""
            for k, v in filter_tags.items():
                if k not in metric_tags or metric_tags[k] != v:
                    return False
            return True
        
        def _format_prometheus_tags(self, tags):
            """Format tags for Prometheus export."""
            if not tags:
                return ''
            tag_pairs = [f'{k}="{v}"' for k, v in sorted(tags.items())]
            return '{' + ','.join(tag_pairs) + '}'
        
        def _start_background_tasks(self):
            """Start background aggregation tasks."""
            self._running = True
            self._aggregation_thread = threading.Thread(
                target=self._background_aggregation,
                daemon=True
            )
            self._aggregation_thread.start()
        
        def _background_aggregation(self):
            """Background thread for periodic aggregation and cleanup."""
            while self._running:
                try:
                    # Clear old metrics every hour
                    self.clear_old_metrics()
                    
                    # Sleep for 5 minutes
                    for _ in range(300):
                        if not self._running:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    # Log error but continue
                    time.sleep(60)

# AlertManager import removed to avoid circular dependency
# Import it directly where needed: from main.monitoring.alerts.alert_manager import AlertManager

from .function_tracker import FunctionTracker

from .monitor import PerformanceMonitor

from .global_monitor import (
    get_global_monitor,
    set_global_monitor,
    reset_global_monitor,
    is_global_monitor_initialized,
    record_metric,
    time_function,
    timer,
    sync_timer,
    start_monitoring,
    stop_monitoring,
    get_system_summary,
    get_function_summary,
    get_alerts_summary,
    set_default_thresholds,
    clear_metrics,
    export_metrics
)

from .memory import (
    MemoryMonitor,
    MemorySnapshot,
    MemoryThresholds,
    get_memory_monitor,
    memory_profiled
)

from .dashboard_factory import DashboardFactory

# Import metrics utilities from the utils subdirectory
from .metrics_utils import MetricsBuffer, MetricsExporter

__all__ = [
    # Metric Types (from metrics.py)
    'MetricType',
    'MetricRecord', 
    'ValidationMetric',
    'PipelineStatus',
    
    # Alert and Performance Types (from types.py)
    'AlertLevel',
    'PerformanceMetric',
    'SystemResources',
    'FunctionMetrics',
    'Alert',
    
    # Components
    'SystemMetricsCollector',
    'MetricsCollector',
    'MetricsBuffer',
    'MetricsExporter',
    # 'AlertManager', # Removed to avoid circular dependency
    'FunctionTracker',
    'PerformanceMonitor',
    
    # Global Monitor
    'get_global_monitor',
    'set_global_monitor',
    'reset_global_monitor',
    'is_global_monitor_initialized',
    
    # Convenience Functions
    'record_metric',
    'time_function',
    'timer',
    'sync_timer',
    'start_monitoring',
    'stop_monitoring',
    'get_system_summary',
    'get_function_summary',
    'get_alerts_summary',
    'set_default_thresholds',
    'clear_metrics',
    'export_metrics',
    
    # Memory Monitoring
    'MemoryMonitor',
    'MemorySnapshot',
    'MemoryThresholds',
    'get_memory_monitor',
    'memory_profiled',
    
    # Dashboard Factory
    'DashboardFactory'
]
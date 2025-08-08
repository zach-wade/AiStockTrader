"""
Repository Metrics Collector

Collects and reports metrics for repository operations.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from main.utils.core import secure_uniform
import time

from main.utils.core import get_logger
from main.utils.monitoring import record_metric, MetricType

logger = get_logger(__name__)


class RepositoryMetricsCollector:
    """
    Collects metrics for repository operations including performance,
    errors, cache hits, and data volume.
    """
    
    def __init__(
        self,
        repository_name: str,
        enable_metrics: bool = True,
        sample_rate: float = 1.0
    ):
        """
        Initialize the metrics collector.
        
        Args:
            repository_name: Name of the repository
            enable_metrics: Whether to collect metrics
            sample_rate: Sampling rate for metrics (0.0 to 1.0)
        """
        self.repository_name = repository_name
        self.enable_metrics = enable_metrics
        self.sample_rate = sample_rate
        
        # Local statistics
        self.operation_counts = {}
        self.operation_times = {}
        self.error_counts = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
        
        logger.debug(f"MetricsCollector initialized for {repository_name}")
    
    async def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool,
        records: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record metrics for a repository operation.
        
        Args:
            operation: Operation name (e.g., 'get_by_id', 'create')
            duration: Operation duration in seconds
            success: Whether operation succeeded
            records: Number of records affected
            metadata: Optional additional metadata
        """
        if not self.enable_metrics:
            return
        
        # Update local statistics
        if operation not in self.operation_counts:
            self.operation_counts[operation] = {'success': 0, 'failure': 0}
            self.operation_times[operation] = []
        
        if success:
            self.operation_counts[operation]['success'] += 1
        else:
            self.operation_counts[operation]['failure'] += 1
            self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
        
        self.operation_times[operation].append(duration)
        
        # Sample and record to monitoring system
        if secure_uniform(0, 1) <= self.sample_rate:
            try:
                # Record latency histogram
                await record_metric(
                    MetricType.HISTOGRAM,
                    f"repository_operation_duration_seconds",
                    duration,
                    labels={
                        'repository': self.repository_name,
                        'operation': operation,
                        'success': str(success).lower()
                    }
                )
                
                # Record operation counter
                await record_metric(
                    MetricType.COUNTER,
                    f"repository_operations_total",
                    1,
                    labels={
                        'repository': self.repository_name,
                        'operation': operation,
                        'success': str(success).lower()
                    }
                )
                
                # Record records processed if applicable
                if records > 0:
                    await record_metric(
                        MetricType.COUNTER,
                        f"repository_records_processed_total",
                        records,
                        labels={
                            'repository': self.repository_name,
                            'operation': operation
                        }
                    )
                
            except Exception as e:
                logger.debug(f"Failed to record metrics: {e}")
    
    async def record_cache_access(self, hit: bool) -> None:
        """
        Record cache hit or miss.
        
        Args:
            hit: True for cache hit, False for miss
        """
        if not self.enable_metrics:
            return
        
        # Update local statistics
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Record to monitoring system
        try:
            metric_name = "repository_cache_hits_total" if hit else "repository_cache_misses_total"
            await record_metric(
                MetricType.COUNTER,
                metric_name,
                1,
                labels={'repository': self.repository_name}
            )
        except Exception as e:
            logger.debug(f"Failed to record cache metric: {e}")
    
    async def record_error(
        self,
        operation: str,
        error_type: str,
        error_message: str
    ) -> None:
        """
        Record an error occurrence.
        
        Args:
            operation: Operation that failed
            error_type: Type of error
            error_message: Error message
        """
        if not self.enable_metrics:
            return
        
        # Update error counts
        error_key = f"{operation}:{error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Record to monitoring system
        try:
            await record_metric(
                MetricType.COUNTER,
                "repository_errors_total",
                1,
                labels={
                    'repository': self.repository_name,
                    'operation': operation,
                    'error_type': error_type
                }
            )
        except Exception as e:
            logger.debug(f"Failed to record error metric: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Returns:
            Dictionary with collected statistics
        """
        uptime = time.time() - self.start_time
        
        # Calculate operation statistics
        operation_stats = {}
        for op, counts in self.operation_counts.items():
            total = counts['success'] + counts['failure']
            times = self.operation_times.get(op, [])
            
            operation_stats[op] = {
                'total': total,
                'success': counts['success'],
                'failure': counts['failure'],
                'success_rate': counts['success'] / total if total > 0 else 0,
                'avg_duration': sum(times) / len(times) if times else 0,
                'min_duration': min(times) if times else 0,
                'max_duration': max(times) if times else 0
            }
        
        # Calculate cache statistics
        total_cache = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0
        
        return {
            'repository': self.repository_name,
            'uptime_seconds': uptime,
            'operations': operation_stats,
            'cache': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': cache_hit_rate
            },
            'errors': dict(self.error_counts)
        }
    
    def reset_statistics(self) -> None:
        """Reset all collected statistics."""
        self.operation_counts = {}
        self.operation_times = {}
        self.error_counts = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
        
        logger.debug(f"Statistics reset for {self.repository_name}")
    
    async def report_summary(self) -> None:
        """Report a summary of metrics to logs."""
        if not self.enable_metrics:
            return
        
        stats = self.get_statistics()
        
        logger.info(
            f"Repository metrics summary for {self.repository_name}: "
            f"Operations: {sum(op['total'] for op in stats['operations'].values())}, "
            f"Cache hit rate: {stats['cache']['hit_rate']:.2%}, "
            f"Errors: {sum(stats['errors'].values())}"
        )
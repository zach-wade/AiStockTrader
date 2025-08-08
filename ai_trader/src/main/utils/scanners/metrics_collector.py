"""
Scanner metrics collection utilities.

Provides comprehensive metrics tracking for scanner performance,
alert generation, and system health monitoring.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from collections import defaultdict, deque
import asyncio

from main.interfaces.scanners import IScannerMetrics
from main.utils.monitoring import MetricsCollector
from main.utils.core import timer

logger = logging.getLogger(__name__)


@dataclass
class ScannerMetric:
    """Individual scanner metric."""
    scanner_name: str
    metric_type: str
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScannerPerformance:
    """Scanner performance statistics."""
    scanner_name: str
    total_scans: int = 0
    total_symbols_scanned: int = 0
    total_alerts_generated: int = 0
    total_errors: int = 0
    avg_scan_duration_ms: float = 0.0
    p95_scan_duration_ms: float = 0.0
    p99_scan_duration_ms: float = 0.0
    alerts_by_type: Dict[str, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    last_scan_time: Optional[datetime] = None
    success_rate: float = 1.0


class ScannerMetricsCollector(IScannerMetrics):
    """
    Comprehensive metrics collector for scanner operations.
    
    Features:
    - Performance tracking per scanner
    - Alert generation statistics
    - Error tracking and analysis
    - Time-series metrics for monitoring
    - Integration with global metrics system
    """
    
    def __init__(
        self,
        global_metrics: Optional[MetricsCollector] = None,
        retention_minutes: int = 60,
        aggregation_interval_seconds: int = 60
    ):
        """
        Initialize scanner metrics collector.
        
        Args:
            global_metrics: Global metrics collector for integration
            retention_minutes: How long to keep detailed metrics
            aggregation_interval_seconds: Interval for metric aggregation
        """
        self.global_metrics = global_metrics
        self.retention_minutes = retention_minutes
        self.aggregation_interval_seconds = aggregation_interval_seconds
        
        # Performance tracking
        self.scanner_performance: Dict[str, ScannerPerformance] = {}
        
        # Time-series metrics (using deques for automatic size limiting)
        max_metrics = (retention_minutes * 60) // aggregation_interval_seconds
        self.time_series_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_metrics)
        )
        
        # Recent scan durations for percentile calculations
        self.recent_durations: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # Alert tracking
        self.alert_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        # Start aggregation task
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
    
    def record_scan_duration(
        self,
        scanner_name: str,
        duration_ms: float,
        symbol_count: int
    ) -> None:
        """
        Record scan execution time.
        
        Args:
            scanner_name: Name of scanner
            duration_ms: Duration in milliseconds
            symbol_count: Number of symbols scanned
        """
        # Update performance stats
        perf = self._get_or_create_performance(scanner_name)
        perf.total_scans += 1
        perf.total_symbols_scanned += symbol_count
        perf.last_scan_time = datetime.now(timezone.utc)
        
        # Track duration
        self.recent_durations[scanner_name].append(duration_ms)
        
        # Update average duration
        durations = list(self.recent_durations[scanner_name])
        perf.avg_scan_duration_ms = sum(durations) / len(durations)
        
        # Calculate percentiles
        if len(durations) >= 10:
            sorted_durations = sorted(durations)
            p95_idx = int(len(sorted_durations) * 0.95)
            p99_idx = int(len(sorted_durations) * 0.99)
            perf.p95_scan_duration_ms = sorted_durations[p95_idx]
            perf.p99_scan_duration_ms = sorted_durations[p99_idx]
        
        # Record time-series metric
        metric = ScannerMetric(
            scanner_name=scanner_name,
            metric_type='scan_duration',
            timestamp=datetime.now(timezone.utc),
            value=duration_ms,
            tags={'symbol_count': str(symbol_count)}
        )
        self._record_metric(metric)
        
        # Send to global metrics if available
        if self.global_metrics:
            self.global_metrics.record_gauge(
                'scanner.scan_duration_ms',
                duration_ms,
                tags={'scanner': scanner_name}
            )
            self.global_metrics.record_counter(
                'scanner.symbols_scanned',
                symbol_count,
                tags={'scanner': scanner_name}
            )
        
        # Log slow scans
        if duration_ms > 5000:  # > 5 seconds
            logger.warning(
                f"Slow scan detected: {scanner_name} took {duration_ms:.2f}ms "
                f"for {symbol_count} symbols"
            )
    
    def record_alert_generated(
        self,
        scanner_name: str,
        alert_type: str,
        symbol: str,
        confidence: float
    ) -> None:
        """
        Record alert generation.
        
        Args:
            scanner_name: Name of scanner
            alert_type: Type of alert
            symbol: Symbol for alert
            confidence: Alert confidence score
        """
        # Update performance stats
        perf = self._get_or_create_performance(scanner_name)
        perf.total_alerts_generated += 1
        perf.alerts_by_type[alert_type] = perf.alerts_by_type.get(alert_type, 0) + 1
        
        # Track alert counts
        self.alert_counts[scanner_name][alert_type] += 1
        
        # Record metric
        metric = ScannerMetric(
            scanner_name=scanner_name,
            metric_type='alert_generated',
            timestamp=datetime.now(timezone.utc),
            value=confidence,
            tags={
                'alert_type': alert_type,
                'symbol': symbol
            }
        )
        self._record_metric(metric)
        
        # Send to global metrics
        if self.global_metrics:
            self.global_metrics.record_counter(
                'scanner.alerts_generated',
                1,
                tags={
                    'scanner': scanner_name,
                    'alert_type': alert_type,
                    'symbol': symbol
                }
            )
            self.global_metrics.record_gauge(
                'scanner.alert_confidence',
                confidence,
                tags={
                    'scanner': scanner_name,
                    'alert_type': alert_type
                }
            )
    
    def record_scan_error(
        self,
        scanner_name: str,
        error_type: str,
        error_message: str
    ) -> None:
        """
        Record scan error.
        
        Args:
            scanner_name: Name of scanner
            error_type: Type of error
            error_message: Error message
        """
        # Update performance stats
        perf = self._get_or_create_performance(scanner_name)
        perf.total_errors += 1
        perf.errors_by_type[error_type] = perf.errors_by_type.get(error_type, 0) + 1
        
        # Update success rate
        total_attempts = perf.total_scans + perf.total_errors
        if total_attempts > 0:
            perf.success_rate = perf.total_scans / total_attempts
        
        # Record metric
        metric = ScannerMetric(
            scanner_name=scanner_name,
            metric_type='scan_error',
            timestamp=datetime.now(timezone.utc),
            value=1.0,
            tags={
                'error_type': error_type,
                'error_message': error_message[:100]  # Truncate long messages
            }
        )
        self._record_metric(metric)
        
        # Send to global metrics
        if self.global_metrics:
            self.global_metrics.record_counter(
                'scanner.errors',
                1,
                tags={
                    'scanner': scanner_name,
                    'error_type': error_type
                }
            )
        
        # Log error
        logger.error(f"Scanner error in {scanner_name}: {error_type} - {error_message}")
    
    def get_metrics_summary(
        self,
        scanner_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metrics summary for scanner(s).
        
        Args:
            scanner_name: Specific scanner (None for all)
            
        Returns:
            Dictionary with metrics summary
        """
        if scanner_name:
            # Single scanner summary
            perf = self.scanner_performance.get(scanner_name)
            if not perf:
                return {'error': f'No metrics for scanner: {scanner_name}'}
            
            return self._format_performance_summary(perf)
        else:
            # All scanners summary
            summaries = {}
            for name, perf in self.scanner_performance.items():
                summaries[name] = self._format_performance_summary(perf)
            
            # Add aggregate statistics
            total_scans = sum(p.total_scans for p in self.scanner_performance.values())
            total_alerts = sum(p.total_alerts_generated for p in self.scanner_performance.values())
            total_errors = sum(p.total_errors for p in self.scanner_performance.values())
            
            return {
                'scanners': summaries,
                'aggregate': {
                    'total_scanners': len(self.scanner_performance),
                    'total_scans': total_scans,
                    'total_alerts': total_alerts,
                    'total_errors': total_errors,
                    'overall_success_rate': (
                        total_scans / (total_scans + total_errors)
                        if (total_scans + total_errors) > 0 else 0
                    )
                }
            }
    
    def get_time_series_metrics(
        self,
        scanner_name: str,
        metric_type: str,
        minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get time series metrics for a scanner.
        
        Args:
            scanner_name: Scanner name
            metric_type: Type of metric
            minutes: Minutes of history
            
        Returns:
            List of metric data points
        """
        key = f"{scanner_name}:{metric_type}"
        metrics = self.time_series_metrics.get(key, deque())
        
        # Filter by time window
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        return [
            {
                'timestamp': m.timestamp.isoformat(),
                'value': m.value,
                'tags': m.tags
            }
            for m in metrics
            if m.timestamp >= cutoff_time
        ]
    
    def get_alert_statistics(
        self,
        scanner_name: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get alert generation statistics.
        
        Args:
            scanner_name: Specific scanner (None for all)
            hours: Hours of history
            
        Returns:
            Alert statistics
        """
        stats = {}
        
        if scanner_name:
            counts = self.alert_counts.get(scanner_name, {})
            stats[scanner_name] = dict(counts)
        else:
            stats = {
                name: dict(counts)
                for name, counts in self.alert_counts.items()
            }
        
        # Add totals
        total_by_type = defaultdict(int)
        for scanner_counts in stats.values():
            for alert_type, count in scanner_counts.items():
                total_by_type[alert_type] += count
        
        return {
            'by_scanner': stats,
            'by_type': dict(total_by_type),
            'total': sum(total_by_type.values())
        }
    
    async def _aggregation_loop(self) -> None:
        """Periodic metrics aggregation."""
        while True:
            try:
                await asyncio.sleep(self.aggregation_interval_seconds)
                
                # Clean old metrics
                cutoff_time = datetime.now(timezone.utc) - timedelta(
                    minutes=self.retention_minutes
                )
                
                # This would clean up old time series data
                # For now, deques handle this automatically with maxlen
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
    
    def _get_or_create_performance(self, scanner_name: str) -> ScannerPerformance:
        """Get or create performance stats for scanner."""
        if scanner_name not in self.scanner_performance:
            self.scanner_performance[scanner_name] = ScannerPerformance(
                scanner_name=scanner_name
            )
        return self.scanner_performance[scanner_name]
    
    def _record_metric(self, metric: ScannerMetric) -> None:
        """Record metric in time series."""
        key = f"{metric.scanner_name}:{metric.metric_type}"
        self.time_series_metrics[key].append(metric)
    
    def _format_performance_summary(self, perf: ScannerPerformance) -> Dict[str, Any]:
        """Format performance stats for output."""
        return {
            'scanner_name': perf.scanner_name,
            'total_scans': perf.total_scans,
            'total_symbols_scanned': perf.total_symbols_scanned,
            'total_alerts_generated': perf.total_alerts_generated,
            'total_errors': perf.total_errors,
            'success_rate': round(perf.success_rate, 4),
            'performance': {
                'avg_scan_duration_ms': round(perf.avg_scan_duration_ms, 2),
                'p95_scan_duration_ms': round(perf.p95_scan_duration_ms, 2),
                'p99_scan_duration_ms': round(perf.p99_scan_duration_ms, 2),
            },
            'alerts_by_type': dict(perf.alerts_by_type),
            'errors_by_type': dict(perf.errors_by_type),
            'last_scan_time': (
                perf.last_scan_time.isoformat() if perf.last_scan_time else None
            ),
            'alerts_per_scan': (
                round(perf.total_alerts_generated / perf.total_scans, 2)
                if perf.total_scans > 0 else 0
            ),
            'symbols_per_scan': (
                round(perf.total_symbols_scanned / perf.total_scans, 2)
                if perf.total_scans > 0 else 0
            )
        }
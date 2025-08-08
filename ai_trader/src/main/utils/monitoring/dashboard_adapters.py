"""
Dashboard Adapters for Utils Monitoring

This module provides adapters that allow dashboards to use the utils monitoring
system directly, eliminating the need for duplicate monitoring implementations.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from main.utils.core import get_logger
from main.utils.database import DatabasePool

from .global_monitor import get_global_monitor
from .migration import MigrationMonitor
from .types import MetricType, AlertLevel, Alert

logger = get_logger(__name__)


class DashboardMetricsAdapter:
    """
    Adapter that provides dashboard-friendly interface to utils monitoring.
    
    This allows dashboards to use the centralized monitoring system without
    depending on the separate monitoring module implementations.
    """
    
    def __init__(self, monitor: Optional[MigrationMonitor] = None):
        """
        Initialize dashboard metrics adapter.
        
        Args:
            monitor: Optional monitor instance (uses global if not provided)
        """
        self.monitor = monitor or get_global_monitor()
    
    async def get_system_health_score(self) -> Dict[str, Any]:
        """Get system health score in dashboard format."""
        if hasattr(self.monitor, 'get_system_health_score'):
            return self.monitor.get_system_health_score()
        else:
            # Fallback for basic monitor
            summary = self.monitor.get_system_summary()
            alerts = self.monitor.alert_manager.get_active_alerts()
            
            # Calculate basic health score
            cpu_usage = summary.get('current', {}).get('cpu_percent', 0)
            memory_usage = summary.get('current', {}).get('memory_percent', 0)
            
            score = 100
            if cpu_usage > 90:
                score -= 30
            elif cpu_usage > 70:
                score -= 15
            
            if memory_usage > 90:
                score -= 30
            elif memory_usage > 80:
                score -= 15
            
            score -= len(alerts) * 10
            score = max(0, score)
            
            return {
                'overall_score': score,
                'status': 'healthy' if score > 80 else 'warning' if score > 50 else 'critical',
                'active_alerts': len(alerts),
                'metric_scores': {
                    'cpu': {'score': 100 - cpu_usage, 'value': cpu_usage},
                    'memory': {'score': 100 - memory_usage, 'value': memory_usage}
                }
            }
    
    async def get_metric_value(
        self,
        name: str,
        aggregation: str = "last",
        period_minutes: int = 5,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Get metric value with aggregation."""
        if hasattr(self.monitor, 'get_metric_value'):
            return await self.monitor.get_metric_value(
                name, aggregation, period_minutes, tags
            )
        else:
            # Fallback for basic monitor
            history = self.monitor.get_metric_history(name, minutes=period_minutes)
            if not history:
                return None
            
            values = [m.value for m in history]
            if aggregation == "last":
                return values[-1] if values else None
            elif aggregation == "avg":
                return sum(values) / len(values) if values else None
            elif aggregation == "min":
                return min(values) if values else None
            elif aggregation == "max":
                return max(values) if values else None
            elif aggregation == "sum":
                return sum(values) if values else None
            else:
                return None
    
    async def get_metric_series(
        self,
        name: str,
        period_minutes: int = 60,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Get metric time series data."""
        if hasattr(self.monitor, 'get_metric_series'):
            return await self.monitor.get_metric_series(name, period_minutes, tags)
        else:
            # Fallback for basic monitor
            history = self.monitor.get_metric_history(name, minutes=period_minutes)
            return [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'value': m.value,
                    'tags': {}
                }
                for m in history
            ]
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return self.monitor.alert_manager.get_active_alerts()
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric (dashboard convenience method)."""
        self.monitor.record_metric(name, value, MetricType.GAUGE, tags)
    
    def get_current_value(self, name: str) -> Optional[float]:
        """Get current metric value (synchronous for dashboard widgets)."""
        history = self.monitor.get_metric_history(name, minutes=1)
        return history[-1].value if history else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        if hasattr(self.monitor, 'enhanced_monitor') and self.monitor.enhanced_monitor:
            definitions = self.monitor.enhanced_monitor.get_metric_definitions()
            return {
                'registered_metrics': len(definitions),
                'metrics_processed': sum(
                    len(metrics) 
                    for metrics in self.monitor.enhanced_monitor._metrics.values()
                ),
                'enhanced_features': True
            }
        else:
            return {
                'registered_metrics': 0,
                'metrics_processed': len(self.monitor.metrics),
                'enhanced_features': False
            }


class DashboardHealthReporter:
    """
    Health reporter adapter for dashboards.
    
    Provides health reporting functionality using utils monitoring.
    """
    
    def __init__(self, adapter: DashboardMetricsAdapter):
        """Initialize health reporter."""
        self.adapter = adapter
        self.monitor = adapter.monitor
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        # Get health score
        health_score = await self.adapter.get_system_health_score()
        
        # Get system metrics
        system_summary = self.monitor.get_system_summary()
        
        # Get alerts
        alerts = await self.adapter.get_active_alerts()
        
        # Get metric statistics
        stats = self.adapter.get_statistics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,
            'system_metrics': {
                'cpu': {
                    'current': system_summary.get('current', {}).get('cpu_percent', 0),
                    'average': system_summary.get('average', {}).get('cpu_percent', 0)
                },
                'memory': {
                    'current': system_summary.get('current', {}).get('memory_percent', 0),
                    'average': system_summary.get('average', {}).get('memory_percent', 0),
                    'used_mb': system_summary.get('current', {}).get('memory_used_mb', 0)
                },
                'disk': {
                    'usage_percent': system_summary.get('current', {}).get('disk_usage_percent', 0)
                }
            },
            'active_alerts': [
                {
                    'message': alert.message,
                    'level': alert.level.value,
                    'timestamp': alert.timestamp.isoformat(),
                    'source': alert.source
                }
                for alert in alerts
            ],
            'statistics': stats,
            'recommendations': self._generate_recommendations(health_score, alerts)
        }
    
    def _generate_recommendations(
        self,
        health_score: Dict[str, Any],
        alerts: List[Alert]
    ) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        if health_score['overall_score'] < 50:
            recommendations.append("System health is critical. Immediate attention required.")
        elif health_score['overall_score'] < 80:
            recommendations.append("System health is degraded. Review active alerts.")
        
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            recommendations.append(f"Address {len(critical_alerts)} critical alerts immediately.")
        
        return recommendations


class DashboardPerformanceTracker:
    """
    Performance tracker adapter for dashboards.
    
    Provides performance tracking using utils monitoring.
    """
    
    def __init__(self, adapter: DashboardMetricsAdapter):
        """Initialize performance tracker."""
        self.adapter = adapter
    
    async def get_performance_metrics(self, timeframe: str = "1h") -> Dict[str, Any]:
        """Get performance metrics for specified timeframe."""
        # Convert timeframe to minutes
        timeframe_map = {
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080
        }
        minutes = timeframe_map.get(timeframe, 60)
        
        # Get various metrics
        metrics = {}
        
        # API performance
        api_latency = await self.adapter.get_metric_value(
            "api.request_duration", "avg", minutes
        )
        if api_latency:
            metrics['api_latency_ms'] = api_latency
        
        api_requests = await self.adapter.get_metric_value(
            "api.request_count", "sum", minutes
        )
        if api_requests:
            metrics['api_requests_total'] = api_requests
        
        api_errors = await self.adapter.get_metric_value(
            "api.error_count", "sum", minutes
        )
        if api_errors:
            metrics['api_errors_total'] = api_errors
            if api_requests and api_requests > 0:
                metrics['api_error_rate'] = api_errors / api_requests
        
        # System performance
        cpu_avg = await self.adapter.get_metric_value(
            "system.cpu_usage", "avg", minutes
        )
        if cpu_avg:
            metrics['cpu_usage_avg'] = cpu_avg
        
        memory_avg = await self.adapter.get_metric_value(
            "system.memory_usage", "avg", minutes
        )
        if memory_avg:
            metrics['memory_usage_avg'] = memory_avg
        
        return {
            'timeframe': timeframe,
            'period_minutes': minutes,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }


def create_dashboard_adapter(
    db_pool: Optional[DatabasePool] = None
) -> DashboardMetricsAdapter:
    """
    Create a dashboard metrics adapter.
    
    Args:
        db_pool: Optional database pool for enhanced features
    
    Returns:
        DashboardMetricsAdapter instance
    """
    # Get or create monitor with appropriate features
    monitor = get_global_monitor()
    
    # If we have a db_pool and the monitor doesn't have enhanced features,
    # recreate it with enhanced features
    if db_pool and not getattr(monitor, 'use_enhanced', False):
        from .migration import create_monitor
        from .global_monitor import set_global_monitor
        
        enhanced_monitor = create_monitor(db_pool=db_pool)
        set_global_monitor(enhanced_monitor)
        monitor = enhanced_monitor
    
    return DashboardMetricsAdapter(monitor)


# Convenience function for dashboards
def get_dashboard_adapter() -> DashboardMetricsAdapter:
    """Get a dashboard adapter using the global monitor."""
    return DashboardMetricsAdapter()
"""
Database Pool Health Monitoring

Focused module for monitoring connection pool health and generating recommendations.
"""

import logging
from typing import Dict, Any, List
from .connection_metrics import ConnectionHealthStatus, MetricsCollector

logger = logging.getLogger(__name__)


class PoolHealthMonitor:
    """Monitor database connection pool health and generate recommendations"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def assess_health(self, pool_info: Dict[str, Any]) -> ConnectionHealthStatus:
        """Assess comprehensive health status of the connection pool"""
        metrics = self.metrics_collector.get_metrics_snapshot()
        
        # Calculate health indicators
        pool_size = pool_info.get('pool_size', 1)
        active_connections = metrics['pool_status']['active_connections']
        pool_utilization = (active_connections / max(pool_size, 1)) * 100
        
        avg_response_time = metrics['performance_metrics']['recent_avg_time'] * 1000  # Convert to ms
        slow_query_rate = metrics['performance_metrics']['slow_query_rate']
        
        # Determine health status
        is_healthy = True
        warnings = []
        recommendations = []
        
        # Check pool utilization
        if pool_utilization > 80:
            warnings.append(f"High pool utilization: {pool_utilization:.1f}%")
            if pool_utilization > 90:
                is_healthy = False
                recommendations.append("Consider increasing pool_size or max_overflow")
        
        # Check response times
        if avg_response_time > 100:  # 100ms threshold
            warnings.append(f"High average response time: {avg_response_time:.1f}ms")
            if avg_response_time > 500:
                is_healthy = False
                recommendations.append("Investigate slow queries and database performance")
        
        # Check slow query rate
        if slow_query_rate > 10:  # 10% threshold
            warnings.append(f"High slow query rate: {slow_query_rate:.1f}%")
            if slow_query_rate > 25:
                is_healthy = False
                recommendations.append("Optimize slow queries and add database indexes")
        
        # Check for errors
        error_count = (
            metrics['error_metrics']['connection_errors'] + 
            metrics['error_metrics']['connection_timeouts'] + 
            metrics['error_metrics']['pool_exhaustions']
        )
        
        total_operations = metrics['performance_metrics']['total_queries'] + error_count
        error_rate = (error_count / max(total_operations, 1)) * 100
        
        if error_rate > 5:  # 5% error rate threshold
            warnings.append(f"High error rate: {error_rate:.1f}%")
            if error_rate > 15:
                is_healthy = False
                recommendations.append("Investigate connection errors and database connectivity")
        
        return ConnectionHealthStatus(
            is_healthy=is_healthy,
            pool_utilization=pool_utilization,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def check_connection_leaks(self, pool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential connection leaks"""
        metrics = self.metrics_collector.get_metrics_snapshot()
        
        pool_size = pool_info.get('pool_size', 0)
        max_overflow = pool_info.get('max_overflow', 0)
        checked_out = metrics['pool_status']['active_connections']
        overflow = metrics['pool_status']['overflow_connections']
        
        leak_indicators = []
        
        if checked_out > pool_size * 0.8:
            leak_indicators.append(
                f"High number of checked out connections: {checked_out}/{pool_size}"
            )
        
        if overflow > max_overflow * 0.5:
            leak_indicators.append(
                f"High overflow usage: {overflow}/{max_overflow}"
            )
        
        # Check for long-running connections
        uptime_hours = metrics['uptime_info']['uptime_seconds'] / 3600
        if checked_out > 0 and uptime_hours > 1:
            total_queries = metrics['performance_metrics']['total_queries']
            avg_checkout_time = uptime_hours / max(total_queries, 1)
            if avg_checkout_time > 0.1:  # 6 minutes average is suspicious
                leak_indicators.append(
                    f"Suspicious average connection checkout time: {avg_checkout_time:.2f} hours"
                )
        
        return {
            'potential_leaks': len(leak_indicators) > 0,
            'indicators': leak_indicators,
            'pool_status': {
                'checked_out': checked_out,
                'pool_size': pool_size,
                'overflow': overflow,
                'max_overflow': max_overflow
            },
            'recommendations': [
                "Ensure all database connections are properly closed",
                "Use context managers or try/finally blocks for connection handling",
                "Monitor for long-running queries that might hold connections"
            ] if leak_indicators else []
        }
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current metrics"""
        metrics = self.metrics_collector.get_metrics_snapshot()
        recommendations = []
        
        # Performance-based recommendations
        slow_query_rate = metrics['performance_metrics']['slow_query_rate']
        if slow_query_rate > 15:
            recommendations.append(
                "High slow query rate detected. Consider implementing query caching."
            )
        
        avg_response_time = metrics['performance_metrics']['recent_avg_time']
        if avg_response_time > 0.2:  # 200ms
            recommendations.append(
                "High average response time. Consider connection pool tuning."
            )
        
        # Error-based recommendations
        error_metrics = metrics['error_metrics']
        if error_metrics['connection_timeouts'] > 10:
            recommendations.append(
                "Frequent connection timeouts. Consider increasing timeout values."
            )
        
        if error_metrics['pool_exhaustions'] > 5:
            recommendations.append(
                "Pool exhaustion detected. Consider increasing pool size."
            )
        
        # Usage-based recommendations
        pool_status = metrics['pool_status']
        if pool_status['overflow_connections'] > 0:
            recommendations.append(
                "Overflow connections in use. Monitor for traffic spikes."
            )
        
        if not recommendations:
            recommendations.append("Database pool performance appears optimal.")
        
        return recommendations
"""
Database Query Performance Tracker

This module provides comprehensive query performance tracking, analysis, and optimization
suggestions for database operations. Integrates with the database pool to monitor
query execution times, identify slow queries, and provide optimization recommendations.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import re
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Database query types."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    DROP = "DROP"
    ALTER = "ALTER"
    UNKNOWN = "UNKNOWN"


class QueryPriority(Enum):
    """Query performance priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QueryExecution:
    """Individual query execution record."""
    query_hash: str
    normalized_query: str
    execution_time: float
    timestamp: datetime
    query_type: QueryType
    params_count: int
    result_count: Optional[int] = None
    error: Optional[str] = None
    connection_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_hash': self.query_hash,
            'normalized_query': self.normalized_query,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'query_type': self.query_type.value,
            'params_count': self.params_count,
            'result_count': self.result_count,
            'error': self.error,
            'connection_id': self.connection_id
        }


@dataclass
class QueryStats:
    """Aggregated statistics for a query pattern."""
    query_hash: str
    normalized_query: str
    query_type: QueryType
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    error_count: int = 0
    last_execution: Optional[datetime] = None
    recent_executions: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return self.total_time / self.execution_count if self.execution_count > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Error rate percentage."""
        return (self.error_count / self.execution_count * 100) if self.execution_count > 0 else 0.0
    
    @property
    def recent_avg_time(self) -> float:
        """Recent average execution time."""
        if not self.recent_executions:
            return 0.0
        return sum(self.recent_executions) / len(self.recent_executions)
    
    def get_priority(self) -> QueryPriority:
        """Determine query optimization priority."""
        if self.error_rate > 10:
            return QueryPriority.CRITICAL
        elif self.avg_time > 5.0:
            return QueryPriority.HIGH
        elif self.avg_time > 1.0 or self.execution_count > 1000:
            return QueryPriority.MEDIUM
        else:
            return QueryPriority.LOW


@dataclass
class QueryAnalysis:
    """Query analysis and optimization suggestions."""
    query_hash: str
    normalized_query: str
    priority: QueryPriority
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    estimated_improvement: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_hash': self.query_hash,
            'normalized_query': self.normalized_query,
            'priority': self.priority.value,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'estimated_improvement': self.estimated_improvement
        }


class QueryTracker:
    """
    Database query performance tracker.
    
    Monitors query execution times, identifies slow queries, and provides
    optimization suggestions based on query patterns and performance metrics.
    """
    
    def __init__(self, 
                 slow_query_threshold: float = 1.0,
                 history_size: int = 10000,
                 enable_analysis: bool = True):
        """
        Initialize query tracker.
        
        Args:
            slow_query_threshold: Threshold in seconds for slow queries
            history_size: Maximum number of query executions to track
            enable_analysis: Whether to enable query analysis
        """
        self.slow_query_threshold = slow_query_threshold
        self.history_size = history_size
        self.enable_analysis = enable_analysis
        
        # Query tracking
        self.query_stats: Dict[str, QueryStats] = {}
        self.query_executions: deque = deque(maxlen=history_size)
        self.slow_queries: deque = deque(maxlen=1000)
        
        # Analysis results
        self.query_analyses: Dict[str, QueryAnalysis] = {}
        
        # Performance metrics
        self.total_queries = 0
        self.total_query_time = 0.0
        self.error_count = 0
        
        # Callbacks
        self.slow_query_callbacks: List[Callable[[QueryExecution], None]] = []
        self.analysis_callbacks: List[Callable[[QueryAnalysis], None]] = []
        
        logger.info(f"Query tracker initialized with {slow_query_threshold}s slow query threshold")
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query for pattern matching.
        
        Removes parameters and normalizes formatting to group similar queries.
        
        Args:
            query: Original SQL query
            
        Returns:
            Normalized query pattern
        """
        # Remove extra whitespace and newlines
        normalized = re.sub(r'\s+', ' ', query.strip())
        
        # Replace parameter placeholders with generic markers
        normalized = re.sub(r'\$\d+', '?', normalized)  # PostgreSQL parameters
        normalized = re.sub(r'%\([^)]+\)s', '?', normalized)  # Python string formatting
        normalized = re.sub(r"'[^']*'", "'?'", normalized)  # String literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)  # Numeric literals
        
        # Normalize IN clauses
        normalized = re.sub(r'IN\s*\([^)]+\)', 'IN (?)', normalized, flags=re.IGNORECASE)
        
        # Normalize common patterns
        normalized = re.sub(r'LIMIT\s+\d+', 'LIMIT ?', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'OFFSET\s+\d+', 'OFFSET ?', normalized, flags=re.IGNORECASE)
        
        return normalized.upper()
    
    def get_query_hash(self, normalized_query: str) -> str:
        """Generate hash for normalized query."""
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def detect_query_type(self, query: str) -> QueryType:
        """Detect query type from SQL statement."""
        query_upper = query.strip().upper()
        
        if query_upper.startswith('SELECT'):
            return QueryType.SELECT
        elif query_upper.startswith('INSERT'):
            return QueryType.INSERT
        elif query_upper.startswith('UPDATE'):
            return QueryType.UPDATE
        elif query_upper.startswith('DELETE'):
            return QueryType.DELETE
        elif query_upper.startswith('CREATE'):
            return QueryType.CREATE
        elif query_upper.startswith('DROP'):
            return QueryType.DROP
        elif query_upper.startswith('ALTER'):
            return QueryType.ALTER
        else:
            return QueryType.UNKNOWN
    
    def track_query(self, 
                   query: str, 
                   execution_time: float,
                   params: Optional[List[Any]] = None,
                   result_count: Optional[int] = None,
                   error: Optional[str] = None,
                   connection_id: Optional[str] = None):
        """
        Track a query execution.
        
        Args:
            query: SQL query string
            execution_time: Execution time in seconds
            params: Query parameters
            result_count: Number of results returned
            error: Error message if query failed
            connection_id: Database connection identifier
        """
        # Normalize query and generate hash
        normalized_query = self.normalize_query(query)
        query_hash = self.get_query_hash(normalized_query)
        query_type = self.detect_query_type(query)
        
        # Create execution record
        execution = QueryExecution(
            query_hash=query_hash,
            normalized_query=normalized_query,
            execution_time=execution_time,
            timestamp=datetime.now(),
            query_type=query_type,
            params_count=len(params) if params else 0,
            result_count=result_count,
            error=error,
            connection_id=connection_id
        )
        
        # Store execution
        self.query_executions.append(execution)
        
        # Update statistics
        self._update_query_stats(execution)
        
        # Update global metrics
        self.total_queries += 1
        self.total_query_time += execution_time
        
        if error:
            self.error_count += 1
        
        # Check for slow queries
        if execution_time >= self.slow_query_threshold:
            self.slow_queries.append(execution)
            self._fire_slow_query_callbacks(execution)
            logger.warning(f"Slow query detected: {execution_time:.3f}s - {normalized_query[:100]}...")
        
        # Trigger analysis if enabled
        if self.enable_analysis and query_hash not in self.query_analyses:
            asyncio.create_task(self._analyze_query(query_hash))
    
    def _update_query_stats(self, execution: QueryExecution):
        """Update statistics for a query pattern."""
        query_hash = execution.query_hash
        
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = QueryStats(
                query_hash=query_hash,
                normalized_query=execution.normalized_query,
                query_type=execution.query_type
            )
        
        stats = self.query_stats[query_hash]
        stats.execution_count += 1
        stats.total_time += execution.execution_time
        stats.min_time = min(stats.min_time, execution.execution_time)
        stats.max_time = max(stats.max_time, execution.execution_time)
        stats.last_execution = execution.timestamp
        stats.recent_executions.append(execution.execution_time)
        
        if execution.error:
            stats.error_count += 1
    
    async def _analyze_query(self, query_hash: str):
        """Analyze a query pattern for optimization opportunities."""
        if query_hash not in self.query_stats:
            return
        
        stats = self.query_stats[query_hash]
        analysis = QueryAnalysis(
            query_hash=query_hash,
            normalized_query=stats.normalized_query,
            priority=stats.get_priority()
        )
        
        # Analyze query for issues and recommendations
        self._analyze_query_issues(stats, analysis)
        self._generate_recommendations(stats, analysis)
        
        # Store analysis
        self.query_analyses[query_hash] = analysis
        
        # Fire callbacks
        self._fire_analysis_callbacks(analysis)
    
    def _analyze_query_issues(self, stats: QueryStats, analysis: QueryAnalysis):
        """Analyze query for potential issues."""
        
        # High execution time
        if stats.avg_time > self.slow_query_threshold:
            analysis.issues.append(f"High average execution time: {stats.avg_time:.3f}s")
        
        # High error rate
        if stats.error_rate > 5:
            analysis.issues.append(f"High error rate: {stats.error_rate:.1f}%")
        
        # High frequency
        if stats.execution_count > 1000:
            analysis.issues.append(f"High frequency query: {stats.execution_count} executions")
        
        # Performance degradation
        if len(stats.recent_executions) > 50:
            recent_avg = stats.recent_avg_time
            if recent_avg > stats.avg_time * 1.5:
                analysis.issues.append(f"Performance degradation: recent avg {recent_avg:.3f}s vs overall {stats.avg_time:.3f}s")
        
        # Query pattern analysis
        query_upper = stats.normalized_query.upper()
        
        # Missing WHERE clause on UPDATE/DELETE
        if (stats.query_type in [QueryType.UPDATE, QueryType.DELETE] and 
            'WHERE' not in query_upper):
            analysis.issues.append("Missing WHERE clause on UPDATE/DELETE query")
        
        # SELECT *
        if stats.query_type == QueryType.SELECT and 'SELECT *' in query_upper:
            analysis.issues.append("Using SELECT * instead of specific columns")
        
        # Multiple JOINs
        join_count = query_upper.count('JOIN')
        if join_count > 3:
            analysis.issues.append(f"Complex query with {join_count} JOINs")
        
        # Subqueries
        if 'SELECT' in query_upper.replace('SELECT', '', 1):
            analysis.issues.append("Contains subqueries that might be optimizable")
    
    def _generate_recommendations(self, stats: QueryStats, analysis: QueryAnalysis):
        """Generate optimization recommendations."""
        
        # High execution time recommendations
        if stats.avg_time > self.slow_query_threshold:
            analysis.recommendations.append("Consider adding database indexes")
            analysis.recommendations.append("Review query execution plan")
            analysis.recommendations.append("Consider query rewriting or optimization")
        
        # High frequency recommendations
        if stats.execution_count > 1000:
            analysis.recommendations.append("Consider caching results")
            analysis.recommendations.append("Review if query frequency can be reduced")
        
        # Query pattern recommendations
        query_upper = stats.normalized_query.upper()
        
        if 'SELECT *' in query_upper:
            analysis.recommendations.append("Use specific column names instead of SELECT *")
        
        if stats.query_type == QueryType.SELECT and 'ORDER BY' in query_upper:
            analysis.recommendations.append("Ensure ORDER BY columns are indexed")
        
        if 'LIKE' in query_upper:
            analysis.recommendations.append("Consider full-text search for LIKE queries")
        
        if stats.query_type == QueryType.SELECT and 'GROUP BY' in query_upper:
            analysis.recommendations.append("Ensure GROUP BY columns are indexed")
        
        # Error rate recommendations
        if stats.error_rate > 5:
            analysis.recommendations.append("Investigate and fix query errors")
            analysis.recommendations.append("Add proper error handling")
        
        # Estimate potential improvement
        if stats.avg_time > 1.0 and stats.execution_count > 100:
            potential_improvement = min(stats.avg_time * 0.5, stats.avg_time - 0.1)
            analysis.estimated_improvement = potential_improvement
    
    def _fire_slow_query_callbacks(self, execution: QueryExecution):
        """Fire slow query callbacks."""
        for callback in self.slow_query_callbacks:
            try:
                callback(execution)
            except Exception as e:
                logger.error(f"Error in slow query callback: {e}")
    
    def _fire_analysis_callbacks(self, analysis: QueryAnalysis):
        """Fire query analysis callbacks."""
        for callback in self.analysis_callbacks:
            try:
                callback(analysis)
            except Exception as e:
                logger.error(f"Error in analysis callback: {e}")
    
    def add_slow_query_callback(self, callback: Callable[[QueryExecution], None]):
        """Add callback for slow query detection."""
        self.slow_query_callbacks.append(callback)
    
    def add_analysis_callback(self, callback: Callable[[QueryAnalysis], None]):
        """Add callback for query analysis completion."""
        self.analysis_callbacks.append(callback)
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent slow queries."""
        return [q.to_dict() for q in list(self.slow_queries)[-limit:]]
    
    def get_query_stats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get query statistics sorted by priority."""
        sorted_stats = sorted(
            self.query_stats.values(),
            key=lambda x: (x.get_priority().value, x.avg_time),
            reverse=True
        )
        
        return [
            {
                'query_hash': stats.query_hash,
                'normalized_query': stats.normalized_query,
                'query_type': stats.query_type.value,
                'execution_count': stats.execution_count,
                'avg_time': round(stats.avg_time, 4),
                'min_time': round(stats.min_time, 4),
                'max_time': round(stats.max_time, 4),
                'error_rate': round(stats.error_rate, 2),
                'priority': stats.get_priority().value,
                'last_execution': stats.last_execution.isoformat() if stats.last_execution else None
            }
            for stats in sorted_stats[:limit]
        ]
    
    def get_query_analyses(self, priority: Optional[QueryPriority] = None) -> List[Dict[str, Any]]:
        """Get query analyses, optionally filtered by priority."""
        analyses = self.query_analyses.values()
        
        if priority:
            analyses = [a for a in analyses if a.priority == priority]
        
        return [analysis.to_dict() for analysis in analyses]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall query performance summary."""
        if self.total_queries == 0:
            return {'status': 'no_queries'}
        
        # Calculate averages
        avg_query_time = self.total_query_time / self.total_queries
        error_rate = (self.error_count / self.total_queries) * 100
        
        # Query type distribution
        type_counts = defaultdict(int)
        for stats in self.query_stats.values():
            type_counts[stats.query_type.value] += stats.execution_count
        
        # Priority distribution
        priority_counts = defaultdict(int)
        for analysis in self.query_analyses.values():
            priority_counts[analysis.priority.value] += 1
        
        return {
            'total_queries': self.total_queries,
            'total_query_time': round(self.total_query_time, 2),
            'avg_query_time': round(avg_query_time, 4),
            'error_count': self.error_count,
            'error_rate': round(error_rate, 2),
            'slow_queries_count': len(self.slow_queries),
            'unique_query_patterns': len(self.query_stats),
            'analyzed_queries': len(self.query_analyses),
            'query_type_distribution': dict(type_counts),
            'priority_distribution': dict(priority_counts),
            'slow_query_threshold': self.slow_query_threshold
        }
    
    def export_analysis_report(self) -> str:
        """Export comprehensive analysis report."""
        report = []
        report.append("=== Query Performance Analysis Report ===\n")
        
        # Summary
        summary = self.get_performance_summary()
        report.append(f"Total Queries: {summary['total_queries']}")
        report.append(f"Average Query Time: {summary['avg_query_time']:.4f}s")
        report.append(f"Error Rate: {summary['error_rate']:.2f}%")
        report.append(f"Slow Queries: {summary['slow_queries_count']}")
        report.append("")
        
        # Top slow queries
        report.append("=== Top Slow Queries ===")
        for i, stats in enumerate(self.get_query_stats(10), 1):
            report.append(f"{i}. {stats['normalized_query'][:100]}...")
            report.append(f"   Avg Time: {stats['avg_time']:.4f}s, Count: {stats['execution_count']}")
            report.append(f"   Priority: {stats['priority']}")
            report.append("")
        
        # Analysis with recommendations
        report.append("=== Query Analysis & Recommendations ===")
        for analysis in self.get_query_analyses():
            if analysis['priority'] in ['high', 'critical']:
                report.append(f"Query: {analysis['normalized_query'][:100]}...")
                report.append(f"Priority: {analysis['priority']}")
                
                if analysis['issues']:
                    report.append("Issues:")
                    for issue in analysis['issues']:
                        report.append(f"  - {issue}")
                
                if analysis['recommendations']:
                    report.append("Recommendations:")
                    for rec in analysis['recommendations']:
                        report.append(f"  - {rec}")
                
                report.append("")
        
        return "\n".join(report)
    
    def reset_metrics(self):
        """Reset all tracking metrics."""
        self.query_stats.clear()
        self.query_executions.clear()
        self.slow_queries.clear()
        self.query_analyses.clear()
        self.total_queries = 0
        self.total_query_time = 0.0
        self.error_count = 0
        
        logger.info("Query tracker metrics reset")


def track_query(tracker: QueryTracker):
    """
    Decorator to automatically track query execution.
    
    Args:
        tracker: QueryTracker instance
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract query from arguments (assuming first arg is query)
            query = args[0] if args else kwargs.get('query', 'UNKNOWN')
            
            start_time = time.time()
            error = None
            result = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                execution_time = time.time() - start_time
                
                # Extract additional info from result if available
                result_count = None
                if hasattr(result, '__len__'):
                    try:
                        result_count = len(result)
                    except TypeError:
                        pass  # Result doesn't support len()
                
                # Track the query
                tracker.track_query(
                    query=query,
                    execution_time=execution_time,
                    params=args[1:] if len(args) > 1 else None,
                    result_count=result_count,
                    error=error
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            query = args[0] if args else kwargs.get('query', 'UNKNOWN')
            
            start_time = time.time()
            error = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                execution_time = time.time() - start_time
                
                result_count = None
                if hasattr(result, '__len__'):
                    try:
                        result_count = len(result)
                    except TypeError:
                        pass  # Result doesn't support len()
                
                tracker.track_query(
                    query=query,
                    execution_time=execution_time,
                    params=args[1:] if len(args) > 1 else None,
                    result_count=result_count,
                    error=error
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global query tracker instance
_global_tracker = QueryTracker()


def get_global_tracker() -> QueryTracker:
    """Get the global query tracker instance."""
    return _global_tracker


# Alias for backward compatibility
QueryPerformanceTracker = QueryTracker
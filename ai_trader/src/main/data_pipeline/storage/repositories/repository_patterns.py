"""
Repository Patterns and Mixins

Reusable patterns to reduce code duplication across repositories.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
import hashlib
import json

from main.utils.core import get_logger
from main.utils.cache import get_global_cache, CacheTier
from main.utils.monitoring import record_metric, MetricType

logger = get_logger(__name__)


class RepositoryMixin:
    """Base mixin providing common repository functionality."""
    
    def _generate_unique_id(self, *args) -> str:
        """
        Generate a unique ID from multiple fields.
        
        Args:
            *args: Fields to include in ID generation
            
        Returns:
            Unique ID string
        """
        # Concatenate all arguments as strings
        id_parts = []
        for arg in args:
            if isinstance(arg, datetime):
                id_parts.append(arg.isoformat())
            elif arg is not None:
                id_parts.append(str(arg))
        
        # Create hash of concatenated parts
        id_string = "_".join(id_parts)
        return hashlib.md5(id_string.encode()).hexdigest()
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to uppercase."""
        return symbol.upper() if symbol else symbol
    
    
    def _validate_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """
        Validate date range.
        
        Raises:
            ValueError: If date range is invalid
        """
        if start_date >= end_date:
            raise ValueError(f"Start date {start_date} must be before end date {end_date}")
        
        # Check for unreasonable ranges
        max_days = 365 * 5  # 5 years
        if (end_date - start_date).days > max_days:
            raise ValueError(f"Date range exceeds maximum of {max_days} days")


class DualStorageMixin:
    """Mixin for repositories with hot/cold storage support."""
    
    def __init__(self, *args, **kwargs):
        """Initialize dual storage settings."""
        super().__init__(*args, **kwargs)
        self.hot_storage_days = getattr(self.config, 'hot_storage_days', 30)
        self.enable_dual_storage = getattr(self.config, 'enable_dual_storage', False)
    
    def _get_hot_storage_cutoff(self) -> datetime:
        """Get the cutoff date for hot storage."""
        return datetime.now(timezone.utc) - timedelta(days=self.hot_storage_days)
    
    def _is_hot_data(self, timestamp: datetime) -> bool:
        """Check if data should be in hot storage."""
        if not self.enable_dual_storage:
            return True
        
        cutoff = self._get_hot_storage_cutoff()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        return timestamp >= cutoff
    
    async def _route_to_storage(
        self,
        query_filter: 'QueryFilter',
        operation: str
    ) -> str:
        """
        Route operation to appropriate storage.
        
        Args:
            query_filter: Query filter with date range
            operation: Operation type (read/write)
            
        Returns:
            Storage type ('hot', 'cold', or 'both')
        """
        if not self.enable_dual_storage:
            return 'hot'
        
        # For writes, always go to hot storage
        if operation == 'write':
            return 'hot'
        
        # For reads, check date range
        cutoff = self._get_hot_storage_cutoff()
        
        if query_filter.start_date and query_filter.end_date:
            from main.utils.core import ensure_utc
            start = ensure_utc(query_filter.start_date)
            end = ensure_utc(query_filter.end_date)
            
            if end < cutoff:
                return 'cold'
            elif start >= cutoff:
                return 'hot'
            else:
                return 'both'
        
        return 'hot'


class CacheMixin:
    """Mixin for repositories with caching support."""
    
    def __init__(self, *args, **kwargs):
        """Initialize cache settings."""
        super().__init__(*args, **kwargs)
        self.enable_caching = getattr(self.config, 'enable_caching', True)
        self.cache_ttl = getattr(self.config, 'cache_ttl_seconds', 300)
        self._cache = get_global_cache() if self.enable_caching else None
    
    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            prefix: Cache key prefix
            **kwargs: Parameters to include in key
            
        Returns:
            Cache key string
        """
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        param_str = "_".join(f"{k}={v}" for k, v in sorted_params if v is not None)
        return f"{prefix}:{param_str}"
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._cache or not self.enable_caching:
            return None
        
        try:
            return await self._cache.get(key, tier=CacheTier.MEMORY)
        except Exception as e:
            logger.debug(f"Cache get error for {key}: {e}")
            return None
    
    async def _set_in_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if not self._cache or not self.enable_caching:
            return
        
        try:
            await self._cache.set(
                key,
                value,
                ttl=self.cache_ttl,
                tier=CacheTier.MEMORY
            )
        except Exception as e:
            logger.debug(f"Cache set error for {key}: {e}")
    
    async def _invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        if not self._cache or not self.enable_caching:
            return
        
        try:
            if pattern:
                await self._cache.delete_pattern(pattern)
            else:
                await self._cache.clear()
        except Exception as e:
            logger.debug(f"Cache invalidation error: {e}")


class MetricsMixin:
    """Mixin for repositories with metrics collection."""
    
    def __init__(self, *args, **kwargs):
        """Initialize metrics settings."""
        super().__init__(*args, **kwargs)
        self.enable_metrics = getattr(self.config, 'enable_metrics', True)
        self._repo_name = self.__class__.__name__
    
    async def _record_operation_metric(
        self,
        operation: str,
        duration: float,
        success: bool,
        records: int = 0
    ) -> None:
        """
        Record repository operation metric.
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            success: Whether operation succeeded
            records: Number of records affected
        """
        if not self.enable_metrics:
            return
        
        try:
            # Record operation latency
            await record_metric(
                MetricType.HISTOGRAM,
                f"repository_{operation}_duration",
                duration,
                labels={
                    'repository': self._repo_name,
                    'operation': operation,
                    'success': str(success).lower()
                }
            )
            
            # Record operation count
            await record_metric(
                MetricType.COUNTER,
                f"repository_{operation}_total",
                1,
                labels={
                    'repository': self._repo_name,
                    'operation': operation,
                    'success': str(success).lower()
                }
            )
            
            # Record records processed if applicable
            if records > 0:
                await record_metric(
                    MetricType.COUNTER,
                    f"repository_records_processed",
                    records,
                    labels={
                        'repository': self._repo_name,
                        'operation': operation
                    }
                )
        except Exception as e:
            logger.debug(f"Metrics recording error: {e}")
    
    async def _record_cache_metric(self, hit: bool) -> None:
        """Record cache hit/miss metric."""
        if not self.enable_metrics:
            return
        
        try:
            await record_metric(
                MetricType.COUNTER,
                "repository_cache_hits" if hit else "repository_cache_misses",
                1,
                labels={'repository': self._repo_name}
            )
        except Exception as e:
            logger.debug(f"Cache metrics error: {e}")
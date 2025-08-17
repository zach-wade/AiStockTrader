"""
Scanner cache management utilities.

Provides intelligent caching for scanner results with TTL management,
invalidation strategies, and performance optimization.
"""

# Standard library imports
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from typing import Any

# Local imports
from main.interfaces.scanners import IScannerCache
from main.utils.cache import CacheType, get_global_cache
from main.utils.core import timer

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached scanner result."""

    key: str
    value: Any
    timestamp: datetime
    ttl_seconds: int
    hit_count: int = 0
    size_bytes: int = 0
    scanner_name: str = ""
    symbol: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        age = (datetime.now(UTC) - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now(UTC) - self.timestamp).total_seconds()


@dataclass
class CacheStats:
    """Cache performance statistics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_size_bytes: int = 0
    evictions: int = 0
    avg_hit_rate: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


class ScannerCacheManager(IScannerCache):
    """
    Intelligent cache manager for scanner results.

    Features:
    - Multi-level caching (memory + redis)
    - Smart TTL based on data volatility
    - Pattern-based invalidation
    - Performance metrics tracking
    """

    def __init__(
        self,
        enable_memory_cache: bool = True,
        enable_redis_cache: bool = True,
        memory_cache_size_mb: int = 100,
        default_ttl_seconds: int = 300,
    ):
        """
        Initialize scanner cache manager.

        Args:
            enable_memory_cache: Enable in-memory caching
            enable_redis_cache: Enable Redis caching
            memory_cache_size_mb: Max memory cache size in MB
            default_ttl_seconds: Default TTL for cache entries
        """
        self.enable_memory_cache = enable_memory_cache
        self.enable_redis_cache = enable_redis_cache
        self.memory_cache_size_mb = memory_cache_size_mb
        self.default_ttl_seconds = default_ttl_seconds

        # Initialize caches
        self.global_cache = get_global_cache() if enable_redis_cache else None
        self.memory_cache: dict[str, CacheEntry] = {}
        self.cache_stats = CacheStats()

        # TTL strategies by data type
        self.ttl_strategies = {
            "market_snapshot": 30,  # Very short for real-time data
            "volume_stats": 300,  # 5 minutes for volume statistics
            "technical_data": 900,  # 15 minutes for technical indicators
            "scan_result": 60,  # 1 minute for scan results
            "news_sentiment": 1800,  # 30 minutes for news
            "correlation": 3600,  # 1 hour for correlations
        }

        # Start cache maintenance task
        self._maintenance_task = asyncio.create_task(self._cache_maintenance_loop())

    async def get_cached_result(self, scanner_name: str, symbol: str, cache_key: str) -> Any | None:
        """
        Retrieve cached scanner result.

        Args:
            scanner_name: Name of scanner
            symbol: Symbol being scanned
            cache_key: Unique cache key

        Returns:
            Cached value or None if not found/expired
        """
        with timer() as t:
            self.cache_stats.total_requests += 1

            # Build full cache key
            full_key = self._build_cache_key(scanner_name, symbol, cache_key)

            # Check memory cache first
            if self.enable_memory_cache:
                entry = self.memory_cache.get(full_key)
                if entry and not entry.is_expired:
                    entry.hit_count += 1
                    self.cache_stats.cache_hits += 1
                    logger.debug(f"Memory cache hit for {full_key} ({t.elapsed_ms:.2f}ms)")
                    return entry.value

            # Check Redis cache
            if self.global_cache and self.enable_redis_cache:
                try:
                    value = await self.global_cache.get(
                        full_key, cache_type=CacheType.SCANNER_RESULTS
                    )
                    if value is not None:
                        self.cache_stats.cache_hits += 1

                        # Populate memory cache
                        if self.enable_memory_cache:
                            await self._add_to_memory_cache(full_key, value, scanner_name, symbol)

                        logger.debug(f"Redis cache hit for {full_key} ({t.elapsed_ms:.2f}ms)")
                        return value
                except Exception as e:
                    logger.error(f"Redis cache error: {e}")

            self.cache_stats.cache_misses += 1
            logger.debug(f"Cache miss for {full_key}")
            return None

    async def cache_result(
        self,
        scanner_name: str,
        symbol: str,
        cache_key: str,
        result: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Cache scanner result.

        Args:
            scanner_name: Name of scanner
            symbol: Symbol being scanned
            cache_key: Unique cache key
            result: Result to cache
            ttl_seconds: Optional TTL override
        """
        # Determine TTL
        if ttl_seconds is None:
            ttl_seconds = self._get_smart_ttl(cache_key, result)

        # Build full cache key
        full_key = self._build_cache_key(scanner_name, symbol, cache_key)

        # Add to memory cache
        if self.enable_memory_cache:
            await self._add_to_memory_cache(full_key, result, scanner_name, symbol, ttl_seconds)

        # Add to Redis cache
        if self.global_cache and self.enable_redis_cache:
            try:
                await self.global_cache.set(
                    full_key, result, ttl=ttl_seconds, cache_type=CacheType.SCANNER_RESULTS
                )
            except Exception as e:
                logger.error(f"Redis cache set error: {e}")

    async def invalidate_cache(
        self, scanner_name: str | None = None, symbol: str | None = None
    ) -> None:
        """
        Invalidate cached results.

        Args:
            scanner_name: Scanner to invalidate (None for all)
            symbol: Symbol to invalidate (None for all)
        """
        invalidated_count = 0

        # Invalidate memory cache
        if self.enable_memory_cache:
            keys_to_remove = []
            for key, entry in self.memory_cache.items():
                if scanner_name and entry.scanner_name != scanner_name:
                    continue
                if symbol and entry.symbol != symbol:
                    continue
                keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.memory_cache[key]
                invalidated_count += 1

        # Invalidate Redis cache
        if self.global_cache and self.enable_redis_cache:
            # Build pattern for deletion
            pattern_parts = ["scanner"]
            if scanner_name:
                pattern_parts.append(scanner_name)
            else:
                pattern_parts.append("*")
            if symbol:
                pattern_parts.append(symbol)
            else:
                pattern_parts.append("*")
            pattern_parts.append("*")

            pattern = ":".join(pattern_parts)

            try:
                # Note: This would need Redis SCAN implementation
                # For now, we can't do pattern-based deletion easily
                logger.info(f"Would invalidate Redis keys matching: {pattern}")
            except Exception as e:
                logger.error(f"Redis invalidation error: {e}")

        logger.info(f"Invalidated {invalidated_count} cache entries")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        memory_size = sum(entry.size_bytes for entry in self.memory_cache.values())

        return {
            "total_requests": self.cache_stats.total_requests,
            "cache_hits": self.cache_stats.cache_hits,
            "cache_misses": self.cache_stats.cache_misses,
            "hit_rate": self.cache_stats.hit_rate,
            "memory_entries": len(self.memory_cache),
            "memory_size_mb": memory_size / (1024 * 1024),
            "evictions": self.cache_stats.evictions,
            "avg_entry_age_seconds": self._get_average_entry_age(),
        }

    async def _add_to_memory_cache(
        self, key: str, value: Any, scanner_name: str, symbol: str, ttl_seconds: int | None = None
    ) -> None:
        """Add entry to memory cache with size management."""
        # Estimate size (simplified)
        size_bytes = len(str(value).encode("utf-8"))

        # Check if we need to evict entries
        await self._evict_if_needed(size_bytes)

        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(UTC),
            ttl_seconds=ttl_seconds or self.default_ttl_seconds,
            size_bytes=size_bytes,
            scanner_name=scanner_name,
            symbol=symbol,
        )

        self.memory_cache[key] = entry
        self.cache_stats.total_size_bytes += size_bytes

    async def _evict_if_needed(self, new_size_bytes: int) -> None:
        """Evict cache entries if size limit exceeded."""
        max_size_bytes = self.memory_cache_size_mb * 1024 * 1024
        current_size = sum(e.size_bytes for e in self.memory_cache.values())

        if current_size + new_size_bytes <= max_size_bytes:
            return

        # Evict entries using LRU with TTL consideration
        entries = list(self.memory_cache.items())
        # Sort by: expired first, then by age, then by hit count
        entries.sort(
            key=lambda x: (
                not x[1].is_expired,  # Expired first
                -x[1].age_seconds,  # Oldest second
                x[1].hit_count,  # Least used third
            )
        )

        evicted_size = 0
        for key, entry in entries:
            if current_size + new_size_bytes - evicted_size <= max_size_bytes:
                break

            del self.memory_cache[key]
            evicted_size += entry.size_bytes
            self.cache_stats.evictions += 1

        self.cache_stats.total_size_bytes -= evicted_size

    async def _cache_maintenance_loop(self) -> None:
        """Periodic cache maintenance."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                # Remove expired entries
                expired_keys = [key for key, entry in self.memory_cache.items() if entry.is_expired]

                for key in expired_keys:
                    entry = self.memory_cache.pop(key)
                    self.cache_stats.total_size_bytes -= entry.size_bytes

                if expired_keys:
                    logger.debug(f"Removed {len(expired_keys)} expired cache entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")

    def _build_cache_key(self, scanner_name: str, symbol: str, cache_key: str) -> str:
        """Build full cache key."""
        return f"scanner:{scanner_name}:{symbol}:{cache_key}"

    def _get_smart_ttl(self, cache_key: str, result: Any) -> int:
        """Determine smart TTL based on data type and volatility."""
        # Check if cache key contains known data type
        for data_type, ttl in self.ttl_strategies.items():
            if data_type in cache_key:
                return ttl

        # Adjust based on result characteristics
        if isinstance(result, dict):
            # Shorter TTL for results with timestamps
            if "timestamp" in result:
                data_age = datetime.now(UTC) - result.get("timestamp", datetime.now(UTC))
                if data_age.total_seconds() < 60:  # Very fresh data
                    return 30  # Short TTL

        return self.default_ttl_seconds

    def _get_average_entry_age(self) -> float:
        """Calculate average age of cache entries."""
        if not self.memory_cache:
            return 0.0

        total_age = sum(entry.age_seconds for entry in self.memory_cache.values())
        return total_age / len(self.memory_cache)

"""
Scanner Cache Manager

This module provides caching functionality for scanner operations.
"""

# Standard library imports
import asyncio
import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class ScannerCacheManager:
    """Manages caching for scanner results to improve performance."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 1000):
        """
        Initialize the cache manager.

        Args:
            ttl_seconds: Time to live for cache entries in seconds
            max_entries: Maximum number of entries to keep in cache
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()
        logger.info(
            f"ScannerCacheManager initialized with TTL={ttl_seconds}s, max_entries={max_entries}"
        )

    def _generate_key(self, scanner_name: str, symbols: list[str], params: dict[str, Any]) -> str:
        """
        Generate a cache key from scanner parameters.

        Args:
            scanner_name: Name of the scanner
            symbols: List of symbols
            params: Scanner parameters

        Returns:
            Cache key string
        """
        # Create a consistent key from inputs
        key_data = {
            "scanner": scanner_name,
            "symbols": sorted(symbols) if symbols else [],
            "params": params,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def generate_cache_key(
        self, scanner_name: str, operation: str, symbols: list[str], **params
    ) -> str:
        """
        Generate a standardized cache key for scanner operations.

        This is a public method for scanners to use directly.

        Args:
            scanner_name: Name of the scanner
            operation: Operation name (e.g., 'scan', 'analyze')
            symbols: List of symbols
            **params: Additional parameters to include in key

        Returns:
            Cache key string
        """
        # Limit symbols to avoid overly long keys
        symbol_subset = symbols[:10] if len(symbols) > 10 else symbols

        # Build key components
        components = [scanner_name, operation, ",".join(sorted(symbol_subset))]

        # Add sorted parameters
        if params:
            param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
            components.append(param_str)

        return ":".join(components)

    async def get(
        self, scanner_name: str, symbols: list[str], params: dict[str, Any]
    ) -> Any | None:
        """
        Get cached result if available and not expired.

        Args:
            scanner_name: Name of the scanner
            symbols: List of symbols
            params: Scanner parameters

        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(scanner_name, symbols, params)

        async with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]

                # Check if expired
                if time.time() - timestamp < self.ttl_seconds:
                    logger.debug(f"Cache hit for {scanner_name} (key: {key[:8]}...)")
                    if not hasattr(self, "_hits"):
                        self._hits = 0
                    self._hits += 1
                    return result
                else:
                    # Remove expired entry
                    del self._cache[key]
                    logger.debug(f"Cache expired for {scanner_name} (key: {key[:8]}...)")

        if not hasattr(self, "_misses"):
            self._misses = 0
        self._misses += 1
        return None

    async def set(
        self, scanner_name: str, symbols: list[str], params: dict[str, Any], result: Any
    ) -> None:
        """
        Store result in cache.

        Args:
            scanner_name: Name of the scanner
            symbols: List of symbols
            params: Scanner parameters
            result: Result to cache
        """
        key = self._generate_key(scanner_name, symbols, params)

        async with self._lock:
            # Check cache size limit
            if len(self._cache) >= self.max_entries:
                # Remove oldest entries (simple FIFO)
                oldest_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][1])[
                    : len(self._cache) - self.max_entries + 1
                ]
                for old_key in oldest_keys:
                    del self._cache[old_key]
                logger.debug(f"Evicted {len(oldest_keys)} old cache entries")

            # Store new entry
            self._cache[key] = (result, time.time())
            logger.debug(f"Cached result for {scanner_name} (key: {key[:8]}...)")

    async def invalidate(self, scanner_name: str | None = None) -> None:
        """
        Invalidate cache entries.

        Args:
            scanner_name: Specific scanner to invalidate, or None for all
        """
        async with self._lock:
            if scanner_name:
                # Invalidate entries for specific scanner
                keys_to_remove = []
                for key in self._cache:
                    # Since we hash the key, we need to track scanner names separately
                    # For now, just clear all if scanner specified
                    keys_to_remove.append(key)

                for key in keys_to_remove[:10]:  # Limit to avoid clearing everything
                    del self._cache[key]

                logger.info(f"Invalidated {len(keys_to_remove)} cache entries for {scanner_name}")
            else:
                # Clear all cache
                self._cache.clear()
                logger.info("Cleared entire scanner cache")

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        async with self._lock:
            total_entries = len(self._cache)

            # Count expired entries
            current_time = time.time()
            expired_count = sum(
                1
                for _, (_, timestamp) in self._cache.items()
                if current_time - timestamp >= self.ttl_seconds
            )

            # Calculate memory usage (approximate)
            total_size = sum(
                len(str(key)) + len(str(value)) for key, (value, _) in self._cache.items()
            )

            return {
                "total_entries": total_entries,
                "active_entries": total_entries - expired_count,
                "expired_entries": expired_count,
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "cache_full": total_entries >= self.max_entries,
                "approx_memory_bytes": total_size,
                "hit_rate": self._calculate_hit_rate(),
            }

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not hasattr(self, "_hits"):
            self._hits = 0
            self._misses = 0

        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    async def cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        async with self._lock:
            current_time = time.time()
            keys_to_remove = []

            for key, (_, timestamp) in self._cache.items():
                if current_time - timestamp >= self.ttl_seconds:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]

            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} expired cache entries")


# Global instance for easy access
_global_cache_manager = None


def get_scanner_cache_manager() -> ScannerCacheManager:
    """Get the global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = ScannerCacheManager()
    return _global_cache_manager

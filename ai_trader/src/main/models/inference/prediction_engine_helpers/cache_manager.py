"""
Cache Manager for Prediction Engine

Provides caching functionality for model predictions and feature data
to improve performance and reduce redundant calculations.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta

# Use secure serializer instead of pickle for security
from main.utils.core.secure_serializer import SecureSerializer
from typing import Any, Dict, Optional, Union, Callable
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for prediction engine operations."""
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 max_age_hours: float = 24,
                 max_size_mb: float = 1000,
                 enable_disk_cache: bool = True,
                 enable_memory_cache: bool = True):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for disk cache
            max_age_hours: Maximum age of cache entries
            max_size_mb: Maximum cache size in MB
            enable_disk_cache: Whether to use disk caching
            enable_memory_cache: Whether to use memory caching
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.ai_trader' / 'cache'
        self.max_age = timedelta(hours=max_age_hours)
        self.max_size_mb = max_size_mb
        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        
        # Memory cache
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Create cache directory if needed
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Track cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        # Check memory cache first
        if self.enable_memory_cache and key in self._memory_cache:
            entry = self._memory_cache[key]
            if self._is_valid_entry(entry):
                self._stats['hits'] += 1
                logger.debug(f"Memory cache hit for key: {key}")
                return entry['data']
            else:
                # Remove expired entry
                del self._memory_cache[key]
        
        # Check disk cache
        if self.enable_disk_cache:
            cache_file = self._get_cache_path(key)
            if cache_file.exists():
                try:
                    # Use secure serializer for loading
                    serializer = SecureSerializer()
                    with open(cache_file, 'rb') as f:
                        serialized_data = f.read()
                    entry = serializer.deserialize(serialized_data)
                    
                    if self._is_valid_entry(entry):
                        self._stats['hits'] += 1
                        logger.debug(f"Disk cache hit for key: {key}")
                        
                        # Add to memory cache
                        if self.enable_memory_cache:
                            self._memory_cache[key] = entry
                        
                        return entry['data']
                    else:
                        # Remove expired file
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error reading cache file {cache_file}: {e}")
                    cache_file.unlink()
        
        self._stats['misses'] += 1
        return None
    
    def set(self, key: str, data: Any) -> None:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        entry = {
            'data': data,
            'timestamp': datetime.now(),
            'key': key
        }
        
        # Add to memory cache
        if self.enable_memory_cache:
            self._memory_cache[key] = entry
            self._check_memory_size()
        
        # Add to disk cache
        if self.enable_disk_cache:
            try:
                cache_file = self._get_cache_path(key)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Use secure serializer for saving
                serializer = SecureSerializer()
                serialized_data = serializer.serialize(entry)
                with open(cache_file, 'wb') as f:
                    f.write(serialized_data)
                
                self._check_disk_size()
            except Exception as e:
                logger.warning(f"Error writing cache file: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        # Clear memory cache
        self._memory_cache.clear()
        
        # Clear disk cache
        if self.enable_disk_cache and self.cache_dir.exists():
            for cache_file in self.cache_dir.rglob('*.pkl'):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting cache file {cache_file}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'hit_rate': hit_rate,
            'memory_entries': len(self._memory_cache),
            'disk_size_mb': self._get_disk_cache_size()
        }
    
    def cached(self, key_func: Optional[Callable] = None):
        """
        Decorator for caching function results.
        
        Args:
            key_func: Function to generate cache key from args/kwargs
            
        Returns:
            Decorated function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    key = hashlib.md5('_'.join(key_parts).encode()).hexdigest()
                
                # Check cache
                result = self.get(key)
                if result is not None:
                    return result
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def _is_valid_entry(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        age = datetime.now() - entry['timestamp']
        return age < self.max_age
    
    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache file."""
        # Use first 2 chars of key for subdirectory
        subdir = key[:2] if len(key) >= 2 else 'misc'
        return self.cache_dir / subdir / f"{key}.pkl"
    
    def _check_memory_size(self) -> None:
        """Check memory cache size and evict if needed."""
        # Simple LRU eviction based on timestamp
        if len(self._memory_cache) > 1000:  # Arbitrary limit
            # Sort by timestamp and remove oldest
            sorted_keys = sorted(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k]['timestamp']
            )
            
            # Remove oldest 10%
            to_remove = sorted_keys[:len(sorted_keys) // 10]
            for key in to_remove:
                del self._memory_cache[key]
                self._stats['evictions'] += 1
    
    def _check_disk_size(self) -> None:
        """Check disk cache size and evict if needed."""
        size_mb = self._get_disk_cache_size()
        
        if size_mb > self.max_size_mb:
            # Get all cache files with modification times
            cache_files = []
            for cache_file in self.cache_dir.rglob('*.pkl'):
                try:
                    mtime = cache_file.stat().st_mtime
                    size = cache_file.stat().st_size
                    cache_files.append((cache_file, mtime, size))
                except (OSError, FileNotFoundError):
                    continue  # File may have been deleted
            
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Remove oldest files until under limit
            removed_size = 0
            target_remove = (size_mb - self.max_size_mb * 0.8) * 1024 * 1024  # Remove to 80% of limit
            
            for cache_file, _, size in cache_files:
                try:
                    cache_file.unlink()
                    removed_size += size
                    self._stats['evictions'] += 1
                    
                    if removed_size >= target_remove:
                        break
                except (OSError, FileNotFoundError):
                    continue  # File may have been deleted
    
    def _get_disk_cache_size(self) -> float:
        """Get total disk cache size in MB."""
        if not self.cache_dir.exists():
            return 0.0
        
        total_size = 0
        for cache_file in self.cache_dir.rglob('*.pkl'):
            try:
                total_size += cache_file.stat().st_size
            except (OSError, FileNotFoundError):
                continue  # File may have been deleted
        
        return total_size / (1024 * 1024)


# Singleton instance
_cache_manager = None


def get_cache_manager(**kwargs) -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(**kwargs)
    return _cache_manager


def clear_cache():
    """Clear all caches."""
    manager = get_cache_manager()
    manager.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    manager = get_cache_manager()
    return manager.get_stats()
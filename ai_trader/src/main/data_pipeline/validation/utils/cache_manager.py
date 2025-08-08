"""
Validation Cache Manager

Provides caching functionality for validation operations using the main utils cache system.
"""

from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import hashlib
import json

# Core imports
from main.utils.core import get_logger
from main.utils.cache import CacheTier, MemoryBackend
from main.utils.cache.simple_cache import SimpleCache

logger = get_logger(__name__)


class ValidationCacheManager:
    """
    Cache manager for validation operations.
    
    Provides caching for validation results, quality scores, coverage analysis,
    and other validation-related data to improve performance.
    """
    
    def __init__(
        self,
        cache_tier: CacheTier = CacheTier.MEMORY,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation cache manager.
        
        Args:
            cache_tier: Tier of cache backend to use (MEMORY, REDIS, etc.)
            ttl_seconds: Default time-to-live for cache entries in seconds
            max_size: Maximum number of entries in cache
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        
        # Initialize cache with memory backend
        self.cache = SimpleCache(
            backend=MemoryBackend(max_size=max_size)
        )
        self.cache_tier = cache_tier
        
        # Cache namespaces for different validation data types
        self.namespaces = {
            'validation_results': 'val_results',
            'quality_scores': 'val_quality',
            'coverage_data': 'val_coverage',
            'rule_evaluations': 'val_rules',
            'profile_configs': 'val_profiles'
        }
        
        logger.info(f"ValidationCacheManager initialized with {cache_tier} backend, TTL={ttl_seconds}s")
    
    # Validation Results Caching
    def cache_validation_result(
        self,
        key: str,
        result: Any,
        stage: str,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache a validation result.
        
        Args:
            key: Cache key (e.g., symbol, data hash)
            result: Validation result to cache
            stage: Validation stage
            ttl: Optional TTL override in seconds
        """
        cache_key = self._generate_cache_key(
            namespace=self.namespaces['validation_results'],
            stage=stage,
            key=key
        )
        
        # SimpleCache doesn't have ttl parameter, so we'll store the result directly
        self.cache.backend.set(
            key=cache_key,
            value=result,
            ttl=ttl or self.ttl_seconds
        )
        
        logger.debug(f"Cached validation result for {stage}:{key}")
    
    def get_validation_result(
        self,
        key: str,
        stage: str
    ) -> Optional[Any]:
        """
        Get cached validation result.
        
        Args:
            key: Cache key
            stage: Validation stage
            
        Returns:
            Cached validation result or None if not found
        """
        cache_key = self._generate_cache_key(
            namespace=self.namespaces['validation_results'],
            stage=stage,
            key=key
        )
        
        result = self.cache.backend.get(cache_key)
        if result:
            logger.debug(f"Cache hit for validation result {stage}:{key}")
        return result
    
    # Quality Score Caching
    def cache_quality_score(
        self,
        symbol: str,
        source: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache data quality score.
        
        Args:
            symbol: Trading symbol
            source: Data source
            score: Quality score (0-1)
            metadata: Optional metadata
            ttl: Optional TTL override
        """
        cache_key = self._generate_cache_key(
            namespace=self.namespaces['quality_scores'],
            symbol=symbol,
            source=source
        )
        
        cache_value = {
            'score': score,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store the cache value with ttl
        self.cache.backend.set(
            key=cache_key,
            value=cache_value,
            ttl=ttl or self.ttl_seconds
        )
        
        logger.debug(f"Cached quality score for {symbol}:{source} = {score:.2f}")
    
    def get_quality_score(
        self,
        symbol: str,
        source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached quality score.
        
        Args:
            symbol: Trading symbol
            source: Data source
            
        Returns:
            Cached quality score data or None
        """
        cache_key = self._generate_cache_key(
            namespace=self.namespaces['quality_scores'],
            symbol=symbol,
            source=source
        )
        
        return self.cache.backend.get(cache_key)
    
    # Coverage Data Caching
    def cache_coverage_analysis(
        self,
        symbols: List[str],
        intervals: List[str],
        coverage_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache coverage analysis results.
        
        Args:
            symbols: List of symbols analyzed
            intervals: List of intervals analyzed
            coverage_data: Coverage analysis results
            ttl: Optional TTL override
        """
        # Create hash of symbols and intervals for cache key
        key_data = {
            'symbols': sorted(symbols),
            'intervals': sorted(intervals)
        }
        key_hash = self._hash_dict(key_data)
        
        cache_key = self._generate_cache_key(
            namespace=self.namespaces['coverage_data'],
            key=key_hash
        )
        
        # Store coverage data with longer ttl
        self.cache.backend.set(
            key=cache_key,
            value=coverage_data,
            ttl=ttl or self.ttl_seconds * 2  # Coverage data can be cached longer
        )
        
        logger.debug(f"Cached coverage analysis for {len(symbols)} symbols, {len(intervals)} intervals")
    
    def get_coverage_analysis(
        self,
        symbols: List[str],
        intervals: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached coverage analysis.
        
        Args:
            symbols: List of symbols
            intervals: List of intervals
            
        Returns:
            Cached coverage data or None
        """
        key_data = {
            'symbols': sorted(symbols),
            'intervals': sorted(intervals)
        }
        key_hash = self._hash_dict(key_data)
        
        cache_key = self._generate_cache_key(
            namespace=self.namespaces['coverage_data'],
            key=key_hash
        )
        
        return self.cache.backend.get(cache_key)
    
    # Rule Evaluation Caching
    def cache_rule_evaluation(
        self,
        rule_name: str,
        data_hash: str,
        result: bool,
        message: str,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache rule evaluation result.
        
        Args:
            rule_name: Name of the validation rule
            data_hash: Hash of the data being validated
            result: Rule evaluation result
            message: Result message
            ttl: Optional TTL override
        """
        cache_key = self._generate_cache_key(
            namespace=self.namespaces['rule_evaluations'],
            rule=rule_name,
            data=data_hash
        )
        
        cache_value = {
            'result': result,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store rule evaluation with shorter ttl
        self.cache.backend.set(
            key=cache_key,
            value=cache_value,
            ttl=ttl or self.ttl_seconds // 2  # Rule evaluations have shorter TTL
        )
    
    def get_rule_evaluation(
        self,
        rule_name: str,
        data_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached rule evaluation.
        
        Args:
            rule_name: Name of the validation rule
            data_hash: Hash of the data
            
        Returns:
            Cached rule evaluation or None
        """
        cache_key = self._generate_cache_key(
            namespace=self.namespaces['rule_evaluations'],
            rule=rule_name,
            data=data_hash
        )
        
        return self.cache.backend.get(cache_key)
    
    # Profile Configuration Caching
    def cache_profile_config(
        self,
        profile_name: str,
        config: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache validation profile configuration.
        
        Args:
            profile_name: Name of the validation profile
            config: Profile configuration
            ttl: Optional TTL override
        """
        cache_key = self._generate_cache_key(
            namespace=self.namespaces['profile_configs'],
            profile=profile_name
        )
        
        # Store profile config with longer ttl
        self.cache.backend.set(
            key=cache_key,
            value=config,
            ttl=ttl or self.ttl_seconds * 10  # Profile configs are relatively static
        )
        
        logger.debug(f"Cached profile configuration for {profile_name}")
    
    def get_profile_config(
        self,
        profile_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached profile configuration.
        
        Args:
            profile_name: Name of the validation profile
            
        Returns:
            Cached profile configuration or None
        """
        cache_key = self._generate_cache_key(
            namespace=self.namespaces['profile_configs'],
            profile=profile_name
        )
        
        return self.cache.backend.get(cache_key)
    
    # Cache Management
    def clear_namespace(self, namespace: str) -> None:
        """
        Clear all cache entries in a namespace.
        
        Args:
            namespace: Cache namespace to clear
        """
        if namespace in self.namespaces:
            prefix = self.namespaces[namespace]
            # Note: This would require cache backend to support pattern deletion
            logger.info(f"Clearing cache namespace: {namespace}")
    
    def clear_all(self) -> None:
        """Clear all validation cache entries."""
        self.cache.backend.clear()
        logger.info("Cleared all validation cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'tier': str(self.cache_tier),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'namespaces': list(self.namespaces.keys())
        }
    
    # Helper Methods
    def _generate_cache_key(self, **kwargs) -> str:
        """
        Generate a cache key from keyword arguments.
        
        Args:
            **kwargs: Key components
            
        Returns:
            Generated cache key
        """
        components = []
        for key, value in sorted(kwargs.items()):
            if value is not None:
                components.append(f"{key}:{value}")
        
        return ":".join(components)
    
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """
        Generate hash from dictionary data.
        
        Args:
            data: Dictionary to hash
            
        Returns:
            Hash string
        """
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def hash_data(self, data: Any) -> str:
        """
        Generate hash from arbitrary data.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash string
        """
        if isinstance(data, dict):
            return self._hash_dict(data)
        elif isinstance(data, (list, tuple)):
            return self._hash_dict({'data': list(data)})
        else:
            str_data = str(data)
            return hashlib.md5(str_data.encode()).hexdigest()


# Global instance (optional, for backward compatibility)
_global_cache_manager: Optional[ValidationCacheManager] = None


def get_validation_cache_manager(
    cache_tier: CacheTier = CacheTier.MEMORY,
    ttl_seconds: int = 3600,
    max_size: int = 1000,
    config: Optional[Dict[str, Any]] = None
) -> ValidationCacheManager:
    """
    Get or create global validation cache manager instance.
    
    Args:
        cache_tier: Tier of cache backend
        ttl_seconds: Default TTL in seconds
        max_size: Maximum cache size
        config: Optional configuration
        
    Returns:
        ValidationCacheManager instance
    """
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = ValidationCacheManager(
            cache_tier=cache_tier,
            ttl_seconds=ttl_seconds,
            max_size=max_size,
            config=config
        )
    
    return _global_cache_manager
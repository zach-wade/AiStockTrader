"""
Storage backends for rate limiting system.

Provides Redis and in-memory storage implementations for distributed
and local rate limiting scenarios.
"""

import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

try:
    import redis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import RedisError

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .config import RateLimitConfig
from .exceptions import RateLimitStorageError


class RateLimitStorage(ABC):
    """Abstract base class for rate limit storage backends."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Get value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value with optional TTL."""
        pass

    @abstractmethod
    def increment(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        """Atomically increment counter."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        pass

    @abstractmethod
    def ttl(self, key: str) -> int:
        """Get TTL for key (-1 if no TTL, -2 if key doesn't exist)."""
        pass

    @abstractmethod
    def keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Clean up expired keys and return count removed."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if storage backend is healthy."""
        pass


class MemoryRateLimitStorage(RateLimitStorage):
    """
    In-memory storage backend for rate limiting.

    Suitable for single-instance deployments or development.
    Data is lost when application restarts.
    """

    def __init__(self, cleanup_interval: int = 3600) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}  # key -> (value, expires_at)
        self._lock = threading.RLock()
        self.cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def _cleanup_if_needed(self) -> Any:
        """Perform cleanup if enough time has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self.cleanup_expired()
            self._last_cleanup = current_time

    def _is_expired(self, expires_at: float | None) -> bool:
        """Check if entry is expired."""
        if expires_at is None:
            return False
        return time.time() > expires_at

    def get(self, key: str) -> Any | None:
        """Get value by key."""
        with self._lock:
            self._cleanup_if_needed()

            if key not in self._store:
                return None

            value, expires_at = self._store[key]

            if self._is_expired(expires_at):
                del self._store[key]
                return None

            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value with optional TTL."""
        with self._lock:
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl

            self._store[key] = (value, expires_at)
            return True

    def increment(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        """Atomically increment counter."""
        with self._lock:
            current_value = self.get(key) or 0
            new_value = current_value + amount
            self.set(key, new_value, ttl)
            return new_value

    def delete(self, key: str) -> bool:
        """Delete key."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        with self._lock:
            if key not in self._store:
                return False

            value, _ = self._store[key]
            expires_at = time.time() + ttl
            self._store[key] = (value, expires_at)
            return True

    def ttl(self, key: str) -> int:
        """Get TTL for key."""
        with self._lock:
            if key not in self._store:
                return -2

            _, expires_at = self._store[key]
            if expires_at is None:
                return -1

            remaining = int(expires_at - time.time())
            return max(0, remaining)

    def keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern (simple prefix matching)."""
        with self._lock:
            self._cleanup_if_needed()

            # Simple pattern matching (only supports prefix with *)
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                return [key for key in self._store.keys() if key.startswith(prefix)]
            else:
                return [key for key in self._store.keys() if key == pattern]

    def cleanup_expired(self) -> int:
        """Clean up expired keys."""
        with self._lock:
            current_time = time.time()
            expired_keys = []

            for key, (_, expires_at) in self._store.items():
                if expires_at is not None and current_time > expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._store[key]

            return len(expired_keys)

    def health_check(self) -> bool:
        """Memory storage is always healthy."""
        return True


class RedisRateLimitStorage(RateLimitStorage):
    """
    Redis storage backend for rate limiting.

    Provides distributed rate limiting across multiple application instances.
    Supports Redis Cluster and Sentinel configurations.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        if not REDIS_AVAILABLE:
            raise RateLimitStorageError(
                "Redis is not available. Install redis-py: pip install redis",
                storage_backend="redis",
            )

        self.config = config
        self.key_prefix = config.redis_key_prefix

        # Parse Redis URL and create connection
        try:
            self.redis_client = redis.from_url(
                config.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test connection
            self.redis_client.ping()

        except Exception as e:
            raise RateLimitStorageError(f"Failed to connect to Redis: {e}", storage_backend="redis")

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Any | None:
        """Get value by key."""
        try:
            prefixed_key = self._make_key(key)
            value = self.redis_client.get(prefixed_key)

            if value is None:
                return None

            # Try to deserialize JSON, fallback to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except RedisError as e:
            raise RateLimitStorageError(
                f"Redis GET failed: {e}", operation="get", storage_backend="redis"
            )

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value with optional TTL."""
        try:
            prefixed_key = self._make_key(key)

            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)

            if ttl is not None:
                result = self.redis_client.setex(prefixed_key, ttl, serialized_value)
            else:
                result = self.redis_client.set(prefixed_key, serialized_value)

            return bool(result)

        except RedisError as e:
            raise RateLimitStorageError(
                f"Redis SET failed: {e}", operation="set", storage_backend="redis"
            )

    def increment(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        """Atomically increment counter."""
        try:
            prefixed_key = self._make_key(key)

            # Use Redis pipeline for atomic operation
            with self.redis_client.pipeline() as pipe:
                pipe.multi()
                pipe.incrby(prefixed_key, amount)

                if ttl is not None:
                    pipe.expire(prefixed_key, ttl)

                results = pipe.execute()
                return int(results[0])

        except RedisError as e:
            raise RateLimitStorageError(
                f"Redis INCREMENT failed: {e}", operation="increment", storage_backend="redis"
            )

    def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            prefixed_key = self._make_key(key)
            result = self.redis_client.delete(prefixed_key)
            return bool(result)

        except RedisError as e:
            raise RateLimitStorageError(
                f"Redis DELETE failed: {e}", operation="delete", storage_backend="redis"
            )

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            prefixed_key = self._make_key(key)
            return bool(self.redis_client.exists(prefixed_key))

        except RedisError as e:
            raise RateLimitStorageError(
                f"Redis EXISTS failed: {e}", operation="exists", storage_backend="redis"
            )

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        try:
            prefixed_key = self._make_key(key)
            result = self.redis_client.expire(prefixed_key, ttl)
            return bool(result)

        except RedisError as e:
            raise RateLimitStorageError(
                f"Redis EXPIRE failed: {e}", operation="expire", storage_backend="redis"
            )

    def ttl(self, key: str) -> int:
        """Get TTL for key."""
        try:
            prefixed_key = self._make_key(key)
            return int(self.redis_client.ttl(prefixed_key))

        except RedisError as e:
            raise RateLimitStorageError(
                f"Redis TTL failed: {e}", operation="ttl", storage_backend="redis"
            )

    def keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        try:
            prefixed_pattern = self._make_key(pattern)
            keys = self.redis_client.keys(prefixed_pattern)

            # Remove prefix from returned keys
            prefix_len = len(self.key_prefix)
            return [key[prefix_len:] for key in keys]

        except RedisError as e:
            raise RateLimitStorageError(
                f"Redis KEYS failed: {e}", operation="keys", storage_backend="redis"
            )

    def cleanup_expired(self) -> int:
        """Redis automatically cleans up expired keys."""
        # Redis handles TTL automatically, so this is a no-op
        # We could implement active cleanup here if needed
        return 0

    def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            result = self.redis_client.ping()
            return bool(result)
        except Exception:
            return False

    def get_info(self) -> dict[str, Any]:
        """Get Redis server information."""
        try:
            return dict(self.redis_client.info())
        except RedisError as e:
            raise RateLimitStorageError(
                f"Redis INFO failed: {e}", operation="info", storage_backend="redis"
            )

    def flush_rate_limits(self) -> int:
        """Delete all rate limit keys."""
        try:
            pattern = self._make_key("*")
            keys = self.redis_client.keys(pattern)

            if keys:
                deleted = self.redis_client.delete(*keys)
                return int(deleted)
            return 0

        except RedisError as e:
            raise RateLimitStorageError(
                f"Redis flush failed: {e}", operation="flush", storage_backend="redis"
            )


def create_storage(config: RateLimitConfig) -> RateLimitStorage:
    """Factory function to create appropriate storage backend."""
    if config.storage_backend.lower() == "redis":
        return RedisRateLimitStorage(config)
    elif config.storage_backend.lower() == "memory":
        return MemoryRateLimitStorage(config.cleanup_interval)
    else:
        raise RateLimitStorageError(
            f"Unknown storage backend: {config.storage_backend}",
            storage_backend=config.storage_backend,
        )

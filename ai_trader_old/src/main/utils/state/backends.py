"""
Storage Backends

Implementation of various storage backends for state management.
"""

# Standard library imports
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timedelta
import hashlib
import json
import logging
from pathlib import Path

# Third-party imports
import redis.asyncio as redis

from .types import StateMetadata

logger = logging.getLogger(__name__)


class StorageBackendInterface(ABC):
    """Abstract interface for storage backends."""

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Get value by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> bool:
        """Set value with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup storage backend."""
        pass


class MemoryBackend(StorageBackendInterface):
    """In-memory storage backend."""

    def __init__(self, max_size_mb: float = 100.0):
        """Initialize memory backend."""
        self._storage: dict[str, bytes] = {}
        self._metadata: dict[str, StateMetadata] = {}
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> bytes | None:
        """Get value by key."""
        async with self._lock:
            if key not in self._storage:
                return None

            # Check expiration
            metadata = self._metadata.get(key)
            if metadata and metadata.is_expired:
                await self._remove_key(key)
                return None

            # Update access metadata
            if metadata:
                metadata.access_count += 1
                metadata.last_accessed = datetime.utcnow()

            return self._storage[key]

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> bool:
        """Set value with optional TTL."""
        async with self._lock:
            # Check if we need to make space
            value_size = len(value)
            if key not in self._storage:
                if self._current_size + value_size > self._max_size_bytes:
                    await self._evict_lru()

            # Store value
            old_size = len(self._storage.get(key, b""))
            self._storage[key] = value
            self._current_size = self._current_size - old_size + value_size

            # Update metadata
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None

            if key in self._metadata:
                metadata = self._metadata[key]
                metadata.updated_at = now
                metadata.ttl_seconds = ttl_seconds
                metadata.expires_at = expires_at
                metadata.size_bytes = value_size
            else:
                self._metadata[key] = StateMetadata(
                    key=key,
                    namespace="",
                    created_at=now,
                    updated_at=now,
                    ttl_seconds=ttl_seconds,
                    expires_at=expires_at,
                    size_bytes=value_size,
                )

            return True

    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        async with self._lock:
            if key in self._storage:
                await self._remove_key(key)
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        async with self._lock:
            if key not in self._storage:
                return False

            # Check expiration
            metadata = self._metadata.get(key)
            if metadata and metadata.is_expired:
                await self._remove_key(key)
                return False

            return True

    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        # Standard library imports
        import fnmatch

        async with self._lock:
            # Clean up expired keys first
            expired_keys = [key for key, metadata in self._metadata.items() if metadata.is_expired]
            for key in expired_keys:
                await self._remove_key(key)

            # Return matching keys
            if pattern == "*":
                return list(self._storage.keys())
            else:
                return [key for key in self._storage.keys() if fnmatch.fnmatch(key, pattern)]

    async def cleanup(self) -> None:
        """Cleanup expired entries."""
        async with self._lock:
            expired_keys = [key for key, metadata in self._metadata.items() if metadata.is_expired]
            for key in expired_keys:
                await self._remove_key(key)

    async def _remove_key(self, key: str) -> None:
        """Remove key and update size tracking."""
        if key in self._storage:
            self._current_size -= len(self._storage[key])
            del self._storage[key]
        if key in self._metadata:
            del self._metadata[key]

    async def _evict_lru(self) -> None:
        """Evict least recently used items to make space."""
        if not self._metadata:
            return

        # Sort by last accessed time
        sorted_keys = sorted(
            self._metadata.keys(),
            key=lambda k: self._metadata[k].last_accessed or self._metadata[k].created_at,
        )

        # Evict oldest 25% of items
        num_to_evict = max(1, len(sorted_keys) // 4)
        for key in sorted_keys[:num_to_evict]:
            await self._remove_key(key)


class RedisBackend(StorageBackendInterface):
    """Redis storage backend."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis backend."""
        self.redis_url = redis_url
        self._redis: redis.Redis | None = None
        self._lock = asyncio.Lock()

    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            async with self._lock:
                if self._redis is None:
                    self._redis = redis.from_url(self.redis_url)
        return self._redis

    async def get(self, key: str) -> bytes | None:
        """Get value by key."""
        redis_client = await self._get_redis()
        value = await redis_client.get(key)
        return value

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> bool:
        """Set value with optional TTL."""
        redis_client = await self._get_redis()
        if ttl_seconds:
            result = await redis_client.setex(key, ttl_seconds, value)
        else:
            result = await redis_client.set(key, value)
        return result

    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        redis_client = await self._get_redis()
        result = await redis_client.delete(key)
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        redis_client = await self._get_redis()
        result = await redis_client.exists(key)
        return result > 0

    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        redis_client = await self._get_redis()
        keys = await redis_client.keys(pattern)
        return [key.decode() if isinstance(key, bytes) else key for key in keys]

    async def cleanup(self) -> None:
        """Cleanup is handled by Redis TTL automatically."""
        pass


class FileBackend(StorageBackendInterface):
    """File-based storage backend."""

    def __init__(self, base_path: str = "./state_storage"):
        """Initialize file backend."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _get_file_path(self, key: str) -> Path:
        """Get file path for key."""
        # Hash the key to create safe filename
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.base_path / f"{safe_key}.state"

    async def get(self, key: str) -> bytes | None:
        """Get value by key."""
        file_path = self._get_file_path(key)
        if not file_path.exists():
            return None

        try:
            async with self._lock:
                return file_path.read_bytes()
        except Exception as e:
            logger.error(f"Failed to read state file {file_path}: {e}")
            return None

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> bool:
        """Set value with optional TTL."""
        file_path = self._get_file_path(key)

        try:
            async with self._lock:
                file_path.write_bytes(value)

            # TTL is handled by periodic cleanup
            return True

        except Exception as e:
            logger.error(f"Failed to write state file {file_path}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        file_path = self._get_file_path(key)

        try:
            async with self._lock:
                if file_path.exists():
                    file_path.unlink()
                    return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete state file {file_path}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        file_path = self._get_file_path(key)
        return file_path.exists()

    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        # For file backend, we store a mapping file
        mapping_file = self.base_path / "key_mapping.json"

        if not mapping_file.exists():
            return []

        try:
            mapping = json.loads(mapping_file.read_text())
            # Standard library imports
            import fnmatch

            return [key for key in mapping.keys() if fnmatch.fnmatch(key, pattern)]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to read mapping file: {e}")
            return []

    async def cleanup(self) -> None:
        """Cleanup old files."""
        # This would implement TTL cleanup based on file timestamps
        pass

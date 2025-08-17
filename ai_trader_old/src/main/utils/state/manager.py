"""
State Manager

Main state management orchestrator.
"""

# Standard library imports
import asyncio
from collections.abc import Callable
import logging
from typing import Any

from .backends import FileBackend, MemoryBackend, RedisBackend, StorageBackendInterface
from .context import StateContext
from .persistence import StatePersistence
from .types import SerializationFormat, StateConfig, StorageBackend

logger = logging.getLogger(__name__)


class StateManager:
    """
    Unified state management system.

    Consolidates patterns from:
    - config_manager.py (configuration state)
    - cache.py (Redis caching with TTL)
    - feature_store.py (versioned state with metadata)
    - portfolio_manager.py (immutable state with locks)
    - timestamp_tracker.py (JSON file persistence)
    """

    def __init__(self, config: StateConfig):
        """Initialize state manager."""
        self.config = config

        # Storage backends
        self._backends: dict[StorageBackend, StorageBackendInterface] = {}
        self._init_backends()

        # Serializers
        self._serializers: dict[type, Callable] = {}
        self._deserializers: dict[type, Callable] = {}
        self._init_serializers()

        # Validators
        self._validators: dict[str, Callable] = {}

        # Sub-managers
        self.persistence = StatePersistence(self)
        self.context = StateContext(self)

        # Metrics
        self._metrics = {
            "operations": {"get": 0, "set": 0, "delete": 0},
            "cache_hits": 0,
            "cache_misses": 0,
            "serialization_errors": 0,
            "validation_errors": 0,
        }

        # Background tasks
        self._cleanup_task: asyncio.Task | None = None
        self._checkpoint_task: asyncio.Task | None = None

        # Start background tasks
        self._start_background_tasks()

        logger.info("StateManager initialized")

    def _init_backends(self) -> None:
        """Initialize storage backends."""
        # Memory backend (always available)
        self._backends[StorageBackend.MEMORY] = MemoryBackend(
            max_size_mb=self.config.max_memory_size_mb
        )

        # Redis backend
        if self.config.redis_url:
            self._backends[StorageBackend.REDIS] = RedisBackend(self.config.redis_url)

        # File backend
        if self.config.file_storage_path:
            self._backends[StorageBackend.FILE] = FileBackend(self.config.file_storage_path)

    def _init_serializers(self) -> None:
        """Initialize serialization handlers."""
        # Local imports
        from main.utils.core import from_json, to_json

        # JSON serializer
        self._serializers[SerializationFormat.JSON] = lambda obj: to_json(obj).encode()
        self._deserializers[SerializationFormat.JSON] = lambda data: from_json(data.decode())

        # Pickle serializer with security
        # Local imports
        from main.utils.core import secure_dumps, secure_loads

        self._serializers[SerializationFormat.PICKLE] = secure_dumps
        self._deserializers[SerializationFormat.PICKLE] = secure_loads

        # Binary (no serialization)
        self._serializers[SerializationFormat.BINARY] = lambda obj: (
            obj if isinstance(obj, bytes) else str(obj).encode()
        )
        self._deserializers[SerializationFormat.BINARY] = lambda data: data

    # Core state operations
    async def get(
        self, key: str, namespace: str = "default", backend: StorageBackend | None = None
    ) -> Any | None:
        """
        Get state value by key.

        Args:
            key: State key
            namespace: State namespace
            backend: Specific backend to use

        Returns:
            Deserialized state value or None
        """
        full_key = self._build_key(key, namespace)
        backend = backend or self.config.default_backend

        try:
            storage = self._backends.get(backend)
            if not storage:
                raise ValueError(f"Backend {backend} not available")

            # Get raw data
            raw_data = await storage.get(full_key)
            if raw_data is None:
                self._metrics["cache_misses"] += 1
                return None

            # Deserialize
            value = await self._deserialize(raw_data, self.config.default_serialization)

            self._metrics["operations"]["get"] += 1
            self._metrics["cache_hits"] += 1

            return value

        except Exception as e:
            logger.error(f"Failed to get state {full_key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl_seconds: int | None = None,
        backend: StorageBackend | None = None,
        serialization: SerializationFormat | None = None,
    ) -> bool:
        """
        Set state value.

        Args:
            key: State key
            value: Value to store
            namespace: State namespace
            ttl_seconds: Time to live in seconds
            backend: Specific backend to use
            serialization: Serialization format

        Returns:
            True if successful
        """
        full_key = self._build_key(key, namespace)
        backend = backend or self.config.default_backend
        serialization = serialization or self.config.default_serialization
        ttl_seconds = ttl_seconds or self.config.default_ttl_seconds

        try:
            # Validate if validator exists
            if namespace in self._validators:
                if not await self._validate_value(namespace, value):
                    self._metrics["validation_errors"] += 1
                    return False

            # Serialize
            raw_data = await self._serialize(value, serialization)

            # Store
            storage = self._backends.get(backend)
            if not storage:
                raise ValueError(f"Backend {backend} not available")

            result = await storage.set(full_key, raw_data, ttl_seconds)

            if result:
                self._metrics["operations"]["set"] += 1

            return result

        except Exception as e:
            logger.error(f"Failed to set state {full_key}: {e}")
            self._metrics["serialization_errors"] += 1
            return False

    async def delete(
        self, key: str, namespace: str = "default", backend: StorageBackend | None = None
    ) -> bool:
        """
        Delete state value.

        Args:
            key: State key
            namespace: State namespace
            backend: Specific backend to use

        Returns:
            True if successful
        """
        full_key = self._build_key(key, namespace)
        backend = backend or self.config.default_backend

        try:
            storage = self._backends.get(backend)
            if not storage:
                raise ValueError(f"Backend {backend} not available")

            result = await storage.delete(full_key)

            if result:
                self._metrics["operations"]["delete"] += 1

            return result

        except Exception as e:
            logger.error(f"Failed to delete state {full_key}: {e}")
            return False

    async def exists(
        self, key: str, namespace: str = "default", backend: StorageBackend | None = None
    ) -> bool:
        """Check if state key exists."""
        full_key = self._build_key(key, namespace)
        backend = backend or self.config.default_backend

        try:
            storage = self._backends.get(backend)
            if not storage:
                return False

            return await storage.exists(full_key)

        except Exception as e:
            logger.error(f"Failed to check existence of {full_key}: {e}")
            return False

    async def keys(
        self, pattern: str = "*", namespace: str = "default", backend: StorageBackend | None = None
    ) -> list[str]:
        """List keys matching pattern."""
        namespace_pattern = self._build_key(pattern, namespace)
        backend = backend or self.config.default_backend

        try:
            storage = self._backends.get(backend)
            if not storage:
                return []

            full_keys = await storage.keys(namespace_pattern)

            # Strip namespace prefix
            prefix = f"{namespace}:"
            return [key[len(prefix) :] for key in full_keys if key.startswith(prefix)]

        except Exception as e:
            logger.error(f"Failed to list keys with pattern {namespace_pattern}: {e}")
            return []

    # Proxy methods for sub-managers
    async def checkpoint(self, namespace: str = "default") -> str:
        """Create a checkpoint of namespace state."""
        return await self.persistence.checkpoint(namespace)

    async def restore(self, checkpoint_id: str, namespace: str = "default") -> bool:
        """Restore state from checkpoint."""
        return await self.persistence.restore(checkpoint_id, namespace)

    def lock(self, resource: str, timeout: float = 10.0):
        """Acquire distributed lock for resource."""
        return self.context.lock(resource, timeout)

    # Validation and serialization
    def register_validator(self, namespace: str, validator: Callable[[Any], bool]) -> None:
        """Register validator for namespace."""
        self._validators[namespace] = validator

    async def _validate_value(self, namespace: str, value: Any) -> bool:
        """Validate value for namespace."""
        validator = self._validators.get(namespace)
        if validator:
            try:
                if asyncio.iscoroutinefunction(validator):
                    return await validator(value)
                else:
                    return validator(value)
            except Exception as e:
                logger.error(f"Validation failed for namespace {namespace}: {e}")
                return False
        return True

    async def _serialize(self, value: Any, format: SerializationFormat) -> bytes:
        """Serialize value to bytes."""
        serializer = self._serializers.get(format)
        if not serializer:
            raise ValueError(f"No serializer for format {format}")

        return serializer(value)

    async def _deserialize(self, data: bytes, format: SerializationFormat) -> Any:
        """Deserialize bytes to value."""
        deserializer = self._deserializers.get(format)
        if not deserializer:
            raise ValueError(f"No deserializer for format {format}")

        return deserializer(data)

    # Utility methods
    def _build_key(self, key: str, namespace: str) -> str:
        """Build full key with namespace."""
        return f"{namespace}:{key}"

    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        if self.config.cleanup_interval_seconds > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        if self.config.checkpoint_interval_seconds > 0:
            self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                await self._cleanup_expired_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _checkpoint_loop(self) -> None:
        """Background checkpoint loop."""
        while True:
            try:
                await asyncio.sleep(self.config.checkpoint_interval_seconds)
                # Auto-checkpoint critical namespaces
                await self.persistence.auto_checkpoint(["config", "portfolio", "risk"])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}")

    async def _cleanup_expired_state(self) -> None:
        """Clean up expired state across all backends."""
        for backend in self._backends.values():
            try:
                await backend.cleanup()
            except Exception as e:
                logger.error(f"Backend cleanup failed: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get state management metrics."""
        return {
            "operations": self._metrics["operations"].copy(),
            "cache_hits": self._metrics["cache_hits"],
            "cache_misses": self._metrics["cache_misses"],
            "cache_hit_rate": (
                self._metrics["cache_hits"]
                / (self._metrics["cache_hits"] + self._metrics["cache_misses"])
                if (self._metrics["cache_hits"] + self._metrics["cache_misses"]) > 0
                else 0
            ),
            "serialization_errors": self._metrics["serialization_errors"],
            "validation_errors": self._metrics["validation_errors"],
            "active_backends": list(self._backends.keys()),
            "active_locks": len(self.context._locks),
            "checkpoints": len(self.persistence._checkpoints),
        }

    async def cleanup(self) -> None:
        """Cleanup state manager."""
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._checkpoint_task:
            self._checkpoint_task.cancel()

        # Cleanup backends
        for backend in self._backends.values():
            await backend.cleanup()

        logger.info("StateManager cleaned up")


# Convenience functions
def create_state_config(
    backend: str = "memory",
    redis_url: str | None = None,
    file_path: str | None = None,
    enable_metrics: bool = True,
) -> StateConfig:
    """Create state configuration with common settings."""
    return StateConfig(
        default_backend=StorageBackend(backend),
        redis_url=redis_url,
        file_storage_path=file_path,
        enable_metrics=enable_metrics,
    )


# Global state manager instance
_global_state_manager: StateManager | None = None


def get_state_manager(config: StateConfig | None = None) -> StateManager:
    """Get or create global state manager."""
    global _global_state_manager

    if _global_state_manager is None:
        if config is None:
            config = StateConfig()
        _global_state_manager = StateManager(config)

    return _global_state_manager

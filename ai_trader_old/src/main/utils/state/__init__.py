"""State management package."""

from .backends import FileBackend, MemoryBackend, RedisBackend, StorageBackendInterface
from .context import StateContext
from .manager import StateManager, create_state_config, get_state_manager
from .persistence import StatePersistence
from .types import (
    SerializationFormat,
    StateCheckpoint,
    StateConfig,
    StateMetadata,
    StateScope,
    StorageBackend,
)

__all__ = [
    # Types
    "StorageBackend",
    "SerializationFormat",
    "StateScope",
    "StateMetadata",
    "StateCheckpoint",
    "StateConfig",
    # Backends
    "StorageBackendInterface",
    "MemoryBackend",
    "RedisBackend",
    "FileBackend",
    # Manager
    "StateManager",
    "create_state_config",
    "get_state_manager",
    # Persistence
    "StatePersistence",
    # Context
    "StateContext",
]

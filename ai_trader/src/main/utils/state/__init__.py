"""State management package."""

from .types import (
    StorageBackend,
    SerializationFormat,
    StateScope,
    StateMetadata,
    StateCheckpoint,
    StateConfig
)

from .backends import (
    StorageBackendInterface,
    MemoryBackend,
    RedisBackend,
    FileBackend
)

from .manager import (
    StateManager,
    create_state_config,
    get_state_manager
)

from .persistence import StatePersistence

from .context import StateContext

__all__ = [
    # Types
    'StorageBackend',
    'SerializationFormat',
    'StateScope',
    'StateMetadata',
    'StateCheckpoint',
    'StateConfig',
    
    # Backends
    'StorageBackendInterface',
    'MemoryBackend',
    'RedisBackend',
    'FileBackend',
    
    # Manager
    'StateManager',
    'create_state_config',
    'get_state_manager',
    
    # Persistence
    'StatePersistence',
    
    # Context
    'StateContext'
]
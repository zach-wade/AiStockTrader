"""
State Management Types

Data classes and enums for state management system.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StorageBackend(Enum):
    """Storage backend types."""

    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"
    DATABASE = "database"
    HYBRID = "hybrid"


class SerializationFormat(Enum):
    """Serialization format options."""

    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    BINARY = "binary"
    AUTO = "auto"


class StateScope(Enum):
    """State scope levels."""

    SESSION = "session"  # Temporary, cleared on restart
    PERSISTENT = "persistent"  # Survives restarts
    SHARED = "shared"  # Shared across processes
    DISTRIBUTED = "distributed"  # Distributed across nodes


@dataclass
class StateMetadata:
    """Metadata for stored state."""

    key: str
    namespace: str
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    last_accessed: datetime | None = None
    ttl_seconds: int | None = None
    expires_at: datetime | None = None
    size_bytes: int = 0
    checksum: str | None = None
    tags: list[str] = field(default_factory=list)
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if state has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get age of state in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()


@dataclass
class StateCheckpoint:
    """State checkpoint for recovery."""

    checkpoint_id: str
    namespace: str
    created_at: datetime
    state_keys: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat(),
            "state_keys": self.state_keys,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateCheckpoint":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            namespace=data["namespace"],
            created_at=datetime.fromisoformat(data["created_at"]),
            state_keys=data["state_keys"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class StateConfig:
    """Configuration for state management."""

    default_backend: StorageBackend = StorageBackend.MEMORY
    default_serialization: SerializationFormat = SerializationFormat.JSON
    default_ttl_seconds: int | None = None
    enable_compression: bool = False
    enable_encryption: bool = False
    max_memory_size_mb: float = 100.0
    checkpoint_interval_seconds: int = 300
    cleanup_interval_seconds: int = 3600
    redis_url: str | None = None
    database_url: str | None = None
    file_storage_path: str | None = None
    enable_metrics: bool = True
    enable_validation: bool = True

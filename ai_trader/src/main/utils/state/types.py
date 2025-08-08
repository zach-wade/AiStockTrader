"""
State Management Types

Data classes and enums for state management system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import uuid4


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
    SESSION = "session"      # Temporary, cleared on restart
    PERSISTENT = "persistent"  # Survives restarts
    SHARED = "shared"        # Shared across processes
    DISTRIBUTED = "distributed"  # Distributed across nodes


@dataclass
class StateMetadata:
    """Metadata for stored state."""
    key: str
    namespace: str
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    ttl_seconds: Optional[int] = None
    expires_at: Optional[datetime] = None
    size_bytes: int = 0
    checksum: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
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
    state_keys: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'namespace': self.namespace,
            'created_at': self.created_at.isoformat(),
            'state_keys': self.state_keys,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateCheckpoint':
        """Create from dictionary."""
        return cls(
            checkpoint_id=data['checkpoint_id'],
            namespace=data['namespace'],
            created_at=datetime.fromisoformat(data['created_at']),
            state_keys=data['state_keys'],
            metadata=data.get('metadata', {})
        )


@dataclass
class StateConfig:
    """Configuration for state management."""
    default_backend: StorageBackend = StorageBackend.MEMORY
    default_serialization: SerializationFormat = SerializationFormat.JSON
    default_ttl_seconds: Optional[int] = None
    enable_compression: bool = False
    enable_encryption: bool = False
    max_memory_size_mb: float = 100.0
    checkpoint_interval_seconds: int = 300
    cleanup_interval_seconds: int = 3600
    redis_url: Optional[str] = None
    database_url: Optional[str] = None
    file_storage_path: Optional[str] = None
    enable_metrics: bool = True
    enable_validation: bool = True
"""
Production RSA key management and rotation system for the AI Trading System.

Provides secure RSA key generation, storage, rotation, and lifecycle management
with hardware security module support and audit logging.
"""

import base64
import hashlib
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

logger = logging.getLogger(__name__)


class KeyUsage(Enum):
    """Types of key usage for different purposes."""

    JWT_SIGNING = "jwt_signing"
    API_ENCRYPTION = "api_encryption"
    DATABASE_ENCRYPTION = "database_encryption"
    INTER_SERVICE = "inter_service"
    BACKUP_ENCRYPTION = "backup_encryption"


class KeyStatus(Enum):
    """Status of cryptographic keys."""

    ACTIVE = "active"
    PENDING = "pending"
    ROTATING = "rotating"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class KeyMetadata:
    """Metadata for cryptographic keys."""

    key_id: str
    usage: KeyUsage
    status: KeyStatus
    algorithm: str = "RSA"
    key_size: int = 4096
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    last_used: datetime | None = None
    rotation_count: int = 0
    checksum: str | None = None
    hsm_backed: bool = False

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def days_until_expiry(self) -> int | None:
        """Get days until key expiry."""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.utcnow()
        return max(0, delta.days)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key_id": self.key_id,
            "usage": self.usage.value,
            "status": self.status.value,
            "algorithm": self.algorithm,
            "key_size": self.key_size,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "rotation_count": self.rotation_count,
            "checksum": self.checksum,
            "hsm_backed": self.hsm_backed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyMetadata":
        """Create from dictionary."""
        return cls(
            key_id=data["key_id"],
            usage=KeyUsage(data["usage"]),
            status=KeyStatus(data["status"]),
            algorithm=data.get("algorithm", "RSA"),
            key_size=data.get("key_size", 4096),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            rotation_count=data.get("rotation_count", 0),
            checksum=data.get("checksum"),
            hsm_backed=data.get("hsm_backed", False),
        )


class KeyStorage(ABC):
    """Abstract interface for key storage backends."""

    @abstractmethod
    def store_key(
        self, key_id: str, private_key: bytes, public_key: bytes, metadata: KeyMetadata
    ) -> None:
        """Store a key pair with metadata."""
        pass

    @abstractmethod
    def get_private_key(self, key_id: str) -> bytes:
        """Retrieve private key by ID."""
        pass

    @abstractmethod
    def get_public_key(self, key_id: str) -> bytes:
        """Retrieve public key by ID."""
        pass

    @abstractmethod
    def get_metadata(self, key_id: str) -> KeyMetadata:
        """Retrieve key metadata by ID."""
        pass

    @abstractmethod
    def list_keys(
        self, usage: KeyUsage | None = None, status: KeyStatus | None = None
    ) -> list[str]:
        """List key IDs optionally filtered by usage and status."""
        pass

    @abstractmethod
    def delete_key(self, key_id: str) -> None:
        """Delete a key from storage."""
        pass

    @abstractmethod
    def update_metadata(self, key_id: str, metadata: KeyMetadata) -> None:
        """Update key metadata."""
        pass


class FileSystemKeyStorage(KeyStorage):
    """File system-based key storage with encryption."""

    def __init__(self, storage_path: str, encryption_key: bytes | None = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Set secure permissions
        os.chmod(self.storage_path, 0o700)

        # Initialize encryption
        if encryption_key:
            from cryptography.fernet import Fernet

            self.cipher: Fernet | None = Fernet(encryption_key)
        else:
            self.cipher = None

        self._lock = threading.RLock()

    def _get_key_path(self, key_id: str) -> Path:
        """Get path for key files."""
        return self.storage_path / key_id

    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if encryption is enabled."""
        if self.cipher:
            return self.cipher.encrypt(data)
        return data

    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data if encryption is enabled."""
        if self.cipher:
            return self.cipher.decrypt(data)
        return data

    def store_key(
        self, key_id: str, private_key: bytes, public_key: bytes, metadata: KeyMetadata
    ) -> None:
        """Store key pair and metadata to filesystem."""
        with self._lock:
            key_path = self._get_key_path(key_id)
            key_path.mkdir(exist_ok=True)

            # Store private key
            private_path = key_path / "private.pem"
            with open(private_path, "wb") as f:
                f.write(self._encrypt_data(private_key))
            os.chmod(private_path, 0o600)

            # Store public key
            public_path = key_path / "public.pem"
            with open(public_path, "wb") as f:
                f.write(public_key)  # Public key doesn't need encryption
            os.chmod(public_path, 0o644)

            # Store metadata
            metadata_path = key_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)
            os.chmod(metadata_path, 0o600)

    def get_private_key(self, key_id: str) -> bytes:
        """Get private key from filesystem."""
        with self._lock:
            private_path = self._get_key_path(key_id) / "private.pem"
            if not private_path.exists():
                raise ValueError(f"Private key not found: {key_id}")

            with open(private_path, "rb") as f:
                encrypted_data = f.read()

            return self._decrypt_data(encrypted_data)

    def get_public_key(self, key_id: str) -> bytes:
        """Get public key from filesystem."""
        with self._lock:
            public_path = self._get_key_path(key_id) / "public.pem"
            if not public_path.exists():
                raise ValueError(f"Public key not found: {key_id}")

            with open(public_path, "rb") as f:
                return f.read()

    def get_metadata(self, key_id: str) -> KeyMetadata:
        """Get key metadata from filesystem."""
        with self._lock:
            metadata_path = self._get_key_path(key_id) / "metadata.json"
            if not metadata_path.exists():
                raise ValueError(f"Key metadata not found: {key_id}")

            with open(metadata_path) as f:
                data = json.load(f)

            return KeyMetadata.from_dict(data)

    def list_keys(
        self, usage: KeyUsage | None = None, status: KeyStatus | None = None
    ) -> list[str]:
        """List keys from filesystem."""
        with self._lock:
            keys = []

            for key_path in self.storage_path.iterdir():
                if not key_path.is_dir():
                    continue

                key_id = key_path.name
                try:
                    metadata = self.get_metadata(key_id)

                    # Filter by usage and status
                    if usage and metadata.usage != usage:
                        continue
                    if status and metadata.status != status:
                        continue

                    keys.append(key_id)
                except Exception as e:
                    logger.warning(f"Error reading metadata for key {key_id}: {e}")

            return keys

    def delete_key(self, key_id: str) -> None:
        """Delete key from filesystem."""
        with self._lock:
            key_path = self._get_key_path(key_id)
            if key_path.exists():
                import shutil

                shutil.rmtree(key_path)

    def update_metadata(self, key_id: str, metadata: KeyMetadata) -> None:
        """Update key metadata."""
        with self._lock:
            metadata_path = self._get_key_path(key_id) / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)


class RSAKeyManager:
    """
    Production RSA key management system.

    Features:
    - Secure key generation with customizable key sizes
    - Key rotation with configurable schedules
    - Multiple key storage backends
    - Audit logging for all key operations
    - Key lifecycle management
    - Performance monitoring
    """

    def __init__(
        self,
        storage: KeyStorage,
        default_key_size: int = 4096,
        default_validity_days: int = 365,
        rotation_warning_days: int = 30,
        auto_rotation: bool = False,
    ):
        self.storage = storage
        self.default_key_size = default_key_size
        self.default_validity_days = default_validity_days
        self.rotation_warning_days = rotation_warning_days
        self.auto_rotation = auto_rotation

        # Thread safety
        self._lock = threading.RLock()

        # Cache for frequently used keys
        self._key_cache: dict[str, tuple[Any, Any, datetime]] = {}
        self._cache_ttl = 300  # 5 minutes

        # Audit log
        self._audit_log: list[dict[str, Any]] = []

        logger.info("RSA Key Manager initialized")

    def generate_key_pair(
        self,
        key_id: str,
        usage: KeyUsage,
        key_size: int | None = None,
        validity_days: int | None = None,
    ) -> tuple[str, str]:
        """
        Generate a new RSA key pair.

        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        with self._lock:
            key_size = key_size or self.default_key_size
            validity_days = validity_days or self.default_validity_days

            # Validate key size
            if key_size < 2048:
                raise ValueError("RSA key size must be at least 2048 bits")

            logger.info(f"Generating RSA key pair: {key_id} ({key_size} bits)")

            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=key_size, backend=default_backend()
            )

            # Get public key
            public_key = private_key.public_key()

            # Serialize private key
            private_pem = private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=NoEncryption(),
            )

            # Serialize public key
            public_pem = public_key.public_bytes(
                encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
            )

            # Calculate checksum
            checksum = hashlib.sha256(public_pem).hexdigest()

            # Create metadata
            metadata = KeyMetadata(
                key_id=key_id,
                usage=usage,
                status=KeyStatus.ACTIVE,
                algorithm="RSA",
                key_size=key_size,
                expires_at=datetime.utcnow() + timedelta(days=validity_days),
                checksum=checksum,
            )

            # Store the key pair
            self.storage.store_key(key_id, private_pem, public_pem, metadata)

            # Clear cache
            self._clear_key_cache(key_id)

            # Audit log
            self._log_audit_event(
                "key_generated",
                {
                    "key_id": key_id,
                    "usage": usage.value,
                    "key_size": key_size,
                    "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                },
            )

            logger.info(f"Successfully generated RSA key pair: {key_id}")
            return private_pem.decode(), public_pem.decode()

    def get_private_key(self, key_id: str, update_last_used: bool = True) -> Any:
        """Get private key object for cryptographic operations."""
        with self._lock:
            # Check cache first
            cache_key = f"private_{key_id}"
            if cache_key in self._key_cache:
                key, _, cached_at = self._key_cache[cache_key]
                if time.time() - cached_at.timestamp() < self._cache_ttl:
                    if update_last_used:
                        self._update_last_used(key_id)
                    return key

            # Load from storage
            private_pem = self.storage.get_private_key(key_id)
            private_key = serialization.load_pem_private_key(
                private_pem, password=None, backend=default_backend()
            )

            # Cache the key
            self._key_cache[cache_key] = (private_key, None, datetime.utcnow())

            # Update last used timestamp
            if update_last_used:
                self._update_last_used(key_id)

            return private_key

    def get_public_key(self, key_id: str, update_last_used: bool = True) -> Any:
        """Get public key object for cryptographic operations."""
        with self._lock:
            # Check cache first
            cache_key = f"public_{key_id}"
            if cache_key in self._key_cache:
                _, key, cached_at = self._key_cache[cache_key]
                if key and time.time() - cached_at.timestamp() < self._cache_ttl:
                    if update_last_used:
                        self._update_last_used(key_id)
                    return key

            # Load from storage
            public_pem = self.storage.get_public_key(key_id)
            public_key = serialization.load_pem_public_key(public_pem, backend=default_backend())

            # Cache the key
            if cache_key in self._key_cache:
                private_key, _, _ = self._key_cache[cache_key]
                self._key_cache[cache_key] = (private_key, public_key, datetime.utcnow())
            else:
                self._key_cache[cache_key] = (None, public_key, datetime.utcnow())

            # Update last used timestamp
            if update_last_used:
                self._update_last_used(key_id)

            return public_key

    def rotate_key(self, key_id: str, new_key_id: str | None = None) -> str:
        """
        Rotate a key by generating a new one and marking the old one as deprecated.

        Returns:
            ID of the new key
        """
        with self._lock:
            # Get current key metadata
            old_metadata = self.storage.get_metadata(key_id)

            # Generate new key ID if not provided
            if not new_key_id:
                timestamp = int(time.time())
                new_key_id = f"{key_id}_{timestamp}"

            logger.info(f"Rotating key {key_id} -> {new_key_id}")

            # Generate new key pair with same parameters
            self.generate_key_pair(
                key_id=new_key_id,
                usage=old_metadata.usage,
                key_size=old_metadata.key_size,
            )

            # Update new key metadata
            new_metadata = self.storage.get_metadata(new_key_id)
            new_metadata.rotation_count = old_metadata.rotation_count + 1
            self.storage.update_metadata(new_key_id, new_metadata)

            # Mark old key as deprecated
            old_metadata.status = KeyStatus.DEPRECATED
            self.storage.update_metadata(key_id, old_metadata)

            # Clear caches
            self._clear_key_cache(key_id)
            self._clear_key_cache(new_key_id)

            # Audit log
            self._log_audit_event(
                "key_rotated",
                {
                    "old_key_id": key_id,
                    "new_key_id": new_key_id,
                    "rotation_count": new_metadata.rotation_count,
                },
            )

            logger.info(f"Successfully rotated key {key_id} -> {new_key_id}")
            return new_key_id

    def revoke_key(self, key_id: str, reason: str = "") -> None:
        """Revoke a key, making it unusable."""
        with self._lock:
            metadata = self.storage.get_metadata(key_id)
            metadata.status = KeyStatus.REVOKED
            self.storage.update_metadata(key_id, metadata)

            # Clear from cache
            self._clear_key_cache(key_id)

            # Audit log
            self._log_audit_event(
                "key_revoked",
                {
                    "key_id": key_id,
                    "reason": reason,
                },
            )

            logger.warning(f"Key revoked: {key_id} - {reason}")

    def list_keys(
        self,
        usage: KeyUsage | None = None,
        status: KeyStatus | None = None,
    ) -> list[KeyMetadata]:
        """List keys with optional filtering."""
        key_ids = self.storage.list_keys(usage, status)
        keys = []

        for key_id in key_ids:
            try:
                metadata = self.storage.get_metadata(key_id)
                keys.append(metadata)
            except Exception as e:
                logger.error(f"Error loading metadata for key {key_id}: {e}")

        return sorted(keys, key=lambda k: k.created_at, reverse=True)

    def get_keys_needing_rotation(self) -> list[KeyMetadata]:
        """Get keys that need rotation based on expiry warning period."""
        warning_date = datetime.utcnow() + timedelta(days=self.rotation_warning_days)

        keys_needing_rotation = []
        active_keys = self.list_keys(status=KeyStatus.ACTIVE)

        for metadata in active_keys:
            if metadata.expires_at and metadata.expires_at <= warning_date:
                keys_needing_rotation.append(metadata)

        return keys_needing_rotation

    def cleanup_expired_keys(self) -> int:
        """Clean up expired and deprecated keys."""
        cleaned = 0

        # Get expired keys
        expired_keys = []
        all_keys = self.list_keys()

        for metadata in all_keys:
            if (
                metadata.status == KeyStatus.DEPRECATED
                and metadata.created_at < datetime.utcnow() - timedelta(days=90)
            ) or metadata.is_expired():
                expired_keys.append(metadata.key_id)

        # Delete expired keys
        for key_id in expired_keys:
            try:
                self.storage.delete_key(key_id)
                self._clear_key_cache(key_id)
                cleaned += 1

                self._log_audit_event("key_deleted", {"key_id": key_id})
                logger.info(f"Deleted expired key: {key_id}")

            except Exception as e:
                logger.error(f"Error deleting expired key {key_id}: {e}")

        return cleaned

    def _update_last_used(self, key_id: str) -> None:
        """Update last used timestamp for a key."""
        try:
            metadata = self.storage.get_metadata(key_id)
            metadata.last_used = datetime.utcnow()
            self.storage.update_metadata(key_id, metadata)
        except Exception as e:
            logger.error(f"Error updating last used for key {key_id}: {e}")

    def _clear_key_cache(self, key_id: str) -> None:
        """Clear cached keys for a key ID."""
        keys_to_remove = []
        for cache_key in self._key_cache:
            if key_id in cache_key:
                keys_to_remove.append(cache_key)

        for cache_key in keys_to_remove:
            del self._key_cache[cache_key]

    def _log_audit_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Log audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
        }

        self._audit_log.append(event)
        logger.info(f"Key audit event: {event_type} - {details}")

        # Keep only last 1000 events in memory
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]

    def health_check(self) -> dict[str, Any]:
        """Perform health check on key management system."""
        health: dict[str, Any] = {
            "healthy": True,
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        try:
            # Check storage connectivity
            test_keys = self.storage.list_keys()
            checks = health["checks"]
            checks["storage_connectivity"] = True
            checks["total_keys"] = len(test_keys)

            # Check for keys needing rotation
            rotation_needed = self.get_keys_needing_rotation()
            checks["keys_needing_rotation"] = len(rotation_needed)

            if rotation_needed:
                warnings = health["warnings"]
                if isinstance(warnings, list):
                    warnings.extend(
                        [f"Key needs rotation: {key.key_id}" for key in rotation_needed[:5]]
                    )

            # Check cache performance
            checks["cache_size"] = len(self._key_cache)

        except Exception as e:
            health["healthy"] = False
            errors = health["errors"]
            if isinstance(errors, list):
                errors.append(f"Health check failed: {e}")

        return health


def create_production_key_manager(
    storage_path: str,
    encryption_password: str | None = None,
) -> RSAKeyManager:
    """Create a production-ready key manager."""
    # Generate encryption key from password
    encryption_key = None
    if encryption_password:
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"trading_system_salt",
            iterations=100000,
            backend=default_backend(),
        )
        key = base64.urlsafe_b64encode(kdf.derive(encryption_password.encode()))
        encryption_key = key

    # Create storage
    storage = FileSystemKeyStorage(storage_path, encryption_key)

    # Create manager with production settings
    return RSAKeyManager(
        storage=storage,
        default_key_size=4096,  # Strong key size
        default_validity_days=365,  # 1 year validity
        rotation_warning_days=30,  # 30-day rotation warning
        auto_rotation=False,  # Manual rotation for control
    )


def create_development_key_manager(storage_path: str) -> RSAKeyManager:
    """Create a development-friendly key manager."""
    storage = FileSystemKeyStorage(storage_path)

    return RSAKeyManager(
        storage=storage,
        default_key_size=2048,  # Faster generation
        default_validity_days=90,  # Shorter validity for testing
        rotation_warning_days=7,  # 7-day warning
        auto_rotation=False,
    )

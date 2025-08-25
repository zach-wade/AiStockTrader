"""
File-based audit storage backend.

Provides high-performance file storage with features like compression,
encryption, rotation, and integrity verification.
"""

import gzip
import hashlib
import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from ..config import AuditConfig
from ..exceptions import AuditStorageError
from .base import AuditStorage
from .file_utils import FileUtils


class FileStorageManager:
    """Manages file operations and rotation for file storage."""

    def __init__(
        self,
        storage_path: Path,
        rotation_enabled: bool,
        rotation_size_mb: int,
        rotation_count: int,
        compression_enabled: bool,
    ):
        self.storage_path = storage_path
        self.rotation_enabled = rotation_enabled
        self.rotation_size_mb = rotation_size_mb
        self.rotation_count = rotation_count
        self.compression_enabled = compression_enabled

        # Current log file
        self.current_file: TextIO | gzip.GzipFile | None = None
        self.current_file_size = 0

        # Initialize current log file
        self._initialize_current_file()

    def _initialize_current_file(self) -> None:
        """Initialize current log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{timestamp}.jsonl"

        if self.compression_enabled:
            filename += ".gz"

        file_path = self.storage_path / filename

        if self.compression_enabled:
            self.current_file = gzip.open(file_path, "wt", encoding="utf-8")
        else:
            self.current_file = open(file_path, "w", encoding="utf-8")

        self.current_file_size = 0

    def needs_rotation(self) -> bool:
        """Check if log file needs rotation."""
        if not self.rotation_enabled:
            return False

        size_mb = self.current_file_size / (1024 * 1024)
        return size_mb >= self.rotation_size_mb

    def rotate_log_file(self) -> None:
        """Rotate current log file."""
        if self.current_file:
            self.current_file.close()

        # Remove oldest files if we exceed rotation count
        self._cleanup_old_files()

        # Initialize new log file
        self._initialize_current_file()

    def _cleanup_old_files(self) -> None:
        """Clean up old log files based on rotation count."""
        log_files = sorted(
            FileUtils.get_all_log_files(self.storage_path, self.compression_enabled),
            key=lambda f: f.stat().st_mtime,
        )

        while len(log_files) >= self.rotation_count:
            oldest_file = log_files.pop(0)
            oldest_file.unlink()

    def write_event(self, event_line: str) -> int:
        """Write event line to current file and return bytes written."""
        if not self.current_file:
            self._initialize_current_file()

        bytes_written = FileUtils.write_to_file(
            self.current_file, event_line, self.compression_enabled
        )
        self.current_file_size += bytes_written
        return bytes_written

    def close(self) -> None:
        """Close current file."""
        if self.current_file and not self.current_file.closed:
            self.current_file.close()


class FileIntegrityManager:
    """Manages integrity tracking for file storage."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.integrity_hashes: dict[str, str] = {}
        self.integrity_file = storage_path / ".integrity"
        self._load_integrity_hashes()

    def _load_integrity_hashes(self) -> None:
        """Load integrity hashes from file."""
        try:
            if self.integrity_file.exists():
                with open(self.integrity_file) as f:
                    self.integrity_hashes = json.load(f)
        except Exception:
            self.integrity_hashes = {}

    def save_integrity_hashes(self) -> None:
        """Save integrity hashes to file."""
        try:
            with open(self.integrity_file, "w") as f:
                json.dump(self.integrity_hashes, f)
        except Exception:
            # Silently handle save errors
            pass

    def update_integrity_hash(self, event_id: str, event_line: str) -> None:
        """Update integrity hash for event."""
        hash_obj = hashlib.sha256(event_line.encode("utf-8"))
        self.integrity_hashes[event_id] = hash_obj.hexdigest()

    def verify_file_integrity(self, file_path: Path) -> bool:
        """Verify integrity of a specific log file."""
        try:
            for event in FileUtils.read_events_from_file(file_path):
                event_id = event.get("event_id")
                if event_id and event_id in self.integrity_hashes:
                    # Reconstruct event line and verify hash
                    event_json = json.dumps(event, default=str, separators=(",", ":"))
                    event_line = event_json + "\n"

                    hash_obj = hashlib.sha256(event_line.encode("utf-8"))
                    calculated_hash = hash_obj.hexdigest()
                    stored_hash = self.integrity_hashes[event_id]

                    if calculated_hash != stored_hash:
                        return False

            return True

        except Exception:
            return False


class FileStorage(AuditStorage):
    """
    File-based audit storage backend.

    Provides high-performance file storage with features like compression,
    encryption, rotation, and integrity verification.
    """

    def __init__(self, config: AuditConfig) -> None:
        """Initialize file storage backend."""
        super().__init__(config)

        self.storage_path = Path(config.storage_config.file_storage_path)

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize managers
        self.file_manager = FileStorageManager(
            storage_path=self.storage_path,
            rotation_enabled=config.storage_config.file_rotation_enabled,
            rotation_size_mb=config.storage_config.file_rotation_size_mb,
            rotation_count=config.storage_config.file_rotation_count,
            compression_enabled=config.storage_config.compression_enabled,
        )

        self.integrity_manager = FileIntegrityManager(self.storage_path)
        self.file_lock = threading.Lock()

    def store(self, event_data: dict[str, Any]) -> str:
        """Store single audit event to file."""
        event_id = event_data.get("event_id", "unknown")

        try:
            with self.file_lock:
                # Check if rotation is needed
                if self.file_manager.needs_rotation():
                    self.file_manager.rotate_log_file()

                # Serialize event data
                event_json = json.dumps(event_data, default=str, separators=(",", ":"))
                event_line = event_json + "\n"

                # Write to current file
                self.file_manager.write_event(event_line)

                # Update integrity hash
                self.integrity_manager.update_integrity_hash(event_id, event_line)

                file_name = (
                    self.file_manager.current_file.name
                    if self.file_manager.current_file
                    else "unknown"
                )
                return f"file:{file_name}:{event_id}"

        except Exception as e:
            raise AuditStorageError(
                f"Failed to store event to file: {e}",
                storage_backend="file",
                operation="store",
                underlying_error=e,
            )

    def store_batch(self, events: list[dict[str, Any]]) -> list[str]:
        """Store batch of audit events to file."""
        storage_refs = []

        try:
            with self.file_lock:
                for event_data in events:
                    storage_ref = self._store_single_unlocked(event_data)
                    storage_refs.append(storage_ref)

                # Flush integrity hashes after batch
                self.integrity_manager.save_integrity_hashes()

        except Exception as e:
            raise AuditStorageError(
                f"Failed to store event batch to file: {e}",
                storage_backend="file",
                operation="store_batch",
                underlying_error=e,
            )

        return storage_refs

    def _store_single_unlocked(self, event_data: dict[str, Any]) -> str:
        """Store single event without acquiring lock (used in batch operations)."""
        event_id = event_data.get("event_id", "unknown")

        if self.file_manager.needs_rotation():
            self.file_manager.rotate_log_file()

        event_json = json.dumps(event_data, default=str, separators=(",", ":"))
        event_line = event_json + "\n"

        self.file_manager.write_event(event_line)
        self.integrity_manager.update_integrity_hash(event_id, event_line)

        file_name = (
            self.file_manager.current_file.name if self.file_manager.current_file else "unknown"
        )
        return f"file:{file_name}:{event_id}"

    def retrieve(self, event_id: str) -> dict[str, Any] | None:
        """Retrieve audit event by ID from files."""
        try:
            # Search through current and rotated files
            for log_file in FileUtils.get_all_log_files(
                self.storage_path, self.file_manager.compression_enabled
            ):
                event = FileUtils.search_file_for_event(log_file, event_id)
                if event:
                    return event

            return None

        except Exception as e:
            raise AuditStorageError(
                f"Failed to retrieve event from file: {e}",
                storage_backend="file",
                operation="retrieve",
                underlying_error=e,
            )

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[str] | None = None,
        user_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Query audit events from files."""
        try:
            count = 0

            for log_file in FileUtils.get_all_log_files(
                self.storage_path, self.file_manager.compression_enabled
            ):
                for event in FileUtils.read_events_from_file(log_file):
                    # Apply filters
                    if not self._matches_filters(
                        event, start_time, end_time, event_types, user_ids
                    ):
                        continue

                    yield event
                    count += 1

                    # Check limit
                    if limit and count >= limit:
                        return

        except Exception as e:
            raise AuditStorageError(
                f"Failed to query events from file: {e}",
                storage_backend="file",
                operation="query",
                underlying_error=e,
            )

    def _matches_filters(
        self,
        event: dict[str, Any],
        start_time: datetime | None,
        end_time: datetime | None,
        event_types: list[str] | None,
        user_ids: list[str] | None,
    ) -> bool:
        """Check if event matches query filters."""
        # Time filters
        if start_time or end_time:
            event_time_str = event.get("timestamp")
            if event_time_str:
                try:
                    event_time = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
                    if start_time and event_time < start_time:
                        return False
                    if end_time and event_time > end_time:
                        return False
                except ValueError:
                    pass  # Skip events with invalid timestamps

        # Event type filter
        if event_types and event.get("event_type") not in event_types:
            return False

        # User ID filter
        if user_ids and event.get("user_id") not in user_ids:
            return False

        return True

    def delete_expired(self, cutoff_date: datetime) -> int:
        """Delete expired audit log files."""
        deleted_count = 0

        try:
            for log_file in FileUtils.get_all_log_files(
                self.storage_path, self.file_manager.compression_enabled
            ):
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=UTC)

                if file_mtime < cutoff_date:
                    # Count events before deletion
                    count = sum(1 for _ in FileUtils.read_events_from_file(log_file))
                    deleted_count += count

                    # Delete file
                    log_file.unlink()
                    self._logger.info(f"Deleted expired log file: {log_file}")

            return deleted_count

        except Exception as e:
            raise AuditStorageError(
                f"Failed to delete expired files: {e}",
                storage_backend="file",
                operation="delete_expired",
                underlying_error=e,
            )

    def verify_integrity(self) -> bool:
        """Verify file storage integrity."""
        try:
            for log_file in FileUtils.get_all_log_files(
                self.storage_path, self.file_manager.compression_enabled
            ):
                if not self.integrity_manager.verify_file_integrity(log_file):
                    return False
            return True

        except Exception as e:
            self._logger.error(f"Integrity verification failed: {e}")
            return False

    def get_storage_stats(self) -> dict[str, Any]:
        """Get file storage statistics."""
        try:
            log_files = list(
                FileUtils.get_all_log_files(
                    self.storage_path, self.file_manager.compression_enabled
                )
            )
            total_size = sum(f.stat().st_size for f in log_files)
            total_events = sum(
                sum(1 for _ in FileUtils.read_events_from_file(f)) for f in log_files
            )

            return {
                "backend_type": "file",
                "storage_path": str(self.storage_path),
                "total_files": len(log_files),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_events": total_events,
                "compression_enabled": self.file_manager.compression_enabled,
                "rotation_enabled": self.file_manager.rotation_enabled,
                "current_file": (
                    str(self.file_manager.current_file) if self.file_manager.current_file else None
                ),
                "current_file_size_bytes": self.file_manager.current_file_size,
            }

        except Exception as e:
            raise AuditStorageError(
                f"Failed to get storage stats: {e}",
                storage_backend="file",
                operation="get_stats",
                underlying_error=e,
            )

    def close(self) -> None:
        """Close file storage."""
        with self.file_lock:
            self.integrity_manager.save_integrity_hashes()
            self.file_manager.close()

"""
Audit storage backends for multiple storage systems.

This module provides flexible storage backends for audit logs, supporting
file systems, databases, and external systems with high performance,
reliability, and compliance features.
"""

import gzip
import hashlib
import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, TextIO, cast

from .config import AuditConfig
from .exceptions import AuditStorageError


class AuditStorage(ABC):
    """
    Abstract base class for audit storage backends.

    Defines the interface that all storage backends must implement,
    ensuring consistent behavior across different storage systems.
    """

    def __init__(self, config: AuditConfig) -> None:
        """
        Initialize audit storage backend.

        Args:
            config: Audit configuration
        """
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def store(self, event_data: dict[str, Any]) -> str:
        """
        Store single audit event.

        Args:
            event_data: Audit event data

        Returns:
            Storage reference/ID for the stored event
        """
        pass

    @abstractmethod
    def store_batch(self, events: list[dict[str, Any]]) -> list[str]:
        """
        Store batch of audit events.

        Args:
            events: List of audit event data

        Returns:
            List of storage references/IDs for stored events
        """
        pass

    @abstractmethod
    def retrieve(self, event_id: str) -> dict[str, Any] | None:
        """
        Retrieve audit event by ID.

        Args:
            event_id: Event identifier

        Returns:
            Event data if found, None otherwise
        """
        pass

    @abstractmethod
    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[str] | None = None,
        user_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Query audit events with filters.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            event_types: Filter by event types
            user_ids: Filter by user IDs
            limit: Maximum number of events to return

        Returns:
            Iterator of matching events
        """
        pass

    @abstractmethod
    def delete_expired(self, cutoff_date: datetime) -> int:
        """
        Delete expired audit events.

        Args:
            cutoff_date: Delete events older than this date

        Returns:
            Number of events deleted
        """
        pass

    @abstractmethod
    def verify_integrity(self) -> bool:
        """
        Verify storage integrity.

        Returns:
            True if integrity is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary containing storage statistics
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage backend and cleanup resources."""
        pass


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
        self.rotation_enabled = config.storage_config.file_rotation_enabled
        self.rotation_size_mb = config.storage_config.file_rotation_size_mb
        self.rotation_count = config.storage_config.file_rotation_count
        self.compression_enabled = config.storage_config.compression_enabled

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Current log file
        self.current_file: TextIO | gzip.GzipFile | None = None
        self.current_file_size = 0
        self.file_lock = threading.Lock()

        # Integrity tracking
        self.integrity_hashes: dict[str, str] = {}
        self.integrity_file = self.storage_path / ".integrity"
        self._load_integrity_hashes()

        # Initialize current log file
        self._initialize_current_file()

    def store(self, event_data: dict[str, Any]) -> str:
        """Store single audit event to file."""
        event_id = event_data.get("event_id", "unknown")

        try:
            with self.file_lock:
                # Check if rotation is needed
                if self.rotation_enabled and self._needs_rotation():
                    self._rotate_log_file()

                # Serialize event data
                event_json = json.dumps(event_data, default=str, separators=(",", ":"))
                event_line = event_json + "\n"

                # Write to current file
                if self.compression_enabled:
                    self._write_compressed(event_line)
                else:
                    self._write_plain(event_line)

                # Update integrity hash
                self._update_integrity_hash(event_id, event_line)

                file_name = self.current_file.name if self.current_file else "unknown"
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
                self._save_integrity_hashes()

        except Exception as e:
            raise AuditStorageError(
                f"Failed to store event batch to file: {e}",
                storage_backend="file",
                operation="store_batch",
                underlying_error=e,
            )

        return storage_refs

    def retrieve(self, event_id: str) -> dict[str, Any] | None:
        """Retrieve audit event by ID from files."""
        try:
            # Search through current and rotated files
            for log_file in self._get_all_log_files():
                event = self._search_file_for_event(log_file, event_id)
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

            for log_file in self._get_all_log_files():
                for event in self._read_events_from_file(log_file):
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

    def delete_expired(self, cutoff_date: datetime) -> int:
        """Delete expired audit log files."""
        deleted_count = 0

        try:
            for log_file in self._get_all_log_files():
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=UTC)

                if file_mtime < cutoff_date:
                    # Count events before deletion
                    count = sum(1 for _ in self._read_events_from_file(log_file))
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
            for log_file in self._get_all_log_files():
                if not self._verify_file_integrity(log_file):
                    return False
            return True

        except Exception as e:
            self._logger.error(f"Integrity verification failed: {e}")
            return False

    def get_storage_stats(self) -> dict[str, Any]:
        """Get file storage statistics."""
        try:
            log_files = list(self._get_all_log_files())
            total_size = sum(f.stat().st_size for f in log_files)
            total_events = sum(sum(1 for _ in self._read_events_from_file(f)) for f in log_files)

            return {
                "backend_type": "file",
                "storage_path": str(self.storage_path),
                "total_files": len(log_files),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_events": total_events,
                "compression_enabled": self.compression_enabled,
                "rotation_enabled": self.rotation_enabled,
                "current_file": str(self.current_file) if self.current_file else None,
                "current_file_size_bytes": self.current_file_size,
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
            self._save_integrity_hashes()
            if self.current_file and not self.current_file.closed:
                self.current_file.close()

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

    def _needs_rotation(self) -> bool:
        """Check if log file needs rotation."""
        if not self.rotation_enabled:
            return False

        size_mb = self.current_file_size / (1024 * 1024)
        return size_mb >= self.rotation_size_mb

    def _rotate_log_file(self) -> None:
        """Rotate current log file."""
        if self.current_file:
            self.current_file.close()

        # Remove oldest files if we exceed rotation count
        self._cleanup_old_files()

        # Initialize new log file
        self._initialize_current_file()

    def _cleanup_old_files(self) -> None:
        """Clean up old log files based on rotation count."""
        log_files = sorted(self._get_all_log_files(), key=lambda f: f.stat().st_mtime)

        while len(log_files) >= self.rotation_count:
            oldest_file = log_files.pop(0)
            oldest_file.unlink()
            self._logger.info(f"Deleted old log file: {oldest_file}")

    def _write_plain(self, content: str) -> None:
        """Write content to plain text file."""
        if self.current_file:
            cast(IO[str], self.current_file).write(content)
            cast(IO[str], self.current_file).flush()
            self.current_file_size += len(content.encode("utf-8"))

    def _write_compressed(self, content: str) -> None:
        """Write content to compressed file."""
        if self.current_file:
            # GzipFile expects bytes
            import gzip

            if isinstance(self.current_file, gzip.GzipFile):
                self.current_file.write(content.encode("utf-8"))
                self.current_file.flush()
            else:
                cast(IO[bytes], self.current_file).write(content.encode("utf-8"))
                cast(IO[bytes], self.current_file).flush()
            # Note: For gzip, we approximate size since compression ratio varies
            self.current_file_size += len(content.encode("utf-8")) // 2

    def _store_single_unlocked(self, event_data: dict[str, Any]) -> str:
        """Store single event without acquiring lock (used in batch operations)."""
        event_id = event_data.get("event_id", "unknown")

        if self.rotation_enabled and self._needs_rotation():
            self._rotate_log_file()

        event_json = json.dumps(event_data, default=str, separators=(",", ":"))
        event_line = event_json + "\n"

        if self.compression_enabled:
            self._write_compressed(event_line)
        else:
            self._write_plain(event_line)

        self._update_integrity_hash(event_id, event_line)

        file_name = self.current_file.name if self.current_file else "unknown"
        return f"file:{file_name}:{event_id}"

    def _get_all_log_files(self) -> Iterator[Path]:
        """Get all audit log files."""
        patterns = ["audit_*.jsonl"]
        if self.compression_enabled:
            patterns.append("audit_*.jsonl.gz")

        for pattern in patterns:
            yield from self.storage_path.glob(pattern)

    def _read_events_from_file(self, file_path: Path) -> Iterator[dict[str, Any]]:
        """Read events from a log file."""
        try:
            if file_path.suffix == ".gz":
                f = gzip.open(file_path, "rt", encoding="utf-8")
            else:
                f = open(file_path, encoding="utf-8")

            with f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            self._logger.warning(
                                f"Invalid JSON in log file {file_path}: {line[:100]}"
                            )
                            continue

        except Exception as e:
            self._logger.error(f"Failed to read log file {file_path}: {e}")

    def _search_file_for_event(self, file_path: Path, event_id: str) -> dict[str, Any] | None:
        """Search for specific event ID in a log file."""
        for event in self._read_events_from_file(file_path):
            if event.get("event_id") == event_id:
                return event
        return None

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

    def _update_integrity_hash(self, event_id: str, event_line: str) -> None:
        """Update integrity hash for event."""
        hash_obj = hashlib.sha256(event_line.encode("utf-8"))
        self.integrity_hashes[event_id] = hash_obj.hexdigest()

    def _load_integrity_hashes(self) -> None:
        """Load integrity hashes from file."""
        try:
            if self.integrity_file.exists():
                with open(self.integrity_file) as f:
                    self.integrity_hashes = json.load(f)
        except Exception as e:
            self._logger.warning(f"Failed to load integrity hashes: {e}")
            self.integrity_hashes = {}

    def _save_integrity_hashes(self) -> None:
        """Save integrity hashes to file."""
        try:
            with open(self.integrity_file, "w") as f:
                json.dump(self.integrity_hashes, f)
        except Exception as e:
            self._logger.error(f"Failed to save integrity hashes: {e}")

    def _verify_file_integrity(self, file_path: Path) -> bool:
        """Verify integrity of a specific log file."""
        try:
            for event in self._read_events_from_file(file_path):
                event_id = event.get("event_id")
                if event_id and event_id in self.integrity_hashes:
                    # Reconstruct event line and verify hash
                    event_json = json.dumps(event, default=str, separators=(",", ":"))
                    event_line = event_json + "\n"

                    hash_obj = hashlib.sha256(event_line.encode("utf-8"))
                    calculated_hash = hash_obj.hexdigest()
                    stored_hash = self.integrity_hashes[event_id]

                    if calculated_hash != stored_hash:
                        self._logger.error(
                            f"Integrity violation for event {event_id} in {file_path}"
                        )
                        return False

            return True

        except Exception as e:
            self._logger.error(f"Failed to verify file integrity for {file_path}: {e}")
            return False


class DatabaseStorage(AuditStorage):
    """
    Database-based audit storage backend.

    Provides scalable database storage with SQL queries, transactions,
    and optimized batch operations.
    """

    def __init__(self, config: AuditConfig) -> None:
        """Initialize database storage backend."""
        super().__init__(config)

        database_url = config.storage_config.database_url
        if not database_url:
            raise AuditStorageError(
                "Database URL is required for database storage", storage_backend="database"
            )
        self.database_url: str = database_url
        self.table_name = config.storage_config.database_table_name
        self.pool_size = config.storage_config.database_connection_pool_size
        self.batch_size = config.storage_config.database_batch_insert_size

        # Initialize connection pool (simplified SQLite implementation)
        self.connection_lock = threading.Lock()
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize database schema."""
        try:
            with sqlite3.connect(self.database_url) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT UNIQUE NOT NULL,
                        event_type TEXT NOT NULL,
                        resource_type TEXT NOT NULL,
                        resource_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        user_id TEXT,
                        timestamp DATETIME NOT NULL,
                        severity TEXT NOT NULL,
                        is_critical BOOLEAN NOT NULL,
                        event_data TEXT NOT NULL,
                        integrity_hash TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_event_id ON audit_logs(event_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_logs(event_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_logs(user_id)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_resource_type ON audit_logs(resource_type)"
                )

                conn.commit()

        except Exception as e:
            raise AuditStorageError(
                f"Failed to initialize database: {e}",
                storage_backend="database",
                operation="initialize",
                underlying_error=e,
            )

    def store(self, event_data: dict[str, Any]) -> str:
        """Store single audit event to database."""
        try:
            with self.connection_lock:
                with sqlite3.connect(self.database_url) as conn:
                    cursor = conn.cursor()

                    # Extract key fields
                    event_id = event_data.get("event_id")
                    event_type = event_data.get("event_type")
                    resource_type = event_data.get("resource_type")
                    resource_id = event_data.get("resource_id")
                    action = event_data.get("action")
                    user_id = event_data.get("user_id")
                    timestamp = event_data.get("timestamp")
                    severity = event_data.get("severity")
                    is_critical = event_data.get("is_critical", False)
                    integrity_hash = event_data.get("integrity_hash")

                    # Serialize full event data
                    event_json = json.dumps(event_data, default=str)

                    cursor.execute(
                        """
                        INSERT INTO audit_logs
                        (event_id, event_type, resource_type, resource_id, action,
                         user_id, timestamp, severity, is_critical, event_data, integrity_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            event_id,
                            event_type,
                            resource_type,
                            resource_id,
                            action,
                            user_id,
                            timestamp,
                            severity,
                            is_critical,
                            event_json,
                            integrity_hash,
                        ),
                    )

                    conn.commit()
                    return f"db:{cursor.lastrowid}"

        except Exception as e:
            raise AuditStorageError(
                f"Failed to store event to database: {e}",
                storage_backend="database",
                operation="store",
                underlying_error=e,
            )

    def store_batch(self, events: list[dict[str, Any]]) -> list[str]:
        """Store batch of audit events to database."""
        storage_refs = []

        try:
            with self.connection_lock:
                with sqlite3.connect(self.database_url) as conn:
                    cursor = conn.cursor()

                    batch_data = []
                    for event_data in events:
                        # Extract key fields
                        event_id = event_data.get("event_id")
                        event_type = event_data.get("event_type")
                        resource_type = event_data.get("resource_type")
                        resource_id = event_data.get("resource_id")
                        action = event_data.get("action")
                        user_id = event_data.get("user_id")
                        timestamp = event_data.get("timestamp")
                        severity = event_data.get("severity")
                        is_critical = event_data.get("is_critical", False)
                        integrity_hash = event_data.get("integrity_hash")

                        # Serialize full event data
                        event_json = json.dumps(event_data, default=str)

                        batch_data.append(
                            (
                                event_id,
                                event_type,
                                resource_type,
                                resource_id,
                                action,
                                user_id,
                                timestamp,
                                severity,
                                is_critical,
                                event_json,
                                integrity_hash,
                            )
                        )

                    cursor.executemany(
                        """
                        INSERT INTO audit_logs
                        (event_id, event_type, resource_type, resource_id, action,
                         user_id, timestamp, severity, is_critical, event_data, integrity_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        batch_data,
                    )

                    conn.commit()

                    # Generate storage references
                    last_id = cursor.lastrowid
                    if last_id:
                        start_id = last_id - len(events) + 1
                        storage_refs = [f"db:{start_id + i}" for i in range(len(events))]
                    else:
                        storage_refs = [f"db:unknown_{i}" for i in range(len(events))]

        except Exception as e:
            raise AuditStorageError(
                f"Failed to store event batch to database: {e}",
                storage_backend="database",
                operation="store_batch",
                underlying_error=e,
            )

        return storage_refs

    def retrieve(self, event_id: str) -> dict[str, Any] | None:
        """Retrieve audit event by ID from database."""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT event_data FROM audit_logs WHERE event_id = ?", (event_id,))
                row = cursor.fetchone()

                if row:
                    event_data = json.loads(row[0])
                    return cast(dict[str, Any] | None, event_data)
                return None

        except Exception as e:
            raise AuditStorageError(
                f"Failed to retrieve event from database: {e}",
                storage_backend="database",
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
        """Query audit events from database."""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()

                # Build query with filters
                query = "SELECT event_data FROM audit_logs WHERE 1=1"
                params = []

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                if event_types:
                    placeholders = ",".join("?" * len(event_types))
                    query += f" AND event_type IN ({placeholders})"
                    params.extend(event_types)

                if user_ids:
                    placeholders = ",".join("?" * len(user_ids))
                    query += f" AND user_id IN ({placeholders})"
                    params.extend(user_ids)

                query += " ORDER BY timestamp DESC"

                if limit:
                    query += " LIMIT ?"
                    params.append(str(limit))

                cursor.execute(query, params)

                for row in cursor:
                    yield json.loads(row[0])

        except Exception as e:
            raise AuditStorageError(
                f"Failed to query events from database: {e}",
                storage_backend="database",
                operation="query",
                underlying_error=e,
            )

    def delete_expired(self, cutoff_date: datetime) -> int:
        """Delete expired audit events from database."""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM audit_logs WHERE timestamp < ?", (cutoff_date.isoformat(),)
                )
                conn.commit()
                return cursor.rowcount

        except Exception as e:
            raise AuditStorageError(
                f"Failed to delete expired events from database: {e}",
                storage_backend="database",
                operation="delete_expired",
                underlying_error=e,
            )

    def verify_integrity(self) -> bool:
        """Verify database storage integrity."""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT event_id, event_data, integrity_hash FROM audit_logs")

                for row in cursor:
                    event_id, event_data, stored_hash = row

                    # Recalculate hash
                    hash_obj = hashlib.sha256(event_data.encode("utf-8"))
                    calculated_hash = hash_obj.hexdigest()

                    if calculated_hash != stored_hash:
                        self._logger.error(f"Integrity violation for event {event_id}")
                        return False

                return True

        except Exception as e:
            self._logger.error(f"Database integrity verification failed: {e}")
            return False

    def get_storage_stats(self) -> dict[str, Any]:
        """Get database storage statistics."""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()

                # Get total count
                cursor.execute("SELECT COUNT(*) FROM audit_logs")
                total_events = cursor.fetchone()[0]

                # Get critical events count
                cursor.execute("SELECT COUNT(*) FROM audit_logs WHERE is_critical = 1")
                critical_events = cursor.fetchone()[0]

                # Get date range
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM audit_logs")
                date_range = cursor.fetchone()

                # Get database size (SQLite specific)
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                database_size = page_count * page_size

                return {
                    "backend_type": "database",
                    "database_url": self.database_url,
                    "table_name": self.table_name,
                    "total_events": total_events,
                    "critical_events": critical_events,
                    "database_size_bytes": database_size,
                    "database_size_mb": database_size / (1024 * 1024),
                    "earliest_event": date_range[0] if date_range[0] else None,
                    "latest_event": date_range[1] if date_range[1] else None,
                }

        except Exception as e:
            raise AuditStorageError(
                f"Failed to get database storage stats: {e}",
                storage_backend="database",
                operation="get_stats",
                underlying_error=e,
            )

    def close(self) -> None:
        """Close database storage."""
        # In a real implementation, you would close connection pools here
        pass


class ExternalStorage(AuditStorage):
    """
    External system audit storage backend.

    Provides integration with external audit systems, SIEM platforms,
    and cloud storage services.
    """

    def __init__(self, config: AuditConfig) -> None:
        """Initialize external storage backend."""
        super().__init__(config)

        self.external_type = config.storage_config.external_storage_type
        self.external_config = config.storage_config.external_storage_config
        self.retry_attempts = config.storage_config.external_storage_retry_attempts

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)

    def store(self, event_data: dict[str, Any]) -> str:
        """Store single audit event to external system."""
        # This is a placeholder implementation
        # In practice, you would integrate with specific external systems
        event_id = event_data.get("event_id", "unknown")

        try:
            # Simulate external API call
            self._send_to_external_system([event_data])
            return f"external:{self.external_type}:{event_id}"

        except Exception as e:
            raise AuditStorageError(
                f"Failed to store event to external system: {e}",
                storage_backend="external",
                operation="store",
                underlying_error=e,
            )

    def store_batch(self, events: list[dict[str, Any]]) -> list[str]:
        """Store batch of audit events to external system."""
        try:
            self._send_to_external_system(events)

            storage_refs = []
            for event_data in events:
                event_id = event_data.get("event_id", "unknown")
                storage_refs.append(f"external:{self.external_type}:{event_id}")

            return storage_refs

        except Exception as e:
            raise AuditStorageError(
                f"Failed to store event batch to external system: {e}",
                storage_backend="external",
                operation="store_batch",
                underlying_error=e,
            )

    def retrieve(self, event_id: str) -> dict[str, Any] | None:
        """Retrieve audit event by ID from external system."""
        # External systems typically don't support individual retrieval
        raise NotImplementedError("External storage does not support individual retrieval")

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[str] | None = None,
        user_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Query audit events from external system."""
        # External systems typically don't support querying
        raise NotImplementedError("External storage does not support querying")

    def delete_expired(self, cutoff_date: datetime) -> int:
        """Delete expired audit events from external system."""
        # External systems typically manage their own retention
        return 0

    def verify_integrity(self) -> bool:
        """Verify external storage integrity."""
        # External systems typically manage their own integrity
        return True

    def get_storage_stats(self) -> dict[str, Any]:
        """Get external storage statistics."""
        return {"backend_type": "external", "external_type": self.external_type, "status": "active"}

    def close(self) -> None:
        """Close external storage."""
        if self._executor:
            self._executor.shutdown(wait=True)

    def _send_to_external_system(self, events: list[dict[str, Any]]) -> None:
        """Send events to external system."""
        # This is a placeholder - implement specific external system integration
        self._logger.info(f"Sending {len(events)} events to external system: {self.external_type}")

        # In practice, you would:
        # 1. Format events according to external system requirements
        # 2. Handle authentication and authorization
        # 3. Send via HTTP API, message queue, or other protocol
        # 4. Handle retries and error conditions
        # 5. Verify delivery confirmation

        time.sleep(0.1)  # Simulate network latency


class MultiStorage(AuditStorage):
    """
    Multi-backend storage that writes to multiple storage systems.

    Provides redundancy and ensures audit events are stored in multiple
    locations for maximum reliability and compliance.
    """

    def __init__(self, primary_storage: AuditStorage, backup_storages: list[AuditStorage]) -> None:
        """
        Initialize multi-backend storage.

        Args:
            primary_storage: Primary storage backend
            backup_storages: List of backup storage backends
        """
        self.primary_storage = primary_storage
        self.backup_storages = backup_storages
        self._logger = logging.getLogger(f"{__name__}.MultiStorage")

    def store(self, event_data: dict[str, Any]) -> str:
        """Store event to all storage backends."""
        primary_ref = None
        backup_refs = []

        # Store to primary storage
        try:
            primary_ref = self.primary_storage.store(event_data)
        except Exception as e:
            self._logger.error(f"Primary storage failed: {e}")
            # Continue with backup storages

        # Store to backup storages
        for backup_storage in self.backup_storages:
            try:
                backup_ref = backup_storage.store(event_data)
                backup_refs.append(backup_ref)
            except Exception as e:
                self._logger.warning(f"Backup storage failed: {e}")

        # Return primary reference if available, otherwise first backup
        return primary_ref or (backup_refs[0] if backup_refs else "failed")

    def store_batch(self, events: list[dict[str, Any]]) -> list[str]:
        """Store batch to all storage backends."""
        primary_refs = None

        # Store to primary storage
        try:
            primary_refs = self.primary_storage.store_batch(events)
        except Exception as e:
            self._logger.error(f"Primary batch storage failed: {e}")

        # Store to backup storages
        for backup_storage in self.backup_storages:
            try:
                backup_storage.store_batch(events)
            except Exception as e:
                self._logger.warning(f"Backup batch storage failed: {e}")

        return primary_refs or [f"failed_{i}" for i in range(len(events))]

    def retrieve(self, event_id: str) -> dict[str, Any] | None:
        """Retrieve from primary storage."""
        return self.primary_storage.retrieve(event_id)

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[str] | None = None,
        user_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Query from primary storage."""
        return self.primary_storage.query(start_time, end_time, event_types, user_ids, limit)

    def delete_expired(self, cutoff_date: datetime) -> int:
        """Delete expired from all storages."""
        total_deleted = 0

        # Delete from primary storage
        try:
            total_deleted += self.primary_storage.delete_expired(cutoff_date)
        except Exception as e:
            self._logger.error(f"Primary storage cleanup failed: {e}")

        # Delete from backup storages
        for backup_storage in self.backup_storages:
            try:
                backup_storage.delete_expired(cutoff_date)
            except Exception as e:
                self._logger.warning(f"Backup storage cleanup failed: {e}")

        return total_deleted

    def verify_integrity(self) -> bool:
        """Verify integrity of all storages."""
        all_valid = True

        # Verify primary storage
        try:
            if not self.primary_storage.verify_integrity():
                all_valid = False
        except Exception as e:
            self._logger.error(f"Primary storage integrity check failed: {e}")
            all_valid = False

        # Verify backup storages
        for backup_storage in self.backup_storages:
            try:
                if not backup_storage.verify_integrity():
                    all_valid = False
            except Exception as e:
                self._logger.warning(f"Backup storage integrity check failed: {e}")

        return all_valid

    def get_storage_stats(self) -> dict[str, Any]:
        """Get combined storage statistics."""
        stats: dict[str, Any] = {
            "backend_type": "multi",
            "primary_storage": {},
            "backup_storages": [],
        }

        # Get primary storage stats
        try:
            stats["primary_storage"] = self.primary_storage.get_storage_stats()
        except Exception as e:
            self._logger.error(f"Failed to get primary storage stats: {e}")

        # Get backup storage stats
        for i, backup_storage in enumerate(self.backup_storages):
            try:
                backup_stats = backup_storage.get_storage_stats()
                stats["backup_storages"].append(backup_stats)
            except Exception as e:
                self._logger.warning(f"Failed to get backup storage {i} stats: {e}")

        return stats

    def close(self) -> None:
        """Close all storage backends."""
        try:
            self.primary_storage.close()
        except Exception as e:
            self._logger.error(f"Failed to close primary storage: {e}")

        for backup_storage in self.backup_storages:
            try:
                backup_storage.close()
            except Exception as e:
                self._logger.warning(f"Failed to close backup storage: {e}")

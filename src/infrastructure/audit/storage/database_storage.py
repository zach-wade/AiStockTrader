"""
Database-based audit storage backend.

Provides scalable database storage with SQL queries, transactions,
and optimized batch operations.
"""

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime
from typing import Any, cast

from ..config import AuditConfig
from ..exceptions import AuditStorageError
from .base import AuditStorage


class DatabaseQueryBuilder:
    """Builds SQL queries for database operations."""

    @staticmethod
    def build_query_with_filters(
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[str] | None = None,
        user_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> tuple[str, list[Any]]:
        """Build query with filters and return query string and parameters."""
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

        return query, params

    @staticmethod
    def get_insert_query() -> str:
        """Get the insert query for audit events."""
        return """
            INSERT INTO audit_logs
            (event_id, event_type, resource_type, resource_id, action,
             user_id, timestamp, severity, is_critical, event_data, integrity_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """


class DatabaseSchemaManager:
    """Manages database schema operations."""

    @staticmethod
    def initialize_schema(database_url: str) -> None:
        """Initialize database schema."""
        try:
            with sqlite3.connect(database_url) as conn:
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
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_event_id ON audit_logs(event_id)",
                    "CREATE INDEX IF NOT EXISTS idx_event_type ON audit_logs(event_type)",
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_user_id ON audit_logs(user_id)",
                    "CREATE INDEX IF NOT EXISTS idx_resource_type ON audit_logs(resource_type)",
                ]

                for index_sql in indexes:
                    conn.execute(index_sql)

                conn.commit()

        except Exception as e:
            raise AuditStorageError(
                f"Failed to initialize database: {e}",
                storage_backend="database",
                operation="initialize",
                underlying_error=e,
            )


class DatabaseEventProcessor:
    """Processes audit events for database storage."""

    @staticmethod
    def extract_event_fields(event_data: dict[str, Any]) -> dict[str, Any]:
        """Extract key fields from event data."""
        return {
            "event_id": event_data.get("event_id"),
            "event_type": event_data.get("event_type"),
            "resource_type": event_data.get("resource_type"),
            "resource_id": event_data.get("resource_id"),
            "action": event_data.get("action"),
            "user_id": event_data.get("user_id"),
            "timestamp": event_data.get("timestamp"),
            "severity": event_data.get("severity"),
            "is_critical": event_data.get("is_critical", False),
            "integrity_hash": event_data.get("integrity_hash"),
        }

    @staticmethod
    def prepare_batch_data(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        """Prepare batch data for database insertion."""
        batch_data = []

        for event_data in events:
            fields = DatabaseEventProcessor.extract_event_fields(event_data)
            event_json = json.dumps(event_data, default=str)

            batch_data.append(
                (
                    fields["event_id"],
                    fields["event_type"],
                    fields["resource_type"],
                    fields["resource_id"],
                    fields["action"],
                    fields["user_id"],
                    fields["timestamp"],
                    fields["severity"],
                    fields["is_critical"],
                    event_json,
                    fields["integrity_hash"],
                )
            )

        return batch_data


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

        # Initialize database schema
        DatabaseSchemaManager.initialize_schema(self.database_url)

    def store(self, event_data: dict[str, Any]) -> str:
        """Store single audit event to database."""
        try:
            with self.connection_lock:
                with sqlite3.connect(self.database_url) as conn:
                    cursor = conn.cursor()

                    # Extract and prepare event data
                    fields = DatabaseEventProcessor.extract_event_fields(event_data)
                    event_json = json.dumps(event_data, default=str)

                    cursor.execute(
                        DatabaseQueryBuilder.get_insert_query(),
                        (
                            fields["event_id"],
                            fields["event_type"],
                            fields["resource_type"],
                            fields["resource_id"],
                            fields["action"],
                            fields["user_id"],
                            fields["timestamp"],
                            fields["severity"],
                            fields["is_critical"],
                            event_json,
                            fields["integrity_hash"],
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

                    # Prepare batch data
                    batch_data = DatabaseEventProcessor.prepare_batch_data(events)

                    cursor.executemany(DatabaseQueryBuilder.get_insert_query(), batch_data)

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
                query, params = DatabaseQueryBuilder.build_query_with_filters(
                    start_time, end_time, event_types, user_ids, limit
                )

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

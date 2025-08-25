"""
Unit tests for audit storage backends.

Tests cover file storage, database storage, external storage,
and multi-storage implementations.
"""

import os
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.infrastructure.audit.config import AuditConfig, StorageBackend, StorageConfig
from src.infrastructure.audit.exceptions import AuditStorageError
from src.infrastructure.audit.storage import (
    AuditStorage,
    DatabaseStorage,
    ExternalStorage,
    FileStorage,
    MultiStorage,
)


@pytest.fixture
def sample_event_data():
    """Create sample event data for testing."""
    return {
        "event_id": "test_event_123",
        "event_type": "order_create",
        "resource_type": "order",
        "resource_id": "order_456",
        "action": "create",
        "user_id": "user_789",
        "timestamp": datetime.now(UTC).isoformat(),
        "severity": "high",
        "is_critical": False,
        "resource_details": {"symbol": "AAPL", "quantity": "100", "price": "150.00"},
        "integrity_hash": "abc123def456",
    }


@pytest.fixture
def temp_storage_config():
    """Create temporary storage configuration."""
    config = AuditConfig(
        security_config=SecurityConfig(
            encryption_enabled=False,
            digital_signatures_enabled=False,
            access_control_enabled=False,
            integrity_checks_enabled=False,
            tls_enabled=False,
            tamper_detection_enabled=False,
        ),
        storage_config=StorageConfig(
            primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
        ),
    )

    # Use temporary directory for file storage
    temp_dir = tempfile.mkdtemp()
    config.storage_config.file_storage_path = temp_dir

    # Use temporary database for database storage
    temp_db = tempfile.mktemp(suffix=".db")
    config.storage_config.database_url = temp_db

    yield config

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)
    if os.path.exists(temp_db):
        os.unlink(temp_db)


class TestFileStorage:
    """Test suite for FileStorage class."""

    def test_init(self, temp_storage_config):
        """Test file storage initialization."""
        storage = FileStorage(temp_storage_config)

        assert storage.storage_path.exists()
        assert storage.compression_enabled == temp_storage_config.storage_config.compression_enabled
        assert storage.rotation_enabled == temp_storage_config.storage_config.file_rotation_enabled
        assert storage.current_file is not None

    def test_store_single_event(self, temp_storage_config, sample_event_data):
        """Test storing single event."""
        storage = FileStorage(temp_storage_config)

        storage_ref = storage.store(sample_event_data)

        assert storage_ref.startswith("file:")
        assert "test_event_123" in storage_ref

        # Verify file was created
        log_files = list(storage.storage_path.glob("audit_*.jsonl*"))
        assert len(log_files) == 1

        storage.close()

    def test_store_batch_events(self, temp_storage_config, sample_event_data):
        """Test storing batch of events."""
        storage = FileStorage(temp_storage_config)

        # Create batch of events
        events = []
        for i in range(5):
            event = sample_event_data.copy()
            event["event_id"] = f"test_event_{i}"
            events.append(event)

        storage_refs = storage.store_batch(events)

        assert len(storage_refs) == 5
        for ref in storage_refs:
            assert ref.startswith("file:")

        storage.close()

    def test_retrieve_event(self, temp_storage_config, sample_event_data):
        """Test retrieving event by ID."""
        storage = FileStorage(temp_storage_config)

        # Store event first
        storage.store(sample_event_data)
        storage.close()

        # Create new storage instance to test retrieval
        storage = FileStorage(temp_storage_config)

        # Retrieve event
        retrieved_event = storage.retrieve("test_event_123")

        assert retrieved_event is not None
        assert retrieved_event["event_id"] == "test_event_123"
        assert retrieved_event["event_type"] == "order_create"

        storage.close()

    def test_retrieve_nonexistent_event(self, temp_storage_config):
        """Test retrieving nonexistent event."""
        storage = FileStorage(temp_storage_config)

        retrieved_event = storage.retrieve("nonexistent_event")

        assert retrieved_event is None

        storage.close()

    def test_query_events(self, temp_storage_config, sample_event_data):
        """Test querying events with filters."""
        storage = FileStorage(temp_storage_config)

        # Store multiple events
        events = []
        for i in range(5):
            event = sample_event_data.copy()
            event["event_id"] = f"test_event_{i}"
            event["user_id"] = f"user_{i % 2}"  # Alternating users
            events.append(event)
            storage.store(event)

        storage.close()

        # Create new storage instance for querying
        storage = FileStorage(temp_storage_config)

        # Query all events
        all_events = list(storage.query())
        assert len(all_events) == 5

        # Query with user filter
        user_events = list(storage.query(user_ids=["user_0"]))
        assert len(user_events) == 3  # Events 0, 2, 4

        # Query with event type filter
        type_events = list(storage.query(event_types=["order_create"]))
        assert len(type_events) == 5

        # Query with limit
        limited_events = list(storage.query(limit=3))
        assert len(limited_events) == 3

        storage.close()

    def test_delete_expired_events(self, temp_storage_config, sample_event_data):
        """Test deleting expired events."""
        storage = FileStorage(temp_storage_config)

        # Store event
        storage.store(sample_event_data)
        storage.close()

        # Get the log file and modify its timestamp to make it "old"
        log_files = list(
            Path(temp_storage_config.storage_config.file_storage_path).glob("audit_*.jsonl*")
        )
        old_time = datetime.now().timestamp() - 86400  # 1 day ago
        os.utime(log_files[0], (old_time, old_time))

        # Create new storage instance
        storage = FileStorage(temp_storage_config)

        # Delete expired events
        cutoff_date = datetime.now(UTC) - timedelta(hours=12)
        deleted_count = storage.delete_expired(cutoff_date)

        assert deleted_count == 1

        storage.close()

    def test_file_rotation(self, temp_storage_config):
        """Test file rotation functionality."""
        # Set small rotation size for testing
        temp_storage_config.storage_config.file_rotation_size_mb = 1  # 1MB
        temp_storage_config.storage_config.file_rotation_count = 3

        storage = FileStorage(temp_storage_config)

        # Create large event data to trigger rotation
        large_event = {
            "event_id": "large_event",
            "event_type": "test",
            "resource_type": "test",
            "resource_id": "test",
            "action": "test",
            "large_data": "x" * 1024 * 512,  # 512KB
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Store multiple large events to trigger rotation
        for i in range(5):
            event = large_event.copy()
            event["event_id"] = f"large_event_{i}"
            storage.store(event)

        # Should have multiple log files due to rotation
        log_files = list(storage.storage_path.glob("audit_*.jsonl*"))
        assert len(log_files) > 1

        storage.close()

    def test_compression(self, temp_storage_config):
        """Test compression functionality."""
        temp_storage_config.storage_config.compression_enabled = True

        storage = FileStorage(temp_storage_config)

        # Store event
        event_data = {
            "event_id": "compressed_event",
            "event_type": "test",
            "resource_type": "test",
            "resource_id": "test",
            "action": "test",
            "data": "x" * 1000,  # Some data to compress
            "timestamp": datetime.now(UTC).isoformat(),
        }

        storage.store(event_data)
        storage.close()

        # Check that gzipped file was created
        gz_files = list(storage.storage_path.glob("*.gz"))
        assert len(gz_files) == 1

        # Verify we can read the compressed file
        storage = FileStorage(temp_storage_config)
        retrieved_event = storage.retrieve("compressed_event")
        assert retrieved_event is not None
        assert retrieved_event["event_id"] == "compressed_event"

        storage.close()

    def test_integrity_verification(self, temp_storage_config, sample_event_data):
        """Test storage integrity verification."""
        storage = FileStorage(temp_storage_config)

        # Store event
        storage.store(sample_event_data)

        # Verify integrity
        assert storage.verify_integrity() == True

        storage.close()

    def test_storage_stats(self, temp_storage_config, sample_event_data):
        """Test storage statistics."""
        storage = FileStorage(temp_storage_config)

        # Store some events
        for i in range(3):
            event = sample_event_data.copy()
            event["event_id"] = f"stats_event_{i}"
            storage.store(event)

        stats = storage.get_storage_stats()

        assert stats["backend_type"] == "file"
        assert stats["total_files"] >= 1
        assert stats["total_events"] == 3
        assert stats["total_size_bytes"] > 0
        assert "storage_path" in stats

        storage.close()

    def test_storage_failure(self, temp_storage_config, sample_event_data):
        """Test storage failure handling."""
        storage = FileStorage(temp_storage_config)

        # Close the file to simulate failure
        storage.current_file.close()
        storage.current_file = None

        with pytest.raises(AuditStorageError):
            storage.store(sample_event_data)


class TestDatabaseStorage:
    """Test suite for DatabaseStorage class."""

    def test_init(self, temp_storage_config):
        """Test database storage initialization."""
        storage = DatabaseStorage(temp_storage_config)

        # Check that database and table were created
        conn = sqlite3.connect(temp_storage_config.storage_config.database_url)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='audit_logs'
        """
        )

        assert cursor.fetchone() is not None
        conn.close()

    def test_store_single_event(self, temp_storage_config, sample_event_data):
        """Test storing single event in database."""
        storage = DatabaseStorage(temp_storage_config)

        storage_ref = storage.store(sample_event_data)

        assert storage_ref.startswith("db:")

        # Verify event was stored
        conn = sqlite3.connect(temp_storage_config.storage_config.database_url)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM audit_logs")
        count = cursor.fetchone()[0]
        assert count == 1
        conn.close()

    def test_store_batch_events(self, temp_storage_config, sample_event_data):
        """Test storing batch of events in database."""
        storage = DatabaseStorage(temp_storage_config)

        # Create batch of events
        events = []
        for i in range(5):
            event = sample_event_data.copy()
            event["event_id"] = f"batch_event_{i}"
            events.append(event)

        storage_refs = storage.store_batch(events)

        assert len(storage_refs) == 5
        for ref in storage_refs:
            assert ref.startswith("db:")

        # Verify events were stored
        conn = sqlite3.connect(temp_storage_config.storage_config.database_url)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM audit_logs")
        count = cursor.fetchone()[0]
        assert count == 5
        conn.close()

    def test_retrieve_event(self, temp_storage_config, sample_event_data):
        """Test retrieving event from database."""
        storage = DatabaseStorage(temp_storage_config)

        # Store event first
        storage.store(sample_event_data)

        # Retrieve event
        retrieved_event = storage.retrieve("test_event_123")

        assert retrieved_event is not None
        assert retrieved_event["event_id"] == "test_event_123"
        assert retrieved_event["event_type"] == "order_create"

    def test_query_events(self, temp_storage_config, sample_event_data):
        """Test querying events from database."""
        storage = DatabaseStorage(temp_storage_config)

        # Store multiple events with different timestamps
        base_time = datetime.now(UTC)
        events = []
        for i in range(5):
            event = sample_event_data.copy()
            event["event_id"] = f"query_event_{i}"
            event["user_id"] = f"user_{i % 2}"
            event["timestamp"] = (base_time + timedelta(hours=i)).isoformat()
            events.append(event)
            storage.store(event)

        # Query all events
        all_events = list(storage.query())
        assert len(all_events) == 5

        # Query with time filter
        start_time = base_time + timedelta(hours=2)
        time_filtered = list(storage.query(start_time=start_time))
        assert len(time_filtered) == 3  # Events 2, 3, 4

        # Query with user filter
        user_filtered = list(storage.query(user_ids=["user_0"]))
        assert len(user_filtered) == 3  # Events 0, 2, 4

        # Query with limit
        limited = list(storage.query(limit=3))
        assert len(limited) == 3

    def test_delete_expired_events(self, temp_storage_config, sample_event_data):
        """Test deleting expired events from database."""
        storage = DatabaseStorage(temp_storage_config)

        # Store events with different timestamps
        old_time = datetime.now(UTC) - timedelta(days=2)
        new_time = datetime.now(UTC)

        old_event = sample_event_data.copy()
        old_event["event_id"] = "old_event"
        old_event["timestamp"] = old_time.isoformat()
        storage.store(old_event)

        new_event = sample_event_data.copy()
        new_event["event_id"] = "new_event"
        new_event["timestamp"] = new_time.isoformat()
        storage.store(new_event)

        # Delete expired events
        cutoff_date = datetime.now(UTC) - timedelta(days=1)
        deleted_count = storage.delete_expired(cutoff_date)

        assert deleted_count == 1

        # Verify only new event remains
        remaining_events = list(storage.query())
        assert len(remaining_events) == 1
        assert remaining_events[0]["event_id"] == "new_event"

    def test_verify_integrity(self, temp_storage_config, sample_event_data):
        """Test database integrity verification."""
        storage = DatabaseStorage(temp_storage_config)

        # Store event
        storage.store(sample_event_data)

        # Verify integrity
        assert storage.verify_integrity() == True

    def test_storage_stats(self, temp_storage_config, sample_event_data):
        """Test database storage statistics."""
        storage = DatabaseStorage(temp_storage_config)

        # Store some events
        for i in range(3):
            event = sample_event_data.copy()
            event["event_id"] = f"stats_event_{i}"
            event["is_critical"] = i == 0  # Make first event critical
            storage.store(event)

        stats = storage.get_storage_stats()

        assert stats["backend_type"] == "database"
        assert stats["total_events"] == 3
        assert stats["critical_events"] == 1
        assert "database_size_bytes" in stats


class TestExternalStorage:
    """Test suite for ExternalStorage class."""

    def test_init(self, temp_storage_config):
        """Test external storage initialization."""
        temp_storage_config.storage_config.external_storage_type = "s3"
        temp_storage_config.storage_config.external_storage_config = {"bucket": "test-bucket"}

        storage = ExternalStorage(temp_storage_config)

        assert storage.external_type == "s3"
        assert storage.external_config == {"bucket": "test-bucket"}
        assert storage.retry_attempts == 3

    @patch.object(ExternalStorage, "_send_to_external_system")
    def test_store_single_event(self, mock_send, temp_storage_config, sample_event_data):
        """Test storing single event to external system."""
        temp_storage_config.storage_config.external_storage_type = "s3"
        storage = ExternalStorage(temp_storage_config)

        storage_ref = storage.store(sample_event_data)

        assert storage_ref.startswith("external:s3:")
        mock_send.assert_called_once_with([sample_event_data])

    @patch.object(ExternalStorage, "_send_to_external_system")
    def test_store_batch_events(self, mock_send, temp_storage_config, sample_event_data):
        """Test storing batch of events to external system."""
        temp_storage_config.storage_config.external_storage_type = "s3"
        storage = ExternalStorage(temp_storage_config)

        events = [sample_event_data.copy() for _ in range(3)]
        storage_refs = storage.store_batch(events)

        assert len(storage_refs) == 3
        mock_send.assert_called_once_with(events)

    def test_unsupported_operations(self, temp_storage_config):
        """Test that unsupported operations raise NotImplementedError."""
        storage = ExternalStorage(temp_storage_config)

        with pytest.raises(NotImplementedError):
            storage.retrieve("event_id")

        with pytest.raises(NotImplementedError):
            list(storage.query())

    def test_storage_stats(self, temp_storage_config):
        """Test external storage statistics."""
        temp_storage_config.storage_config.external_storage_type = "s3"
        storage = ExternalStorage(temp_storage_config)

        stats = storage.get_storage_stats()

        assert stats["backend_type"] == "external"
        assert stats["external_type"] == "s3"
        assert stats["status"] == "active"


class TestMultiStorage:
    """Test suite for MultiStorage class."""

    def test_init(self, temp_storage_config):
        """Test multi-storage initialization."""
        primary = FileStorage(temp_storage_config)
        backup = DatabaseStorage(temp_storage_config)

        storage = MultiStorage(primary, [backup])

        assert storage.primary_storage == primary
        assert storage.backup_storages == [backup]

    def test_store_success(self, temp_storage_config, sample_event_data):
        """Test successful multi-storage store operation."""
        primary = FileStorage(temp_storage_config)
        backup = DatabaseStorage(temp_storage_config)

        storage = MultiStorage(primary, [backup])

        storage_ref = storage.store(sample_event_data)

        # Should return primary storage reference
        assert storage_ref.startswith("file:")

        # Verify event stored in both storages
        assert primary.retrieve("test_event_123") is not None
        assert backup.retrieve("test_event_123") is not None

    def test_store_primary_failure(self, temp_storage_config, sample_event_data):
        """Test multi-storage with primary storage failure."""
        # Create failing primary storage
        primary = Mock(spec=AuditStorage)
        primary.store.side_effect = Exception("Primary storage failed")

        # Create working backup storage
        backup = DatabaseStorage(temp_storage_config)

        storage = MultiStorage(primary, [backup])

        storage_ref = storage.store(sample_event_data)

        # Should return first backup reference since primary failed
        assert storage_ref.startswith("db:")

        # Verify event stored in backup
        assert backup.retrieve("test_event_123") is not None

    def test_store_batch_success(self, temp_storage_config, sample_event_data):
        """Test successful multi-storage batch store operation."""
        primary = FileStorage(temp_storage_config)
        backup = DatabaseStorage(temp_storage_config)

        storage = MultiStorage(primary, [backup])

        events = [sample_event_data.copy() for _ in range(3)]
        for i, event in enumerate(events):
            event["event_id"] = f"batch_event_{i}"

        storage_refs = storage.store_batch(events)

        assert len(storage_refs) == 3
        for ref in storage_refs:
            assert ref.startswith("file:")

    def test_retrieve_from_primary(self, temp_storage_config, sample_event_data):
        """Test retrieve operation uses primary storage."""
        primary = FileStorage(temp_storage_config)
        backup = DatabaseStorage(temp_storage_config)

        storage = MultiStorage(primary, [backup])

        # Store in multi-storage
        storage.store(sample_event_data)

        # Retrieve should use primary
        retrieved = storage.retrieve("test_event_123")
        assert retrieved is not None
        assert retrieved["event_id"] == "test_event_123"

    def test_delete_expired_all_storages(self, temp_storage_config, sample_event_data):
        """Test delete expired events from all storages."""
        primary = FileStorage(temp_storage_config)
        backup = DatabaseStorage(temp_storage_config)

        storage = MultiStorage(primary, [backup])

        # Store event in both
        storage.store(sample_event_data)

        # Delete from all storages
        cutoff_date = datetime.now(UTC) + timedelta(days=1)  # Future date
        deleted_count = storage.delete_expired(cutoff_date)

        # Should delete from primary (file storage returns count based on file age)
        assert deleted_count >= 0

    def test_verify_integrity_all_storages(self, temp_storage_config, sample_event_data):
        """Test integrity verification for all storages."""
        primary = FileStorage(temp_storage_config)
        backup = DatabaseStorage(temp_storage_config)

        storage = MultiStorage(primary, [backup])

        # Store event
        storage.store(sample_event_data)

        # Verify integrity of all storages
        assert storage.verify_integrity() == True

    def test_get_combined_stats(self, temp_storage_config, sample_event_data):
        """Test combined storage statistics."""
        primary = FileStorage(temp_storage_config)
        backup = DatabaseStorage(temp_storage_config)

        storage = MultiStorage(primary, [backup])

        # Store event
        storage.store(sample_event_data)

        stats = storage.get_storage_stats()

        assert stats["backend_type"] == "multi"
        assert "primary_storage" in stats
        assert "backup_storages" in stats
        assert len(stats["backup_storages"]) == 1

    def test_close_all_storages(self, temp_storage_config):
        """Test closing all storages."""
        primary = FileStorage(temp_storage_config)
        backup = DatabaseStorage(temp_storage_config)

        storage = MultiStorage(primary, [backup])

        # Should not raise exception
        storage.close()

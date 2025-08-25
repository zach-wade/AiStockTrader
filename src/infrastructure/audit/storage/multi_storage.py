"""
Multi-backend storage that writes to multiple storage systems.

Provides redundancy and ensures audit events are stored in multiple
locations for maximum reliability and compliance.
"""

import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from .base import AuditStorage


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

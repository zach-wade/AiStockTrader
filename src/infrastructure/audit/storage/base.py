"""
Abstract base class for audit storage backends.

Defines the interface that all storage backends must implement,
ensuring consistent behavior across different storage systems.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from ..config import AuditConfig


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

"""
External system audit storage backend.

Provides integration with external audit systems, SIEM platforms,
and cloud storage services.
"""

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from ..config import AuditConfig
from ..exceptions import AuditStorageError
from .base import AuditStorage


class ExternalSystemClient:
    """Client for communicating with external systems."""

    def __init__(self, external_type: str, external_config: dict[str, Any], retry_attempts: int):
        self.external_type = external_type
        self.external_config = external_config
        self.retry_attempts = retry_attempts

    def send_events(self, events: list[dict[str, Any]]) -> None:
        """Send events to external system with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                self._send_to_external_system(events)
                return
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise e
                # Wait before retry (exponential backoff)
                time.sleep(2**attempt)

    def _send_to_external_system(self, events: list[dict[str, Any]]) -> None:
        """Send events to external system."""
        # This is a placeholder - implement specific external system integration
        # In practice, you would:
        # 1. Format events according to external system requirements
        # 2. Handle authentication and authorization
        # 3. Send via HTTP API, message queue, or other protocol
        # 4. Handle retries and error conditions
        # 5. Verify delivery confirmation

        time.sleep(0.1)  # Simulate network latency


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

        # Initialize external system client
        if self.external_type is None:
            raise ValueError("External storage type is required but was None")
        self.client = ExternalSystemClient(
            self.external_type, self.external_config, self.retry_attempts
        )

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)

    def store(self, event_data: dict[str, Any]) -> str:
        """Store single audit event to external system."""
        # This is a placeholder implementation
        # In practice, you would integrate with specific external systems
        event_id = event_data.get("event_id", "unknown")

        try:
            # Send to external system
            self.client.send_events([event_data])
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
            self.client.send_events(events)

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
        return {
            "backend_type": "external",
            "external_type": self.external_type,
            "status": "active",
            "retry_attempts": self.retry_attempts,
        }

    def close(self) -> None:
        """Close external storage."""
        if self._executor:
            self._executor.shutdown(wait=True)

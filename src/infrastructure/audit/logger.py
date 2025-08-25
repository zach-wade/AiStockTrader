"""
Core audit logging functionality for financial trading operations.

This module implements high-performance audit logging with enterprise-grade
features including multiple storage backends, compliance reporting, and
real-time monitoring capabilities.
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from asyncio import Task
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from .config import AuditConfig
from .events import AuditEvent
from .exceptions import AuditException, AuditStorageError, AuditValidationError
from .formatters import AuditFormatter
from .storage import AuditStorage


@dataclass
class AuditContext:
    """Context information for audit operations."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str | None = None
    user_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


class AuditLogger:
    """
    High-performance audit logger for financial trading operations.

    Features:
        - Sub-millisecond logging performance
        - Multiple storage backends
        - Compliance reporting
        - Real-time monitoring
        - Integrity verification
        - Tamper detection
    """

    def __init__(
        self,
        config: AuditConfig,
        storage: AuditStorage,
        formatter: AuditFormatter,
        enable_async: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize audit logger.

        Args:
            config: Audit configuration
            storage: Storage backend
            formatter: Event formatter
            enable_async: Enable asynchronous processing
            max_workers: Maximum number of worker threads
        """
        self.config = config
        self.storage = storage
        self.formatter = formatter
        self.enable_async = enable_async
        self.max_workers = max_workers

        # Performance tracking
        self._event_count = 0
        self._error_count = 0
        self._start_time = time.time()
        self._lock = threading.RLock()

        # Thread pool for async processing
        self._executor: ThreadPoolExecutor | None
        if self.enable_async:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self._executor = None

        # Context storage for thread-local audit context
        self._context_storage = threading.local()

        # Monitoring hooks
        self._monitoring_hooks: list[Callable[[dict[str, Any]], None]] = []

        # Initialize logger
        self._logger = logging.getLogger(f"{__name__}.AuditLogger")

    def __enter__(self) -> "AuditLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close audit logger and cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)

    @property
    def current_context(self) -> AuditContext | None:
        """Get current audit context for this thread."""
        return getattr(self._context_storage, "context", None)

    @contextmanager
    def audit_context(self, context: AuditContext) -> Any:
        """
        Set audit context for current thread.

        Args:
            context: Audit context to set
        """
        old_context = getattr(self._context_storage, "context", None)
        self._context_storage.context = context
        try:
            yield context
        finally:
            self._context_storage.context = old_context

    def log_event(
        self, event: AuditEvent, context: AuditContext | None = None, immediate: bool = False
    ) -> str:
        """
        Log audit event.

        Args:
            event: Audit event to log
            context: Optional context override
            immediate: Force immediate synchronous processing

        Returns:
            Event ID for tracking
        """
        # Generate unique event ID
        event_id = str(uuid.uuid4())

        # Use provided context or current thread context
        audit_context = context or self.current_context

        try:
            # Validate event
            self._validate_event(event)

            # Enrich event with metadata
            enriched_event = self._enrich_event(event, event_id, audit_context)

            # Process event (async or sync)
            if self.enable_async and not immediate:
                self._process_event_async(enriched_event)
            else:
                self._process_event_sync(enriched_event)

            # Update metrics
            with self._lock:
                self._event_count += 1

            return event_id

        except Exception as e:
            with self._lock:
                self._error_count += 1

            self._logger.error(
                f"Failed to log audit event: {e}",
                exc_info=True,
                extra={"event_type": event.event_type, "event_id": event_id},
            )

            # For critical events, try to log failure to backup storage
            if event.is_critical:
                self._log_failure(event_id, event, e)

            raise AuditException(
                f"Failed to log audit event: {e}",
                error_code="AUDIT_LOG_FAILURE",
                context={"event_id": event_id, "event_type": event.event_type},
            )

    def log_order_event(
        self,
        event_type: str,
        order_id: str,
        symbol: str,
        quantity: float,
        price: float | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience method for logging order events."""
        from .events import OrderEvent

        event = OrderEvent(
            event_type=event_type,
            order_id=order_id,
            symbol=symbol,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)) if price is not None else None,
            user_id=user_id,
            **kwargs,
        )
        return self.log_event(event)

    def log_position_event(
        self,
        event_type: str,
        position_id: str,
        symbol: str,
        quantity: float,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience method for logging position events."""
        from .events import PositionEvent

        event = PositionEvent(
            event_type=event_type,
            position_id=position_id,
            symbol=symbol,
            quantity=Decimal(str(quantity)),
            user_id=user_id,
            **kwargs,
        )
        return self.log_event(event)

    def log_portfolio_event(
        self, event_type: str, portfolio_id: str, user_id: str | None = None, **kwargs: Any
    ) -> str:
        """Convenience method for logging portfolio events."""
        from .events import PortfolioEvent

        event = PortfolioEvent(
            event_type=event_type, portfolio_id=portfolio_id, user_id=user_id, **kwargs
        )
        return self.log_event(event)

    def log_risk_event(
        self,
        event_type: str,
        risk_type: str,
        threshold_value: float,
        current_value: float,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience method for logging risk events."""
        from .events import RiskEvent

        event = RiskEvent(
            event_type=event_type,
            risk_type=risk_type,
            threshold_value=Decimal(str(threshold_value)),
            current_value=Decimal(str(current_value)),
            user_id=user_id,
            **kwargs,
        )
        return self.log_event(event)

    def add_monitoring_hook(self, hook: Callable[[dict[str, Any]], None]) -> None:
        """Add monitoring hook for real-time event processing."""
        self._monitoring_hooks.append(hook)

    def remove_monitoring_hook(self, hook: Callable[[dict[str, Any]], None]) -> None:
        """Remove monitoring hook."""
        if hook in self._monitoring_hooks:
            self._monitoring_hooks.remove(hook)

    def get_metrics(self) -> dict[str, Any]:
        """Get audit logger performance metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            events_per_second = self._event_count / max(uptime, 1)
            error_rate = self._error_count / max(self._event_count, 1)

            return {
                "event_count": self._event_count,
                "error_count": self._error_count,
                "uptime_seconds": uptime,
                "events_per_second": events_per_second,
                "error_rate": error_rate,
                "async_enabled": self.enable_async,
                "max_workers": self.max_workers,
            }

    def _validate_event(self, event: AuditEvent) -> None:
        """Validate audit event before processing."""
        if not event.event_type:
            raise AuditValidationError(
                "Event type is required", field_name="event_type", validation_rule="not_empty"
            )

        if not event.resource_type:
            raise AuditValidationError(
                "Resource type is required", field_name="resource_type", validation_rule="not_empty"
            )

        # Additional validation based on configuration
        if self.config.strict_validation:
            event.validate()

    def _enrich_event(
        self, event: AuditEvent, event_id: str, context: AuditContext | None
    ) -> dict[str, Any]:
        """Enrich event with metadata and context."""
        # Convert event to dictionary
        event_data = event.to_dict()

        # Add system metadata
        event_data.update(
            {
                "event_id": event_id,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "system_version": self.config.system_version,
                "audit_version": "1.0.0",
            }
        )

        # Add context information
        if context:
            event_data["context"] = context.to_dict()

        # Add integrity hash
        event_data["integrity_hash"] = self._calculate_integrity_hash(event_data)

        return event_data

    def _calculate_integrity_hash(self, event_data: dict[str, Any]) -> str:
        """Calculate integrity hash for event data."""
        # Create deterministic JSON representation
        json_str = json.dumps(event_data, sort_keys=True, separators=(",", ":"))

        # Calculate SHA-256 hash
        hash_obj = hashlib.sha256(json_str.encode("utf-8"))
        return hash_obj.hexdigest()

    def _process_event_sync(self, event_data: dict[str, Any]) -> None:
        """Process audit event synchronously."""
        try:
            # Format event
            formatted_event = self.formatter.format(event_data)

            # Store event
            self.storage.store(formatted_event)

            # Trigger monitoring hooks
            self._trigger_monitoring_hooks(event_data)

        except Exception as e:
            raise AuditStorageError(
                f"Failed to process audit event: {e}", operation="process_sync", underlying_error=e
            )

    def _process_event_async(self, event_data: dict[str, Any]) -> None:
        """Process audit event asynchronously."""
        if self._executor:
            future = self._executor.submit(self._process_event_sync, event_data)
            # Don't wait for completion to maintain low latency
        else:
            # Fall back to synchronous processing
            self._process_event_sync(event_data)

    def _trigger_monitoring_hooks(self, event_data: dict[str, Any]) -> None:
        """Trigger monitoring hooks for real-time processing."""
        for hook in self._monitoring_hooks:
            try:
                hook(event_data)
            except Exception as e:
                self._logger.warning(
                    f"Monitoring hook failed: {e}",
                    exc_info=True,
                    extra={"event_id": event_data.get("event_id")},
                )

    def _log_failure(self, event_id: str, event: AuditEvent, error: Exception) -> None:
        """Log audit failure to backup storage."""
        try:
            failure_data = {
                "event_id": event_id,
                "event_type": event.event_type,
                "failure_timestamp": datetime.now(UTC).isoformat(),
                "error": str(error),
                "error_type": type(error).__name__,
            }

            # Use backup storage if available
            if hasattr(self.storage, "backup_storage"):
                self.storage.backup_storage.store(failure_data)
            else:
                # Log to file system as last resort
                self._logger.critical(f"Critical audit failure: {failure_data}", extra=failure_data)

        except Exception as backup_error:
            self._logger.critical(
                f"Failed to log audit failure: {backup_error}",
                exc_info=True,
                extra={"original_error": str(error)},
            )


class AsyncAuditLogger:
    """
    Asynchronous audit logger for high-throughput environments.

    Provides asyncio-compatible audit logging with minimal performance overhead.
    """

    def __init__(self, config: AuditConfig, storage: AuditStorage, formatter: AuditFormatter):
        """
        Initialize async audit logger.

        Args:
            config: Audit configuration
            storage: Storage backend
            formatter: Event formatter
        """
        self.config = config
        self.storage = storage
        self.formatter = formatter

        # Event queue for batch processing
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10000)
        self._processing_task: Task[Any] | None = None
        self._shutdown_event = asyncio.Event()

        # Performance tracking
        self._event_count = 0
        self._error_count = 0
        self._start_time = time.time()

        # Logger
        self._logger = logging.getLogger(f"{__name__}.AsyncAuditLogger")

    async def start(self) -> None:
        """Start async processing task."""
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._process_events())

    async def stop(self) -> None:
        """Stop async processing task."""
        self._shutdown_event.set()
        if self._processing_task:
            await self._processing_task
            self._processing_task = None

    async def log_event(self, event: AuditEvent) -> str:
        """
        Log audit event asynchronously.

        Args:
            event: Audit event to log

        Returns:
            Event ID for tracking
        """
        event_id = str(uuid.uuid4())

        try:
            # Enrich event
            enriched_event = self._enrich_event(event, event_id)

            # Add to queue (non-blocking)
            try:
                self._event_queue.put_nowait(enriched_event)
                self._event_count += 1
                return event_id
            except asyncio.QueueFull:
                # Handle queue overflow
                self._error_count += 1
                self._logger.error(
                    "Audit event queue full, dropping event",
                    extra={"event_id": event_id, "event_type": event.event_type},
                )
                raise AuditException(
                    "Audit event queue full",
                    error_code="QUEUE_FULL",
                    context={"event_id": event_id},
                )

        except Exception as e:
            self._error_count += 1
            self._logger.error(f"Failed to queue audit event: {e}", exc_info=True)
            raise AuditException(
                f"Failed to queue audit event: {e}",
                error_code="QUEUE_ERROR",
                context={"event_id": event_id},
            )

    async def _process_events(self) -> None:
        """Process events from queue in batches."""
        batch_size = self.config.batch_size
        batch_timeout = self.config.batch_timeout

        while not self._shutdown_event.is_set():
            try:
                events: list[dict[str, Any]] = []

                # Collect events for batch processing
                deadline = asyncio.get_event_loop().time() + batch_timeout

                while len(events) < batch_size and asyncio.get_event_loop().time() < deadline:
                    try:
                        remaining_time = deadline - asyncio.get_event_loop().time()
                        if remaining_time <= 0:
                            break

                        event = await asyncio.wait_for(
                            self._event_queue.get(), timeout=remaining_time
                        )
                        events.append(event)

                    except TimeoutError:
                        break

                # Process batch if we have events
                if events:
                    await self._process_batch(events)

            except Exception as e:
                self._logger.error(f"Error in event processing loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause before retrying

    async def _process_batch(self, events: list[dict[str, Any]]) -> None:
        """Process batch of events."""
        try:
            # Format events
            formatted_events = [self.formatter.format(event) for event in events]

            # Store batch
            await self._store_batch_async(formatted_events)

        except Exception as e:
            self._error_count += len(events)
            self._logger.error(f"Failed to process event batch: {e}", exc_info=True)

    async def _store_batch_async(self, events: list[dict[str, Any]]) -> Any:
        """Store batch of events asynchronously."""
        # Run storage operation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.storage.store_batch, events)

    def _enrich_event(self, event: AuditEvent, event_id: str) -> dict[str, Any]:
        """Enrich event with metadata."""
        event_data = event.to_dict()
        event_data.update(
            {
                "event_id": event_id,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "system_version": self.config.system_version,
                "audit_version": "1.0.0",
            }
        )
        return event_data

    def get_metrics(self) -> dict[str, Any]:
        """Get async audit logger metrics."""
        uptime = time.time() - self._start_time
        events_per_second = self._event_count / max(uptime, 1)
        error_rate = self._error_count / max(self._event_count, 1)

        return {
            "event_count": self._event_count,
            "error_count": self._error_count,
            "uptime_seconds": uptime,
            "events_per_second": events_per_second,
            "error_rate": error_rate,
            "queue_size": self._event_queue.qsize(),
            "queue_maxsize": self._event_queue.maxsize,
        }

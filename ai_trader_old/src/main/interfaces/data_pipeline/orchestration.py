"""
Data Pipeline Orchestration Interfaces

Interfaces for orchestration components that coordinate data pipeline operations
with layer-based architecture and event-driven processing.
"""

# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

# Local imports
# Import from our new core enums
from main.data_pipeline.core.enums import DataLayer, DataType, ProcessingPriority


class OrchestrationStatus(Enum):
    """Status values for orchestration operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ILayerManager(ABC):
    """Interface for managing layer-based data processing."""

    @abstractmethod
    async def get_layer_config(self, layer: DataLayer) -> dict[str, Any]:
        """Get configuration for a specific layer."""
        pass

    @abstractmethod
    async def get_symbols_for_layer(self, layer: DataLayer) -> list[str]:
        """Get symbols assigned to a specific layer."""
        pass

    @abstractmethod
    async def promote_symbol(self, symbol: str, from_layer: DataLayer, to_layer: DataLayer) -> bool:
        """Promote a symbol from one layer to another."""
        pass

    @abstractmethod
    async def demote_symbol(self, symbol: str, from_layer: DataLayer, to_layer: DataLayer) -> bool:
        """Demote a symbol from one layer to another."""
        pass

    @abstractmethod
    async def get_layer_limits(self, layer: DataLayer) -> dict[str, Any]:
        """Get processing limits for a layer."""
        pass

    @abstractmethod
    async def validate_layer_capacity(self, layer: DataLayer) -> bool:
        """Check if layer has capacity for new symbols."""
        pass


class IRetentionManager(ABC):
    """Interface for managing data retention policies."""

    @abstractmethod
    async def get_retention_policy(self, layer: DataLayer, data_type: DataType) -> dict[str, Any]:
        """Get retention policy for layer and data type."""
        pass

    @abstractmethod
    async def apply_retention_policy(self, symbol: str, layer: DataLayer) -> dict[str, Any]:
        """Apply retention policy to a symbol's data."""
        pass

    @abstractmethod
    async def cleanup_expired_data(self, layer: DataLayer) -> dict[str, Any]:
        """Clean up expired data for a layer."""
        pass

    @abstractmethod
    async def move_to_cold_storage(self, symbol: str, layer: DataLayer) -> bool:
        """Move data from hot to cold storage."""
        pass

    @abstractmethod
    async def archive_old_data(self, symbol: str, layer: DataLayer) -> bool:
        """Archive old data according to policy."""
        pass


class IUnifiedOrchestrator(ABC):
    """
    Interface for the unified orchestrator that replaces multiple separate orchestrators.

    Coordinates all data pipeline operations with layer-aware processing.
    """

    @abstractmethod
    async def start(self) -> None:
        """Start the orchestrator."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the orchestrator."""
        pass

    @abstractmethod
    async def process_symbol_data(
        self,
        symbol: str,
        data_types: list[DataType],
        layer: DataLayer | None = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
    ) -> dict[str, Any]:
        """Process data for a symbol across specified data types."""
        pass

    @abstractmethod
    async def backfill_symbol(
        self,
        symbol: str,
        data_types: list[DataType],
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer | None = None,
    ) -> dict[str, Any]:
        """Backfill historical data for a symbol."""
        pass

    @abstractmethod
    async def process_layer(
        self, layer: DataLayer, data_types: list[DataType], symbols: list[str] | None = None
    ) -> dict[str, Any]:
        """Process all symbols in a layer for specified data types."""
        pass

    @abstractmethod
    async def get_orchestration_status(self) -> dict[str, Any]:
        """Get current orchestration status."""
        pass

    @abstractmethod
    async def get_active_operations(self) -> list[dict[str, Any]]:
        """Get list of active operations."""
        pass

    @abstractmethod
    async def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a specific operation."""
        pass


class IDataProcessor(ABC):
    """Interface for data processing components with layer awareness."""

    @abstractmethod
    async def process(
        self,
        data: Any,
        data_type: DataType,
        layer: DataLayer,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Process data with layer-specific logic."""
        pass

    @abstractmethod
    async def validate_input(self, data: Any, data_type: DataType) -> bool:
        """Validate input data."""
        pass

    @abstractmethod
    async def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        pass


class IBackfillOrchestrator(ABC):
    """Interface for backfill orchestration operations."""

    @abstractmethod
    async def schedule_backfill(
        self,
        symbols: list[str],
        data_types: list[DataType],
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer | None = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
    ) -> str:
        """Schedule a backfill operation."""
        pass

    @abstractmethod
    async def execute_backfill(self, backfill_id: str) -> dict[str, Any]:
        """Execute a scheduled backfill."""
        pass

    @abstractmethod
    async def get_backfill_status(self, backfill_id: str) -> dict[str, Any]:
        """Get status of a backfill operation."""
        pass

    @abstractmethod
    async def cancel_backfill(self, backfill_id: str) -> bool:
        """Cancel a backfill operation."""
        pass

    @abstractmethod
    async def get_backfill_queue(self) -> list[dict[str, Any]]:
        """Get the current backfill queue."""
        pass


class IProgressTracker(ABC):
    """Interface for tracking progress of operations."""

    @abstractmethod
    async def start_tracking(self, operation_id: str, operation_type: str) -> None:
        """Start tracking an operation."""
        pass

    @abstractmethod
    async def update_progress(
        self, operation_id: str, progress_percent: float, details: dict[str, Any] | None = None
    ) -> None:
        """Update operation progress."""
        pass

    @abstractmethod
    async def complete_operation(
        self, operation_id: str, success: bool, results: dict[str, Any] | None = None
    ) -> None:
        """Mark operation as completed."""
        pass

    @abstractmethod
    async def get_operation_progress(self, operation_id: str) -> dict[str, Any]:
        """Get progress for a specific operation."""
        pass

    @abstractmethod
    async def get_all_operations(self) -> list[dict[str, Any]]:
        """Get all tracked operations."""
        pass

    @abstractmethod
    async def cleanup_completed_operations(self, older_than_hours: int = 24) -> int:
        """Clean up old completed operations."""
        pass


class IEventCoordinator(ABC):
    """Interface for coordinating event-driven operations."""

    @abstractmethod
    async def handle_symbol_qualified(self, symbol: str, layer: DataLayer) -> None:
        """Handle symbol qualification event."""
        pass

    @abstractmethod
    async def handle_symbol_promoted(
        self, symbol: str, from_layer: DataLayer, to_layer: DataLayer
    ) -> None:
        """Handle symbol promotion event."""
        pass

    @abstractmethod
    async def handle_data_gap_detected(
        self, symbol: str, data_type: DataType, gap_info: dict[str, Any]
    ) -> None:
        """Handle data gap detection event."""
        pass

    @abstractmethod
    async def schedule_automatic_backfill(self, symbol: str, layer: DataLayer) -> str:
        """Schedule automatic backfill based on layer policies."""
        pass

    @abstractmethod
    async def get_event_statistics(self) -> dict[str, Any]:
        """Get event processing statistics."""
        pass

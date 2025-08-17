"""
Data Pipeline Historical Interfaces

Interfaces for historical data management including gap detection,
backfill operations, and historical analysis with layer-aware processing.
"""

# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.data_pipeline.core.enums import DataLayer, DataType, ProcessingPriority


class IGapDetector(ABC):
    """Interface for detecting gaps in historical data."""

    @abstractmethod
    async def detect_gaps(
        self,
        symbol: str,
        data_type: DataType,
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer,
        interval: str | None = None,
    ) -> list[dict[str, Any]]:
        """Detect data gaps for a symbol."""
        pass

    @abstractmethod
    async def detect_gaps_batch(
        self,
        symbols: list[str],
        data_type: DataType,
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer,
    ) -> dict[str, list[dict[str, Any]]]:
        """Detect gaps for multiple symbols."""
        pass

    @abstractmethod
    async def analyze_gap_patterns(
        self, symbol: str, data_type: DataType, layer: DataLayer, lookback_days: int = 30
    ) -> dict[str, Any]:
        """Analyze patterns in data gaps."""
        pass

    @abstractmethod
    async def prioritize_gaps(
        self, gaps: list[dict[str, Any]], layer: DataLayer
    ) -> list[dict[str, Any]]:
        """Prioritize gaps for backfill based on layer policies."""
        pass

    @abstractmethod
    async def get_gap_statistics(
        self, layer: DataLayer | None = None, data_type: DataType | None = None
    ) -> dict[str, Any]:
        """Get gap detection statistics."""
        pass


class IHistoricalManager(ABC):
    """Interface for managing historical data operations."""

    @abstractmethod
    async def start(self) -> None:
        """Start the historical manager."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the historical manager."""
        pass

    @abstractmethod
    async def backfill_symbol(
        self,
        symbol: str,
        data_types: list[DataType],
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
    ) -> str:
        """Backfill historical data for a symbol."""
        pass

    @abstractmethod
    async def backfill_layer(
        self,
        layer: DataLayer,
        data_types: list[DataType],
        start_date: datetime,
        end_date: datetime,
        symbols: list[str] | None = None,
    ) -> str:
        """Backfill historical data for an entire layer."""
        pass

    @abstractmethod
    async def schedule_maintenance_backfill(
        self, layer: DataLayer, maintenance_type: str = "gap_fill"
    ) -> str:
        """Schedule maintenance backfill for a layer."""
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
    async def get_historical_coverage(
        self, symbol: str, data_type: DataType, layer: DataLayer
    ) -> dict[str, Any]:
        """Get historical data coverage for a symbol."""
        pass


class IDataFetcher(ABC):
    """Interface for fetching historical data from sources."""

    @abstractmethod
    async def fetch_historical_data(
        self,
        symbol: str,
        data_type: DataType,
        start_date: datetime,
        end_date: datetime,
        interval: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical data for a symbol."""
        pass

    @abstractmethod
    async def fetch_batch_historical_data(
        self,
        symbols: list[str],
        data_type: DataType,
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols."""
        pass

    @abstractmethod
    async def get_available_date_range(
        self, symbol: str, data_type: DataType, source: str | None = None
    ) -> tuple[datetime | None, datetime | None]:
        """Get available date range for a symbol from source."""
        pass

    @abstractmethod
    async def validate_data_availability(
        self,
        symbol: str,
        data_type: DataType,
        date_range: tuple[datetime, datetime],
        source: str | None = None,
    ) -> dict[str, Any]:
        """Validate data availability for a symbol."""
        pass

    @abstractmethod
    async def get_fetch_statistics(self) -> dict[str, Any]:
        """Get data fetching statistics."""
        pass


class IDataRouter(ABC):
    """Interface for routing historical data operations."""

    @abstractmethod
    async def route_data_request(
        self,
        symbol: str,
        data_type: DataType,
        layer: DataLayer,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Route a data request to appropriate handler."""
        pass

    @abstractmethod
    async def get_optimal_source(
        self,
        symbol: str,
        data_type: DataType,
        layer: DataLayer,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> str:
        """Get optimal data source for a request."""
        pass

    @abstractmethod
    async def route_batch_request(
        self, symbols: list[str], data_type: DataType, layer: DataLayer, operation: str
    ) -> dict[str, dict[str, Any]]:
        """Route batch data requests."""
        pass

    @abstractmethod
    async def get_routing_statistics(self) -> dict[str, Any]:
        """Get data routing statistics."""
        pass


class IHistoricalAnalyzer(ABC):
    """Interface for analyzing historical data patterns."""

    @abstractmethod
    async def analyze_data_quality_trends(
        self, symbol: str, data_type: DataType, layer: DataLayer, lookback_days: int = 90
    ) -> dict[str, Any]:
        """Analyze data quality trends over time."""
        pass

    @abstractmethod
    async def analyze_coverage_gaps(
        self, layer: DataLayer, data_type: DataType, lookback_days: int = 30
    ) -> dict[str, Any]:
        """Analyze coverage gaps across a layer."""
        pass

    @abstractmethod
    async def generate_backfill_recommendations(
        self, layer: DataLayer, analysis_period_days: int = 30
    ) -> list[dict[str, Any]]:
        """Generate backfill recommendations for a layer."""
        pass

    @abstractmethod
    async def analyze_symbol_activity(
        self, symbol: str, layer: DataLayer, lookback_days: int = 90
    ) -> dict[str, Any]:
        """Analyze historical activity patterns for a symbol."""
        pass

    @abstractmethod
    async def compare_layer_coverage(
        self, layers: list[DataLayer], data_type: DataType
    ) -> dict[str, Any]:
        """Compare data coverage across layers."""
        pass


class IArchiveManager(ABC):
    """Interface for managing historical data archives."""

    @abstractmethod
    async def archive_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        data_type: DataType,
        layer: DataLayer,
        archive_date: datetime,
    ) -> str:
        """Archive historical data."""
        pass

    @abstractmethod
    async def retrieve_archived_data(
        self,
        symbol: str,
        data_type: DataType,
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer,
    ) -> pd.DataFrame:
        """Retrieve data from archive."""
        pass

    @abstractmethod
    async def list_archived_data(
        self,
        symbol: str | None = None,
        data_type: DataType | None = None,
        layer: DataLayer | None = None,
    ) -> list[dict[str, Any]]:
        """List available archived data."""
        pass

    @abstractmethod
    async def cleanup_old_archives(self, layer: DataLayer, older_than_days: int) -> dict[str, Any]:
        """Clean up old archived data based on retention policy."""
        pass

    @abstractmethod
    async def get_archive_statistics(self) -> dict[str, Any]:
        """Get archive storage statistics."""
        pass


class IMaintenanceScheduler(ABC):
    """Interface for scheduling historical data maintenance tasks."""

    @abstractmethod
    async def schedule_gap_detection(
        self, layer: DataLayer, data_type: DataType, schedule: str = "daily"
    ) -> str:
        """Schedule regular gap detection."""
        pass

    @abstractmethod
    async def schedule_data_validation(
        self, layer: DataLayer, validation_type: str = "quality_check", schedule: str = "weekly"
    ) -> str:
        """Schedule data validation tasks."""
        pass

    @abstractmethod
    async def schedule_archive_cleanup(self, layer: DataLayer, schedule: str = "monthly") -> str:
        """Schedule archive cleanup tasks."""
        pass

    @abstractmethod
    async def get_scheduled_tasks(self, layer: DataLayer | None = None) -> list[dict[str, Any]]:
        """Get scheduled maintenance tasks."""
        pass

    @abstractmethod
    async def cancel_scheduled_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        pass

    @abstractmethod
    async def execute_maintenance_task(self, task_id: str) -> dict[str, Any]:
        """Execute a maintenance task immediately."""
        pass

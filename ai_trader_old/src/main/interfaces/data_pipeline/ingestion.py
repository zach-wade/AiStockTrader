"""
Data Pipeline Ingestion Interfaces

Interfaces for data ingestion components including clients, processors,
and coordinators with standardized patterns and layer awareness.
"""

# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.data_pipeline.core.enums import DataLayer, DataType, ProcessingPriority


class IDataSource(ABC):
    """Interface for data source configuration and management."""

    @abstractmethod
    async def get_source_info(self) -> dict[str, Any]:
        """Get information about the data source."""
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate source credentials."""
        pass

    @abstractmethod
    async def get_rate_limits(self) -> dict[str, Any]:
        """Get API rate limits for the source."""
        pass

    @abstractmethod
    async def get_supported_data_types(self) -> list[DataType]:
        """Get supported data types for this source."""
        pass

    @abstractmethod
    async def get_available_symbols(self, data_type: DataType) -> list[str]:
        """Get available symbols for a data type."""
        pass

    @abstractmethod
    async def test_connection(self) -> dict[str, Any]:
        """Test connection to the data source."""
        pass


class IDataClient(ABC):
    """Interface for data client implementations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the client."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the client."""
        pass

    @abstractmethod
    async def fetch_data(
        self,
        symbol: str,
        data_type: DataType,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        interval: str | None = None,
        **kwargs,
    ) -> Any:
        """Fetch data for a symbol."""
        pass

    @abstractmethod
    async def fetch_batch_data(
        self,
        symbols: list[str],
        data_type: DataType,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Fetch data for multiple symbols."""
        pass

    @abstractmethod
    async def get_client_status(self) -> dict[str, Any]:
        """Get client status information."""
        pass

    @abstractmethod
    async def get_usage_statistics(self) -> dict[str, Any]:
        """Get client usage statistics."""
        pass


class IIngestionClient(ABC):
    """Interface for high-level ingestion client operations."""

    @abstractmethod
    async def ingest_symbol_data(
        self,
        symbol: str,
        data_types: list[DataType],
        layer: DataLayer,
        date_range: tuple[datetime, datetime] | None = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
    ) -> dict[str, Any]:
        """Ingest data for a symbol."""
        pass

    @abstractmethod
    async def ingest_layer_data(
        self,
        layer: DataLayer,
        data_types: list[DataType],
        symbols: list[str] | None = None,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """Ingest data for an entire layer."""
        pass

    @abstractmethod
    async def schedule_ingestion(
        self,
        symbols: list[str],
        data_types: list[DataType],
        layer: DataLayer,
        schedule_config: dict[str, Any],
    ) -> str:
        """Schedule recurring ingestion."""
        pass

    @abstractmethod
    async def get_ingestion_status(self, ingestion_id: str) -> dict[str, Any]:
        """Get status of an ingestion operation."""
        pass

    @abstractmethod
    async def cancel_ingestion(self, ingestion_id: str) -> bool:
        """Cancel an ingestion operation."""
        pass


class IAssetClient(ABC):
    """Interface for asset information client."""

    @abstractmethod
    async def get_asset_info(self, symbol: str) -> dict[str, Any]:
        """Get asset information for a symbol."""
        pass

    @abstractmethod
    async def get_assets_by_exchange(self, exchange: str) -> list[dict[str, Any]]:
        """Get assets for a specific exchange."""
        pass

    @abstractmethod
    async def get_active_assets(self) -> list[dict[str, Any]]:
        """Get all currently active assets."""
        pass

    @abstractmethod
    async def search_assets(
        self, query: str, asset_class: str | None = None, exchange: str | None = None
    ) -> list[dict[str, Any]]:
        """Search for assets matching criteria."""
        pass

    @abstractmethod
    async def validate_symbol(self, symbol: str) -> dict[str, Any]:
        """Validate if a symbol exists and is tradeable."""
        pass


class IIngestionProcessor(ABC):
    """Interface for processing ingested data."""

    @abstractmethod
    async def process_raw_data(
        self,
        raw_data: Any,
        symbol: str,
        data_type: DataType,
        layer: DataLayer,
        source_metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Process raw data into standardized format."""
        pass

    @abstractmethod
    async def validate_processed_data(
        self, processed_data: pd.DataFrame, data_type: DataType, layer: DataLayer
    ) -> dict[str, Any]:
        """Validate processed data."""
        pass

    @abstractmethod
    async def enrich_data(
        self, data: pd.DataFrame, symbol: str, data_type: DataType, layer: DataLayer
    ) -> pd.DataFrame:
        """Enrich data with additional fields."""
        pass

    @abstractmethod
    async def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        pass


class IIngestionCoordinator(ABC):
    """Interface for coordinating ingestion operations."""

    @abstractmethod
    async def start_coordinator(self) -> None:
        """Start the ingestion coordinator."""
        pass

    @abstractmethod
    async def stop_coordinator(self) -> None:
        """Stop the ingestion coordinator."""
        pass

    @abstractmethod
    async def coordinate_ingestion(
        self, ingestion_request: dict[str, Any], layer: DataLayer
    ) -> str:
        """Coordinate an ingestion request."""
        pass

    @abstractmethod
    async def manage_ingestion_queue(self, layer: DataLayer) -> dict[str, Any]:
        """Manage the ingestion queue for a layer."""
        pass

    @abstractmethod
    async def balance_ingestion_load(self) -> dict[str, Any]:
        """Balance ingestion load across sources and clients."""
        pass

    @abstractmethod
    async def get_coordinator_status(self) -> dict[str, Any]:
        """Get coordinator status."""
        pass

    @abstractmethod
    async def get_ingestion_metrics(self, layer: DataLayer | None = None) -> dict[str, Any]:
        """Get ingestion metrics."""
        pass


class IRateLimiter(ABC):
    """Interface for rate limiting ingestion operations."""

    @abstractmethod
    async def acquire_permit(self, source: str, operation_type: str, layer: DataLayer) -> bool:
        """Acquire a rate limit permit."""
        pass

    @abstractmethod
    async def release_permit(self, source: str, operation_type: str, layer: DataLayer) -> None:
        """Release a rate limit permit."""
        pass

    @abstractmethod
    async def get_remaining_quota(self, source: str, operation_type: str) -> int:
        """Get remaining quota for an operation."""
        pass

    @abstractmethod
    async def wait_for_quota(self, source: str, operation_type: str, layer: DataLayer) -> float:
        """Wait for quota availability and return wait time."""
        pass

    @abstractmethod
    async def get_rate_limit_status(self) -> dict[str, Any]:
        """Get rate limiting status."""
        pass


class IIngestionValidator(ABC):
    """Interface for validating ingested data."""

    @abstractmethod
    async def validate_data_schema(
        self, data: pd.DataFrame, data_type: DataType, source: str
    ) -> dict[str, Any]:
        """Validate data against expected schema."""
        pass

    @abstractmethod
    async def validate_data_quality(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        layer: DataLayer,
        quality_profile: str | None = None,
    ) -> dict[str, Any]:
        """Validate data quality."""
        pass

    @abstractmethod
    async def validate_data_completeness(
        self,
        data: pd.DataFrame,
        symbol: str,
        data_type: DataType,
        expected_date_range: tuple[datetime, datetime],
    ) -> dict[str, Any]:
        """Validate data completeness."""
        pass

    @abstractmethod
    async def validate_business_rules(
        self, data: pd.DataFrame, data_type: DataType, layer: DataLayer
    ) -> dict[str, Any]:
        """Validate business rules."""
        pass

    @abstractmethod
    async def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        pass


class IIngestionMonitor(ABC):
    """Interface for monitoring ingestion operations."""

    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start ingestion monitoring."""
        pass

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop ingestion monitoring."""
        pass

    @abstractmethod
    async def monitor_ingestion_health(self) -> dict[str, Any]:
        """Monitor ingestion system health."""
        pass

    @abstractmethod
    async def monitor_source_availability(self) -> dict[str, Any]:
        """Monitor data source availability."""
        pass

    @abstractmethod
    async def monitor_ingestion_performance(self, layer: DataLayer) -> dict[str, Any]:
        """Monitor ingestion performance for a layer."""
        pass

    @abstractmethod
    async def detect_ingestion_anomalies(self) -> list[dict[str, Any]]:
        """Detect anomalies in ingestion patterns."""
        pass

    @abstractmethod
    async def generate_ingestion_alerts(self) -> list[dict[str, Any]]:
        """Generate alerts for ingestion issues."""
        pass

    @abstractmethod
    async def get_monitoring_dashboard(self) -> dict[str, Any]:
        """Get monitoring dashboard data."""
        pass

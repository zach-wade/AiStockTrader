"""
Ingestion System Interfaces

Defines interfaces for data ingestion operations including
bulk loading, data transformation, and ETL pipeline components.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class LoadStrategy(Enum):
    """Strategy for loading data to storage."""

    COPY = "copy"  # Use PostgreSQL COPY command
    INSERT = "insert"  # Use batch INSERT statements
    UPSERT = "upsert"  # Use INSERT ON CONFLICT UPDATE
    REPLACE = "replace"  # Delete and insert


@dataclass
class BulkLoadConfig:
    """Configuration for bulk data loading operations."""

    buffer_size: int = 10000  # Records before flushing
    use_copy_command: bool = True  # Use COPY vs INSERT
    batch_timeout_seconds: float = 30.0  # Max time before flush
    max_memory_mb: int = 500  # Max memory before flush
    parallel_archives: int = 3  # Parallel archive operations
    retry_on_failure: bool = True  # Retry failed operations
    max_retries: int = 3  # Maximum retry attempts
    recovery_enabled: bool = True  # Save failed records for recovery
    recovery_directory: str = "data/recovery"  # Where to save recovery files


@dataclass
class BulkLoadResult:
    """Result of a bulk load operation."""

    success: bool
    records_loaded: int = 0
    records_failed: int = 0
    symbols_processed: set[str] = field(default_factory=set)
    load_time_seconds: float = 0.0
    archive_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    data_type: str = ""
    skipped: bool = False
    skip_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IBulkLoader(Protocol):
    """
    Interface for bulk data loading operations.

    Bulk loaders handle efficient loading of large datasets
    into storage systems with buffering, batching, and error recovery.
    """

    async def load(self, data: Any, **kwargs) -> BulkLoadResult:
        """
        Load data into storage system.

        Args:
            data: Data to load (format depends on implementation)
            **kwargs: Additional parameters specific to data type

        Returns:
            BulkLoadResult with operation details
        """
        ...

    async def flush_all(self) -> BulkLoadResult:
        """
        Flush all buffered data to storage.

        Returns:
            BulkLoadResult with flush operation details
        """
        ...

    def get_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for the loader.

        Returns:
            Dictionary with metrics like records/sec, memory usage, etc.
        """
        ...

    def get_buffer_status(self) -> dict[str, Any]:
        """
        Get current buffer status.

        Returns:
            Dictionary with buffer size, memory usage, symbols, etc.
        """
        ...

    async def recover_failed_records(self, recovery_file: str) -> BulkLoadResult:
        """
        Attempt to reload previously failed records.

        Args:
            recovery_file: Path to recovery file

        Returns:
            BulkLoadResult for recovery operation
        """
        ...


@runtime_checkable
class IDataTransformer(Protocol):
    """
    Interface for data transformation operations.

    Transformers convert data from source format to storage format.
    """

    def transform(self, raw_data: Any, source: str) -> list[dict[str, Any]]:
        """
        Transform raw data to storage format.

        Args:
            raw_data: Raw data from source
            source: Source identifier (e.g., 'polygon', 'yahoo')

        Returns:
            List of transformed records ready for storage
        """
        ...

    def validate(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Validate transformed records.

        Args:
            records: Records to validate

        Returns:
            List of valid records (invalid ones filtered out)
        """
        ...


@runtime_checkable
class IIngestionClient(Protocol):
    """
    Interface for data source ingestion clients.

    Clients handle fetching data from external sources.
    """

    def can_fetch(self, data_type: str) -> bool:
        """
        Check if this client can fetch the specified data type.

        Args:
            data_type: Type of data (e.g., 'market_data', 'news')

        Returns:
            True if client supports this data type
        """
        ...

    async def fetch_and_archive(
        self,
        data_type: str,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        archive: Any,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Fetch data and archive to cold storage.

        Args:
            data_type: Type of data to fetch
            symbols: List of symbols to fetch
            start_date: Start of date range
            end_date: End of date range
            archive: Archive instance for cold storage
            **kwargs: Additional parameters

        Returns:
            Dictionary with fetch results and statistics
        """
        ...

    def get_supported_data_types(self) -> list[str]:
        """
        Get list of supported data types.

        Returns:
            List of data type strings this client supports
        """
        ...


@runtime_checkable
class IIngestionOrchestrator(Protocol):
    """
    Interface for ingestion orchestration.

    Orchestrators coordinate the ETL pipeline from source to storage.
    """

    async def ingest(
        self,
        data_type: str,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        sources: list[str] | None = None,
        load_to_hot: bool = True,
        archive_to_cold: bool = True,
    ) -> dict[str, Any]:
        """
        Orchestrate data ingestion from sources to storage.

        Args:
            data_type: Type of data to ingest
            symbols: Symbols to process
            start_date: Start of date range
            end_date: End of date range
            sources: Specific sources to use (None = all available)
            load_to_hot: Whether to load to hot storage (database)
            archive_to_cold: Whether to archive to cold storage

        Returns:
            Dictionary with ingestion results and statistics
        """
        ...

    def get_pipeline_status(self) -> dict[str, Any]:
        """
        Get current pipeline status.

        Returns:
            Dictionary with active operations, queues, metrics
        """
        ...


@runtime_checkable
class IBulkLoaderFactory(Protocol):
    """
    Factory interface for creating bulk loader instances.
    """

    def create_loader(self, data_type: str, config: BulkLoadConfig | None = None) -> IBulkLoader:
        """
        Create a bulk loader for the specified data type.

        Args:
            data_type: Type of data to load
            config: Optional configuration override

        Returns:
            Configured bulk loader instance
        """
        ...

    def get_supported_types(self) -> list[str]:
        """
        Get list of supported data types.

        Returns:
            List of data types this factory can create loaders for
        """
        ...

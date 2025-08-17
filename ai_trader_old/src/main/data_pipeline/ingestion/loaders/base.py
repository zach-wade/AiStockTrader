"""
Base class for bulk data loaders.

This module provides the abstract base class for all bulk loaders,
defining the common interface and shared functionality.
"""

# Standard library imports
from abc import ABC, abstractmethod
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Generic, TypeVar

# Local imports
from main.data_pipeline.storage.archive import DataArchive
from main.interfaces.database import IAsyncDatabase
from main.interfaces.ingestion import BulkLoadConfig, BulkLoadResult, IBulkLoader
from main.utils.core import AsyncCircuitBreaker, get_logger, timer
from main.utils.processing.historical import ProcessingUtils

logger = get_logger(__name__)

T = TypeVar("T")  # Generic type for data records


class BaseBulkLoader(IBulkLoader, Generic[T], ABC):
    """
    Abstract base class for bulk data loaders.

    Provides common functionality for accumulation, buffering,
    and metrics tracking. Subclasses implement specific loading logic.
    """

    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        archive: DataArchive | None = None,
        config: BulkLoadConfig | None = None,
        data_type: str = "unknown",
    ):
        """
        Initialize base bulk loader.

        Args:
            db_adapter: Database adapter for PostgreSQL operations
            archive: Optional archive for cold storage
            config: Bulk loading configuration
            data_type: Type of data this loader handles
        """
        self.db_adapter = db_adapter
        self.archive = archive
        self.config = config or BulkLoadConfig()
        self.data_type = data_type

        # Processing utilities
        self.processing_utils = ProcessingUtils()

        # Circuit breaker for database operations
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=5, recovery_timeout=30.0, expected_exception=Exception
        )

        # Accumulation buffers - subclasses can add more specific buffers
        self._buffer: list[T] = []
        self._buffer_size_bytes = 0
        self._symbols_in_buffer: set[str] = set()
        self._last_flush_time = datetime.now(UTC)

        # Metrics
        self._total_records_loaded = 0
        self._total_records_failed = 0
        self._total_load_time = 0.0
        self._total_archive_time = 0.0
        self._flush_count = 0

        logger.info(
            f"{self.__class__.__name__} initialized for {data_type} with config: {self.config}"
        )

    @abstractmethod
    async def load(self, data: Any, **kwargs) -> BulkLoadResult:
        """
        Main entry point for loading data.

        Args:
            data: Data to load (format depends on subclass)
            **kwargs: Additional parameters specific to data type

        Returns:
            BulkLoadResult with operation details
        """
        pass

    @abstractmethod
    def _prepare_records(self, data: Any, **kwargs) -> list[T]:
        """
        Prepare data records for loading.

        Args:
            data: Raw data to prepare
            **kwargs: Additional parameters

        Returns:
            List of prepared records
        """
        pass

    @abstractmethod
    async def _load_to_database(self, records: list[T]) -> int:
        """
        Load records to database.

        Args:
            records: Records to load

        Returns:
            Number of records loaded
        """
        pass

    @abstractmethod
    async def _archive_records(self, records: list[T]) -> None:
        """
        Archive records to cold storage.

        Args:
            records: Records to archive
        """
        pass

    def _estimate_record_size(self, record: T) -> int:
        """
        Estimate memory size of a record in bytes.

        Override in subclasses for more accurate estimates.

        Args:
            record: Record to estimate

        Returns:
            Estimated size in bytes
        """
        return 200  # Default estimate

    def _add_to_buffer(self, records: list[T], symbol: str | None = None):
        """Add records to the accumulation buffer."""
        self._buffer.extend(records)
        if symbol:
            self._symbols_in_buffer.add(symbol)

        # Estimate buffer size
        for record in records:
            self._buffer_size_bytes += self._estimate_record_size(record)

        # Track records added even if not flushing yet
        self._total_records_loaded += len(records)

        logger.debug(
            f"{self.__class__.__name__} buffer now contains {len(self._buffer)} records "
            f"for {len(self._symbols_in_buffer)} symbols"
        )

    def _should_flush(self) -> bool:
        """Determine if buffer should be flushed."""
        # Check accumulation size
        if len(self._buffer) >= self.config.buffer_size:
            logger.info(f"Buffer size {len(self._buffer)} reached limit {self.config.buffer_size}")
            return True

        # Check memory usage
        memory_mb = self._buffer_size_bytes / (1024 * 1024)
        if memory_mb >= self.config.max_memory_mb:
            logger.info(
                f"Buffer memory {memory_mb:.1f}MB reached limit {self.config.max_memory_mb}MB"
            )
            return True

        # Check timeout
        time_since_flush = (datetime.now(UTC) - self._last_flush_time).total_seconds()
        if time_since_flush >= self.config.batch_timeout_seconds and self._buffer:
            logger.info(
                f"Buffer timeout {time_since_flush:.1f}s reached limit "
                f"{self.config.batch_timeout_seconds}s"
            )
            return True

        return False

    async def _flush_buffer(self) -> BulkLoadResult:
        """Flush accumulated data to database."""
        result = BulkLoadResult(success=False, data_type=self.data_type)

        if not self._buffer:
            result.success = True
            return result

        # Save buffer state in case we need to restore
        buffer_backup = self._buffer.copy()
        symbols_backup = self._symbols_in_buffer.copy()
        buffer_size_backup = self._buffer_size_bytes

        with timer("bulk_load_flush") as t:
            try:
                # Archive to cold storage first (if configured)
                if self.archive:
                    try:
                        with timer("bulk_archive") as archive_timer:
                            await self._archive_records(self._buffer)
                            result.archive_time_seconds = archive_timer.elapsed
                    except Exception as e:
                        # Archive failure is non-critical, log and continue
                        logger.error(f"Archive failed during flush: {e}")
                        result.errors.append(f"Archive error: {e!s}")

                # Load to hot storage
                try:
                    logger.info(
                        f"{self.__class__.__name__} flushing {len(self._buffer)} records to database"
                    )
                    records_loaded = await self._load_to_database(self._buffer)

                    result.records_loaded = records_loaded
                    result.symbols_processed = self._symbols_in_buffer.copy()
                    result.success = True

                    if records_loaded != len(self._buffer):
                        logger.warning(
                            f"{self.__class__.__name__} only loaded {records_loaded} of "
                            f"{len(self._buffer)} records"
                        )
                        result.records_failed = len(self._buffer) - records_loaded
                        self._total_records_failed += result.records_failed

                    # Update metrics
                    self._total_load_time += t.elapsed
                    self._flush_count += 1

                    # Clear buffer only on success
                    self._buffer.clear()
                    self._symbols_in_buffer.clear()
                    self._buffer_size_bytes = 0
                    self._last_flush_time = datetime.now(UTC)

                    logger.info(
                        f"✓ {self.__class__.__name__} flushed {records_loaded} {self.data_type} records "
                        f"for {len(result.symbols_processed)} symbols in {t.elapsed:.2f}s"
                    )

                except Exception as e:
                    # Database load failed
                    logger.error(f"Failed to load to database: {e}")
                    result.errors.append(f"Database error: {e!s}")

                    # Restore buffer state
                    self._buffer = buffer_backup
                    self._symbols_in_buffer = symbols_backup
                    self._buffer_size_bytes = buffer_size_backup

                    # Save failed records for recovery if enabled
                    if self.config.recovery_enabled and self._buffer:
                        await self._save_recovery_file(self._buffer)

                    raise

            except Exception as e:
                logger.error(f"Flush operation failed: {e}")
                result.errors.append(str(e))

            finally:
                result.load_time_seconds = t.elapsed

        return result

    async def flush_all(self) -> BulkLoadResult:
        """
        Flush all buffered data to storage.

        Returns:
            BulkLoadResult with flush operation details
        """
        if not self._buffer:
            return BulkLoadResult(success=True, data_type=self.data_type)

        logger.info(f"Force flushing {len(self._buffer)} buffered records")
        return await self._flush_buffer()

    def get_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for the loader.

        Returns:
            Dictionary with metrics
        """
        avg_load_time = (self._total_load_time / self._flush_count) if self._flush_count > 0 else 0
        records_per_second = (
            (self._total_records_loaded / self._total_load_time) if self._total_load_time > 0 else 0
        )

        return {
            "data_type": self.data_type,
            "total_records_loaded": self._total_records_loaded,
            "total_records_failed": self._total_records_failed,
            "total_flushes": self._flush_count,
            "total_load_time_seconds": self._total_load_time,
            "total_archive_time_seconds": self._total_archive_time,
            "average_flush_time_seconds": avg_load_time,
            "records_per_second": records_per_second,
            "buffer_config": {
                "buffer_size": self.config.buffer_size,
                "max_memory_mb": self.config.max_memory_mb,
                "timeout_seconds": self.config.batch_timeout_seconds,
            },
        }

    def get_buffer_status(self) -> dict[str, Any]:
        """
        Get current buffer status.

        Returns:
            Dictionary with buffer information
        """
        memory_mb = self._buffer_size_bytes / (1024 * 1024)
        time_since_flush = (datetime.now(UTC) - self._last_flush_time).total_seconds()

        return {
            "records_in_buffer": len(self._buffer),
            "symbols_in_buffer": len(self._symbols_in_buffer),
            "buffer_memory_mb": memory_mb,
            "time_since_flush_seconds": time_since_flush,
            "will_flush_on_next_add": self._should_flush(),
        }

    async def _save_recovery_file(self, records: list[T]) -> str:
        """
        Save failed records to a recovery file.

        Args:
            records: Records to save

        Returns:
            Path to recovery file
        """
        try:
            # Create recovery directory if it doesn't exist
            recovery_dir = Path(self.config.recovery_directory)
            recovery_dir.mkdir(parents=True, exist_ok=True)

            # Generate recovery filename
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_type}_{timestamp}_recovery.json"
            filepath = recovery_dir / filename

            # Convert records to JSON-serializable format
            serializable_records = []
            for record in records:
                if isinstance(record, dict):
                    serializable_records.append(record)
                elif hasattr(record, "__dict__"):
                    serializable_records.append(record.__dict__)
                else:
                    serializable_records.append(str(record))

            # Save to file
            with open(filepath, "w") as f:
                json.dump(
                    {
                        "data_type": self.data_type,
                        "timestamp": timestamp,
                        "record_count": len(records),
                        "records": serializable_records,
                    },
                    f,
                    indent=2,
                    default=str,
                )

            logger.info(f"Saved {len(records)} failed records to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save recovery file: {e}")
            return ""

    async def recover_failed_records(self, recovery_file: str) -> BulkLoadResult:
        """
        Attempt to reload previously failed records.

        Args:
            recovery_file: Path to recovery file

        Returns:
            BulkLoadResult for recovery operation
        """
        result = BulkLoadResult(success=False, data_type=self.data_type)

        try:
            with open(recovery_file) as f:
                recovery_data = json.load(f)

            records = recovery_data.get("records", [])
            if not records:
                result.success = True
                result.skip_reason = "No records in recovery file"
                return result

            logger.info(f"Attempting to recover {len(records)} records from {recovery_file}")

            # Process recovered records
            self._buffer = records
            self._buffer_size_bytes = len(records) * 200  # Rough estimate

            # Attempt to load
            flush_result = await self._flush_buffer()

            result.success = flush_result.success
            result.records_loaded = flush_result.records_loaded
            result.records_failed = flush_result.records_failed
            result.errors = flush_result.errors
            result.metadata["recovery_file"] = recovery_file

            if result.success:
                logger.info(f"Successfully recovered {result.records_loaded} records")
                # Optionally delete recovery file on success
                Path(recovery_file).unlink(missing_ok=True)
            else:
                logger.error(f"Failed to recover records: {result.errors}")

            return result

        except Exception as e:
            logger.error(f"Error during recovery: {e}")
            result.errors.append(str(e))
            return result

"""
Repository Core Operations Service

Handles basic CRUD operations for repositories.
Focused on single-record operations with caching and metrics.
"""

# Standard library imports
import time
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import OperationResult
from main.utils.core import get_logger

from .repository_query_builder import RepositoryQueryBuilder

logger = get_logger(__name__)


class RepositoryCoreOperations:
    """
    Service for basic repository CRUD operations.

    Handles single-record operations with caching, metrics, and validation.
    Composes with query builder for SQL construction.
    """

    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        query_builder: RepositoryQueryBuilder,
        table_name: str,
        validate_record_fn: callable,
        cache_mixin: Any = None,
        metrics_mixin: Any = None,
    ):
        """
        Initialize core operations service.

        Args:
            db_adapter: Database adapter for async operations
            query_builder: Query builder for SQL construction
            table_name: Database table name
            validate_record_fn: Function to validate records
            cache_mixin: Optional cache mixin for caching operations
            metrics_mixin: Optional metrics mixin for recording metrics
        """
        self.db_adapter = db_adapter
        self.query_builder = query_builder
        self.table_name = table_name
        self.validate_record = validate_record_fn
        self.cache_mixin = cache_mixin
        self.metrics_mixin = metrics_mixin
        self.logger = get_logger(__name__)

    async def get_by_id(self, record_id: Any) -> dict[str, Any] | None:
        """
        Get a single record by ID.

        Args:
            record_id: ID of the record to retrieve

        Returns:
            Record data as dictionary, or None if not found
        """
        start_time = time.time()

        try:
            # Check cache first
            cached_result = await self._get_from_cache_if_enabled(
                f"{self.table_name}_id", id=record_id
            )

            if cached_result is not None:
                await self._record_cache_metric_if_enabled(hit=True)
                return cached_result

            await self._record_cache_metric_if_enabled(hit=False)

            # Query database
            query = f"SELECT * FROM {self.table_name} WHERE id = $1"
            result = await self.db_adapter.fetch_one(query, record_id)

            if result:
                result_dict = dict(result)
                # Cache the result
                await self._set_in_cache_if_enabled(
                    f"{self.table_name}_id", result_dict, id=record_id
                )

                # Record metrics
                duration = time.time() - start_time
                await self._record_operation_metric_if_enabled(
                    "get_by_id", duration, success=True, records=1
                )

                return result_dict

            # Record metrics for not found
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                "get_by_id", duration, success=True, records=0
            )

            return None

        except Exception as e:
            logger.error(f"Error getting record by ID {record_id}: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled("get_by_id", duration, success=False)
            raise

    async def create(self, data: dict[str, Any] | pd.DataFrame) -> OperationResult:
        """
        Create new records.

        Args:
            data: Single record dict or DataFrame of records

        Returns:
            OperationResult with creation details
        """
        start_time = time.time()

        try:
            # Convert DataFrame to list of dicts if needed
            if isinstance(data, pd.DataFrame):
                records = data.to_dict("records")
            else:
                records = [data]

            # Validate records
            for record in records:
                errors = self.validate_record(record)
                if errors:
                    return OperationResult(
                        success=False, error=f"Validation failed: {', '.join(errors)}"
                    )

            # Insert records
            created_ids = []
            for record in records:
                query_data = self.query_builder.build_insert_query(record)
                result = await self.db_adapter.execute_query(
                    query_data["sql"], *query_data["params"]
                )
                if result:
                    created_ids.append(result)

            # Invalidate cache
            await self._invalidate_cache_if_enabled(f"{self.table_name}*")

            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                "create", duration, success=True, records=len(created_ids)
            )

            return OperationResult(
                success=True,
                data=created_ids,
                records_affected=len(created_ids),
                records_created=len(created_ids),
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Error creating records: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled("create", duration, success=False)
            return OperationResult(success=False, error=str(e), duration_seconds=duration)

    async def update(self, record_id: Any, data: dict[str, Any]) -> OperationResult:
        """
        Update an existing record.

        Args:
            record_id: ID of record to update
            data: Data to update

        Returns:
            OperationResult with update details
        """
        start_time = time.time()

        try:
            # Build update query
            query_data = self.query_builder.build_update_query(record_id, data)

            # Execute update
            affected = await self.db_adapter.execute_query(query_data["sql"], *query_data["params"])

            # Invalidate cache
            await self._invalidate_cache_if_enabled(f"{self.table_name}_id", id=record_id)

            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                "update", duration, success=True, records=1 if affected else 0
            )

            return OperationResult(
                success=True,
                records_affected=1 if affected else 0,
                records_updated=1 if affected else 0,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Error updating record {record_id}: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled("update", duration, success=False)
            return OperationResult(success=False, error=str(e), duration_seconds=duration)

    async def delete(self, record_id: Any) -> OperationResult:
        """
        Delete a record.

        Args:
            record_id: ID of record to delete

        Returns:
            OperationResult with deletion details
        """
        start_time = time.time()

        try:
            query = f"DELETE FROM {self.table_name} WHERE id = $1"
            affected = await self.db_adapter.execute_query(query, record_id)

            # Invalidate cache
            await self._invalidate_cache_if_enabled(f"{self.table_name}_id", id=record_id)

            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                "delete", duration, success=True, records=1 if affected else 0
            )

            return OperationResult(
                success=True,
                records_affected=1 if affected else 0,
                records_deleted=1 if affected else 0,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Error deleting record {record_id}: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled("delete", duration, success=False)
            return OperationResult(success=False, error=str(e), duration_seconds=duration)

    async def exists(self, record_id: Any) -> bool:
        """
        Check if a record exists.

        Args:
            record_id: ID of record to check

        Returns:
            True if record exists, False otherwise
        """
        try:
            query = f"SELECT 1 FROM {self.table_name} WHERE id = $1 LIMIT 1"
            result = await self.db_adapter.fetch_one(query, record_id)
            return result is not None

        except Exception as e:
            logger.error(f"Error checking existence for {record_id}: {e}")
            return False

    async def count_all(self) -> int:
        """
        Count all records in the table.

        Returns:
            Total number of records
        """
        try:
            query = f"SELECT COUNT(*) as count FROM {self.table_name}"
            result = await self.db_adapter.fetch_one(query)
            return result.get("count", 0) if result else 0

        except Exception as e:
            logger.error(f"Error counting records: {e}")
            return 0

    # Cache helper methods
    async def _get_from_cache_if_enabled(self, prefix: str, **kwargs) -> Any | None:
        """Get from cache if caching is enabled."""
        if not self.cache_mixin:
            return None

        cache_key = self.cache_mixin._get_cache_key(prefix, **kwargs)
        return await self.cache_mixin._get_from_cache(cache_key)

    async def _set_in_cache_if_enabled(self, prefix: str, value: Any, **kwargs) -> None:
        """Set in cache if caching is enabled."""
        if not self.cache_mixin:
            return

        cache_key = self.cache_mixin._get_cache_key(prefix, **kwargs)
        await self.cache_mixin._set_in_cache(cache_key, value)

    async def _invalidate_cache_if_enabled(self, pattern: str, **kwargs) -> None:
        """Invalidate cache if caching is enabled."""
        if not self.cache_mixin:
            return

        if kwargs:
            cache_key = self.cache_mixin._get_cache_key(pattern, **kwargs)
            await self.cache_mixin._invalidate_cache(cache_key)
        else:
            await self.cache_mixin._invalidate_cache(pattern)

    # Metrics helper methods
    async def _record_operation_metric_if_enabled(
        self, operation: str, duration: float, success: bool, records: int = 0
    ) -> None:
        """Record operation metric if metrics are enabled."""
        if not self.metrics_mixin:
            return

        await self.metrics_mixin._record_operation_metric(operation, duration, success, records)

    async def _record_cache_metric_if_enabled(self, hit: bool) -> None:
        """Record cache metric if metrics are enabled."""
        if not self.metrics_mixin:
            return

        await self.metrics_mixin._record_cache_metric(hit)

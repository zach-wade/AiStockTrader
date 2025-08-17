"""
Base Repository Implementation

Abstract base class providing common database operations through composition.
Uses helper classes for specialized tasks following Single Responsibility Principle.
"""

# Standard library imports
from abc import ABC, abstractmethod
import time
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories import IRepository
from main.interfaces.repositories.base import OperationResult, QueryFilter, RepositoryConfig
from main.utils.core import ensure_utc, get_logger

from .helpers import sanitize_order_by_columns, validate_filter_keys
from .repository_patterns import CacheMixin, DualStorageMixin, MetricsMixin, RepositoryMixin

logger = get_logger(__name__)


class BaseRepository(IRepository, RepositoryMixin, DualStorageMixin, CacheMixin, MetricsMixin, ABC):
    """
    Enhanced base repository providing common database operations.

    This class orchestrates interactions between a database adapter and
    specialized helper components for query building, CRUD operations,
    validation, caching, and metrics.
    """

    def __init__(
        self, db_adapter: IAsyncDatabase, model_class: type, config: RepositoryConfig | None = None
    ):
        """
        Initialize the base repository.

        Args:
            db_adapter: Database adapter for async operations
            model_class: SQLAlchemy model class
            config: Optional repository configuration
        """
        self.db_adapter = db_adapter
        self.model_class = model_class
        self.config = config or RepositoryConfig()

        # Initialize mixins
        super().__init__()

        # Table name from model
        self.table_name = getattr(model_class, "__tablename__", model_class.__name__.lower())

        logger.info(f"Initialized {self.__class__.__name__} for table {self.table_name}")

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def get_required_fields(self) -> list[str]:
        """Get list of required fields for this repository."""
        pass

    @abstractmethod
    def validate_record(self, record: dict[str, Any]) -> list[str]:
        """
        Validate a record.

        Args:
            record: Record to validate

        Returns:
            List of validation errors (empty if valid)
        """
        pass

    # IRepository interface implementation
    async def get_by_id(self, record_id: Any) -> dict[str, Any] | None:
        """Get a single record by ID."""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._get_cache_key(f"{self.table_name}_id", id=record_id)
            cached_result = await self._get_from_cache(cache_key)

            if cached_result is not None:
                await self._record_cache_metric(hit=True)
                return cached_result

            await self._record_cache_metric(hit=False)

            # Query database
            query = f"SELECT * FROM {self.table_name} WHERE id = $1"
            result = await self.db_adapter.fetch_one(query, record_id)

            if result:
                # Cache the result
                await self._set_in_cache(cache_key, dict(result))

            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric(
                "get_by_id", duration, success=True, records=1 if result else 0
            )

            return dict(result) if result else None

        except Exception as e:
            logger.error(f"Error getting record by ID {record_id}: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric("get_by_id", duration, success=False)
            raise

    async def get_by_filter(self, query_filter: QueryFilter) -> pd.DataFrame:
        """Get records matching filter criteria."""
        start_time = time.time()

        try:
            # Build query
            query, params = self._build_select_query(query_filter)

            # Execute query
            # Note: Cold storage feature removed for simplicity
            # All queries now use the primary database storage
            results = await self.db_adapter.fetch_all(query, *params)

            # Convert to DataFrame
            df = pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric(
                "get_by_filter", duration, success=True, records=len(df)
            )

            return df

        except Exception as e:
            logger.error(f"Error getting records by filter: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric("get_by_filter", duration, success=False)
            raise

    async def create(self, data: dict[str, Any] | pd.DataFrame) -> OperationResult:
        """Create new records."""
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
                query = self._build_insert_query(record)
                result = await self.db_adapter.execute_query(query["sql"], *query["params"])
                if result:
                    created_ids.append(result)

            # Invalidate cache
            await self._invalidate_cache(f"{self.table_name}*")

            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric(
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
            await self._record_operation_metric("create", duration, success=False)
            return OperationResult(success=False, error=str(e), duration_seconds=duration)

    async def update(self, record_id: Any, data: dict[str, Any]) -> OperationResult:
        """Update an existing record."""
        start_time = time.time()

        try:
            # Build update query
            query = self._build_update_query(record_id, data)

            # Execute update
            affected = await self.db_adapter.execute_query(query["sql"], *query["params"])

            # Invalidate cache
            cache_key = self._get_cache_key(f"{self.table_name}_id", id=record_id)
            await self._invalidate_cache(cache_key)

            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric(
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
            await self._record_operation_metric("update", duration, success=False)
            return OperationResult(success=False, error=str(e), duration_seconds=duration)

    async def upsert(self, data: dict[str, Any] | pd.DataFrame) -> OperationResult:
        """Insert or update records."""
        start_time = time.time()

        try:
            # Convert DataFrame to list of dicts if needed
            if isinstance(data, pd.DataFrame):
                records = data.to_dict("records")
            else:
                records = [data]

            # Perform upsert
            created = 0
            updated = 0

            for record in records:
                # Check if record exists
                record_id = record.get("id") or record.get(f"{self.table_name}_id")
                if record_id and await self.exists(record_id):
                    # Update existing
                    result = await self.update(record_id, record)
                    if result.success:
                        updated += 1
                else:
                    # Create new
                    result = await self.create(record)
                    if result.success:
                        created += 1

            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric(
                "upsert", duration, success=True, records=created + updated
            )

            return OperationResult(
                success=True,
                records_affected=created + updated,
                records_created=created,
                records_updated=updated,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Error upserting records: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric("upsert", duration, success=False)
            return OperationResult(success=False, error=str(e), duration_seconds=duration)

    async def delete(self, record_id: Any) -> OperationResult:
        """Delete a record."""
        start_time = time.time()

        try:
            query = f"DELETE FROM {self.table_name} WHERE id = $1"
            affected = await self.db_adapter.execute_query(query, record_id)

            # Invalidate cache
            cache_key = self._get_cache_key(f"{self.table_name}_id", id=record_id)
            await self._invalidate_cache(cache_key)

            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric(
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
            await self._record_operation_metric("delete", duration, success=False)
            return OperationResult(success=False, error=str(e), duration_seconds=duration)

    async def delete_by_filter(self, query_filter: QueryFilter) -> OperationResult:
        """Delete records matching filter criteria."""
        start_time = time.time()

        try:
            # Build delete query
            conditions = []
            params = []
            param_count = 1

            if query_filter.symbol:
                conditions.append(f"symbol = ${param_count}")
                params.append(query_filter.symbol)
                param_count += 1

            if query_filter.start_date:
                conditions.append(f"timestamp >= ${param_count}")
                params.append(query_filter.start_date)
                param_count += 1

            if query_filter.end_date:
                conditions.append(f"timestamp <= ${param_count}")
                params.append(query_filter.end_date)
                param_count += 1

            where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"DELETE FROM {self.table_name}{where_clause}"

            affected = await self.db_adapter.execute_query(query, *params)

            # Invalidate cache
            await self._invalidate_cache(f"{self.table_name}*")

            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric(
                "delete_by_filter", duration, success=True, records=affected or 0
            )

            return OperationResult(
                success=True,
                records_affected=affected or 0,
                records_deleted=affected or 0,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Error deleting by filter: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric("delete_by_filter", duration, success=False)
            return OperationResult(success=False, error=str(e), duration_seconds=duration)

    async def count(self, query_filter: QueryFilter | None = None) -> int:
        """Count records matching filter criteria."""
        try:
            if query_filter:
                query, params = self._build_count_query(query_filter)
            else:
                query = f"SELECT COUNT(*) FROM {self.table_name}"
                params = []

            result = await self.db_adapter.fetch_one(query, *params)
            return result.get("count", 0) if result else 0

        except Exception as e:
            logger.error(f"Error counting records: {e}")
            return 0

    async def exists(self, record_id: Any) -> bool:
        """Check if a record exists."""
        try:
            query = f"SELECT 1 FROM {self.table_name} WHERE id = $1 LIMIT 1"
            result = await self.db_adapter.fetch_one(query, record_id)
            return result is not None

        except Exception as e:
            logger.error(f"Error checking existence for {record_id}: {e}")
            return False

    # Helper methods for query building
    def _build_select_query(self, query_filter: QueryFilter) -> tuple:
        """Build SELECT query from filter."""
        conditions = []
        params = []
        param_count = 1

        if query_filter.symbol:
            conditions.append(f"symbol = ${param_count}")
            params.append(self._normalize_symbol(query_filter.symbol))
            param_count += 1

        if query_filter.symbols:
            placeholders = [
                f"${i}" for i in range(param_count, param_count + len(query_filter.symbols))
            ]
            conditions.append(f"symbol IN ({','.join(placeholders)})")
            params.extend([self._normalize_symbol(s) for s in query_filter.symbols])
            param_count += len(query_filter.symbols)

        if query_filter.start_date:
            conditions.append(f"timestamp >= ${param_count}")
            params.append(ensure_utc(query_filter.start_date))
            param_count += 1

        if query_filter.end_date:
            conditions.append(f"timestamp <= ${param_count}")
            params.append(ensure_utc(query_filter.end_date))
            param_count += 1

        # Add custom filters
        # Validate all filter keys before using them
        validated_filters = validate_filter_keys(self.table_name, query_filter.filters)
        for key, value in validated_filters.items():
            conditions.append(f"{key} = ${param_count}")
            params.append(value)
            param_count += 1

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        # Add ordering
        order_clause = ""
        if query_filter.order_by:
            # Validate and sanitize ORDER BY columns to prevent SQL injection
            sanitized_columns = sanitize_order_by_columns(self.table_name, query_filter.order_by)
            order_dir = "ASC" if query_filter.ascending else "DESC"
            order_clause = f" ORDER BY {', '.join(sanitized_columns)} {order_dir}"

        # Add pagination
        limit_clause = ""
        if query_filter.limit:
            limit_clause = f" LIMIT {query_filter.limit}"
        if query_filter.offset:
            limit_clause += f" OFFSET {query_filter.offset}"

        query = f"SELECT * FROM {self.table_name}{where_clause}{order_clause}{limit_clause}"

        return query, params

    def _build_count_query(self, query_filter: QueryFilter) -> tuple:
        """Build COUNT query from filter."""
        # Similar to select but without ordering/pagination
        conditions = []
        params = []
        param_count = 1

        if query_filter.symbol:
            conditions.append(f"symbol = ${param_count}")
            params.append(self._normalize_symbol(query_filter.symbol))
            param_count += 1

        if query_filter.start_date:
            conditions.append(f"timestamp >= ${param_count}")
            params.append(ensure_utc(query_filter.start_date))
            param_count += 1

        if query_filter.end_date:
            conditions.append(f"timestamp <= ${param_count}")
            params.append(ensure_utc(query_filter.end_date))
            param_count += 1

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT COUNT(*) as count FROM {self.table_name}{where_clause}"

        return query, params

    def _build_insert_query(self, record: dict[str, Any]) -> dict[str, Any]:
        """Build INSERT query."""
        columns = list(record.keys())
        values = list(record.values())
        placeholders = [f"${i+1}" for i in range(len(values))]

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
        """

        return {"sql": query, "params": values}

    def _build_update_query(self, record_id: Any, data: dict[str, Any]) -> dict[str, Any]:
        """Build UPDATE query."""
        set_clauses = []
        params = []

        for i, (key, value) in enumerate(data.items(), 1):
            set_clauses.append(f"{key} = ${i}")
            params.append(value)

        params.append(record_id)

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_clauses)}
            WHERE id = ${len(params)}
        """

        return {"sql": query, "params": params}

    async def cleanup(self) -> None:
        """
        Clean up repository resources.

        This includes clearing caches, closing connections if needed,
        and releasing any held resources.
        """
        try:
            # Clear any caches
            if hasattr(self, "_cache"):
                self._cache.clear()

            # Clear metrics if applicable
            if hasattr(self, "_metrics"):
                self._metrics.clear()

            logger.info(f"Cleaned up resources for {self.__class__.__name__}")

        except Exception as e:
            logger.error(f"Error during cleanup for {self.__class__.__name__}: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()
        return False

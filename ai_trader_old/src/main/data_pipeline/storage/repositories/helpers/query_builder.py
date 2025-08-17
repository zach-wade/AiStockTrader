"""
Query Builder Helper

Constructs SQL queries from QueryFilter objects for repository operations.
"""

# Standard library imports
from typing import Any

# Local imports
from main.interfaces.repositories.base import QueryFilter
from main.utils.core import ensure_utc, get_logger

from .sql_validator import sanitize_order_by_columns, validate_filter_keys, validate_table_column

logger = get_logger(__name__)


class QueryBuilder:
    """
    Builds SQL queries based on QueryFilter objects.

    Provides methods to construct SELECT, INSERT, UPDATE, DELETE queries
    with proper parameterization for PostgreSQL.
    """

    def __init__(self, table_name: str, model_class: type | None = None):
        """
        Initialize the QueryBuilder.

        Args:
            table_name: Name of the database table
            model_class: Optional SQLAlchemy model class for metadata
        """
        self.table_name = table_name
        self.model_class = model_class
        logger.debug(f"QueryBuilder initialized for table: {table_name}")

    def build_select(
        self, filters: QueryFilter, columns: list[str] | None = None
    ) -> tuple[str, list[Any]]:
        """
        Build a SELECT query with filters.

        Args:
            filters: Query filter criteria
            columns: Optional specific columns to select

        Returns:
            Tuple of (SQL query string, parameter values)
        """
        # Select clause
        if columns:
            select_clause = f"SELECT {', '.join(columns)}"
        else:
            select_clause = "SELECT *"

        # Build WHERE clause
        where_clause, params = self._build_where_clause(filters)

        # Build ORDER BY clause
        order_clause = self._build_order_clause(filters)

        # Build LIMIT/OFFSET clause
        limit_clause = self._build_limit_clause(filters)

        # Combine all parts
        query = f"{select_clause} FROM {self.table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if order_clause:
            query += f" {order_clause}"
        if limit_clause:
            query += f" {limit_clause}"

        logger.debug(f"Built SELECT query: {query[:100]}...")
        return query, params

    def build_count(self, filters: QueryFilter) -> tuple[str, list[Any]]:
        """
        Build a COUNT query with filters.

        Args:
            filters: Query filter criteria

        Returns:
            Tuple of (SQL query string, parameter values)
        """
        where_clause, params = self._build_where_clause(filters)

        query = f"SELECT COUNT(*) as count FROM {self.table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"

        return query, params

    def build_insert(
        self, data: dict[str, Any], returning: list[str] | None = None
    ) -> tuple[str, list[Any]]:
        """
        Build an INSERT query.

        Args:
            data: Data to insert
            returning: Optional columns to return

        Returns:
            Tuple of (SQL query string, parameter values)
        """
        columns = list(data.keys())
        values = list(data.values())
        placeholders = [f"${i+1}" for i in range(len(values))]

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """

        if returning:
            query += f" RETURNING {', '.join(returning)}"
        elif returning is None:
            query += " RETURNING id"

        return query.strip(), values

    def build_update(self, data: dict[str, Any], filters: QueryFilter) -> tuple[str, list[Any]]:
        """
        Build an UPDATE query.

        Args:
            data: Fields to update
            filters: Filter criteria for records to update

        Returns:
            Tuple of (SQL query string, parameter values)
        """
        # Build SET clause
        set_clauses = []
        params = []
        param_count = 1

        for key, value in data.items():
            # Validate column name for SQL injection prevention
            validate_table_column(self.table_name, key)
            set_clauses.append(f"{key} = ${param_count}")
            params.append(value)
            param_count += 1

        # Build WHERE clause
        where_conditions = []

        if filters.symbol:
            where_conditions.append(f"symbol = ${param_count}")
            params.append(filters.symbol.upper())
            param_count += 1

        if hasattr(filters, "record_id"):
            where_conditions.append(f"id = ${param_count}")
            params.append(filters.record_id)
            param_count += 1

        # Add custom filter conditions
        # Validate all filter keys before using them
        validated_filters = validate_filter_keys(self.table_name, filters.filters)
        for key, value in validated_filters.items():
            where_conditions.append(f"{key} = ${param_count}")
            params.append(value)
            param_count += 1

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_clauses)}
        """

        if where_conditions:
            query += f" WHERE {' AND '.join(where_conditions)}"

        return query.strip(), params

    def build_delete(self, filters: QueryFilter) -> tuple[str, list[Any]]:
        """
        Build a DELETE query.

        Args:
            filters: Filter criteria for records to delete

        Returns:
            Tuple of (SQL query string, parameter values)
        """
        where_clause, params = self._build_where_clause(filters)

        query = f"DELETE FROM {self.table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        else:
            # Prevent accidental deletion of all records
            raise ValueError("DELETE query requires at least one filter condition")

        return query, params

    def build_upsert(
        self,
        data: dict[str, Any],
        conflict_columns: list[str],
        update_columns: list[str] | None = None,
    ) -> tuple[str, list[Any]]:
        """
        Build an UPSERT (INSERT ... ON CONFLICT) query.

        Args:
            data: Data to insert/update
            conflict_columns: Columns that define uniqueness
            update_columns: Columns to update on conflict (default: all)

        Returns:
            Tuple of (SQL query string, parameter values)
        """
        # Build INSERT part
        columns = list(data.keys())
        values = list(data.values())
        placeholders = [f"${i+1}" for i in range(len(values))]

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT ({', '.join(conflict_columns)})
        """

        # Build UPDATE part
        if update_columns is None:
            update_columns = [c for c in columns if c not in conflict_columns]

        if update_columns:
            update_set = [f"{col} = EXCLUDED.{col}" for col in update_columns]
            query += f" DO UPDATE SET {', '.join(update_set)}"
        else:
            query += " DO NOTHING"

        query += " RETURNING id"

        return query.strip(), values

    def _build_where_clause(self, filters: QueryFilter) -> tuple[str, list[Any]]:
        """Build WHERE clause from filters."""
        conditions = []
        params = []
        param_count = 1

        # Symbol filter
        if filters.symbol:
            conditions.append(f"symbol = ${param_count}")
            params.append(filters.symbol.upper())
            param_count += 1

        # Multiple symbols filter
        if filters.symbols:
            placeholders = [f"${i}" for i in range(param_count, param_count + len(filters.symbols))]
            conditions.append(f"symbol IN ({','.join(placeholders)})")
            params.extend([s.upper() for s in filters.symbols])
            param_count += len(filters.symbols)

        # Date range filters
        if filters.start_date:
            conditions.append(f"timestamp >= ${param_count}")
            params.append(ensure_utc(filters.start_date))
            param_count += 1

        if filters.end_date:
            conditions.append(f"timestamp <= ${param_count}")
            params.append(ensure_utc(filters.end_date))
            param_count += 1

        # Custom filters
        # Validate all filter keys before using them
        validated_filters = validate_filter_keys(self.table_name, filters.filters)
        for key, value in validated_filters.items():
            if value is not None:
                conditions.append(f"{key} = ${param_count}")
                params.append(value)
                param_count += 1

        where_clause = " AND ".join(conditions) if conditions else ""
        return where_clause, params

    def _build_order_clause(self, filters: QueryFilter) -> str:
        """Build ORDER BY clause from filters."""
        if not filters.order_by:
            return ""

        # Validate and sanitize ORDER BY columns to prevent SQL injection
        sanitized_columns = sanitize_order_by_columns(self.table_name, filters.order_by)

        order_dir = "ASC" if filters.ascending else "DESC"
        order_columns = ", ".join(sanitized_columns)
        return f"ORDER BY {order_columns} {order_dir}"

    def _build_limit_clause(self, filters: QueryFilter) -> str:
        """Build LIMIT/OFFSET clause from filters."""
        parts = []

        if filters.limit is not None:
            parts.append(f"LIMIT {filters.limit}")

        if filters.offset is not None:
            parts.append(f"OFFSET {filters.offset}")

        return " ".join(parts)

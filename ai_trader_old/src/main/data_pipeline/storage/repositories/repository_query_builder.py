"""
Repository Query Builder Service

Pure query building logic for repository operations.
Constructs SQL queries without execution, handling parameter binding and validation.
"""

# Standard library imports
from typing import Any

# Local imports
from main.interfaces.repositories.base import QueryFilter
from main.utils.core import ensure_utc

from .helpers import sanitize_order_by_columns, validate_filter_keys


class RepositoryQueryBuilder:
    """
    Service for building SQL queries for repository operations.

    Focused on pure query construction without database execution.
    Handles parameter binding, validation, and SQL injection prevention.
    """

    def __init__(self, table_name: str):
        """
        Initialize the query builder.

        Args:
            table_name: Name of the database table
        """
        self.table_name = table_name

    def build_select_query(self, query_filter: QueryFilter) -> tuple[str, list[Any]]:
        """
        Build SELECT query from filter.

        Args:
            query_filter: Filter criteria for the query

        Returns:
            Tuple of (SQL query string, parameters list)
        """
        conditions = []
        params = []
        param_count = 1

        # Build WHERE conditions
        param_count = self._add_symbol_conditions(query_filter, conditions, params, param_count)
        param_count = self._add_date_conditions(query_filter, conditions, params, param_count)
        param_count = self._add_custom_filter_conditions(
            query_filter, conditions, params, param_count
        )

        # Build complete query
        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        order_clause = self._build_order_clause(query_filter)
        limit_clause = self._build_limit_clause(query_filter)

        query = f"SELECT * FROM {self.table_name}{where_clause}{order_clause}{limit_clause}"

        return query, params

    def build_count_query(self, query_filter: QueryFilter) -> tuple[str, list[Any]]:
        """
        Build COUNT query from filter.

        Args:
            query_filter: Filter criteria for the query

        Returns:
            Tuple of (SQL query string, parameters list)
        """
        conditions = []
        params = []
        param_count = 1

        # Build WHERE conditions (same as select but without ordering/pagination)
        param_count = self._add_symbol_conditions(query_filter, conditions, params, param_count)
        param_count = self._add_date_conditions(query_filter, conditions, params, param_count)
        param_count = self._add_custom_filter_conditions(
            query_filter, conditions, params, param_count
        )

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT COUNT(*) as count FROM {self.table_name}{where_clause}"

        return query, params

    def build_insert_query(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Build INSERT query.

        Args:
            record: Record data to insert

        Returns:
            Dictionary with 'sql' and 'params' keys
        """
        columns = list(record.keys())
        values = list(record.values())
        placeholders = [f"${i+1}" for i in range(len(values))]

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
        """

        return {"sql": query, "params": values}

    def build_update_query(self, record_id: Any, data: dict[str, Any]) -> dict[str, Any]:
        """
        Build UPDATE query.

        Args:
            record_id: ID of record to update
            data: Data to update

        Returns:
            Dictionary with 'sql' and 'params' keys
        """
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

    def build_delete_by_filter_query(self, query_filter: QueryFilter) -> tuple[str, list[Any]]:
        """
        Build DELETE query with filter conditions.

        Args:
            query_filter: Filter criteria for deletion

        Returns:
            Tuple of (SQL query string, parameters list)
        """
        conditions = []
        params = []
        param_count = 1

        # Build WHERE conditions
        param_count = self._add_symbol_conditions(query_filter, conditions, params, param_count)
        param_count = self._add_date_conditions(query_filter, conditions, params, param_count)
        param_count = self._add_custom_filter_conditions(
            query_filter, conditions, params, param_count
        )

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"DELETE FROM {self.table_name}{where_clause}"

        return query, params

    def _add_symbol_conditions(
        self, query_filter: QueryFilter, conditions: list[str], params: list[Any], param_count: int
    ) -> int:
        """Add symbol-related WHERE conditions."""
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

        return param_count

    def _add_date_conditions(
        self, query_filter: QueryFilter, conditions: list[str], params: list[Any], param_count: int
    ) -> int:
        """Add date-related WHERE conditions."""
        if query_filter.start_date:
            conditions.append(f"timestamp >= ${param_count}")
            params.append(ensure_utc(query_filter.start_date))
            param_count += 1

        if query_filter.end_date:
            conditions.append(f"timestamp <= ${param_count}")
            params.append(ensure_utc(query_filter.end_date))
            param_count += 1

        return param_count

    def _add_custom_filter_conditions(
        self, query_filter: QueryFilter, conditions: list[str], params: list[Any], param_count: int
    ) -> int:
        """Add custom filter WHERE conditions."""
        # Validate all filter keys before using them
        validated_filters = validate_filter_keys(self.table_name, query_filter.filters)

        for key, value in validated_filters.items():
            conditions.append(f"{key} = ${param_count}")
            params.append(value)
            param_count += 1

        return param_count

    def _build_order_clause(self, query_filter: QueryFilter) -> str:
        """Build ORDER BY clause."""
        if not query_filter.order_by:
            return ""

        # Validate and sanitize ORDER BY columns to prevent SQL injection
        sanitized_columns = sanitize_order_by_columns(self.table_name, query_filter.order_by)
        order_dir = "ASC" if query_filter.ascending else "DESC"

        return f" ORDER BY {', '.join(sanitized_columns)} {order_dir}"

    def _build_limit_clause(self, query_filter: QueryFilter) -> str:
        """Build LIMIT/OFFSET clause."""
        limit_clause = ""

        if query_filter.limit:
            limit_clause = f" LIMIT {query_filter.limit}"

        if query_filter.offset:
            limit_clause += f" OFFSET {query_filter.offset}"

        return limit_clause

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to uppercase."""
        return symbol.upper() if symbol else symbol

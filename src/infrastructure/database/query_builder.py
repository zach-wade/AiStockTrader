"""
Type-safe SQL Query Builder - Secure parameterized query construction.

This module provides type-safe query building with automatic parameterization
to prevent SQL injection vulnerabilities. It enforces the use of parameterized
queries and provides validation for SQL identifiers.

Key Security Features:
- All values are automatically parameterized (never concatenated)
- SQL identifiers are validated against allowlists
- Clear separation between structure (identifiers) and data (parameters)
- Compile-time type safety where possible
- Runtime validation of query structure

Usage Examples:
    # SELECT query
    query = (QueryBuilder()
        .select(['id', 'name', 'email'])
        .from_table('users')
        .where('status = %s', ['active'])
        .order_by('name')
        .limit(10)
        .build())

    # INSERT query
    query = (QueryBuilder()
        .insert_into('users', ['name', 'email', 'status'])
        .values('%s, %s, %s', ['John Doe', 'john@example.com', 'active'])
        .build())

    # UPDATE query
    query = (QueryBuilder()
        .update('users')
        .set(['name = %s', 'email = %s'], ['Jane Doe', 'jane@example.com'])
        .where('id = %s', [123])
        .build())
"""

import logging
from enum import Enum
from typing import Any

from src.infrastructure.security.input_sanitizer import InputSanitizer, SanitizationError

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Enumeration of supported query types."""

    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class QueryBuilderError(Exception):
    """Raised when query building fails due to validation or structure errors."""

    pass


class SecurityError(QueryBuilderError):
    """Raised when query building fails due to security validation."""

    pass


class QueryResult:
    """
    Result of query building containing the SQL and parameters.

    This class encapsulates the final SQL query and its parameters,
    ensuring they can only be used together safely.
    """

    def __init__(self, sql: str, parameters: list[Any]):
        self.sql = sql
        self.parameters = parameters
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification after creation for security."""
        if hasattr(self, "_frozen") and self._frozen and name != "_frozen":
            raise AttributeError("QueryResult is immutable after creation")
        super().__setattr__(name, value)

    def execute_with_cursor(self, cursor) -> Any:
        """
        Execute this query with the provided database cursor.

        Args:
            cursor: Database cursor to execute with

        Returns:
            Cursor result
        """
        logger.debug(f"Executing SQL: {self.sql}")
        logger.debug(f"With parameters: {self.parameters}")
        return cursor.execute(self.sql, self.parameters)

    def __str__(self) -> str:
        return f"QueryResult(sql={self.sql!r}, parameters={self.parameters!r})"

    def __repr__(self) -> str:
        return self.__str__()


class QueryBuilder:
    """
    Type-safe SQL query builder with automatic parameterization.

    This class builds SQL queries while enforcing security best practices:
    - All data values are parameterized automatically
    - SQL identifiers (tables, columns) are validated
    - Query structure is validated before building
    - Clear separation between SQL structure and data
    """

    def __init__(self):
        """Initialize a new query builder."""
        self._query_type: QueryType | None = None
        self._select_columns: list[str] = []
        self._from_table: str | None = None
        self._where_clauses: list[str] = []
        self._where_parameters: list[Any] = []
        self._join_clauses: list[str] = []
        self._join_parameters: list[Any] = []
        self._order_by_clauses: list[str] = []
        self._group_by_columns: list[str] = []
        self._having_clauses: list[str] = []
        self._having_parameters: list[Any] = []
        self._limit_count: int | None = None
        self._offset_count: int | None = None

        # INSERT/UPDATE specific
        self._insert_table: str | None = None
        self._insert_columns: list[str] = []
        self._insert_values_placeholder: str | None = None
        self._insert_parameters: list[Any] = []
        self._update_table: str | None = None
        self._set_clauses: list[str] = []
        self._set_parameters: list[Any] = []

        # DELETE specific
        self._delete_table: str | None = None

    def _validate_identifier(self, identifier: str, context: str = "identifier") -> str:
        """
        Validate SQL identifier using InputSanitizer.

        Args:
            identifier: SQL identifier to validate
            context: Context for error messages

        Returns:
            Validated identifier

        Raises:
            SecurityError: If identifier is invalid
        """
        try:
            return InputSanitizer.sanitize_sql_identifier(identifier)
        except SanitizationError as e:
            raise SecurityError(f"Invalid SQL {context}: {e}")

    def _validate_identifiers(
        self, identifiers: list[str], context: str = "identifiers"
    ) -> list[str]:
        """Validate a list of SQL identifiers."""
        return [self._validate_identifier(id, context) for id in identifiers]

    def select(self, columns: list[str] | str) -> "QueryBuilder":
        """
        Add SELECT columns to the query.

        Args:
            columns: Column names to select (list or single string)

        Returns:
            Self for method chaining

        Raises:
            SecurityError: If column names are invalid
            QueryBuilderError: If query structure is invalid
        """
        if self._query_type is not None and self._query_type != QueryType.SELECT:
            raise QueryBuilderError("Cannot mix SELECT with other query types")

        self._query_type = QueryType.SELECT

        if isinstance(columns, str):
            if columns.strip() == "*":
                self._select_columns.append("*")
            else:
                # Validate single column
                validated = self._validate_identifier(columns, "column")
                self._select_columns.append(validated)
        else:
            # Validate all columns
            validated = self._validate_identifiers(columns, "column")
            self._select_columns.extend(validated)

        return self

    def from_table(self, table: str, alias: str | None = None) -> "QueryBuilder":
        """
        Set the FROM table for SELECT queries.

        Args:
            table: Table name
            alias: Optional table alias

        Returns:
            Self for method chaining

        Raises:
            SecurityError: If table name is invalid
        """
        validated_table = self._validate_identifier(table, "table")

        if alias:
            validated_alias = self._validate_identifier(alias, "alias")
            self._from_table = f"{validated_table} AS {validated_alias}"
        else:
            self._from_table = validated_table

        return self

    def where(self, condition: str, parameters: list[Any]) -> "QueryBuilder":
        """
        Add WHERE condition with parameters.

        Args:
            condition: WHERE condition with parameter placeholders (%s)
            parameters: List of parameter values

        Returns:
            Self for method chaining

        Example:
            .where("status = %s AND created > %s", ["active", datetime(2023, 1, 1)])
        """
        if not condition.strip():
            raise QueryBuilderError("WHERE condition cannot be empty")

        # Count placeholders to validate parameter count
        placeholder_count = condition.count("%s")
        if placeholder_count != len(parameters):
            raise QueryBuilderError(
                f"Parameter count mismatch: {placeholder_count} placeholders, {len(parameters)} parameters"
            )

        self._where_clauses.append(condition)
        self._where_parameters.extend(parameters)

        return self

    def join(
        self, table: str, on_condition: str, parameters: list[Any] = None, join_type: str = "INNER"
    ) -> "QueryBuilder":
        """
        Add JOIN clause with optional parameters.

        Args:
            table: Table to join
            on_condition: JOIN condition
            parameters: Optional parameters for the ON condition
            join_type: Type of join (INNER, LEFT, RIGHT, FULL)

        Returns:
            Self for method chaining
        """
        validated_table = self._validate_identifier(table, "join table")

        # Validate join type
        allowed_joins = {"INNER", "LEFT", "RIGHT", "FULL", "CROSS"}
        if join_type.upper() not in allowed_joins:
            raise QueryBuilderError(f"Invalid join type: {join_type}")

        if parameters is None:
            parameters = []

        # Validate parameter count if condition has placeholders
        placeholder_count = on_condition.count("%s")
        if placeholder_count != len(parameters):
            raise QueryBuilderError(
                f"JOIN parameter count mismatch: {placeholder_count} placeholders, {len(parameters)} parameters"
            )

        join_clause = f"{join_type.upper()} JOIN {validated_table} ON {on_condition}"
        self._join_clauses.append(join_clause)
        self._join_parameters.extend(parameters)

        return self

    def order_by(self, column: str, direction: str = "ASC") -> "QueryBuilder":
        """
        Add ORDER BY clause.

        Args:
            column: Column name to order by
            direction: Sort direction (ASC or DESC)

        Returns:
            Self for method chaining
        """
        validated_column = self._validate_identifier(column, "order by column")

        if direction.upper() not in ["ASC", "DESC"]:
            raise QueryBuilderError(f"Invalid sort direction: {direction}")

        self._order_by_clauses.append(f"{validated_column} {direction.upper()}")

        return self

    def group_by(self, columns: list[str] | str) -> "QueryBuilder":
        """
        Add GROUP BY clause.

        Args:
            columns: Column names to group by

        Returns:
            Self for method chaining
        """
        if isinstance(columns, str):
            validated = [self._validate_identifier(columns, "group by column")]
        else:
            validated = self._validate_identifiers(columns, "group by column")

        self._group_by_columns.extend(validated)

        return self

    def having(self, condition: str, parameters: list[Any]) -> "QueryBuilder":
        """
        Add HAVING condition with parameters.

        Args:
            condition: HAVING condition with parameter placeholders
            parameters: List of parameter values

        Returns:
            Self for method chaining
        """
        placeholder_count = condition.count("%s")
        if placeholder_count != len(parameters):
            raise QueryBuilderError(
                f"HAVING parameter count mismatch: {placeholder_count} placeholders, {len(parameters)} parameters"
            )

        self._having_clauses.append(condition)
        self._having_parameters.extend(parameters)

        return self

    def limit(self, count: int) -> "QueryBuilder":
        """
        Add LIMIT clause.

        Args:
            count: Maximum number of rows to return

        Returns:
            Self for method chaining
        """
        if not isinstance(count, int) or count < 0:
            raise QueryBuilderError("LIMIT count must be a non-negative integer")

        self._limit_count = count

        return self

    def offset(self, count: int) -> "QueryBuilder":
        """
        Add OFFSET clause.

        Args:
            count: Number of rows to skip

        Returns:
            Self for method chaining
        """
        if not isinstance(count, int) or count < 0:
            raise QueryBuilderError("OFFSET count must be a non-negative integer")

        self._offset_count = count

        return self

    def insert_into(self, table: str, columns: list[str]) -> "QueryBuilder":
        """
        Start an INSERT query.

        Args:
            table: Table to insert into
            columns: Column names for the insert

        Returns:
            Self for method chaining
        """
        if self._query_type is not None:
            raise QueryBuilderError("Cannot mix INSERT with other query types")

        self._query_type = QueryType.INSERT
        self._insert_table = self._validate_identifier(table, "insert table")
        self._insert_columns = self._validate_identifiers(columns, "insert column")

        return self

    def values(self, placeholder: str, parameters: list[Any]) -> "QueryBuilder":
        """
        Add VALUES clause for INSERT.

        Args:
            placeholder: Value placeholders (e.g., "%s, %s, %s")
            parameters: Parameter values

        Returns:
            Self for method chaining
        """
        if self._query_type != QueryType.INSERT:
            raise QueryBuilderError("VALUES can only be used with INSERT")

        placeholder_count = placeholder.count("%s")
        if placeholder_count != len(parameters):
            raise QueryBuilderError(
                f"VALUES parameter count mismatch: {placeholder_count} placeholders, {len(parameters)} parameters"
            )

        if placeholder_count != len(self._insert_columns):
            raise QueryBuilderError(
                f"VALUES parameter count ({placeholder_count}) must match column count ({len(self._insert_columns)})"
            )

        self._insert_values_placeholder = placeholder
        self._insert_parameters = parameters

        return self

    def update(self, table: str) -> "QueryBuilder":
        """
        Start an UPDATE query.

        Args:
            table: Table to update

        Returns:
            Self for method chaining
        """
        if self._query_type is not None:
            raise QueryBuilderError("Cannot mix UPDATE with other query types")

        self._query_type = QueryType.UPDATE
        self._update_table = self._validate_identifier(table, "update table")

        return self

    def set(self, assignments: list[str], parameters: list[Any]) -> "QueryBuilder":
        """
        Add SET clause for UPDATE.

        Args:
            assignments: Column assignments (e.g., ["name = %s", "email = %s"])
            parameters: Parameter values

        Returns:
            Self for method chaining
        """
        if self._query_type != QueryType.UPDATE:
            raise QueryBuilderError("SET can only be used with UPDATE")

        total_placeholders = sum(assignment.count("%s") for assignment in assignments)
        if total_placeholders != len(parameters):
            raise QueryBuilderError(
                f"SET parameter count mismatch: {total_placeholders} placeholders, {len(parameters)} parameters"
            )

        self._set_clauses.extend(assignments)
        self._set_parameters.extend(parameters)

        return self

    def delete_from(self, table: str) -> "QueryBuilder":
        """
        Start a DELETE query.

        Args:
            table: Table to delete from

        Returns:
            Self for method chaining
        """
        if self._query_type is not None:
            raise QueryBuilderError("Cannot mix DELETE with other query types")

        self._query_type = QueryType.DELETE
        self._delete_table = self._validate_identifier(table, "delete table")

        return self

    def build(self) -> QueryResult:
        """
        Build the final SQL query with parameters.

        Returns:
            QueryResult containing SQL and parameters

        Raises:
            QueryBuilderError: If query structure is invalid
        """
        if self._query_type is None:
            raise QueryBuilderError("No query type specified")

        if self._query_type == QueryType.SELECT:
            return self._build_select()
        elif self._query_type == QueryType.INSERT:
            return self._build_insert()
        elif self._query_type == QueryType.UPDATE:
            return self._build_update()
        elif self._query_type == QueryType.DELETE:
            return self._build_delete()
        else:
            raise QueryBuilderError(f"Unsupported query type: {self._query_type}")

    def _build_select(self) -> QueryResult:
        """Build SELECT query."""
        if not self._select_columns:
            raise QueryBuilderError("SELECT query must have columns")
        if not self._from_table:
            raise QueryBuilderError("SELECT query must have FROM table")

        # Build SQL parts
        sql_parts = ["SELECT " + ", ".join(self._select_columns)]
        sql_parts.append(f"FROM {self._from_table}")

        # Collect all parameters in order
        all_parameters = []

        # Add JOINs
        if self._join_clauses:
            sql_parts.extend(self._join_clauses)
            all_parameters.extend(self._join_parameters)

        # Add WHERE
        if self._where_clauses:
            sql_parts.append(
                "WHERE " + " AND ".join(f"({clause})" for clause in self._where_clauses)
            )
            all_parameters.extend(self._where_parameters)

        # Add GROUP BY
        if self._group_by_columns:
            sql_parts.append("GROUP BY " + ", ".join(self._group_by_columns))

        # Add HAVING
        if self._having_clauses:
            sql_parts.append(
                "HAVING " + " AND ".join(f"({clause})" for clause in self._having_clauses)
            )
            all_parameters.extend(self._having_parameters)

        # Add ORDER BY
        if self._order_by_clauses:
            sql_parts.append("ORDER BY " + ", ".join(self._order_by_clauses))

        # Add LIMIT
        if self._limit_count is not None:
            sql_parts.append(f"LIMIT {self._limit_count}")

        # Add OFFSET
        if self._offset_count is not None:
            sql_parts.append(f"OFFSET {self._offset_count}")

        sql = " ".join(sql_parts)
        return QueryResult(sql, all_parameters)

    def _build_insert(self) -> QueryResult:
        """Build INSERT query."""
        if not self._insert_table:
            raise QueryBuilderError("INSERT query must have table")
        if not self._insert_columns:
            raise QueryBuilderError("INSERT query must have columns")
        if self._insert_values_placeholder is None:
            raise QueryBuilderError("INSERT query must have VALUES")

        columns_str = ", ".join(self._insert_columns)
        sql = f"INSERT INTO {self._insert_table} ({columns_str}) VALUES ({self._insert_values_placeholder})"

        return QueryResult(sql, self._insert_parameters)

    def _build_update(self) -> QueryResult:
        """Build UPDATE query."""
        if not self._update_table:
            raise QueryBuilderError("UPDATE query must have table")
        if not self._set_clauses:
            raise QueryBuilderError("UPDATE query must have SET clauses")

        sql_parts = [f"UPDATE {self._update_table}"]
        sql_parts.append("SET " + ", ".join(self._set_clauses))

        all_parameters = list(self._set_parameters)

        # Add WHERE
        if self._where_clauses:
            sql_parts.append(
                "WHERE " + " AND ".join(f"({clause})" for clause in self._where_clauses)
            )
            all_parameters.extend(self._where_parameters)

        sql = " ".join(sql_parts)
        return QueryResult(sql, all_parameters)

    def _build_delete(self) -> QueryResult:
        """Build DELETE query."""
        if not self._delete_table:
            raise QueryBuilderError("DELETE query must have table")

        sql_parts = [f"DELETE FROM {self._delete_table}"]
        all_parameters = []

        # Add WHERE (strongly recommended for DELETE!)
        if self._where_clauses:
            sql_parts.append(
                "WHERE " + " AND ".join(f"({clause})" for clause in self._where_clauses)
            )
            all_parameters.extend(self._where_parameters)
        else:
            logger.warning("DELETE query without WHERE clause - this will delete ALL rows!")

        sql = " ".join(sql_parts)
        return QueryResult(sql, all_parameters)

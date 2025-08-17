"""
SQL Security Module

Provides validation and sanitization for SQL identifiers to prevent injection attacks.
This module ensures that table names, column names, and other SQL identifiers
are safe to use in dynamic query construction.
"""

# Standard library imports
import re

# Local imports
from main.utils.core import get_logger

logger = get_logger(__name__)


class SQLSecurityError(Exception):
    """Raised when SQL identifier validation fails."""

    pass


# PostgreSQL reserved keywords that should not be used as identifiers
SQL_KEYWORDS: set[str] = {
    "ALL",
    "ALTER",
    "AND",
    "ANY",
    "AS",
    "ASC",
    "BETWEEN",
    "BY",
    "CASE",
    "CHECK",
    "COLUMN",
    "CONSTRAINT",
    "CREATE",
    "CROSS",
    "CURRENT",
    "DELETE",
    "DESC",
    "DISTINCT",
    "DROP",
    "ELSE",
    "END",
    "EXCEPT",
    "EXEC",
    "EXECUTE",
    "EXISTS",
    "FALSE",
    "FETCH",
    "FOR",
    "FOREIGN",
    "FROM",
    "FULL",
    "GRANT",
    "GROUP",
    "HAVING",
    "IN",
    "INDEX",
    "INNER",
    "INSERT",
    "INTERSECT",
    "INTO",
    "IS",
    "JOIN",
    "KEY",
    "LEFT",
    "LIKE",
    "LIMIT",
    "NOT",
    "NULL",
    "OFFSET",
    "ON",
    "OR",
    "ORDER",
    "OUTER",
    "PRIMARY",
    "REFERENCES",
    "REVOKE",
    "RIGHT",
    "SELECT",
    "SET",
    "TABLE",
    "THEN",
    "TO",
    "TRUE",
    "TRUNCATE",
    "UNION",
    "UNIQUE",
    "UPDATE",
    "VALUES",
    "VIEW",
    "WHEN",
    "WHERE",
    "WITH",
}

# Pattern for valid SQL identifiers
IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Maximum identifier length (PostgreSQL limit)
MAX_IDENTIFIER_LENGTH = 63


def validate_table_name(name: str) -> str:
    """
    Validates a table name to prevent SQL injection.

    Rules:
    - Must not be empty
    - Must start with letter or underscore
    - Can contain only letters, numbers, underscores
    - Maximum 63 characters (PostgreSQL limit)
    - Cannot be a SQL keyword

    Args:
        name: The table name to validate

    Returns:
        The validated table name

    Raises:
        SQLSecurityError: If the table name is invalid
    """
    if not name:
        raise SQLSecurityError("Table name cannot be empty")

    # Check length
    if len(name) > MAX_IDENTIFIER_LENGTH:
        raise SQLSecurityError(
            f"Table name too long: {len(name)} characters (max {MAX_IDENTIFIER_LENGTH})"
        )

    # Check pattern
    if not IDENTIFIER_PATTERN.match(name):
        raise SQLSecurityError(
            f"Invalid table name '{name}': must start with letter/underscore "
            f"and contain only letters, numbers, and underscores"
        )

    # Check not a keyword (case-insensitive)
    if name.upper() in SQL_KEYWORDS:
        raise SQLSecurityError(f"Table name '{name}' is a reserved SQL keyword")

    logger.debug(f"Validated table name: {name}")
    return name


def validate_column_name(name: str) -> str:
    """
    Validates a column name to prevent SQL injection.

    Uses the same rules as table name validation.

    Args:
        name: The column name to validate

    Returns:
        The validated column name

    Raises:
        SQLSecurityError: If the column name is invalid
    """
    if not name:
        raise SQLSecurityError("Column name cannot be empty")

    # Check length
    if len(name) > MAX_IDENTIFIER_LENGTH:
        raise SQLSecurityError(
            f"Column name too long: {len(name)} characters (max {MAX_IDENTIFIER_LENGTH})"
        )

    # Check pattern
    if not IDENTIFIER_PATTERN.match(name):
        raise SQLSecurityError(
            f"Invalid column name '{name}': must start with letter/underscore "
            f"and contain only letters, numbers, and underscores"
        )

    # Check not a keyword (case-insensitive)
    if name.upper() in SQL_KEYWORDS:
        raise SQLSecurityError(f"Column name '{name}' is a reserved SQL keyword")

    return name


def validate_identifier_list(names: list[str]) -> list[str]:
    """
    Validates a list of SQL identifiers (column names).

    Args:
        names: List of identifier names to validate

    Returns:
        List of validated identifier names

    Raises:
        SQLSecurityError: If any identifier is invalid
    """
    if not names:
        return []

    validated = []
    for name in names:
        try:
            validated.append(validate_column_name(name))
        except SQLSecurityError as e:
            raise SQLSecurityError(f"Invalid identifier in list: {e}")

    return validated


class SafeQueryBuilder:
    """
    Builds SQL queries with validated identifiers to prevent injection.

    This class provides safe methods for constructing common SQL queries
    with proper validation of all identifiers.
    """

    def __init__(self, table_name: str):
        """
        Initialize the safe query builder.

        Args:
            table_name: The table name (will be validated)

        Raises:
            SQLSecurityError: If the table name is invalid
        """
        self.table_name = validate_table_name(table_name)

    def build_select(
        self,
        columns: list[str] | None = None,
        where_clause: str = "",
        order_by: str | None = None,
        limit: int | None = None,
    ) -> str:
        """
        Build a safe SELECT query.

        Args:
            columns: List of column names to select (None = *)
            where_clause: WHERE clause with parameterized values
            order_by: Column to order by (will be validated)
            limit: Row limit

        Returns:
            Safe SQL query string
        """
        # Validate columns
        if columns:
            safe_columns = ", ".join(validate_identifier_list(columns))
        else:
            safe_columns = "*"

        query = f"SELECT {safe_columns} FROM {self.table_name}"

        if where_clause:
            query += f" {where_clause}"  # Assume WHERE is included

        if order_by:
            safe_order = validate_column_name(order_by)
            query += f" ORDER BY {safe_order}"

        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise SQLSecurityError(f"Invalid limit: {limit}")
            query += f" LIMIT {limit}"

        return query

    def build_insert(self, columns: list[str], num_values: int) -> str:
        """
        Build a safe INSERT query with parameterized values.

        Args:
            columns: List of column names
            num_values: Number of value sets to insert

        Returns:
            Safe SQL query string with $1, $2... placeholders
        """
        safe_columns = validate_identifier_list(columns)
        columns_str = ", ".join(safe_columns)

        # Create placeholders
        placeholders = [f"${i+1}" for i in range(len(columns))]
        placeholders_str = f"({', '.join(placeholders)})"

        query = f"INSERT INTO {self.table_name} ({columns_str}) VALUES {placeholders_str}"

        return query

    def build_update(self, set_columns: list[str], where_clause: str = "") -> tuple[str, int]:
        """
        Build a safe UPDATE query.

        Args:
            set_columns: List of columns to update
            where_clause: WHERE clause with parameterized values

        Returns:
            Tuple of (query string, next parameter number)
        """
        safe_columns = validate_identifier_list(set_columns)

        # Build SET clause with numbered parameters
        set_clauses = []
        param_num = 1
        for col in safe_columns:
            set_clauses.append(f"{col} = ${param_num}")
            param_num += 1

        query = f"UPDATE {self.table_name} SET {', '.join(set_clauses)}"

        if where_clause:
            query += f" {where_clause}"  # Assume WHERE is included

        return query, param_num

    def build_delete(self, where_clause: str = "") -> str:
        """
        Build a safe DELETE query.

        Args:
            where_clause: WHERE clause with parameterized values

        Returns:
            Safe SQL query string
        """
        query = f"DELETE FROM {self.table_name}"

        if where_clause:
            query += f" {where_clause}"  # Assume WHERE is included

        return query

    def build_count(self, where_clause: str = "") -> str:
        """
        Build a safe COUNT query.

        Args:
            where_clause: WHERE clause with parameterized values

        Returns:
            Safe SQL query string
        """
        query = f"SELECT COUNT(*) as count FROM {self.table_name}"

        if where_clause:
            query += f" {where_clause}"  # Assume WHERE is included

        return query


# Convenience functions for one-off validations
def safe_column_list(columns: list[str]) -> str:
    """
    Create a safe comma-separated list of column names.

    Args:
        columns: List of column names

    Returns:
        Comma-separated string of validated column names
    """
    return ", ".join(validate_identifier_list(columns))


def safe_table_column(table: str, column: str) -> str:
    """
    Create a safe table.column reference.

    Args:
        table: Table name
        column: Column name

    Returns:
        Safe table.column string
    """
    safe_table = validate_table_name(table)
    safe_column = validate_column_name(column)
    return f"{safe_table}.{safe_column}"

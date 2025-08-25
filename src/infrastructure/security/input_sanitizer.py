"""
Input Sanitization - Infrastructure layer for cleaning and sanitizing input.

This module provides input sanitization without business logic validation.
It focuses on security concerns like SQL injection and XSS prevention.
Business validation is handled by domain services.
"""

import html
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class SanitizationError(Exception):
    """Raised when input cannot be safely sanitized."""

    pass


class InputSanitizer:
    """
    Input sanitization for security purposes.

    This class only handles security-related sanitization.
    Business validation is handled by domain services.
    """

    # SECURITY WARNING: SQL injection prevention should ONLY use parameterized queries!
    # Manual SQL escaping creates false security and should never be used.

    # XSS patterns to block
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
    ]

    @classmethod
    def sanitize_string(cls, value: Any, max_length: int | None = None) -> str:
        """
        Basic string sanitization for security - no business logic.

        Args:
            value: Value to sanitize (will be converted to string)
            max_length: Optional maximum length

        Returns:
            Sanitized string

        Raises:
            SanitizationError: If string contains dangerous patterns
        """
        # Convert to string and ensure proper typing
        str_value = str(value) if not isinstance(value, str) else value

        # Basic cleanup
        str_value = str_value.strip()

        # XSS security check only - SQL injection prevention uses parameterized queries
        if cls._contains_xss_pattern(str_value):
            logger.warning(f"XSS pattern detected: {str_value[:50]}...")
            raise SanitizationError("Input contains potentially dangerous XSS patterns")

        # HTML escape for safety
        str_value = html.escape(str_value)

        # Simple length truncation
        if max_length and len(str_value) > max_length:
            str_value = str_value[:max_length]

        return str_value

    @classmethod
    def _contains_xss_pattern(cls, value: str) -> bool:
        """Check for XSS patterns only - SQL injection prevention uses parameterized queries."""
        # Only check XSS patterns - SQL injection is prevented by parameterized queries
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False

    @classmethod
    def sanitize_sql_identifier(cls, identifier: str) -> str:
        """
        Validate a SQL identifier (table/column name) using allowlist approach.

        WARNING: This only validates identifiers, not values. Use parameterized queries for values!

        Args:
            identifier: SQL identifier to validate

        Returns:
            Validated identifier

        Raises:
            SanitizationError: If identifier is unsafe
        """
        # Strict allowlist: only alphanumeric and underscore, must start with letter
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", identifier):
            raise SanitizationError(f"Invalid SQL identifier format: {identifier}")

        # Length limit for security
        if len(identifier) > 64:  # Common database limit
            raise SanitizationError(f"SQL identifier too long: {identifier}")

        # Comprehensive reserved words list
        reserved_words = {
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TABLE",
            "DATABASE",
            "INDEX",
            "VIEW",
            "PROCEDURE",
            "FUNCTION",
            "TRIGGER",
            "SCHEMA",
            "CONSTRAINT",
            "PRIMARY",
            "FOREIGN",
            "KEY",
            "UNIQUE",
            "CHECK",
            "DEFAULT",
            "NULL",
            "NOT",
            "AND",
            "OR",
            "WHERE",
            "ORDER",
            "GROUP",
            "HAVING",
            "UNION",
            "JOIN",
            "INNER",
            "OUTER",
            "LEFT",
            "RIGHT",
            "FULL",
            "CROSS",
            "ON",
            "AS",
            "FROM",
            "INTO",
            "VALUES",
            "SET",
            "EXEC",
            "EXECUTE",
            "GRANT",
            "REVOKE",
            "COMMIT",
            "ROLLBACK",
            "TRANSACTION",
            "BEGIN",
            "END",
            "IF",
            "ELSE",
            "CASE",
            "WHEN",
            "THEN",
            "WHILE",
            "FOR",
            "DECLARE",
            "CURSOR",
            "OPEN",
            "CLOSE",
            "FETCH",
            "DEALLOCATE",
        }

        if identifier.upper() in reserved_words:
            raise SanitizationError(f"Reserved SQL word not allowed: {identifier}")

        return identifier

    # DANGEROUS METHOD REMOVED: sanitize_sql_value()
    #
    # This method created false security by trying to manually escape SQL values.
    # Manual SQL escaping is inherently unsafe and can be bypassed.
    #
    # SECURITY REQUIREMENT: Always use parameterized queries for SQL values!
    #
    # Example of CORRECT approach:
    #   cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    #
    # Example of INCORRECT approach (vulnerable to SQL injection):
    #   cursor.execute(f"SELECT * FROM users WHERE id = {sanitize_sql_value(user_id)}")
    #
    # If you need this method, your code architecture needs to be fixed to use
    # parameterized queries instead.

    @classmethod
    def _removed_sanitize_sql_value(cls, value: Any) -> None:
        """
        This method has been removed for security reasons.

        Use parameterized queries instead!
        """
        raise NotImplementedError(
            "sanitize_sql_value() has been removed for security reasons. "
            "Use parameterized queries instead of manual SQL escaping. "
            "Example: cursor.execute('SELECT * FROM table WHERE id = %s', (value,))"
        )

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize a filename for safe filesystem operations.

        Args:
            filename: Filename to sanitize

        Returns:
            Sanitized filename
        """
        # Remove directory traversal attempts
        # Only remove .. when it's part of a path (with slashes) or at start/end
        filename = filename.replace("../", "")
        filename = filename.replace("..\\", "")
        filename = filename.replace("/", "_")
        filename = filename.replace("\\", "_")

        # Remove .. only if it's at the start or end of filename, not in the middle
        if filename == ".." or filename.startswith("../") or filename.startswith("..\\"):
            filename = filename.replace("..", "")
        # For cases like ../../ that might remain after slash replacement
        while filename.startswith(".."):
            if len(filename) == 2 or filename[2] in "/_\\":
                filename = filename[2:]
            else:
                break

        # Only allow safe characters
        filename = re.sub(r"[^a-zA-Z0-9._\-]", "_", filename)

        # Limit length
        if len(filename) > 255:
            if "." in filename:
                # Find the last extension
                name, ext = filename.rsplit(".", 1)
                # Calculate how much space we need for the extension and dot
                ext_space = len(ext) + 1  # +1 for the dot
                # Truncate the name to fit within 255 chars total
                max_name_length = 255 - ext_space
                if max_name_length > 0:
                    filename = name[:max_name_length] + "." + ext
                else:
                    # Extension is too long, just truncate everything
                    filename = filename[:255]
            else:
                # No extension, just truncate
                filename = filename[:255]

        return filename

    @classmethod
    def sanitize_symbol(cls, symbol: str) -> str:
        """
        Sanitize a trading symbol for safe use.

        Args:
            symbol: Trading symbol to sanitize

        Returns:
            Sanitized symbol

        Raises:
            SanitizationError: If symbol is invalid
        """
        # Trading symbols should only contain letters, numbers, dots, and hyphens
        if not re.match(r"^[A-Z0-9.\-]+$", symbol.upper()):
            raise SanitizationError(f"Invalid trading symbol: {symbol}")

        # Limit length (most symbols are under 10 chars)
        if len(symbol) > 20:
            raise SanitizationError(f"Symbol too long: {symbol}")

        # Check for XSS patterns only (SQL injection prevented by parameterized queries)
        if cls._contains_xss_pattern(symbol):
            raise SanitizationError(f"Symbol contains XSS patterns: {symbol}")

        return symbol.upper()

    @classmethod
    def sanitize_identifier(cls, identifier: str) -> str:
        """
        Sanitize a general identifier (like timeframe).

        Args:
            identifier: Identifier to sanitize

        Returns:
            Sanitized identifier

        Raises:
            SanitizationError: If identifier is invalid
        """
        # Allow alphanumeric, underscore, and dash
        if not re.match(r"^[a-zA-Z0-9_\-]+$", identifier):
            raise SanitizationError(f"Invalid identifier: {identifier}")

        # Limit length
        if len(identifier) > 50:
            raise SanitizationError(f"Identifier too long: {identifier}")

        # Check for XSS patterns only (SQL injection prevented by parameterized queries)
        if cls._contains_xss_pattern(identifier):
            raise SanitizationError(f"Identifier contains XSS patterns: {identifier}")

        return identifier

    @classmethod
    def sanitize_url(cls, url: str) -> str:
        """
        Sanitize a URL for safe use.

        Args:
            url: URL to sanitize

        Returns:
            Sanitized URL

        Raises:
            SanitizationError: If URL contains dangerous patterns
        """
        # Check for javascript: and data: URLs
        if url.lower().startswith(("javascript:", "data:", "vbscript:")):
            raise SanitizationError("Potentially dangerous URL scheme")

        # Basic URL encoding of special characters
        url = url.replace(" ", "%20")
        url = url.replace("<", "%3C")
        url = url.replace(">", "%3E")
        url = url.replace('"', "%22")

        return url

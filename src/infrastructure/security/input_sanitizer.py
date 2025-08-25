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

    # SQL injection patterns to block
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE)\b)",
        r"(--|\||;|\/\*|\*\/|xp_|sp_|0x)",
        r"(\bOR\b\s*\d+\s*=\s*\d+)",
        r"(\bAND\b\s*\d+\s*=\s*\d+)",
    ]

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

        # Simple security check - check against known patterns
        if cls._contains_dangerous_pattern(str_value):
            logger.warning(f"Dangerous pattern detected: {str_value[:50]}...")
            raise SanitizationError("Input contains potentially dangerous patterns")

        # HTML escape for safety
        str_value = html.escape(str_value)

        # Simple length truncation
        if max_length and len(str_value) > max_length:
            str_value = str_value[:max_length]

        return str_value

    @classmethod
    def _contains_dangerous_pattern(cls, value: str) -> bool:
        """Simple check for dangerous patterns - no business logic."""
        # Check SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        # Check XSS patterns
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False

    @classmethod
    def sanitize_sql_identifier(cls, identifier: str) -> str:
        """
        Sanitize a SQL identifier (table/column name).

        Args:
            identifier: SQL identifier to sanitize

        Returns:
            Sanitized identifier

        Raises:
            SanitizationError: If identifier is unsafe
        """
        # Only allow alphanumeric and underscore
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", identifier):
            raise SanitizationError(f"Invalid SQL identifier: {identifier}")

        # Check against reserved words (simplified list)
        reserved_words = {"SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"}
        if identifier.upper() in reserved_words:
            raise SanitizationError(f"Reserved SQL word: {identifier}")

        return identifier

    @classmethod
    def sanitize_sql_value(cls, value: Any) -> str:
        """
        Sanitize a value for SQL queries.

        Note: This is a fallback. Use parameterized queries whenever possible.

        Args:
            value: Value to sanitize

        Returns:
            Sanitized value as string
        """
        if value is None:
            return "NULL"

        # Convert to string and escape quotes
        str_value = str(value)
        str_value = str_value.replace("'", "''")
        str_value = str_value.replace("\\", "\\\\")

        return f"'{str_value}'"

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

        # Check for SQL injection patterns
        if cls._contains_dangerous_pattern(symbol):
            raise SanitizationError(f"Symbol contains dangerous patterns: {symbol}")

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

        # Check for SQL injection patterns
        if cls._contains_dangerous_pattern(identifier):
            raise SanitizationError(f"Identifier contains dangerous patterns: {identifier}")

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

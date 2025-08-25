"""
Secure Input Validation - Type-safe validation without dangerous SQL sanitization.

This module provides secure input validation focused on format validation,
length limits, and type safety. It does NOT attempt SQL sanitization, which
should always be handled by parameterized queries.

Security Principles:
1. Validate input format and type - don't try to "clean" it
2. Use allowlists instead of blocklists when possible
3. Fail securely - reject invalid input, don't try to fix it
4. Log security events for monitoring
5. Use parameterized queries for all SQL - never manual escaping
"""

import ipaddress
import json
import logging
import re
import urllib.parse
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None, value: Any = None) -> None:
        self.field = field
        self.value = value
        super().__init__(message)


class InputValidator:
    """
    Secure input validation with type safety.

    This class validates input format and type without attempting dangerous
    SQL sanitization. All SQL operations should use parameterized queries.
    """

    # Common validation patterns
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    PHONE_PATTERN = re.compile(r"^\+?[1-9]\d{1,14}$")  # E.164 format
    UUID_PATTERN = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    )

    # Trading-specific patterns
    SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,20}$")
    CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")

    # Safe identifier patterns (for non-SQL use)
    SAFE_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")

    @classmethod
    def validate_string(
        cls,
        value: Any,
        field_name: str = "field",
        min_length: int = 0,
        max_length: int = 1000,
        pattern: re.Pattern[str] | None = None,
        allow_empty: bool = True,
    ) -> str:
        """
        Validate string input with length and pattern checks.

        Args:
            value: Value to validate
            field_name: Name of field for error messages
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            pattern: Optional regex pattern to match
            allow_empty: Whether empty strings are allowed

        Returns:
            Validated string

        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            if allow_empty:
                return ""
            raise ValidationError(f"{field_name} cannot be None", field_name, value)

        # Convert to string
        str_value = str(value).strip()

        # Check empty string
        if not str_value and not allow_empty:
            raise ValidationError(f"{field_name} cannot be empty", field_name, value)

        # Check length
        if len(str_value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters", field_name, value
            )

        if len(str_value) > max_length:
            raise ValidationError(
                f"{field_name} must be no more than {max_length} characters", field_name, value
            )

        # Check pattern if provided
        if pattern and not pattern.match(str_value):
            raise ValidationError(f"{field_name} does not match required format", field_name, value)

        return str_value

    @classmethod
    def validate_integer(
        cls,
        value: Any,
        field_name: str = "field",
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int:
        """
        Validate integer input with range checks.

        Args:
            value: Value to validate
            field_name: Name of field for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated integer

        Raises:
            ValidationError: If validation fails
        """
        try:
            if isinstance(value, str):
                int_value = int(value.strip())
            else:
                int_value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid integer", field_name, value)

        if min_value is not None and int_value < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}", field_name, value)

        if max_value is not None and int_value > max_value:
            raise ValidationError(
                f"{field_name} must be no more than {max_value}", field_name, value
            )

        return int_value

    @classmethod
    def validate_decimal(
        cls,
        value: Any,
        field_name: str = "field",
        min_value: Decimal | None = None,
        max_value: Decimal | None = None,
        max_decimal_places: int | None = None,
    ) -> Decimal:
        """
        Validate decimal input with range and precision checks.

        Args:
            value: Value to validate
            field_name: Name of field for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            max_decimal_places: Maximum decimal places allowed

        Returns:
            Validated Decimal

        Raises:
            ValidationError: If validation fails
        """
        try:
            if isinstance(value, str):
                decimal_value = Decimal(value.strip())
            else:
                decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid decimal number", field_name, value)

        if min_value is not None and decimal_value < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}", field_name, value)

        if max_value is not None and decimal_value > max_value:
            raise ValidationError(
                f"{field_name} must be no more than {max_value}", field_name, value
            )

        # Check decimal places
        if max_decimal_places is not None:
            _, digits, exponent = decimal_value.as_tuple()
            # Handle special values (exponent can be 'n', 'N', 'F' for special cases)
            if isinstance(exponent, int) and exponent < -max_decimal_places:
                raise ValidationError(
                    f"{field_name} can have at most {max_decimal_places} decimal places",
                    field_name,
                    value,
                )

        return decimal_value

    @classmethod
    def validate_email(cls, value: Any, field_name: str = "email") -> str:
        """
        Validate email address format.

        Args:
            value: Email to validate
            field_name: Name of field for error messages

        Returns:
            Validated email address

        Raises:
            ValidationError: If email format is invalid
        """
        email = cls.validate_string(
            value, field_name, min_length=1, max_length=254, allow_empty=False
        )

        if not cls.EMAIL_PATTERN.match(email):
            raise ValidationError(f"Invalid {field_name} format", field_name, value)

        return email.lower()  # Normalize to lowercase

    @classmethod
    def validate_url(
        cls, value: Any, field_name: str = "url", allowed_schemes: set[str] | None = None
    ) -> str:
        """
        Validate URL format and scheme.

        Args:
            value: URL to validate
            field_name: Name of field for error messages
            allowed_schemes: Set of allowed schemes (default: http, https)

        Returns:
            Validated URL

        Raises:
            ValidationError: If URL format is invalid
        """
        if allowed_schemes is None:
            allowed_schemes = {"http", "https"}

        url = cls.validate_string(
            value, field_name, min_length=1, max_length=2048, allow_empty=False
        )

        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            raise ValidationError(f"Invalid {field_name} format", field_name, value)

        if parsed.scheme.lower() not in allowed_schemes:
            raise ValidationError(
                f"{field_name} must use one of these schemes: {', '.join(allowed_schemes)}",
                field_name,
                value,
            )

        if not parsed.netloc:
            raise ValidationError(f"{field_name} must have a valid domain", field_name, value)

        return url

    @classmethod
    def validate_ip_address(
        cls,
        value: Any,
        field_name: str = "ip_address",
        allow_ipv4: bool = True,
        allow_ipv6: bool = True,
    ) -> str:
        """
        Validate IP address format.

        Args:
            value: IP address to validate
            field_name: Name of field for error messages
            allow_ipv4: Whether IPv4 addresses are allowed
            allow_ipv6: Whether IPv6 addresses are allowed

        Returns:
            Validated IP address

        Raises:
            ValidationError: If IP address format is invalid
        """
        ip_str = cls.validate_string(
            value, field_name, min_length=1, max_length=45, allow_empty=False
        )

        try:
            ip = ipaddress.ip_address(ip_str)

            if isinstance(ip, ipaddress.IPv4Address) and not allow_ipv4:
                raise ValidationError(f"{field_name} IPv4 not allowed", field_name, value)

            if isinstance(ip, ipaddress.IPv6Address) and not allow_ipv6:
                raise ValidationError(f"{field_name} IPv6 not allowed", field_name, value)

        except ValueError:
            raise ValidationError(f"Invalid {field_name} format", field_name, value)

        return str(ip)

    @classmethod
    def validate_trading_symbol(cls, value: Any, field_name: str = "symbol") -> str:
        """
        Validate trading symbol format.

        Args:
            value: Symbol to validate
            field_name: Name of field for error messages

        Returns:
            Validated symbol in uppercase

        Raises:
            ValidationError: If symbol format is invalid
        """
        symbol = cls.validate_string(
            value, field_name, min_length=1, max_length=20, allow_empty=False
        ).upper()

        if not cls.SYMBOL_PATTERN.match(symbol):
            raise ValidationError(
                f"{field_name} must contain only letters, numbers, dots, and hyphens",
                field_name,
                value,
            )

        return symbol

    @classmethod
    def validate_currency_code(cls, value: Any, field_name: str = "currency") -> str:
        """
        Validate ISO 4217 currency code format.

        Args:
            value: Currency code to validate
            field_name: Name of field for error messages

        Returns:
            Validated currency code in uppercase

        Raises:
            ValidationError: If currency code format is invalid
        """
        currency = cls.validate_string(
            value, field_name, min_length=3, max_length=3, allow_empty=False
        ).upper()

        if not cls.CURRENCY_PATTERN.match(currency):
            raise ValidationError(
                f"{field_name} must be a 3-letter currency code", field_name, value
            )

        return currency

    @classmethod
    def validate_json(
        cls, value: Any, field_name: str = "json_data", max_depth: int = 10
    ) -> dict[str, Any]:
        """
        Validate JSON data structure.

        Args:
            value: JSON string or dict to validate
            field_name: Name of field for error messages
            max_depth: Maximum allowed nesting depth

        Returns:
            Validated JSON data as dict

        Raises:
            ValidationError: If JSON is invalid
        """
        if isinstance(value, dict):
            data = value
        elif isinstance(value, str):
            try:
                data = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid {field_name}: {e}", field_name, value)
        else:
            raise ValidationError(f"{field_name} must be JSON string or dict", field_name, value)

        # Check nesting depth
        def check_depth(obj: Any, current_depth: int = 0) -> None:
            if current_depth > max_depth:
                raise ValidationError(
                    f"{field_name} exceeds maximum nesting depth of {max_depth}", field_name, value
                )

            if isinstance(obj, dict):
                for val in obj.values():
                    check_depth(val, current_depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, current_depth + 1)

        check_depth(data)
        return data

    @classmethod
    def validate_safe_identifier(cls, value: Any, field_name: str = "identifier") -> str:
        """
        Validate a safe identifier (for non-SQL use cases).

        Note: This is NOT for SQL identifiers - use proper schema validation
        and parameterized queries for database operations.

        Args:
            value: Identifier to validate
            field_name: Name of field for error messages

        Returns:
            Validated identifier

        Raises:
            ValidationError: If identifier format is invalid
        """
        identifier = cls.validate_string(
            value, field_name, min_length=1, max_length=64, allow_empty=False
        )

        if not cls.SAFE_IDENTIFIER_PATTERN.match(identifier):
            raise ValidationError(
                f"{field_name} must start with letter and contain only letters, numbers, underscore, and hyphen",
                field_name,
                value,
            )

        return identifier

    @classmethod
    def validate_enum(
        cls,
        value: Any,
        allowed_values: set[str],
        field_name: str = "field",
        case_sensitive: bool = True,
    ) -> str:
        """
        Validate value against enumeration of allowed values.

        Args:
            value: Value to validate
            allowed_values: Set of allowed string values
            field_name: Name of field for error messages
            case_sensitive: Whether comparison is case sensitive

        Returns:
            Validated value

        Raises:
            ValidationError: If value is not in allowed set
        """
        str_value = cls.validate_string(value, field_name, min_length=1, allow_empty=False)

        if case_sensitive:
            if str_value not in allowed_values:
                raise ValidationError(
                    f"{field_name} must be one of: {', '.join(sorted(allowed_values))}",
                    field_name,
                    value,
                )
            return str_value
        else:
            lower_allowed = {v.lower() for v in allowed_values}
            if str_value.lower() not in lower_allowed:
                raise ValidationError(
                    f"{field_name} must be one of: {', '.join(sorted(allowed_values))}",
                    field_name,
                    value,
                )
            # Return the original case from allowed_values
            for allowed in allowed_values:
                if allowed.lower() == str_value.lower():
                    return allowed
            return str_value  # Fallback, shouldn't reach here


class SchemaValidator:
    """
    Schema-based validation for complex data structures.
    """

    @classmethod
    def validate_order_data(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate trading order data structure.

        Args:
            data: Order data to validate

        Returns:
            Validated order data

        Raises:
            ValidationError: If validation fails
        """
        validator = InputValidator()

        validated = {}

        # Required fields
        validated["symbol"] = validator.validate_trading_symbol(data.get("symbol"), "symbol")
        validated["quantity"] = str(
            validator.validate_decimal(
                data.get("quantity"), "quantity", min_value=Decimal("0.00001")
            )
        )
        validated["order_type"] = validator.validate_enum(
            data.get("order_type"),
            {"market", "limit", "stop", "stop_limit"},
            "order_type",
            case_sensitive=False,
        )
        validated["side"] = validator.validate_enum(
            data.get("side"), {"buy", "sell"}, "side", case_sensitive=False
        )

        # Optional fields
        if "price" in data and data["price"] is not None:
            validated["price"] = str(
                validator.validate_decimal(data["price"], "price", min_value=Decimal("0.01"))
            )

        if "stop_price" in data and data["stop_price"] is not None:
            validated["stop_price"] = str(
                validator.validate_decimal(
                    data["stop_price"], "stop_price", min_value=Decimal("0.01")
                )
            )

        if "time_in_force" in data and data["time_in_force"] is not None:
            validated["time_in_force"] = validator.validate_enum(
                data["time_in_force"],
                {"day", "gtc", "ioc", "fok"},
                "time_in_force",
                case_sensitive=False,
            )

        return validated

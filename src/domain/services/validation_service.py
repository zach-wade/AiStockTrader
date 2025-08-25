"""
Domain Validation Service - Business validation rules and constraints.

This service contains all business validation logic that was previously
scattered across infrastructure components. It ensures that business rules
are enforced consistently throughout the domain.
"""

import ipaddress
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlparse
from uuid import UUID


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


class DomainValidator:
    """
    Central domain validation service.

    Contains all business validation rules and constraints.
    """

    # Business Rules Constants
    MIN_SYMBOL_LENGTH = 1
    MAX_SYMBOL_LENGTH = 10
    VALID_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9\-\.]+$")

    MIN_PRICE = Decimal("0.01")
    MAX_PRICE = Decimal("999999.99")

    MIN_QUANTITY = Decimal("0.01")
    MAX_QUANTITY = Decimal("999999")

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """
        Validate a trading symbol according to business rules.

        Args:
            symbol: The trading symbol to validate

        Returns:
            The validated and normalized symbol

        Raises:
            ValidationError: If symbol doesn't meet business requirements
        """
        if not symbol:
            raise ValidationError("Symbol cannot be empty")

        # Normalize to uppercase
        symbol = symbol.upper().strip()

        # Check length constraints
        if len(symbol) < DomainValidator.MIN_SYMBOL_LENGTH:
            raise ValidationError(f"Symbol too short: {symbol}")
        if len(symbol) > DomainValidator.MAX_SYMBOL_LENGTH:
            raise ValidationError(f"Symbol too long: {symbol}")

        # Check pattern
        if not DomainValidator.VALID_SYMBOL_PATTERN.match(symbol):
            raise ValidationError(f"Invalid symbol format: {symbol}")

        return symbol

    @staticmethod
    def validate_price(price: Any) -> Decimal:
        """
        Validate a price according to business rules.

        Args:
            price: The price to validate

        Returns:
            The validated price as Decimal

        Raises:
            ValidationError: If price doesn't meet business requirements
        """
        if price is None:
            raise ValidationError("Price cannot be None")

        try:
            decimal_price = Decimal(str(price))
        except (InvalidOperation, ValueError) as e:
            raise ValidationError(f"Invalid price format: {price}") from e

        if decimal_price < DomainValidator.MIN_PRICE:
            raise ValidationError(f"Price too low: {decimal_price}")
        if decimal_price > DomainValidator.MAX_PRICE:
            raise ValidationError(f"Price too high: {decimal_price}")

        return decimal_price

    @staticmethod
    def validate_quantity(quantity: Any) -> Decimal:
        """
        Validate a quantity according to business rules.

        Args:
            quantity: The quantity to validate

        Returns:
            The validated quantity as Decimal

        Raises:
            ValidationError: If quantity doesn't meet business requirements
        """
        if quantity is None:
            raise ValidationError("Quantity cannot be None")

        try:
            decimal_quantity = Decimal(str(quantity))
        except (InvalidOperation, ValueError) as e:
            raise ValidationError(f"Invalid quantity format: {quantity}") from e

        if decimal_quantity < DomainValidator.MIN_QUANTITY:
            raise ValidationError(f"Quantity too low: {decimal_quantity}")
        if decimal_quantity > DomainValidator.MAX_QUANTITY:
            raise ValidationError(f"Quantity too high: {decimal_quantity}")

        return decimal_quantity

    @staticmethod
    def validate_decimal(
        value: Any, min_value: Decimal | None = None, max_value: Decimal | None = None
    ) -> Decimal:
        """
        Validate a decimal value with optional bounds.

        Args:
            value: The value to validate
            min_value: Optional minimum value
            max_value: Optional maximum value

        Returns:
            The validated value as Decimal

        Raises:
            ValidationError: If value doesn't meet requirements
        """
        if value is None:
            raise ValidationError("Value cannot be None")

        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError) as e:
            raise ValidationError(f"Invalid decimal format: {value}") from e

        if min_value is not None and decimal_value < min_value:
            raise ValidationError(f"Value too low: {decimal_value} < {min_value}")
        if max_value is not None and decimal_value > max_value:
            raise ValidationError(f"Value too high: {decimal_value} > {max_value}")

        return decimal_value

    @staticmethod
    def validate_uuid(value: Any) -> UUID:
        """
        Validate a UUID value.

        Args:
            value: The value to validate

        Returns:
            The validated UUID

        Raises:
            ValidationError: If value is not a valid UUID
        """
        if value is None:
            raise ValidationError("UUID cannot be None")

        if isinstance(value, UUID):
            return value

        try:
            return UUID(str(value))
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid UUID format: {value}") from e

    @staticmethod
    def validate_email(email: str) -> str:
        """
        Validate an email address according to business rules.

        Args:
            email: The email to validate

        Returns:
            The validated and normalized email

        Raises:
            ValidationError: If email doesn't meet requirements
        """
        if not email:
            raise ValidationError("Email cannot be empty")

        email = email.lower().strip()

        # Basic email pattern
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        if not email_pattern.match(email):
            raise ValidationError(f"Invalid email format: {email}")

        return email

    @staticmethod
    def validate_percentage(value: Any) -> Decimal:
        """
        Validate a percentage value (0-100).

        Args:
            value: The percentage to validate

        Returns:
            The validated percentage as Decimal

        Raises:
            ValidationError: If value is not a valid percentage
        """
        decimal_value = DomainValidator.validate_decimal(
            value, min_value=Decimal("0"), max_value=Decimal("100")
        )
        return decimal_value


class OrderValidator:
    """
    Validates orders according to business rules.

    This is a specialized validator for order-specific validation.
    """

    def __init__(self, domain_validator: DomainValidator | None = None) -> None:
        """Initialize with optional domain validator."""
        self.domain_validator = domain_validator or DomainValidator()

    def validate_order_params(
        self, symbol: str, quantity: Any, price: Any | None = None
    ) -> tuple[str, Decimal, Decimal | None]:
        """
        Validate order parameters.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Optional order price (for limit orders)

        Returns:
            Tuple of (validated_symbol, validated_quantity, validated_price)

        Raises:
            ValidationError: If any parameter is invalid
        """
        validated_symbol = self.domain_validator.validate_symbol(symbol)
        validated_quantity = self.domain_validator.validate_quantity(quantity)

        validated_price = None
        if price is not None:
            validated_price = self.domain_validator.validate_price(price)

        return validated_symbol, validated_quantity, validated_price


class DatabaseIdentifierValidator:
    """
    Validates database identifiers (table names, schema names, column names).

    This validator ensures identifiers are safe for use in SQL statements.
    """

    # SQL identifier constraints
    MAX_IDENTIFIER_LENGTH = 63  # PostgreSQL limit
    IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    @staticmethod
    def validate_identifier(identifier: str, identifier_type: str = "identifier") -> str:
        """
        Validate a SQL identifier for safety.

        Args:
            identifier: The identifier to validate
            identifier_type: Type description for error messages

        Returns:
            The validated identifier

        Raises:
            ValidationError: If identifier is unsafe for SQL
        """
        if not identifier:
            raise ValidationError(f"{identifier_type} cannot be empty")

        # Check length
        if len(identifier) > DatabaseIdentifierValidator.MAX_IDENTIFIER_LENGTH:
            raise ValidationError(
                f"{identifier_type} too long: {len(identifier)} > {DatabaseIdentifierValidator.MAX_IDENTIFIER_LENGTH}"
            )

        # Check pattern - only alphanumeric and underscore, starting with letter or underscore
        if not DatabaseIdentifierValidator.IDENTIFIER_PATTERN.match(identifier):
            raise ValidationError(
                f"Invalid {identifier_type} format: '{identifier}'. "
                f"Must contain only letters, numbers, and underscores, and start with a letter or underscore."
            )

        return identifier

    @staticmethod
    def validate_schema_name(schema_name: str) -> str:
        """
        Validate a database schema name.

        Args:
            schema_name: The schema name to validate

        Returns:
            The validated schema name

        Raises:
            ValidationError: If schema name is unsafe
        """
        return DatabaseIdentifierValidator.validate_identifier(schema_name, "Schema name")

    @staticmethod
    def validate_table_name(table_name: str) -> str:
        """
        Validate a database table name.

        Args:
            table_name: The table name to validate

        Returns:
            The validated table name

        Raises:
            ValidationError: If table name is unsafe
        """
        return DatabaseIdentifierValidator.validate_identifier(table_name, "Table name")


class ValidationService:
    """Main validation service containing all business validation logic."""

    # Security patterns moved from infrastructure
    SQL_INJECTION_PATTERNS = [
        r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION)\b",
        r"(?i)\b(OR|AND)\s+\d+\s*=\s*\d+",
        r"[';]\s*-{2,}",  # Comment injection
        r"\bunion\b.*\bselect\b",  # Union-based injection
        r"\bwaitfor\s+delay\b",  # Time-based injection
        r"\bxp_cmdshell\b",  # Command execution
    ]

    XSS_PATTERNS = [
        r"(?i)<script[^>]*>.*?</script>",
        r"(?i)javascript:\s*",
        r"(?i)on\w+\s*=\s*[\"'][^\"']*[\"']",
        r"(?i)data:\s*text/html",
        r"(?i)vbscript:\s*",
    ]

    @classmethod
    def validate_sql_injection(cls, value: str) -> bool:
        """Check if string contains SQL injection patterns."""
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value):
                return False
        return True

    @classmethod
    def validate_xss(cls, value: str) -> bool:
        """Check if string contains XSS patterns."""
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value):
                return False
        return True

    @classmethod
    def validate_ip_address(cls, ip: str) -> bool:
        """Validate IP address format."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    @classmethod
    def validate_url(cls, url: str) -> bool:
        """Validate URL format and security."""
        try:
            parsed = urlparse(url)
            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
            # Only allow safe schemes
            if parsed.scheme.lower() not in ["http", "https", "ftp", "ftps"]:
                return False
            return True
        except Exception:
            return False

    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format."""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    @classmethod
    def validate_json_structure(cls, json_str: str, max_depth: int = 10) -> bool:
        """Validate JSON structure and prevent deeply nested attacks."""
        try:
            data = json.loads(json_str)
            return cls._check_json_depth(data, max_depth)
        except (json.JSONDecodeError, ValueError):
            return False

    @classmethod
    def _check_json_depth(cls, obj: Any, max_depth: int, current_depth: int = 0) -> bool:
        """Recursively check JSON depth to prevent attacks."""
        if current_depth > max_depth:
            return False

        if isinstance(obj, dict):
            for value in obj.values():
                if not cls._check_json_depth(value, max_depth, current_depth + 1):
                    return False
        elif isinstance(obj, list):
            for item in obj:
                if not cls._check_json_depth(item, max_depth, current_depth + 1):
                    return False

        return True

    @classmethod
    def validate_field(cls, value: str, field_type: str) -> bool:
        """Validate a field based on its type."""
        # First check for security issues
        if not cls.validate_sql_injection(value):
            return False
        if not cls.validate_xss(value):
            return False

        # Then validate based on type
        if field_type == "email":
            return cls.validate_email(value)
        elif field_type == "url":
            return cls.validate_url(value)
        elif field_type == "ip":
            return cls.validate_ip_address(value)
        elif field_type == "symbol":
            try:
                DomainValidator.validate_symbol(value)
                return True
            except ValidationError:
                return False
        elif field_type == "price":
            try:
                DomainValidator.validate_price(value)
                return True
            except ValidationError:
                return False

        return True  # Default to valid for unknown types

    @classmethod
    def validate_schema(cls, data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
        """Validate data against schema definition."""
        errors = []

        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Required field '{field}' is missing")

        # Validate field types and constraints
        fields = schema.get("fields", {})
        for field_name, field_schema in fields.items():
            if field_name not in data:
                continue

            value = data[field_name]
            field_errors = cls._validate_field_against_schema(field_name, value, field_schema)
            errors.extend(field_errors)

        return errors

    @classmethod
    def _validate_field_against_schema(
        cls, field_name: str, value: Any, field_schema: dict[str, Any]
    ) -> list[str]:
        """Validate a single field against its schema."""
        errors = []

        # Type validation
        expected_type = field_schema.get("type")
        if expected_type:
            type_map = {
                "string": str,
                "integer": int,
                "float": float,
                "boolean": bool,
                "list": list,
                "dict": dict,
            }
            expected_class = type_map.get(expected_type)
            if expected_class and not isinstance(value, expected_class):
                errors.append(f"Field '{field_name}' must be of type {expected_type}")
                return errors  # Skip other validations if type is wrong

        # String validations
        if isinstance(value, str):
            min_length = field_schema.get("min_length")
            if min_length and len(value) < min_length:
                errors.append(f"Field '{field_name}' must be at least {min_length} characters")

            max_length = field_schema.get("max_length")
            if max_length and len(value) > max_length:
                errors.append(f"Field '{field_name}' must be at most {max_length} characters")

            pattern = field_schema.get("pattern")
            if pattern and not re.match(pattern, value):
                errors.append(f"Field '{field_name}' does not match required pattern")

        # Numeric validations
        if isinstance(value, (int, float)):
            minimum = field_schema.get("minimum")
            if minimum is not None and value < minimum:
                errors.append(f"Field '{field_name}' must be at least {minimum}")

            maximum = field_schema.get("maximum")
            if maximum is not None and value > maximum:
                errors.append(f"Field '{field_name}' must be at most {maximum}")

        # Custom validators
        custom_validator = field_schema.get("custom_validator")
        if custom_validator and callable(custom_validator):
            if not custom_validator(value):
                errors.append(f"Field '{field_name}' failed custom validation")

        return errors

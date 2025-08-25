"""
Input validation utilities for infrastructure layer.

This module provides basic security validation for infrastructure components.
Business logic validation is handled by domain services.

SECURITY NOTICE: This module has been updated to remove dangerous SQL sanitization
patterns. All SQL operations must use parameterized queries for security.
"""

import functools
import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any, TypeVar

from src.domain.services.trading_validation_service import TradingValidationService
from src.domain.services.validation_service import ValidationService
from src.infrastructure.security.input_sanitizer import InputSanitizer, SanitizationError
from src.infrastructure.security.input_validation import InputValidator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None, value: Any = None) -> None:
        self.field = field
        self.value = value
        super().__init__(message)


class SchemaValidationError(ValidationError):
    """Raised when schema validation fails."""

    def __init__(self, message: str, schema_errors: list[str]) -> None:
        self.schema_errors = schema_errors
        super().__init__(message)


class SecurityValidationError(ValidationError):
    """Raised when security-related validation fails."""

    pass


def sanitize_input(
    **sanitizers: Callable[[Any], Any],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for sanitizing function inputs - security only, no business logic.

    This decorator uses InputSanitizer for security validation only.
    Business validation should be done in domain services.

    Args:
        **sanitizers: Keyword arguments mapping parameter names to sanitizers

    Example:
        @sanitize_input(
            user_input=lambda x: InputSanitizer.sanitize_string(x, max_length=100)
        )
        def process_input(user_input: str) -> None:
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get function signature
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Sanitize each parameter
            for param_name, sanitizer in sanitizers.items():
                if param_name in bound.arguments:
                    try:
                        value = bound.arguments[param_name]
                        sanitized_value = sanitizer(value)
                        bound.arguments[param_name] = sanitized_value
                    except SanitizationError as e:
                        logger.warning(f"Sanitization failed for {param_name}: {e}")
                        raise ValidationError(f"Invalid {param_name}: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error sanitizing {param_name}: {e}")
                        raise ValidationError(f"Error sanitizing {param_name}: {e}")

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def check_required(**required_params: type) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Simple decorator to check required parameters - no business logic.

    Args:
        **required_params: Parameter names and their expected types

    Example:
        @check_required(api_key=str, api_secret=str)
        def connect_to_api(api_key: str, api_secret: str) -> None:
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Simple type checking - no business logic
            for param_name, expected_type in required_params.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise ValidationError(
                            f"Parameter '{param_name}' must be of type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def validate_sql_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Validate SQL parameters for type safety and basic format checking.

    SECURITY NOTE: This function validates parameter formats but does NOT
    sanitize SQL values. Use parameterized queries for all SQL operations!

    Args:
        params: Dictionary of parameters to validate

    Returns:
        Dictionary with validated parameters

    Raises:
        ValidationError: If any parameter is invalid
    """
    validated: dict[str, Any] = {}
    validator = InputValidator()

    for key, value in params.items():
        try:
            # Validate parameter names (they should be safe identifiers)
            validated_key = validator.validate_safe_identifier(key, f"parameter name '{key}'")

            if isinstance(value, str):
                # Basic string validation (no dangerous sanitization)
                # This only checks for extremely dangerous patterns like XSS
                try:
                    validated_value = InputSanitizer.sanitize_string(value, max_length=10000)
                    validated[validated_key] = validated_value
                except SanitizationError as e:
                    logger.warning(f"Parameter '{key}' contains dangerous patterns: {e}")
                    raise ValidationError(f"Invalid parameter '{key}': contains dangerous patterns")
            elif value is None:
                validated[validated_key] = None
            elif isinstance(value, (int, float, Decimal)) or isinstance(value, bool):
                validated[validated_key] = value
            else:
                # For other types, convert to string and validate
                str_value = str(value)
                try:
                    validated_value = InputSanitizer.sanitize_string(str_value, max_length=10000)
                    validated[validated_key] = validated_value
                except SanitizationError as e:
                    logger.warning(f"Parameter '{key}' contains dangerous patterns: {e}")
                    raise ValidationError(f"Invalid parameter '{key}': contains dangerous patterns")

        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error(f"Unexpected error validating parameter '{key}': {e}")
            raise ValidationError(f"Error validating parameter '{key}': {e}")

    return validated


# Legacy alias for backward compatibility - will be removed in future version
def sanitize_sql_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    DEPRECATED: Use validate_sql_params() instead.

    This function is deprecated because the name implies SQL sanitization,
    which should never be done manually. Use parameterized queries instead!
    """
    logger.warning(
        "sanitize_sql_params() is deprecated - use validate_sql_params() and parameterized queries"
    )
    return validate_sql_params(params)


class SecurityValidator:
    """Thin adapter for security validation - delegates to domain services."""

    @classmethod
    def check_sql_injection(cls, value: str) -> bool:
        """Delegate SQL injection validation to domain service."""
        return ValidationService.validate_sql_injection(value)

    @classmethod
    def check_xss(cls, value: str) -> bool:
        """Delegate XSS validation to domain service."""
        return ValidationService.validate_xss(value)

    @classmethod
    def check_trading_symbol(cls, symbol: str) -> bool:
        """
        Validate trading symbol format - delegates to domain service.

        This method is kept for backward compatibility but delegates
        all business logic to the domain service.
        """
        return TradingValidationService.validate_trading_symbol(symbol)

    @classmethod
    def check_currency_code(cls, currency: str) -> bool:
        """
        Validate ISO currency code format - delegates to domain service.

        This method is kept for backward compatibility but delegates
        all business logic to the domain service.
        """
        return TradingValidationService.validate_currency_code(currency)

    @classmethod
    def check_price(cls, price: str | float | Decimal) -> bool:
        """
        Validate price/monetary amount - delegates to domain service.

        This method is kept for backward compatibility but delegates
        all business logic to the domain service.
        """
        return TradingValidationService.validate_price(price)

    @classmethod
    def check_quantity(cls, quantity: str | int | float) -> bool:
        """
        Validate trading quantity - delegates to domain service.

        This method is kept for backward compatibility but delegates
        all business logic to the domain service.
        """
        return TradingValidationService.validate_quantity(quantity)

    @classmethod
    def check_ip_address(cls, ip: str) -> bool:
        """Delegate IP address validation to domain service."""
        return ValidationService.validate_ip_address(ip)

    @classmethod
    def check_url(cls, url: str) -> bool:
        """Delegate URL validation to domain service."""
        return ValidationService.validate_url(url)

    @classmethod
    def check_email(cls, email: str) -> bool:
        """Delegate email validation to domain service."""
        return ValidationService.validate_email(email)

    @classmethod
    def check_json_structure(cls, json_str: str, max_depth: int = 10) -> bool:
        """Delegate JSON validation to domain service."""
        return ValidationService.validate_json_structure(json_str, max_depth)


class SchemaValidator:
    """Thin adapter for schema validation - delegates to domain service."""

    @classmethod
    def check_schema(cls, data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
        """Delegate schema validation to domain service."""
        return ValidationService.validate_schema(data, schema)


class TradingInputValidator:
    """Trading-specific input validation - delegates to domain service."""

    @classmethod
    def check_order(cls, order_data: dict[str, Any]) -> list[str]:
        """
        Validate trading order data using domain service.

        This method delegates all business logic validation to the domain service.
        The infrastructure layer only handles the delegation.
        """
        return TradingValidationService.validate_order(order_data)

    @classmethod
    def check_portfolio_data(cls, portfolio_data: dict[str, Any]) -> list[str]:
        """
        Validate portfolio data structure using domain service.

        This method delegates all business logic validation to the domain service.
        The infrastructure layer only handles the delegation.
        """
        return TradingValidationService.validate_portfolio_data(portfolio_data)

    @classmethod
    def get_order_schema(cls) -> dict[str, Any]:
        """
        Get order validation schema from domain service.

        Returns:
            Order schema definition from the domain service
        """
        return TradingValidationService.get_order_schema()


def check_and_sanitize(
    **validators: Callable[[Any], Any],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Simple decorator that applies validators - delegates complex logic to domain."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Simple application of validators - no complex logic
            for param_name, validator in validators.items():
                if param_name in bound.arguments and callable(validator):
                    try:
                        result = validator(bound.arguments[param_name])
                        if result is not None:
                            bound.arguments[param_name] = result
                    except ValidationError:
                        raise

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def security_check(field_name: str, validation_type: str = "string") -> Callable[[Any], Any]:
    """Simple security validator factory - delegates to domain service."""

    def validator(value: Any) -> Any:
        if value is None:
            return value

        # Delegate validation to domain service
        str_value = str(value)
        is_valid = ValidationService.validate_field(str_value, validation_type)

        if not is_valid:
            raise ValidationError(f"Validation failed for {field_name} of type {validation_type}")

        # Simple sanitization
        try:
            return InputSanitizer.sanitize_string(str_value)
        except SanitizationError as e:
            raise SecurityValidationError(f"Sanitization failed for {field_name}: {e}")

    return validator

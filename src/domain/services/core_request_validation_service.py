"""
Core Request Validation Service - Domain service for basic request validation.

This service handles core business logic for basic request validation,
focusing only on fundamental request validation after refactoring to
achieve Single Responsibility Principle compliance.

Note: Specific validation concerns have been extracted to focused services:
- RateLimitingService: Rate limiting business rules
- NetworkValidationService: IP/network validation
- ContentValidationService: Request content validation
- ApiVersioningService: API versioning and endpoint policies
- TradingValidationService: Trading-specific validation
- MarketDataValidationService: Market data validation
- WebhookValidationService: Webhook validation
"""

from typing import Any


class RequestValidationError(Exception):
    """Exception raised when request validation fails."""

    pass


class CoreRequestValidationService:
    """
    Core domain service for basic request validation business logic.

    This service has been refactored to follow Single Responsibility Principle,
    containing only core request validation methods. Specific validation concerns
    have been extracted to focused services.
    """

    def validate_basic_request_format(self, request_data: dict[str, Any]) -> bool:
        """
        Validate basic request format and structure.

        Args:
            request_data: Dictionary containing request data

        Returns:
            True if basic format is valid

        Raises:
            RequestValidationError: If basic format is invalid
        """
        if not isinstance(request_data, dict):
            raise RequestValidationError("Request data must be a dictionary")

        # Check for empty request
        if not request_data:
            raise RequestValidationError("Request data cannot be empty")

        return True

    def validate_request_structure(
        self, request_data: dict[str, Any], required_fields: list[str]
    ) -> bool:
        """
        Validate request structure has required fields.

        Args:
            request_data: Dictionary containing request data
            required_fields: List of required field names

        Returns:
            True if all required fields are present

        Raises:
            RequestValidationError: If required fields are missing
        """
        missing_fields = []
        for field in required_fields:
            if field not in request_data:
                missing_fields.append(field)

        if missing_fields:
            raise RequestValidationError(f"Missing required fields: {', '.join(missing_fields)}")

        return True

    def validate_basic_field_types(
        self, request_data: dict[str, Any], field_types: dict[str, type]
    ) -> bool:
        """
        Validate basic field type requirements.

        Args:
            request_data: Dictionary containing request data
            field_types: Dictionary mapping field names to expected types

        Returns:
            True if all fields have correct types

        Raises:
            RequestValidationError: If field types are incorrect
        """
        type_errors = []
        for field, expected_type in field_types.items():
            if field in request_data:
                value = request_data[field]
                if not isinstance(value, expected_type):
                    type_errors.append(
                        f"Field '{field}' must be of type {expected_type.__name__}, got {type(value).__name__}"
                    )

        if type_errors:
            raise RequestValidationError("; ".join(type_errors))

        return True

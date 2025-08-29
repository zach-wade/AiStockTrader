"""
Content Validation Service - Domain service for request content validation business rules.

This service handles business logic for validating request content, headers,
and payload data, implementing the Single Responsibility Principle.
"""

import re
from datetime import datetime
from typing import Any


class ContentValidationError(Exception):
    """Exception raised when content validation fails."""

    pass


class ContentValidationService:
    """
    Domain service for content validation business logic.

    This service contains business rules for validating request content,
    headers, authorization, and payload data.
    """

    # Business rules for request validation
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB default

    # Allowed content types (business rule)
    ALLOWED_CONTENT_TYPES = [
        "application/json",
        "application/xml",
        "multipart/form-data",
        "application/x-www-form-urlencoded",
    ]

    # Business rules for header validation
    SUSPICIOUS_USER_AGENTS = [
        "bot",
        "crawler",
        "scanner",
        "sqlmap",
        "nikto",
        "curl",
        "wget",
        "python-requests",
    ]

    # Pagination limits (business rule)
    MAX_PAGE_SIZE = 100
    DEFAULT_PAGE_SIZE = 50

    # Date range limits (business rule)
    MAX_DATE_RANGE_DAYS = 365  # Maximum 1 year

    @classmethod
    def is_suspicious_user_agent(cls, user_agent: str) -> bool:
        """
        Check if User-Agent looks suspicious according to business rules.

        Args:
            user_agent: The User-Agent string to validate

        Returns:
            True if the User-Agent is considered suspicious
        """
        if not user_agent:
            # Business rule: missing User-Agent is suspicious
            return True

        ua_lower = user_agent.lower()
        for pattern in cls.SUSPICIOUS_USER_AGENTS:
            if pattern in ua_lower:
                return True

        return False

    def validate_request_id(self, request_id: str) -> bool:
        """
        Validate request ID format according to business rules.

        Args:
            request_id: Request ID string

        Returns:
            True if request ID is valid

        Raises:
            ContentValidationError: If request ID format is invalid
        """
        if not request_id:
            raise ContentValidationError("Request ID is required")

        # Minimum length check
        if len(request_id) < 5:
            raise ContentValidationError(f"Request ID too short: {request_id}")

        # Maximum length check
        if len(request_id) > 100:
            raise ContentValidationError(f"Request ID too long: {request_id}")

        # Check for invalid characters (basic alphanumeric + hyphen + underscore)
        if not re.match(r"^[a-zA-Z0-9_-]+$", request_id):
            raise ContentValidationError(f"Request ID contains invalid characters: {request_id}")

        return True

    def validate_authorization_header(self, auth_header: str) -> bool:
        """
        Validate authorization header according to business rules.

        Args:
            auth_header: Authorization header value

        Returns:
            True if authorization is valid

        Raises:
            ContentValidationError: If authorization format is invalid
        """
        if not auth_header:
            raise ContentValidationError("Authorization header is required")

        # Check for Bearer token
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if len(token) < 5:  # Minimum token length
                raise ContentValidationError("Invalid Bearer token")
            return True

        # Check for API key
        if auth_header.startswith("ApiKey "):
            key = auth_header[7:].strip()
            if len(key) < 5:  # Minimum key length
                raise ContentValidationError("Invalid API key")
            return True

        raise ContentValidationError("Invalid authorization format")

    def validate_content_type(self, content_type: str) -> bool:
        """
        Validate content type according to business rules.

        Args:
            content_type: Content-Type header value

        Returns:
            True if content type is valid

        Raises:
            ContentValidationError: If content type is not allowed
        """
        # Extract main content type (ignore charset and other parameters)
        main_type = content_type.split(";")[0].strip().lower()

        for allowed in self.ALLOWED_CONTENT_TYPES:
            if main_type == allowed.lower():
                return True

        raise ContentValidationError(f"Unsupported content type: {content_type}")

    def validate_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Validate date range according to business rules.

        Args:
            start_date: Start date of the range
            end_date: End date of the range

        Returns:
            True if date range is valid

        Raises:
            ContentValidationError: If date range is invalid
        """
        # Check if end date is after start date
        if end_date < start_date:
            raise ContentValidationError("End date must be after start date")

        # Check if range is not too large
        date_diff = (end_date - start_date).days
        if date_diff > self.MAX_DATE_RANGE_DAYS:
            raise ContentValidationError(
                f"Date range exceeds maximum of {self.MAX_DATE_RANGE_DAYS} days"
            )

        # Check if dates are not in the future
        now = datetime.now()
        if start_date > now:
            raise ContentValidationError("Start date cannot be in the future")

        return True

    def validate_pagination_params(self, params: dict[str, Any]) -> bool:
        """
        Validate pagination parameters according to business rules.

        Args:
            params: Dictionary containing pagination parameters

        Returns:
            True if pagination is valid

        Raises:
            ContentValidationError: If pagination parameters are invalid
        """
        # Check page number
        page = params.get("page", 1)
        if not isinstance(page, int) or page < 1:
            raise ContentValidationError(f"Invalid page number: {page}")

        # Check limit
        limit = params.get("limit", self.DEFAULT_PAGE_SIZE)
        if not isinstance(limit, int) or limit < 1:
            raise ContentValidationError(f"Invalid page limit: {limit}")

        if limit > self.MAX_PAGE_SIZE:
            raise ContentValidationError(f"Limit exceeds maximum of {self.MAX_PAGE_SIZE}")

        return True

    def validate_request_size(self, size_bytes: int, max_size: int | None = None) -> bool:
        """
        Validate request size against business rules.

        Args:
            size_bytes: Size of request in bytes
            max_size: Optional custom maximum size (defaults to MAX_REQUEST_SIZE)

        Returns:
            True if size is valid

        Raises:
            ContentValidationError: If size exceeds limits
        """
        limit = max_size if max_size is not None else self.MAX_REQUEST_SIZE
        if size_bytes > limit:
            raise ContentValidationError(
                f"Request size exceeds maximum allowed size of {limit} bytes"
            )
        return True

    @classmethod
    def validate_request_headers(cls, headers: dict[str, str]) -> list[str]:
        """
        Validate request headers according to business rules.

        Args:
            headers: Dictionary of request headers

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for User-Agent (required)
        user_agent = headers.get("User-Agent", "")
        if not user_agent:
            errors.append("User-Agent header is required")
        elif len(user_agent) < 5:
            # Check if User-Agent is too short (business rule)
            errors.append(f"Invalid User-Agent: '{user_agent}' is too short")
        elif cls.is_suspicious_user_agent(user_agent):
            # Check for suspicious User-Agent
            errors.append(f"Invalid User-Agent: '{user_agent}' appears suspicious")

        return errors

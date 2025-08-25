"""
Security Policy Service - Domain Layer

This service encapsulates all security-related business logic and policies.
It determines security requirements based on business rules, not technical implementation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskLevel(Enum):
    """Risk levels for security assessment."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessLevel(Enum):
    """Access levels for resource protection."""

    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    PRIVILEGED = "privileged"
    ADMIN = "admin"


class SanitizationLevel(Enum):
    """Sanitization levels for data processing."""

    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationRules:
    """Validation rules for a specific context."""

    required_fields: list[str]
    optional_fields: list[str]
    field_validators: dict[str, str]  # field_name -> validator_type
    max_length: dict[str, int]
    min_length: dict[str, int]
    patterns: dict[str, str]  # field_name -> regex pattern
    custom_rules: list[str]


@dataclass
class SecurityContext:
    """Context for security decisions."""

    user_role: str | None = None
    source_ip: str | None = None
    request_type: str | None = None
    resource_type: str | None = None
    operation: str | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] | None = None


class SecurityPolicyService:
    """
    Domain service for security policy decisions.

    This service contains all business logic related to security policies,
    extracted from the infrastructure layer to maintain clean architecture.
    """

    # Business rules for validation
    TRADING_CONTEXT_FIELDS = {
        "required": ["symbol", "quantity", "order_type"],
        "optional": ["price", "stop_price", "limit_price", "time_in_force"],
    }

    PORTFOLIO_CONTEXT_FIELDS = {
        "required": ["portfolio_id"],
        "optional": ["positions", "cash_balance", "margin_used"],
    }

    # Risk assessment rules
    HIGH_RISK_OPERATIONS = {"delete", "modify_portfolio", "place_order", "cancel_all_orders"}
    CRITICAL_RISK_OPERATIONS = {"transfer_funds", "close_account", "modify_risk_limits"}

    # Data type sanitization requirements
    SANITIZATION_REQUIREMENTS = {
        "user_input": SanitizationLevel.STRICT,
        "symbol": SanitizationLevel.STANDARD,
        "price": SanitizationLevel.BASIC,
        "quantity": SanitizationLevel.BASIC,
        "database_query": SanitizationLevel.PARANOID,
        "file_path": SanitizationLevel.STRICT,
        "url": SanitizationLevel.STANDARD,
        "email": SanitizationLevel.STANDARD,
        "json": SanitizationLevel.STANDARD,
        "html": SanitizationLevel.PARANOID,
    }

    def determine_validation_rules(self, context: str) -> ValidationRules:
        """
        Determine validation rules based on business context.

        Args:
            context: The business context (e.g., 'trading', 'portfolio', 'user')

        Returns:
            ValidationRules object with context-specific rules
        """
        if context == "trading":
            return ValidationRules(
                required_fields=self.TRADING_CONTEXT_FIELDS["required"],
                optional_fields=self.TRADING_CONTEXT_FIELDS["optional"],
                field_validators={
                    "symbol": "trading_symbol",
                    "quantity": "positive_number",
                    "price": "positive_decimal",
                    "order_type": "enum",
                },
                max_length={"symbol": 10, "order_type": 20},
                min_length={"symbol": 1},
                patterns={
                    "symbol": r"^[A-Z0-9\.]{1,10}$",
                    "order_type": r"^(market|limit|stop|stop_limit)$",
                },
                custom_rules=["validate_trading_hours", "check_symbol_active"],
            )
        elif context == "portfolio":
            return ValidationRules(
                required_fields=self.PORTFOLIO_CONTEXT_FIELDS["required"],
                optional_fields=self.PORTFOLIO_CONTEXT_FIELDS["optional"],
                field_validators={
                    "portfolio_id": "uuid",
                    "cash_balance": "decimal",
                    "margin_used": "decimal",
                },
                max_length={"portfolio_id": 36},
                min_length={"portfolio_id": 36},
                patterns={
                    "portfolio_id": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
                },
                custom_rules=["validate_portfolio_ownership", "check_margin_requirements"],
            )
        else:
            # Default validation rules
            return ValidationRules(
                required_fields=[],
                optional_fields=[],
                field_validators={},
                max_length={},
                min_length={},
                patterns={},
                custom_rules=[],
            )

    def evaluate_request_risk(self, context: SecurityContext) -> RiskLevel:
        """
        Evaluate request risk based on business rules.

        Args:
            context: Security context containing request metadata

        Returns:
            RiskLevel enum indicating the risk assessment
        """
        if not context.operation:
            return RiskLevel.LOW

        # Check for critical operations
        if context.operation in self.CRITICAL_RISK_OPERATIONS:
            return RiskLevel.CRITICAL

        # Check for high-risk operations
        if context.operation in self.HIGH_RISK_OPERATIONS:
            return RiskLevel.HIGH

        # Check user role
        if context.user_role == "admin":
            return RiskLevel.HIGH

        # Check for suspicious patterns
        if self._is_suspicious_request(context):
            return RiskLevel.HIGH

        # Default risk assessment
        if context.operation in ["read", "list", "get"]:
            return RiskLevel.LOW

        return RiskLevel.MEDIUM

    def determine_sanitization_level(self, data_type: str) -> SanitizationLevel:
        """
        Determine sanitization requirements based on data type.

        Args:
            data_type: Type of data being processed

        Returns:
            SanitizationLevel enum indicating required sanitization
        """
        return self.SANITIZATION_REQUIREMENTS.get(data_type, SanitizationLevel.STANDARD)

    def determine_access_level(self, resource: str, operation: str) -> AccessLevel:
        """
        Determine required access level for a resource and operation.

        Args:
            resource: Resource being accessed
            operation: Operation being performed

        Returns:
            AccessLevel enum indicating required access level
        """
        # Public resources
        if resource in ["market_data", "symbols", "trading_hours"]:
            return AccessLevel.PUBLIC

        # Authenticated resources
        if resource in ["portfolio_summary", "order_history"]:
            return AccessLevel.AUTHENTICATED

        # Authorized resources (user owns the resource)
        if resource in ["portfolio", "order", "position"]:
            if operation in ["read", "list"]:
                return AccessLevel.AUTHORIZED
            elif operation in ["create", "update", "delete"]:
                return AccessLevel.PRIVILEGED

        # Admin resources
        if resource in ["system_config", "user_management", "risk_limits"]:
            return AccessLevel.ADMIN

        # Default to authenticated
        return AccessLevel.AUTHENTICATED

    def validate_request_headers(self, headers: dict[str, str]) -> list[str]:
        """
        Validate request headers based on business rules.

        Args:
            headers: Request headers to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Required headers for trading operations
        required_headers = ["X-Request-ID", "X-Client-Version"]
        for header in required_headers:
            if header not in headers:
                errors.append(f"Missing required header: {header}")

        # Validate API version if present
        if "X-API-Version" in headers:
            version = headers["X-API-Version"]
            if not self._is_supported_api_version(version):
                errors.append(f"Unsupported API version: {version}")

        # Check for security headers
        security_headers = ["X-CSRF-Token", "Authorization"]
        has_security = any(h in headers for h in security_headers)
        if not has_security:
            errors.append("Missing security headers")

        return errors

    def should_rate_limit(self, context: SecurityContext) -> bool:
        """
        Determine if request should be rate limited.

        Args:
            context: Security context

        Returns:
            True if rate limiting should be applied
        """
        # Always rate limit high-risk operations
        if self.evaluate_request_risk(context) in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return True

        # Rate limit based on operation type
        rate_limited_operations = ["place_order", "cancel_order", "modify_order"]
        if context.operation in rate_limited_operations:
            return True

        return False

    def _is_suspicious_request(self, context: SecurityContext) -> bool:
        """Check if request shows suspicious patterns."""
        # Example suspicious patterns (business logic)
        suspicious_patterns = [
            context.source_ip and context.source_ip.startswith("10."),  # Internal IP from external
            context.metadata
            and context.metadata.get("request_count", 0) > 100,  # High request rate
            context.operation
            and "admin" in context.operation.lower()
            and context.user_role != "admin",
        ]
        return any(suspicious_patterns)

    def _is_supported_api_version(self, version: str) -> bool:
        """Check if API version is supported."""
        supported_versions = ["1.0", "1.1", "2.0"]
        return version in supported_versions

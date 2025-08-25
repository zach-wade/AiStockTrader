"""
Audit logging system exceptions.

This module defines all exceptions used by the audit logging system,
providing clear error handling and debugging capabilities.
"""

from typing import Any


class AuditException(Exception):
    """Base exception for all audit logging related errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """
        Initialize audit exception.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for categorization
            context: Additional context information for debugging
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
        }


class AuditConfigError(AuditException):
    """Exception raised for audit configuration errors."""

    def __init__(
        self, message: str, config_key: str | None = None, config_value: Any | None = None
    ):
        """
        Initialize audit configuration error.

        Args:
            message: Error description
            config_key: Configuration key that caused the error
            config_value: Invalid configuration value
        """
        context: dict[str, Any] = {}
        if config_key is not None:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = config_value

        super().__init__(message=message, error_code="AUDIT_CONFIG_ERROR", context=context)


class AuditStorageError(AuditException):
    """Exception raised for audit storage operation errors."""

    def __init__(
        self,
        message: str,
        storage_backend: str | None = None,
        operation: str | None = None,
        underlying_error: Exception | None = None,
    ):
        """
        Initialize audit storage error.

        Args:
            message: Error description
            storage_backend: Name of the storage backend that failed
            operation: Storage operation that failed
            underlying_error: Original exception that caused the failure
        """
        context: dict[str, Any] = {}
        if storage_backend:
            context["storage_backend"] = storage_backend
        if operation:
            context["operation"] = operation
        if underlying_error:
            context["underlying_error"] = str(underlying_error)
            context["underlying_error_type"] = type(underlying_error).__name__

        super().__init__(message=message, error_code="AUDIT_STORAGE_ERROR", context=context)


class AuditFormattingError(AuditException):
    """Exception raised for audit event formatting errors."""

    def __init__(
        self,
        message: str,
        formatter_type: str | None = None,
        event_data: dict[str, Any] | None = None,
    ):
        """
        Initialize audit formatting error.

        Args:
            message: Error description
            formatter_type: Type of formatter that failed
            event_data: Event data that failed to format
        """
        context: dict[str, Any] = {}
        if formatter_type:
            context["formatter_type"] = formatter_type
        if event_data:
            context["event_data_keys"] = list(event_data.keys())

        super().__init__(message=message, error_code="AUDIT_FORMATTING_ERROR", context=context)


class AuditValidationError(AuditException):
    """Exception raised for audit event validation errors."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        field_value: Any | None = None,
        validation_rule: str | None = None,
    ):
        """
        Initialize audit validation error.

        Args:
            message: Error description
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            validation_rule: Validation rule that was violated
        """
        context: dict[str, Any] = {}
        if field_name:
            context["field_name"] = field_name
        if field_value is not None:
            context["field_value"] = field_value
        if validation_rule:
            context["validation_rule"] = validation_rule

        super().__init__(message=message, error_code="AUDIT_VALIDATION_ERROR", context=context)


class AuditComplianceError(AuditException):
    """Exception raised for compliance-related audit errors."""

    def __init__(
        self,
        message: str,
        regulation: str | None = None,
        requirement: str | None = None,
        jurisdiction: str | None = None,
    ):
        """
        Initialize audit compliance error.

        Args:
            message: Error description
            regulation: Regulation that was violated (e.g., 'SOX', 'GDPR')
            requirement: Specific requirement that was violated
            jurisdiction: Regulatory jurisdiction
        """
        context: dict[str, Any] = {}
        if regulation:
            context["regulation"] = regulation
        if requirement:
            context["requirement"] = requirement
        if jurisdiction:
            context["jurisdiction"] = jurisdiction

        super().__init__(message=message, error_code="AUDIT_COMPLIANCE_ERROR", context=context)


class AuditIntegrityError(AuditException):
    """Exception raised for audit log integrity violations."""

    def __init__(
        self,
        message: str,
        integrity_check: str | None = None,
        expected_value: str | None = None,
        actual_value: str | None = None,
    ):
        """
        Initialize audit integrity error.

        Args:
            message: Error description
            integrity_check: Type of integrity check that failed
            expected_value: Expected value for the integrity check
            actual_value: Actual value that was found
        """
        context: dict[str, Any] = {}
        if integrity_check:
            context["integrity_check"] = integrity_check
        if expected_value:
            context["expected_value"] = expected_value
        if actual_value:
            context["actual_value"] = actual_value

        super().__init__(message=message, error_code="AUDIT_INTEGRITY_ERROR", context=context)


class AuditQueryError(AuditException):
    """Exception raised for audit query operation errors."""

    def __init__(
        self,
        message: str,
        query_type: str | None = None,
        query_parameters: dict[str, Any] | None = None,
    ):
        """
        Initialize audit query error.

        Args:
            message: Error description
            query_type: Type of query that failed
            query_parameters: Parameters used in the failed query
        """
        context: dict[str, Any] = {}
        if query_type:
            context["query_type"] = query_type
        if query_parameters:
            context["query_parameters"] = query_parameters

        super().__init__(message=message, error_code="AUDIT_QUERY_ERROR", context=context)


class AuditPermissionError(AuditException):
    """Exception raised for audit permission and access control errors."""

    def __init__(
        self,
        message: str,
        user_id: str | None = None,
        required_permission: str | None = None,
        resource: str | None = None,
    ):
        """
        Initialize audit permission error.

        Args:
            message: Error description
            user_id: ID of the user who was denied access
            required_permission: Permission that was required but not granted
            resource: Resource that access was denied to
        """
        context: dict[str, Any] = {}
        if user_id:
            context["user_id"] = user_id
        if required_permission:
            context["required_permission"] = required_permission
        if resource:
            context["resource"] = resource

        super().__init__(message=message, error_code="AUDIT_PERMISSION_ERROR", context=context)

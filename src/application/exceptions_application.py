"""
Application-level exception hierarchy for the AI Trading System.

This module provides exceptions for application-layer errors including
use case failures, validation errors, and service coordination issues.
"""

from typing import Any


class ApplicationException(Exception):
    """Base exception for all application-level errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


# ============================================================================
# Use Case Exceptions
# ============================================================================


class UseCaseException(ApplicationException):
    """Base exception for use case errors."""

    def __init__(
        self, use_case: str, message: str, request_id: str | None = None, **kwargs: Any
    ) -> None:
        details = {"use_case": use_case, "request_id": request_id, **kwargs}
        super().__init__(message, details)
        self.use_case = use_case
        self.request_id = request_id


class UseCaseValidationException(UseCaseException):
    """Raised when use case input validation fails."""

    def __init__(
        self, use_case: str, field: str, value: Any, reason: str, request_id: str | None = None
    ) -> None:
        message = f"Validation failed for {field}: {reason}"
        super().__init__(
            use_case=use_case,
            message=message,
            request_id=request_id,
            field=field,
            value=str(value),
            reason=reason,
        )
        self.field = field
        self.value = value
        self.reason = reason


class UseCaseExecutionException(UseCaseException):
    """Raised when use case execution fails."""

    def __init__(
        self,
        use_case: str,
        operation: str,
        reason: str,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Failed to execute {operation}: {reason}"
        super().__init__(
            use_case=use_case,
            message=message,
            request_id=request_id,
            operation=operation,
            reason=reason,
            **kwargs,
        )
        self.operation = operation
        self.reason = reason


class UseCaseTimeoutException(UseCaseException):
    """Raised when use case execution times out."""

    def __init__(self, use_case: str, timeout_seconds: int, request_id: str | None = None) -> None:
        message = f"Use case timed out after {timeout_seconds}s"
        super().__init__(
            use_case=use_case,
            message=message,
            request_id=request_id,
            timeout_seconds=timeout_seconds,
        )
        self.timeout_seconds = timeout_seconds


# ============================================================================
# Service Coordination Exceptions
# ============================================================================


class ServiceCoordinationException(ApplicationException):
    """Base exception for service coordination errors."""

    pass


class ServiceNotAvailableException(ServiceCoordinationException):
    """Raised when a required service is not available."""

    def __init__(self, service_name: str, reason: str | None = None) -> None:
        message = f"Service {service_name} is not available"
        if reason:
            message += f": {reason}"

        details = {"service_name": service_name, "reason": reason}
        super().__init__(message, details)
        self.service_name = service_name
        self.reason = reason


class ServiceCommunicationException(ServiceCoordinationException):
    """Raised when service communication fails."""

    def __init__(
        self, source_service: str, target_service: str, operation: str, reason: str
    ) -> None:
        message = (
            f"Communication failed from {source_service} to {target_service} "
            f"during {operation}: {reason}"
        )
        details = {
            "source_service": source_service,
            "target_service": target_service,
            "operation": operation,
            "reason": reason,
        }
        super().__init__(message, details)
        self.source_service = source_service
        self.target_service = target_service
        self.operation = operation
        self.reason = reason


class CircuitBreakerOpenException(ServiceCoordinationException):
    """Raised when circuit breaker is open."""

    def __init__(
        self, service_name: str, failure_count: int, threshold: int, retry_after: int | None = None
    ) -> None:
        message = (
            f"Circuit breaker open for {service_name}: " f"{failure_count}/{threshold} failures"
        )
        if retry_after:
            message += f", retry after {retry_after}s"

        details = {
            "service_name": service_name,
            "failure_count": failure_count,
            "threshold": threshold,
            "retry_after": retry_after,
        }
        super().__init__(message, details)
        self.service_name = service_name
        self.failure_count = failure_count
        self.threshold = threshold
        self.retry_after = retry_after


# ============================================================================
# Validation Exceptions
# ============================================================================


class ValidationException(ApplicationException):
    """Base exception for validation errors."""

    def __init__(self, message: str, validation_errors: dict[str, list[str]] | None = None) -> None:
        details = {"validation_errors": validation_errors or {}}
        super().__init__(message, details)
        self.validation_errors = validation_errors or {}


class RequestValidationException(ValidationException):
    """Raised when request validation fails."""

    def __init__(
        self, validation_errors: dict[str, list[str]], request_type: str | None = None
    ) -> None:
        error_count = sum(len(errors) for errors in validation_errors.values())
        message = f"Request validation failed with {error_count} errors"
        if request_type:
            message = f"{request_type} validation failed with {error_count} errors"

        super().__init__(message, validation_errors)
        self.request_type = request_type


class BusinessRuleViolationException(ValidationException):
    """Raised when business rules are violated."""

    def __init__(self, rule: str, context: dict[str, Any] | None = None) -> None:
        message = f"Business rule violated: {rule}"
        super().__init__(message, validation_errors={"business_rule": [rule]})
        self.rule = rule
        self.context = context or {}


# ============================================================================
# Data Transfer Exceptions
# ============================================================================


class DataTransferException(ApplicationException):
    """Base exception for data transfer errors."""

    pass


class SerializationException(DataTransferException):
    """Raised when serialization fails."""

    def __init__(self, data_type: str, reason: str, data: Any = None) -> None:
        message = f"Failed to serialize {data_type}: {reason}"
        details = {
            "data_type": data_type,
            "reason": reason,
            "data_sample": str(data)[:100] if data else None,
        }
        super().__init__(message, details)
        self.data_type = data_type
        self.reason = reason
        self.data = data


class DeserializationException(DataTransferException):
    """Raised when deserialization fails."""

    def __init__(self, data_type: str, reason: str, raw_data: str | None = None) -> None:
        message = f"Failed to deserialize {data_type}: {reason}"
        details = {
            "data_type": data_type,
            "reason": reason,
            "raw_data_sample": raw_data[:100] if raw_data else None,
        }
        super().__init__(message, details)
        self.data_type = data_type
        self.reason = reason
        self.raw_data = raw_data


class DataMappingException(DataTransferException):
    """Raised when data mapping fails."""

    def __init__(
        self,
        source_type: str,
        target_type: str,
        field: str | None = None,
        reason: str | None = None,
    ) -> None:
        message = f"Failed to map {source_type} to {target_type}"
        if field:
            message += f" for field {field}"
        if reason:
            message += f": {reason}"

        details = {
            "source_type": source_type,
            "target_type": target_type,
            "field": field,
            "reason": reason,
        }
        super().__init__(message, details)
        self.source_type = source_type
        self.target_type = target_type
        self.field = field
        self.reason = reason


# ============================================================================
# Workflow Exceptions
# ============================================================================


class WorkflowException(ApplicationException):
    """Base exception for workflow errors."""

    def __init__(
        self,
        workflow_name: str,
        message: str,
        workflow_id: str | None = None,
        step: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = {
            "workflow_name": workflow_name,
            "workflow_id": workflow_id,
            "step": step,
            **kwargs,
        }
        super().__init__(message, details)
        self.workflow_name = workflow_name
        self.workflow_id = workflow_id
        self.step = step


class WorkflowStepFailedException(WorkflowException):
    """Raised when a workflow step fails."""

    def __init__(
        self,
        workflow_name: str,
        step: str,
        reason: str,
        workflow_id: str | None = None,
        retry_count: int = 0,
    ) -> None:
        message = f"Workflow step '{step}' failed: {reason}"
        if retry_count > 0:
            message += f" (after {retry_count} retries)"

        super().__init__(
            workflow_name=workflow_name,
            message=message,
            workflow_id=workflow_id,
            step=step,
            reason=reason,
            retry_count=retry_count,
        )
        self.reason = reason
        self.retry_count = retry_count


class WorkflowCompensationException(WorkflowException):
    """Raised when workflow compensation fails."""

    def __init__(
        self,
        workflow_name: str,
        failed_step: str,
        compensation_step: str,
        reason: str,
        workflow_id: str | None = None,
    ) -> None:
        message = (
            f"Failed to compensate for step '{failed_step}' "
            f"during '{compensation_step}': {reason}"
        )
        super().__init__(
            workflow_name=workflow_name,
            message=message,
            workflow_id=workflow_id,
            step=compensation_step,
            failed_step=failed_step,
            reason=reason,
        )
        self.failed_step = failed_step
        self.compensation_step = compensation_step
        self.reason = reason


# ============================================================================
# Batch Processing Exceptions
# ============================================================================


class BatchProcessingException(ApplicationException):
    """Base exception for batch processing errors."""

    def __init__(
        self,
        batch_id: str,
        message: str,
        total_items: int | None = None,
        failed_items: int | None = None,
        **kwargs: Any,
    ) -> None:
        details = {
            "batch_id": batch_id,
            "total_items": total_items,
            "failed_items": failed_items,
            **kwargs,
        }
        super().__init__(message, details)
        self.batch_id = batch_id
        self.total_items = total_items
        self.failed_items = failed_items


class BatchItemProcessingException(BatchProcessingException):
    """Raised when batch item processing fails."""

    def __init__(
        self,
        batch_id: str,
        item_id: str,
        item_index: int,
        reason: str,
        total_items: int | None = None,
    ) -> None:
        message = f"Failed to process item {item_index} ({item_id}): {reason}"
        super().__init__(
            batch_id=batch_id,
            message=message,
            total_items=total_items,
            item_id=item_id,
            item_index=item_index,
            reason=reason,
        )
        self.item_id = item_id
        self.item_index = item_index
        self.reason = reason


class BatchPartialFailureException(BatchProcessingException):
    """Raised when batch processing partially fails."""

    def __init__(
        self,
        batch_id: str,
        total_items: int,
        successful_items: int,
        failed_items: int,
        failed_item_ids: list[str] | None = None,
    ) -> None:
        message = (
            f"Batch partially failed: {successful_items}/{total_items} succeeded, "
            f"{failed_items} failed"
        )
        super().__init__(
            batch_id=batch_id,
            message=message,
            total_items=total_items,
            failed_items=failed_items,
            successful_items=successful_items,
            failed_item_ids=failed_item_ids,
        )
        self.successful_items = successful_items
        self.failed_item_ids = failed_item_ids or []


# ============================================================================
# Concurrency Exceptions
# ============================================================================


class ConcurrencyException(ApplicationException):
    """Base exception for concurrency-related errors."""

    pass


class ResourceLockException(ConcurrencyException):
    """Raised when resource locking fails."""

    def __init__(
        self,
        resource_id: str,
        resource_type: str,
        lock_holder: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        message = f"Failed to acquire lock on {resource_type} {resource_id}"
        if lock_holder:
            message += f" (held by {lock_holder})"
        if timeout_seconds:
            message += f" after {timeout_seconds}s"

        details = {
            "resource_id": resource_id,
            "resource_type": resource_type,
            "lock_holder": lock_holder,
            "timeout_seconds": timeout_seconds,
        }
        super().__init__(message, details)
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.lock_holder = lock_holder
        self.timeout_seconds = timeout_seconds


class IdempotencyViolationException(ConcurrencyException):
    """Raised when idempotency is violated."""

    def __init__(self, idempotency_key: str, operation: str, previous_result: Any = None) -> None:
        message = (
            f"Idempotency violation: operation '{operation}' already executed "
            f"with key {idempotency_key}"
        )
        details = {
            "idempotency_key": idempotency_key,
            "operation": operation,
            "previous_result": str(previous_result) if previous_result else None,
        }
        super().__init__(message, details)
        self.idempotency_key = idempotency_key
        self.operation = operation
        self.previous_result = previous_result

"""
Data Pipeline Exception Hierarchy

Provides a comprehensive exception hierarchy for the data pipeline with
specific exceptions for each component and proper error context.
"""

# Standard library imports
from typing import Any

# Local imports
from main.utils.core import get_logger

logger = get_logger(__name__)


class DataPipelineError(Exception):
    """Base exception for all data pipeline errors."""

    def __init__(
        self,
        message: str,
        component: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.component = component
        self.error_code = error_code
        self.context = context or {}
        self.original_error = original_error

        # Build comprehensive error message
        error_parts = [message]
        if component:
            error_parts.append(f"Component: {component}")
        if error_code:
            error_parts.append(f"Code: {error_code}")
        if original_error:
            error_parts.append(f"Original: {original_error!s}")

        super().__init__(" | ".join(error_parts))

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "component": self.component,
            "error_code": self.error_code,
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None,
        }


class ValidationError(DataPipelineError):
    """Exceptions related to data validation."""

    def __init__(
        self,
        message: str,
        validation_rule: str | None = None,
        failed_records: int | None = None,
        **kwargs,
    ):
        self.validation_rule = validation_rule
        self.failed_records = failed_records

        # Add validation-specific context
        context = kwargs.get("context", {})
        if validation_rule:
            context["validation_rule"] = validation_rule
        if failed_records is not None:
            context["failed_records"] = failed_records
        kwargs["context"] = context

        super().__init__(message, component="validation", **kwargs)


class StorageError(DataPipelineError):
    """Exceptions related to data storage operations."""

    def __init__(
        self, message: str, storage_type: str | None = None, operation: str | None = None, **kwargs
    ):
        self.storage_type = storage_type
        self.operation = operation

        # Add storage-specific context
        context = kwargs.get("context", {})
        if storage_type:
            context["storage_type"] = storage_type
        if operation:
            context["operation"] = operation
        kwargs["context"] = context

        super().__init__(message, component="storage", **kwargs)


class IngestionError(DataPipelineError):
    """Exceptions related to data ingestion."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        data_type: str | None = None,
        symbol: str | None = None,
        **kwargs,
    ):
        self.source = source
        self.data_type = data_type
        self.symbol = symbol

        # Add ingestion-specific context
        context = kwargs.get("context", {})
        if source:
            context["source"] = source
        if data_type:
            context["data_type"] = data_type
        if symbol:
            context["symbol"] = symbol
        kwargs["context"] = context

        super().__init__(message, component="ingestion", **kwargs)


class ProcessingError(DataPipelineError):
    """Exceptions related to data processing."""

    def __init__(
        self,
        message: str,
        processor_type: str | None = None,
        processing_stage: str | None = None,
        **kwargs,
    ):
        self.processor_type = processor_type
        self.processing_stage = processing_stage

        # Add processing-specific context
        context = kwargs.get("context", {})
        if processor_type:
            context["processor_type"] = processor_type
        if processing_stage:
            context["processing_stage"] = processing_stage
        kwargs["context"] = context

        super().__init__(message, component="processing", **kwargs)


class OrchestrationError(DataPipelineError):
    """Exceptions related to pipeline orchestration."""

    def __init__(
        self,
        message: str,
        orchestrator_type: str | None = None,
        pipeline_stage: str | None = None,
        **kwargs,
    ):
        self.orchestrator_type = orchestrator_type
        self.pipeline_stage = pipeline_stage

        # Add orchestration-specific context
        context = kwargs.get("context", {})
        if orchestrator_type:
            context["orchestrator_type"] = orchestrator_type
        if pipeline_stage:
            context["pipeline_stage"] = pipeline_stage
        kwargs["context"] = context

        super().__init__(message, component="orchestration", **kwargs)


class HistoricalDataError(DataPipelineError):
    """Exceptions related to historical data operations."""

    def __init__(
        self,
        message: str,
        operation_type: str | None = None,
        date_range: str | None = None,
        **kwargs,
    ):
        self.operation_type = operation_type
        self.date_range = date_range

        # Add historical data specific context
        context = kwargs.get("context", {})
        if operation_type:
            context["operation_type"] = operation_type
        if date_range:
            context["date_range"] = date_range
        kwargs["context"] = context

        super().__init__(message, component="historical", **kwargs)


class LayerConfigurationError(DataPipelineError):
    """Exceptions related to layer configuration."""

    def __init__(
        self, message: str, layer: int | None = None, configuration_key: str | None = None, **kwargs
    ):
        self.layer = layer
        self.configuration_key = configuration_key

        # Add layer configuration context
        context = kwargs.get("context", {})
        if layer is not None:
            context["layer"] = layer
        if configuration_key:
            context["configuration_key"] = configuration_key
        kwargs["context"] = context

        super().__init__(message, component="layer_configuration", **kwargs)


class EventProcessingError(DataPipelineError):
    """Exceptions related to event processing."""

    def __init__(
        self,
        message: str,
        event_type: str | None = None,
        event_handler: str | None = None,
        **kwargs,
    ):
        self.event_type = event_type
        self.event_handler = event_handler

        # Add event processing context
        context = kwargs.get("context", {})
        if event_type:
            context["event_type"] = event_type
        if event_handler:
            context["event_handler"] = event_handler
        kwargs["context"] = context

        super().__init__(message, component="event_processing", **kwargs)


# Security-related exceptions
class SecurityError(DataPipelineError):
    """Exceptions related to security issues."""

    def __init__(self, message: str, security_issue: str | None = None, **kwargs):
        self.security_issue = security_issue

        context = kwargs.get("context", {})
        if security_issue:
            context["security_issue"] = security_issue
        kwargs["context"] = context

        super().__init__(message, component="security", **kwargs)


class ConfigurationError(DataPipelineError):
    """Exceptions related to configuration issues."""

    def __init__(
        self,
        message: str,
        config_section: str | None = None,
        config_key: str | None = None,
        **kwargs,
    ):
        self.config_section = config_section
        self.config_key = config_key

        context = kwargs.get("context", {})
        if config_section:
            context["config_section"] = config_section
        if config_key:
            context["config_key"] = config_key
        kwargs["context"] = context

        super().__init__(message, component="configuration", **kwargs)


def convert_exception(
    original_error: Exception, context_message: str, component: str | None = None
) -> DataPipelineError:
    """
    Convert a generic exception to a specific DataPipelineError.

    Args:
        original_error: The original exception
        context_message: Additional context message
        component: Component where the error occurred

    Returns:
        Appropriate DataPipelineError subclass
    """
    # If it's already a DataPipelineError, just return it
    if isinstance(original_error, DataPipelineError):
        return original_error

    error_message = f"{context_message}: {original_error!s}"

    # Map common exception types to specific data pipeline exceptions
    if isinstance(original_error, (ValueError, TypeError)):
        return ValidationError(error_message, original_error=original_error)
    elif isinstance(original_error, (IOError, OSError)):
        return StorageError(error_message, operation="file_system", original_error=original_error)
    elif "database" in str(original_error).lower() or "connection" in str(original_error).lower():
        return StorageError(error_message, storage_type="database", original_error=original_error)
    elif "http" in str(original_error).lower() or "request" in str(original_error).lower():
        return IngestionError(error_message, original_error=original_error)
    else:
        return DataPipelineError(error_message, original_error=original_error, component=component)

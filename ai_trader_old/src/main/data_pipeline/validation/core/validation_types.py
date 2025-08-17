"""
Validation Types - Shared Type Definitions

Common type definitions used across the validation framework to avoid duplication.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Local imports
from main.data_pipeline.core.enums import DataLayer, DataType

# Import from interfaces
from main.interfaces.data_pipeline.validation import ValidationSeverity, ValidationStage


@dataclass
class ValidationResult:
    """
    Shared validation result implementation.

    Implements IValidationResult interface and provides a consistent
    result structure across all validators.
    """

    # Core fields
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional fields
    stage: ValidationStage | None = None
    severity: ValidationSeverity | None = None
    timestamp: datetime | None = None
    duration_ms: float | None = None

    # Statistics
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

    @property
    def error_count(self) -> int:
        """Number of errors encountered."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of warnings encountered."""
        return len(self.warnings)

    @property
    def has_errors(self) -> bool:
        """Indicates if validation has any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Indicates if validation has any warnings."""
        return len(self.warnings) > 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_records == 0:
            return 0.0
        return self.valid_records / self.total_records

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.passed = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.passed = self.passed and other.passed
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.metadata.update(other.metadata)
        self.total_records += other.total_records
        self.valid_records += other.valid_records
        self.invalid_records += other.invalid_records

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "stage": self.stage.value if self.stage else None,
            "severity": self.severity.value if self.severity else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration_ms": self.duration_ms,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "success_rate": self.success_rate,
        }

    def __str__(self) -> str:
        """String representation."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        stage_str = f" [{self.stage.value}]" if self.stage else ""
        return (
            f"ValidationResult{stage_str}: {status}, "
            f"errors={self.error_count}, warnings={self.warning_count}, "
            f"success_rate={self.success_rate:.1%}"
        )


@dataclass
class ValidationContext:
    """
    Shared validation context implementation.

    Implements IValidationContext interface and provides consistent
    context information across all validation operations.
    """

    # Required fields
    stage: ValidationStage
    layer: DataLayer
    data_type: DataType

    # Optional fields
    symbol: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None

    # Run information
    validation_run_id: str | None = None
    batch_id: str | None = None
    pipeline_version: str | None = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)

    def create_child_context(
        self,
        stage: ValidationStage | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> "ValidationContext":
        """Create a child context with updated stage and metadata."""
        child_metadata = self.metadata.copy()
        if additional_metadata:
            child_metadata.update(additional_metadata)

        return ValidationContext(
            stage=stage or self.stage,
            layer=self.layer,
            data_type=self.data_type,
            symbol=self.symbol,
            source=self.source,
            metadata=child_metadata,
            timestamp=datetime.now(UTC),
            validation_run_id=self.validation_run_id,
            batch_id=self.batch_id,
            pipeline_version=self.pipeline_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "stage": self.stage.value,
            "layer": self.layer.value,
            "data_type": self.data_type.value,
            "symbol": self.symbol,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "validation_run_id": self.validation_run_id,
            "batch_id": self.batch_id,
            "pipeline_version": self.pipeline_version,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ValidationContext(stage={self.stage.value}, "
            f"layer={self.layer.value}, data_type={self.data_type.value}, "
            f"symbol={self.symbol}, source={self.source})"
        )


# Validation error types
class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class DataValidationError(ValidationError):
    """Data validation error with details."""

    def __init__(
        self,
        message: str,
        stage: ValidationStage | None = None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = datetime.now(UTC)


class MissingFieldError(ValidationError):
    """Raised when required fields are missing."""

    def __init__(self, missing_fields: list[str], data_type: str):
        self.missing_fields = missing_fields
        self.data_type = data_type
        message = f"Missing required fields for {data_type}: {', '.join(missing_fields)}"
        super().__init__(message)


class DataQualityError(ValidationError):
    """Raised when data quality is below threshold."""

    def __init__(self, quality_score: float, threshold: float, issues: list[str]):
        self.quality_score = quality_score
        self.threshold = threshold
        self.issues = issues
        message = f"Data quality score {quality_score:.2f} below threshold {threshold:.2f}"
        super().__init__(message)


class RuleViolationError(ValidationError):
    """Raised when validation rules are violated."""

    def __init__(self, rule_name: str, message: str, severity: ValidationSeverity):
        self.rule_name = rule_name
        self.severity = severity
        super().__init__(f"Rule '{rule_name}' violation: {message}")


# Factory functions
def create_validation_result(
    passed: bool = True, stage: ValidationStage | None = None, **kwargs
) -> ValidationResult:
    """Factory function to create ValidationResult."""
    return ValidationResult(passed=passed, stage=stage, **kwargs)


def create_validation_context(
    stage: ValidationStage, layer: DataLayer, data_type: DataType, **kwargs
) -> ValidationContext:
    """Factory function to create ValidationContext."""
    return ValidationContext(stage=stage, layer=layer, data_type=data_type, **kwargs)

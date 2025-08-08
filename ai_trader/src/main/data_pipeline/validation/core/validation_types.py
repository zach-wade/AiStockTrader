"""
Validation Types - Shared Type Definitions

Common type definitions used across the validation framework to avoid duplication.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

# Import from interfaces
from main.interfaces.data_pipeline.validation import (
    ValidationStage,
    ValidationSeverity,
    IValidationResult,
    IValidationContext
)
from main.data_pipeline.core.enums import DataLayer, DataType


@dataclass
class ValidationResult:
    """
    Shared validation result implementation.
    
    Implements IValidationResult interface and provides a consistent
    result structure across all validators.
    """
    
    # Core fields
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields
    stage: Optional[ValidationStage] = None
    severity: Optional[ValidationSeverity] = None
    timestamp: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Statistics
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
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
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        self.passed = self.passed and other.passed
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.metadata.update(other.metadata)
        self.total_records += other.total_records
        self.valid_records += other.valid_records
        self.invalid_records += other.invalid_records
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'passed': self.passed,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata,
            'stage': self.stage.value if self.stage else None,
            'severity': self.severity.value if self.severity else None,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'duration_ms': self.duration_ms,
            'total_records': self.total_records,
            'valid_records': self.valid_records,
            'invalid_records': self.invalid_records,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'success_rate': self.success_rate
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
    symbol: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    # Run information
    validation_run_id: Optional[str] = None
    batch_id: Optional[str] = None
    pipeline_version: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)
    
    def create_child_context(
        self,
        stage: Optional[ValidationStage] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> 'ValidationContext':
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
            timestamp=datetime.now(timezone.utc),
            validation_run_id=self.validation_run_id,
            batch_id=self.batch_id,
            pipeline_version=self.pipeline_version
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'stage': self.stage.value,
            'layer': self.layer.value,
            'data_type': self.data_type.value,
            'symbol': self.symbol,
            'source': self.source,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'validation_run_id': self.validation_run_id,
            'batch_id': self.batch_id,
            'pipeline_version': self.pipeline_version
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
        stage: Optional[ValidationStage] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.stage = stage
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = datetime.now(timezone.utc)


class MissingFieldError(ValidationError):
    """Raised when required fields are missing."""
    
    def __init__(self, missing_fields: List[str], data_type: str):
        self.missing_fields = missing_fields
        self.data_type = data_type
        message = f"Missing required fields for {data_type}: {', '.join(missing_fields)}"
        super().__init__(message)


class DataQualityError(ValidationError):
    """Raised when data quality is below threshold."""
    
    def __init__(self, quality_score: float, threshold: float, issues: List[str]):
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
    passed: bool = True,
    stage: Optional[ValidationStage] = None,
    **kwargs
) -> ValidationResult:
    """Factory function to create ValidationResult."""
    return ValidationResult(
        passed=passed,
        stage=stage,
        **kwargs
    )


def create_validation_context(
    stage: ValidationStage,
    layer: DataLayer,
    data_type: DataType,
    **kwargs
) -> ValidationContext:
    """Factory function to create ValidationContext."""
    return ValidationContext(
        stage=stage,
        layer=layer,
        data_type=data_type,
        **kwargs
    )
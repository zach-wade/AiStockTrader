"""
Data Pipeline Validation Interfaces

Core validation interfaces for the data pipeline validation framework.
Provides abstractions for validation orchestration, data quality assessment,
and validation reporting with layer awareness.
"""

# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

# Local imports
from main.data_pipeline.core.enums import DataLayer, DataType


class ValidationStage(Enum):
    """Validation stages in the data pipeline."""

    INGEST = "ingest"
    POST_ETL = "post_etl"
    FEATURE_READY = "feature_ready"
    PRE_STORAGE = "pre_storage"
    POST_STORAGE = "post_storage"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IValidationResult(ABC):
    """Interface for validation results."""

    @property
    @abstractmethod
    def stage(self) -> ValidationStage:
        """Get the validation stage."""
        pass

    @property
    @abstractmethod
    def passed(self) -> bool:
        """Whether validation passed overall."""
        pass

    @property
    @abstractmethod
    def errors(self) -> list[str]:
        """List of validation errors."""
        pass

    @property
    @abstractmethod
    def warnings(self) -> list[str]:
        """List of validation warnings."""
        pass

    @property
    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        """Validation metrics and statistics."""
        pass

    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """When validation was performed."""
        pass

    @property
    @abstractmethod
    def duration_ms(self) -> float:
        """Validation duration in milliseconds."""
        pass


class IValidationContext(ABC):
    """Interface for validation context."""

    @property
    @abstractmethod
    def stage(self) -> ValidationStage:
        """Current validation stage."""
        pass

    @property
    @abstractmethod
    def layer(self) -> DataLayer:
        """Data layer being validated."""
        pass

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """Type of data being validated."""
        pass

    @property
    @abstractmethod
    def symbol(self) -> str | None:
        """Symbol being validated (if applicable)."""
        pass

    @property
    @abstractmethod
    def source(self) -> str | None:
        """Data source being validated."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Additional validation metadata."""
        pass


class IValidator(ABC):
    """Base interface for all validators."""

    @abstractmethod
    async def validate(self, data: Any, context: IValidationContext) -> IValidationResult:
        """Validate data with given context."""
        pass

    @abstractmethod
    async def get_validation_rules(self, context: IValidationContext) -> list[str]:
        """Get applicable validation rules for context."""
        pass

    @abstractmethod
    async def is_applicable(self, context: IValidationContext) -> bool:
        """Check if validator applies to given context."""
        pass


class IValidationPipeline(ABC):
    """Interface for validation pipeline orchestration."""

    @abstractmethod
    async def validate_stage(
        self,
        stage: ValidationStage,
        data: Any,
        layer: DataLayer,
        data_type: DataType,
        symbol: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IValidationResult:
        """Validate data for a specific stage."""
        pass

    @abstractmethod
    async def validate_batch(
        self,
        stage: ValidationStage,
        data_batch: list[Any],
        layer: DataLayer,
        data_type: DataType,
        symbols: list[str] | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[IValidationResult]:
        """Validate a batch of data."""
        pass

    @abstractmethod
    async def get_stage_validators(
        self, stage: ValidationStage, layer: DataLayer, data_type: DataType
    ) -> list[IValidator]:
        """Get validators for a specific stage and context."""
        pass


class IDataQualityCalculator(ABC):
    """Interface for data quality assessment."""

    @abstractmethod
    async def calculate_quality_score(self, data: Any, context: IValidationContext) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        pass

    @abstractmethod
    async def calculate_completeness(self, data: Any, context: IValidationContext) -> float:
        """Calculate data completeness score."""
        pass

    @abstractmethod
    async def calculate_accuracy(self, data: Any, context: IValidationContext) -> float:
        """Calculate data accuracy score."""
        pass

    @abstractmethod
    async def calculate_consistency(self, data: Any, context: IValidationContext) -> float:
        """Calculate data consistency score."""
        pass

    @abstractmethod
    async def get_quality_metrics(self, data: Any, context: IValidationContext) -> dict[str, float]:
        """Get detailed quality metrics."""
        pass


class IDataCleaner(ABC):
    """Interface for data cleaning operations."""

    @abstractmethod
    async def clean_data(
        self, data: Any, context: IValidationContext, in_place: bool = False
    ) -> Any:
        """Clean data according to validation rules."""
        pass

    @abstractmethod
    async def remove_duplicates(self, data: Any, context: IValidationContext) -> Any:
        """Remove duplicate records."""
        pass

    @abstractmethod
    async def handle_missing_values(
        self, data: Any, context: IValidationContext, strategy: str = "drop"
    ) -> Any:
        """Handle missing values with specified strategy."""
        pass

    @abstractmethod
    async def normalize_data(self, data: Any, context: IValidationContext) -> Any:
        """Normalize data formats and values."""
        pass

    @abstractmethod
    async def get_cleaning_summary(
        self, original_data: Any, cleaned_data: Any, context: IValidationContext
    ) -> dict[str, Any]:
        """Get summary of cleaning operations performed."""
        pass


class IValidationMetrics(ABC):
    """Interface for validation metrics collection."""

    @abstractmethod
    async def record_validation(
        self, result: IValidationResult, context: IValidationContext
    ) -> None:
        """Record validation result metrics."""
        pass

    @abstractmethod
    async def get_validation_stats(
        self,
        stage: ValidationStage | None = None,
        layer: DataLayer | None = None,
        data_type: DataType | None = None,
        time_range: tuple | None = None,
    ) -> dict[str, Any]:
        """Get validation statistics."""
        pass

    @abstractmethod
    async def get_quality_trends(
        self, layer: DataLayer, data_type: DataType, time_range: tuple
    ) -> dict[str, list[float]]:
        """Get quality score trends over time."""
        pass

    @abstractmethod
    async def export_metrics(
        self, format: str = "prometheus", include_details: bool = False
    ) -> str:
        """Export metrics in specified format."""
        pass


class IValidationReporter(ABC):
    """Interface for validation reporting."""

    @abstractmethod
    async def generate_validation_report(
        self, results: list[IValidationResult], context: IValidationContext | None = None
    ) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        pass

    @abstractmethod
    async def generate_quality_dashboard(
        self, layer: DataLayer, time_range: tuple | None = None
    ) -> dict[str, Any]:
        """Generate quality dashboard data."""
        pass

    @abstractmethod
    async def send_validation_alerts(
        self,
        results: list[IValidationResult],
        severity_threshold: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        """Send alerts for validation failures."""
        pass

    @abstractmethod
    async def archive_validation_results(
        self, results: list[IValidationResult], retention_days: int = 30
    ) -> None:
        """Archive validation results for historical analysis."""
        pass


class ICoverageAnalyzer(ABC):
    """Interface for data coverage analysis."""

    @abstractmethod
    async def analyze_temporal_coverage(
        self, data: Any, context: IValidationContext, expected_timeframe: tuple | None = None
    ) -> dict[str, Any]:
        """Analyze temporal data coverage."""
        pass

    @abstractmethod
    async def analyze_symbol_coverage(
        self, data: Any, context: IValidationContext, expected_symbols: list[str] | None = None
    ) -> dict[str, Any]:
        """Analyze symbol coverage."""
        pass

    @abstractmethod
    async def analyze_field_coverage(
        self, data: Any, context: IValidationContext, required_fields: list[str] | None = None
    ) -> dict[str, Any]:
        """Analyze field/column coverage."""
        pass

    @abstractmethod
    async def calculate_coverage_score(self, data: Any, context: IValidationContext) -> float:
        """Calculate overall coverage score."""
        pass


class IValidationRule(ABC):
    """Interface for validation rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Rule description."""
        pass

    @property
    @abstractmethod
    def severity(self) -> ValidationSeverity:
        """Rule severity level."""
        pass

    @abstractmethod
    async def evaluate(self, data: Any, context: IValidationContext) -> bool:
        """Evaluate rule against data."""
        pass

    @abstractmethod
    async def get_failure_message(self, data: Any, context: IValidationContext) -> str:
        """Get failure message for rule violation."""
        pass

    @abstractmethod
    async def is_applicable(self, context: IValidationContext) -> bool:
        """Check if rule applies to context."""
        pass


class IRuleEngine(ABC):
    """Interface for validation rule engine."""

    @abstractmethod
    async def register_rule(self, rule: IValidationRule) -> None:
        """Register a validation rule."""
        pass

    @abstractmethod
    async def get_applicable_rules(self, context: IValidationContext) -> list[IValidationRule]:
        """Get rules applicable to context."""
        pass

    @abstractmethod
    async def evaluate_rules(
        self, data: Any, context: IValidationContext, rules: list[IValidationRule] | None = None
    ) -> list[dict[str, Any]]:
        """Evaluate rules against data."""
        pass

    @abstractmethod
    async def create_rule_result(
        self, rule: IValidationRule, passed: bool, message: str, context: IValidationContext
    ) -> dict[str, Any]:
        """Create rule evaluation result."""
        pass


# Factory interfaces for dependency injection
class IValidationFactory(ABC):
    """Factory interface for creating validation components."""

    @abstractmethod
    def create_validator(
        self, validator_type: str, config: dict[str, Any] | None = None
    ) -> IValidator:
        """Create validator instance."""
        pass

    @abstractmethod
    def create_pipeline(self, config: dict[str, Any] | None = None) -> IValidationPipeline:
        """Create validation pipeline."""
        pass

    @abstractmethod
    def create_quality_calculator(
        self, config: dict[str, Any] | None = None
    ) -> IDataQualityCalculator:
        """Create data quality calculator."""
        pass

    @abstractmethod
    def create_data_cleaner(self, config: dict[str, Any] | None = None) -> IDataCleaner:
        """Create data cleaner."""
        pass

    @abstractmethod
    def create_coverage_analyzer(self, config: dict[str, Any] | None = None) -> ICoverageAnalyzer:
        """Create coverage analyzer."""
        pass

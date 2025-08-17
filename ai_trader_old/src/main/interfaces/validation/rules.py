"""
Validation Framework - Rules Engine Interfaces

Rule engine interfaces for flexible validation rule definition,
execution, and management across the validation framework.
"""

# Standard library imports
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any

# Local imports
from main.interfaces.data_pipeline.validation import (
    IRuleEngine,
    IValidationContext,
    IValidationRule,
    ValidationSeverity,
)


class RuleType(Enum):
    """Types of validation rules."""

    FIELD_PRESENCE = "field_presence"  # Field must be present
    FIELD_TYPE = "field_type"  # Field must be of specific type
    FIELD_RANGE = "field_range"  # Field value within range
    FIELD_PATTERN = "field_pattern"  # Field matches pattern
    FIELD_ENUM = "field_enum"  # Field value in enum
    RECORD_COMPLETENESS = "record_completeness"  # Record completeness
    CROSS_FIELD = "cross_field"  # Cross-field validation
    BUSINESS_LOGIC = "business_logic"  # Custom business rules
    DATA_QUALITY = "data_quality"  # Data quality rules
    TEMPORAL = "temporal"  # Time-based rules
    STATISTICAL = "statistical"  # Statistical validation
    CUSTOM = "custom"  # Custom rule implementation


class RuleScope(Enum):
    """Scope of rule application."""

    RECORD = "record"  # Apply to individual records
    DATASET = "dataset"  # Apply to entire dataset
    COLUMN = "column"  # Apply to individual columns
    CROSS_DATASET = "cross_dataset"  # Apply across datasets


class RuleExecutionMode(Enum):
    """Rule execution modes."""

    FAIL_FAST = "fail_fast"  # Stop on first failure
    COLLECT_ALL = "collect_all"  # Collect all failures
    SAMPLE = "sample"  # Sample validation
    PROBABILISTIC = "probabilistic"  # Probabilistic validation


class IRuleDefinition(IValidationRule):
    """Extended interface for rule definitions."""

    @property
    @abstractmethod
    def rule_type(self) -> RuleType:
        """Rule type."""
        pass

    @property
    @abstractmethod
    def rule_scope(self) -> RuleScope:
        """Rule scope."""
        pass

    @property
    @abstractmethod
    def execution_mode(self) -> RuleExecutionMode:
        """Rule execution mode."""
        pass

    @property
    @abstractmethod
    def rule_parameters(self) -> dict[str, Any]:
        """Rule parameters."""
        pass

    @property
    @abstractmethod
    def error_message_template(self) -> str:
        """Error message template."""
        pass

    @abstractmethod
    async def validate_rule_definition(self) -> tuple[bool, list[str]]:
        """Validate rule definition is correct."""
        pass

    @abstractmethod
    async def get_rule_dependencies(self) -> list[str]:
        """Get rule dependencies."""
        pass


class IRuleBuilder(ABC):
    """Interface for building validation rules."""

    @abstractmethod
    def create_field_presence_rule(
        self, field_name: str, severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> IRuleDefinition:
        """Create field presence rule."""
        pass

    @abstractmethod
    def create_field_type_rule(
        self,
        field_name: str,
        expected_type: type,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> IRuleDefinition:
        """Create field type rule."""
        pass

    @abstractmethod
    def create_field_range_rule(
        self,
        field_name: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> IRuleDefinition:
        """Create field range rule."""
        pass

    @abstractmethod
    def create_field_pattern_rule(
        self, field_name: str, pattern: str, severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> IRuleDefinition:
        """Create field pattern rule."""
        pass

    @abstractmethod
    def create_field_enum_rule(
        self,
        field_name: str,
        allowed_values: list[Any],
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> IRuleDefinition:
        """Create field enum rule."""
        pass

    @abstractmethod
    def create_cross_field_rule(
        self,
        rule_expression: str,
        involved_fields: list[str],
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> IRuleDefinition:
        """Create cross-field validation rule."""
        pass

    @abstractmethod
    def create_business_logic_rule(
        self,
        rule_name: str,
        rule_function: Callable,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> IRuleDefinition:
        """Create business logic rule."""
        pass

    @abstractmethod
    def create_custom_rule(self, rule_implementation: IRuleDefinition) -> IRuleDefinition:
        """Create custom rule from implementation."""
        pass


class IRuleRegistry(ABC):
    """Interface for rule registry management."""

    @abstractmethod
    async def register_rule(self, rule: IRuleDefinition, rule_category: str | None = None) -> str:
        """Register validation rule."""
        pass

    @abstractmethod
    async def unregister_rule(self, rule_id: str) -> bool:
        """Unregister validation rule."""
        pass

    @abstractmethod
    async def get_rule(self, rule_id: str) -> IRuleDefinition | None:
        """Get rule by ID."""
        pass

    @abstractmethod
    async def list_rules(
        self,
        category: str | None = None,
        rule_type: RuleType | None = None,
        severity: ValidationSeverity | None = None,
    ) -> list[IRuleDefinition]:
        """List registered rules."""
        pass

    @abstractmethod
    async def search_rules(self, search_criteria: dict[str, Any]) -> list[IRuleDefinition]:
        """Search rules by criteria."""
        pass

    @abstractmethod
    async def get_rules_for_context(self, context: IValidationContext) -> list[IRuleDefinition]:
        """Get applicable rules for validation context."""
        pass

    @abstractmethod
    async def validate_rule_compatibility(
        self, rules: list[IRuleDefinition]
    ) -> tuple[bool, list[str]]:
        """Validate rule compatibility."""
        pass


class IRuleExecutor(ABC):
    """Interface for rule execution."""

    @abstractmethod
    async def execute_rule(
        self, rule: IRuleDefinition, data: Any, context: IValidationContext
    ) -> dict[str, Any]:
        """Execute single validation rule."""
        pass

    @abstractmethod
    async def execute_rule_set(
        self,
        rules: list[IRuleDefinition],
        data: Any,
        context: IValidationContext,
        execution_mode: RuleExecutionMode = RuleExecutionMode.COLLECT_ALL,
    ) -> list[dict[str, Any]]:
        """Execute set of validation rules."""
        pass

    @abstractmethod
    async def execute_rules_parallel(
        self,
        rules: list[IRuleDefinition],
        data: Any,
        context: IValidationContext,
        max_concurrency: int = 5,
    ) -> list[dict[str, Any]]:
        """Execute rules in parallel."""
        pass

    @abstractmethod
    async def execute_conditional_rules(
        self,
        rule_conditions: dict[IRuleDefinition, str],  # rule -> condition
        data: Any,
        context: IValidationContext,
    ) -> list[dict[str, Any]]:
        """Execute rules based on conditions."""
        pass


class IRuleOptimizer(ABC):
    """Interface for rule optimization."""

    @abstractmethod
    async def optimize_rule_execution_order(
        self, rules: list[IRuleDefinition], optimization_criteria: dict[str, Any]
    ) -> list[IRuleDefinition]:
        """Optimize rule execution order."""
        pass

    @abstractmethod
    async def identify_redundant_rules(
        self, rules: list[IRuleDefinition]
    ) -> list[tuple[IRuleDefinition, IRuleDefinition]]:
        """Identify redundant rules."""
        pass

    @abstractmethod
    async def suggest_rule_consolidation(
        self, rules: list[IRuleDefinition]
    ) -> list[dict[str, Any]]:
        """Suggest rule consolidation opportunities."""
        pass

    @abstractmethod
    async def analyze_rule_performance(
        self, rule_execution_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze rule execution performance."""
        pass


class IRuleVersioning(ABC):
    """Interface for rule versioning."""

    @abstractmethod
    async def create_rule_version(self, rule: IRuleDefinition, version_info: dict[str, Any]) -> str:
        """Create new rule version."""
        pass

    @abstractmethod
    async def get_rule_versions(self, rule_id: str) -> list[dict[str, Any]]:
        """Get all versions of a rule."""
        pass

    @abstractmethod
    async def get_rule_version(self, rule_id: str, version: str) -> IRuleDefinition | None:
        """Get specific rule version."""
        pass

    @abstractmethod
    async def promote_rule_version(
        self, rule_id: str, version: str, target_environment: str
    ) -> bool:
        """Promote rule version to environment."""
        pass

    @abstractmethod
    async def rollback_rule_version(self, rule_id: str, target_version: str) -> bool:
        """Rollback rule to previous version."""
        pass


class IRuleTemplate(ABC):
    """Interface for rule templates."""

    @property
    @abstractmethod
    def template_name(self) -> str:
        """Template name."""
        pass

    @property
    @abstractmethod
    def template_parameters(self) -> dict[str, Any]:
        """Template parameters."""
        pass

    @abstractmethod
    async def instantiate_rule(self, parameters: dict[str, Any]) -> IRuleDefinition:
        """Instantiate rule from template."""
        pass

    @abstractmethod
    async def validate_parameters(self, parameters: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate template parameters."""
        pass


class IRuleTemplateRegistry(ABC):
    """Interface for rule template registry."""

    @abstractmethod
    async def register_template(self, template: IRuleTemplate) -> str:
        """Register rule template."""
        pass

    @abstractmethod
    async def get_template(self, template_name: str) -> IRuleTemplate | None:
        """Get rule template."""
        pass

    @abstractmethod
    async def list_templates(self, category: str | None = None) -> list[IRuleTemplate]:
        """List available templates."""
        pass

    @abstractmethod
    async def create_rule_from_template(
        self, template_name: str, parameters: dict[str, Any]
    ) -> IRuleDefinition:
        """Create rule from template."""
        pass


class IAdvancedRuleEngine(IRuleEngine):
    """Extended rule engine with advanced features."""

    @abstractmethod
    async def register_rule_set(
        self,
        rule_set_name: str,
        rules: list[IRuleDefinition],
        execution_config: dict[str, Any] | None = None,
    ) -> str:
        """Register rule set."""
        pass

    @abstractmethod
    async def execute_rule_set_by_name(
        self, rule_set_name: str, data: Any, context: IValidationContext
    ) -> list[dict[str, Any]]:
        """Execute named rule set."""
        pass

    @abstractmethod
    async def create_rule_dependency_graph(
        self, rules: list[IRuleDefinition]
    ) -> dict[str, list[str]]:
        """Create rule dependency graph."""
        pass

    @abstractmethod
    async def execute_rules_with_dependencies(
        self, rules: list[IRuleDefinition], data: Any, context: IValidationContext
    ) -> list[dict[str, Any]]:
        """Execute rules respecting dependencies."""
        pass

    @abstractmethod
    async def get_rule_execution_plan(
        self, rules: list[IRuleDefinition], optimization_hints: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get optimized rule execution plan."""
        pass

    @abstractmethod
    async def execute_with_circuit_breaker(
        self,
        rules: list[IRuleDefinition],
        data: Any,
        context: IValidationContext,
        failure_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Execute rules with circuit breaker pattern."""
        pass

"""
Validation Framework - Rules Engine Interfaces

Rule engine interfaces for flexible validation rule definition,
execution, and management across the validation framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, AsyncIterator
from datetime import datetime
from enum import Enum
import pandas as pd

from main.data_pipeline.core.enums import DataLayer, DataType
from main.interfaces.data_pipeline.validation import (
    ValidationSeverity,
    IValidationRule,
    IRuleEngine,
    IValidationContext,
    IValidationResult
)


class RuleType(Enum):
    """Types of validation rules."""
    FIELD_PRESENCE = "field_presence"      # Field must be present
    FIELD_TYPE = "field_type"              # Field must be of specific type
    FIELD_RANGE = "field_range"            # Field value within range
    FIELD_PATTERN = "field_pattern"        # Field matches pattern
    FIELD_ENUM = "field_enum"              # Field value in enum
    RECORD_COMPLETENESS = "record_completeness"  # Record completeness
    CROSS_FIELD = "cross_field"            # Cross-field validation
    BUSINESS_LOGIC = "business_logic"      # Custom business rules
    DATA_QUALITY = "data_quality"          # Data quality rules
    TEMPORAL = "temporal"                  # Time-based rules
    STATISTICAL = "statistical"           # Statistical validation
    CUSTOM = "custom"                      # Custom rule implementation


class RuleScope(Enum):
    """Scope of rule application."""
    RECORD = "record"          # Apply to individual records
    DATASET = "dataset"        # Apply to entire dataset
    COLUMN = "column"          # Apply to individual columns
    CROSS_DATASET = "cross_dataset"  # Apply across datasets


class RuleExecutionMode(Enum):
    """Rule execution modes."""
    FAIL_FAST = "fail_fast"        # Stop on first failure
    COLLECT_ALL = "collect_all"    # Collect all failures
    SAMPLE = "sample"              # Sample validation
    PROBABILISTIC = "probabilistic" # Probabilistic validation


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
    def rule_parameters(self) -> Dict[str, Any]:
        """Rule parameters."""
        pass
    
    @property
    @abstractmethod
    def error_message_template(self) -> str:
        """Error message template."""
        pass
    
    @abstractmethod
    async def validate_rule_definition(self) -> Tuple[bool, List[str]]:
        """Validate rule definition is correct."""
        pass
    
    @abstractmethod
    async def get_rule_dependencies(self) -> List[str]:
        """Get rule dependencies."""
        pass


class IRuleBuilder(ABC):
    """Interface for building validation rules."""
    
    @abstractmethod
    def create_field_presence_rule(
        self,
        field_name: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> IRuleDefinition:
        """Create field presence rule."""
        pass
    
    @abstractmethod
    def create_field_type_rule(
        self,
        field_name: str,
        expected_type: type,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> IRuleDefinition:
        """Create field type rule."""
        pass
    
    @abstractmethod
    def create_field_range_rule(
        self,
        field_name: str,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> IRuleDefinition:
        """Create field range rule."""
        pass
    
    @abstractmethod
    def create_field_pattern_rule(
        self,
        field_name: str,
        pattern: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> IRuleDefinition:
        """Create field pattern rule."""
        pass
    
    @abstractmethod
    def create_field_enum_rule(
        self,
        field_name: str,
        allowed_values: List[Any],
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> IRuleDefinition:
        """Create field enum rule."""
        pass
    
    @abstractmethod
    def create_cross_field_rule(
        self,
        rule_expression: str,
        involved_fields: List[str],
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> IRuleDefinition:
        """Create cross-field validation rule."""
        pass
    
    @abstractmethod
    def create_business_logic_rule(
        self,
        rule_name: str,
        rule_function: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> IRuleDefinition:
        """Create business logic rule."""
        pass
    
    @abstractmethod
    def create_custom_rule(
        self,
        rule_implementation: IRuleDefinition
    ) -> IRuleDefinition:
        """Create custom rule from implementation."""
        pass


class IRuleRegistry(ABC):
    """Interface for rule registry management."""
    
    @abstractmethod
    async def register_rule(
        self,
        rule: IRuleDefinition,
        rule_category: Optional[str] = None
    ) -> str:
        """Register validation rule."""
        pass
    
    @abstractmethod
    async def unregister_rule(self, rule_id: str) -> bool:
        """Unregister validation rule."""
        pass
    
    @abstractmethod
    async def get_rule(self, rule_id: str) -> Optional[IRuleDefinition]:
        """Get rule by ID."""
        pass
    
    @abstractmethod
    async def list_rules(
        self,
        category: Optional[str] = None,
        rule_type: Optional[RuleType] = None,
        severity: Optional[ValidationSeverity] = None
    ) -> List[IRuleDefinition]:
        """List registered rules."""
        pass
    
    @abstractmethod
    async def search_rules(
        self,
        search_criteria: Dict[str, Any]
    ) -> List[IRuleDefinition]:
        """Search rules by criteria."""
        pass
    
    @abstractmethod
    async def get_rules_for_context(
        self,
        context: IValidationContext
    ) -> List[IRuleDefinition]:
        """Get applicable rules for validation context."""
        pass
    
    @abstractmethod
    async def validate_rule_compatibility(
        self,
        rules: List[IRuleDefinition]
    ) -> Tuple[bool, List[str]]:
        """Validate rule compatibility."""
        pass


class IRuleExecutor(ABC):
    """Interface for rule execution."""
    
    @abstractmethod
    async def execute_rule(
        self,
        rule: IRuleDefinition,
        data: Any,
        context: IValidationContext
    ) -> Dict[str, Any]:
        """Execute single validation rule."""
        pass
    
    @abstractmethod
    async def execute_rule_set(
        self,
        rules: List[IRuleDefinition],
        data: Any,
        context: IValidationContext,
        execution_mode: RuleExecutionMode = RuleExecutionMode.COLLECT_ALL
    ) -> List[Dict[str, Any]]:
        """Execute set of validation rules."""
        pass
    
    @abstractmethod
    async def execute_rules_parallel(
        self,
        rules: List[IRuleDefinition],
        data: Any,
        context: IValidationContext,
        max_concurrency: int = 5
    ) -> List[Dict[str, Any]]:
        """Execute rules in parallel."""
        pass
    
    @abstractmethod
    async def execute_conditional_rules(
        self,
        rule_conditions: Dict[IRuleDefinition, str],  # rule -> condition
        data: Any,
        context: IValidationContext
    ) -> List[Dict[str, Any]]:
        """Execute rules based on conditions."""
        pass


class IRuleOptimizer(ABC):
    """Interface for rule optimization."""
    
    @abstractmethod
    async def optimize_rule_execution_order(
        self,
        rules: List[IRuleDefinition],
        optimization_criteria: Dict[str, Any]
    ) -> List[IRuleDefinition]:
        """Optimize rule execution order."""
        pass
    
    @abstractmethod
    async def identify_redundant_rules(
        self,
        rules: List[IRuleDefinition]
    ) -> List[Tuple[IRuleDefinition, IRuleDefinition]]:
        """Identify redundant rules."""
        pass
    
    @abstractmethod
    async def suggest_rule_consolidation(
        self,
        rules: List[IRuleDefinition]
    ) -> List[Dict[str, Any]]:
        """Suggest rule consolidation opportunities."""
        pass
    
    @abstractmethod
    async def analyze_rule_performance(
        self,
        rule_execution_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze rule execution performance."""
        pass


class IRuleVersioning(ABC):
    """Interface for rule versioning."""
    
    @abstractmethod
    async def create_rule_version(
        self,
        rule: IRuleDefinition,
        version_info: Dict[str, Any]
    ) -> str:
        """Create new rule version."""
        pass
    
    @abstractmethod
    async def get_rule_versions(
        self,
        rule_id: str
    ) -> List[Dict[str, Any]]:
        """Get all versions of a rule."""
        pass
    
    @abstractmethod
    async def get_rule_version(
        self,
        rule_id: str,
        version: str
    ) -> Optional[IRuleDefinition]:
        """Get specific rule version."""
        pass
    
    @abstractmethod
    async def promote_rule_version(
        self,
        rule_id: str,
        version: str,
        target_environment: str
    ) -> bool:
        """Promote rule version to environment."""
        pass
    
    @abstractmethod
    async def rollback_rule_version(
        self,
        rule_id: str,
        target_version: str
    ) -> bool:
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
    def template_parameters(self) -> Dict[str, Any]:
        """Template parameters."""
        pass
    
    @abstractmethod
    async def instantiate_rule(
        self,
        parameters: Dict[str, Any]
    ) -> IRuleDefinition:
        """Instantiate rule from template."""
        pass
    
    @abstractmethod
    async def validate_parameters(
        self,
        parameters: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate template parameters."""
        pass


class IRuleTemplateRegistry(ABC):
    """Interface for rule template registry."""
    
    @abstractmethod
    async def register_template(
        self,
        template: IRuleTemplate
    ) -> str:
        """Register rule template."""
        pass
    
    @abstractmethod
    async def get_template(
        self,
        template_name: str
    ) -> Optional[IRuleTemplate]:
        """Get rule template."""
        pass
    
    @abstractmethod
    async def list_templates(
        self,
        category: Optional[str] = None
    ) -> List[IRuleTemplate]:
        """List available templates."""
        pass
    
    @abstractmethod
    async def create_rule_from_template(
        self,
        template_name: str,
        parameters: Dict[str, Any]
    ) -> IRuleDefinition:
        """Create rule from template."""
        pass


class IAdvancedRuleEngine(IRuleEngine):
    """Extended rule engine with advanced features."""
    
    @abstractmethod
    async def register_rule_set(
        self,
        rule_set_name: str,
        rules: List[IRuleDefinition],
        execution_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register rule set."""
        pass
    
    @abstractmethod
    async def execute_rule_set_by_name(
        self,
        rule_set_name: str,
        data: Any,
        context: IValidationContext
    ) -> List[Dict[str, Any]]:
        """Execute named rule set."""
        pass
    
    @abstractmethod
    async def create_rule_dependency_graph(
        self,
        rules: List[IRuleDefinition]
    ) -> Dict[str, List[str]]:
        """Create rule dependency graph."""
        pass
    
    @abstractmethod
    async def execute_rules_with_dependencies(
        self,
        rules: List[IRuleDefinition],
        data: Any,
        context: IValidationContext
    ) -> List[Dict[str, Any]]:
        """Execute rules respecting dependencies."""
        pass
    
    @abstractmethod
    async def get_rule_execution_plan(
        self,
        rules: List[IRuleDefinition],
        optimization_hints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get optimized rule execution plan."""
        pass
    
    @abstractmethod
    async def execute_with_circuit_breaker(
        self,
        rules: List[IRuleDefinition],
        data: Any,
        context: IValidationContext,
        failure_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Execute rules with circuit breaker pattern."""
        pass
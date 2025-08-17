"""
Configuration Types

Data classes and enums for configuration optimization system.
"""

# Standard library imports
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class OptimizationStrategy(Enum):
    """Configuration optimization strategies."""

    PERFORMANCE = "performance"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    BALANCED = "balanced"
    CUSTOM = "custom"


class ParameterType(Enum):
    """Types of configuration parameters."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    ENUM = "enum"
    LIST = "list"
    DICT = "dict"


@dataclass
class ParameterConstraint:
    """Constraints for configuration parameters."""

    min_value: int | float | None = None
    max_value: int | float | None = None
    allowed_values: list[Any] | None = None
    step_size: int | float | None = None
    validation_func: Callable[[Any], bool] | None = None
    dependencies: list[str] | None = None


@dataclass
class OptimizationTarget:
    """Target metric for optimization."""

    metric_name: str
    target_value: float | None = None
    minimize: bool = True
    weight: float = 1.0
    tolerance: float = 0.05


@dataclass
class ConfigParameter:
    """Configuration parameter definition."""

    name: str
    current_value: Any
    param_type: ParameterType
    constraint: ParameterConstraint | None = None
    description: str | None = None
    impact_score: float = 1.0  # How much this parameter affects performance

    def validate_value(self, value: Any) -> bool:
        """Validate if value meets constraints."""
        if not self.constraint:
            return True

        # Type validation
        if (
            self.param_type == ParameterType.INTEGER
            and not isinstance(value, int)
            or self.param_type == ParameterType.FLOAT
            and not isinstance(value, (int, float))
            or (
                self.param_type == ParameterType.BOOLEAN
                and not isinstance(value, bool)
                or self.param_type == ParameterType.STRING
                and not isinstance(value, str)
            )
        ):
            return False

        # Range validation
        if self.constraint.min_value is not None and value < self.constraint.min_value:
            return False
        if self.constraint.max_value is not None and value > self.constraint.max_value:
            return False

        # Allowed values validation
        if (
            self.constraint.allowed_values is not None
            and value not in self.constraint.allowed_values
        ):
            return False

        # Custom validation
        if self.constraint.validation_func and not self.constraint.validation_func(value):
            return False

        return True


@dataclass
class OptimizationResult:
    """Result of configuration optimization."""

    original_config: dict[str, Any]
    optimized_config: dict[str, Any]
    performance_improvement: float
    optimization_time: float
    iterations: int
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    changed_parameters: list[str]
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_config": self.original_config,
            "optimized_config": self.optimized_config,
            "performance_improvement": self.performance_improvement,
            "optimization_time": self.optimization_time,
            "iterations": self.iterations,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "changed_parameters": self.changed_parameters,
            "timestamp": self.timestamp.isoformat(),
        }

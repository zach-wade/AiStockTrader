"""
Configuration Types

Data classes and enums for configuration optimization system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime


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
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    step_size: Optional[Union[int, float]] = None
    validation_func: Optional[Callable[[Any], bool]] = None
    dependencies: Optional[List[str]] = None


@dataclass
class OptimizationTarget:
    """Target metric for optimization."""
    metric_name: str
    target_value: Optional[float] = None
    minimize: bool = True
    weight: float = 1.0
    tolerance: float = 0.05


@dataclass
class ConfigParameter:
    """Configuration parameter definition."""
    name: str
    current_value: Any
    param_type: ParameterType
    constraint: Optional[ParameterConstraint] = None
    description: Optional[str] = None
    impact_score: float = 1.0  # How much this parameter affects performance
    
    def validate_value(self, value: Any) -> bool:
        """Validate if value meets constraints."""
        if not self.constraint:
            return True
        
        # Type validation
        if self.param_type == ParameterType.INTEGER and not isinstance(value, int):
            return False
        elif self.param_type == ParameterType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif self.param_type == ParameterType.BOOLEAN and not isinstance(value, bool):
            return False
        elif self.param_type == ParameterType.STRING and not isinstance(value, str):
            return False
        
        # Range validation
        if self.constraint.min_value is not None and value < self.constraint.min_value:
            return False
        if self.constraint.max_value is not None and value > self.constraint.max_value:
            return False
        
        # Allowed values validation
        if self.constraint.allowed_values is not None and value not in self.constraint.allowed_values:
            return False
        
        # Custom validation
        if self.constraint.validation_func and not self.constraint.validation_func(value):
            return False
        
        return True


@dataclass
class OptimizationResult:
    """Result of configuration optimization."""
    original_config: Dict[str, Any]
    optimized_config: Dict[str, Any]
    performance_improvement: float
    optimization_time: float
    iterations: int
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    changed_parameters: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_config': self.original_config,
            'optimized_config': self.optimized_config,
            'performance_improvement': self.performance_improvement,
            'optimization_time': self.optimization_time,
            'iterations': self.iterations,
            'metrics_before': self.metrics_before,
            'metrics_after': self.metrics_after,
            'changed_parameters': self.changed_parameters,
            'timestamp': self.timestamp.isoformat()
        }
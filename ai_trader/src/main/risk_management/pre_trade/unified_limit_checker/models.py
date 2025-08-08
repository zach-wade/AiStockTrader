# File: risk_management/pre_trade/unified_limit_checker/models.py

"""
Data models for the Unified Limit Checker.

Provides core data structures for limit definitions, violations, and check results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, List
import uuid

from main.utils.core import ensure_utc, get_logger
from .types import (
    LimitType,
    LimitScope,
    ViolationSeverity,
    LimitAction,
    ComparisonOperator
)

logger = get_logger(__name__)


@dataclass
class LimitDefinition:
    """
    Defines a limit with all necessary parameters.
    
    This is the core model for defining what limits to check and how.
    """
    # Identification
    limit_id: str
    name: str
    description: str
    
    # Limit configuration
    limit_type: LimitType
    scope: LimitScope
    threshold_value: float
    operator: ComparisonOperator = ComparisonOperator.LESS_EQUAL
    
    # Optional thresholds
    soft_threshold: Optional[float] = None  # Warning threshold
    upper_threshold: Optional[float] = None  # For range checks (BETWEEN operator)
    
    # Actions
    violation_action: LimitAction = LimitAction.ALERT
    soft_violation_action: LimitAction = LimitAction.LOG_ONLY
    
    # Scope filtering
    scope_filter: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    enabled: bool = True
    priority: int = 0  # Higher priority limits checked first
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: ensure_utc(datetime.now()))
    updated_at: datetime = field(default_factory=lambda: ensure_utc(datetime.now()))
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate limit definition after initialization."""
        # Validate thresholds
        if self.operator == ComparisonOperator.BETWEEN:
            if self.upper_threshold is None:
                raise ValueError("BETWEEN operator requires upper_threshold")
            if self.threshold_value >= self.upper_threshold:
                raise ValueError("threshold_value must be less than upper_threshold for BETWEEN")
        
        # Validate soft threshold
        if self.soft_threshold is not None:
            if self.operator in [ComparisonOperator.LESS_THAN, ComparisonOperator.LESS_EQUAL]:
                if self.soft_threshold >= self.threshold_value:
                    raise ValueError("soft_threshold must be less than threshold_value for LESS operators")
            elif self.operator in [ComparisonOperator.GREATER_THAN, ComparisonOperator.GREATER_EQUAL]:
                if self.soft_threshold <= self.threshold_value:
                    raise ValueError("soft_threshold must be greater than threshold_value for GREATER operators")
    
    def get_effective_threshold(self, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Get the effective threshold value, potentially adjusted by context.
        
        This allows for dynamic thresholds based on market conditions or other factors.
        """
        base_threshold = self.threshold_value
        
        if context and self.metadata.get('dynamic_adjustment'):
            # Example: Adjust threshold based on volatility
            if 'market_volatility' in context:
                vol_adjustment = context['market_volatility'] / 100  # Convert to decimal
                adjustment_factor = self.metadata.get('volatility_adjustment_factor', 0.5)
                base_threshold = base_threshold * (1 - (vol_adjustment * adjustment_factor))
        
        return base_threshold
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this limit applies to the given context."""
        if not self.enabled:
            return False
        
        # Check scope filters
        for key, value in self.scope_filter.items():
            if key not in context:
                return False
            
            context_value = context[key]
            
            # Handle different filter types
            if isinstance(value, list):
                if context_value not in value:
                    return False
            elif callable(value):
                if not value(context_value):
                    return False
            else:
                if context_value != value:
                    return False
        
        return True


@dataclass
class LimitViolation:
    """
    Records a limit violation with all relevant details.
    """
    # Identification
    violation_id: str
    limit_id: str
    limit_name: str
    
    # Violation details
    current_value: float
    threshold_value: float
    effective_threshold: float
    severity: ViolationSeverity
    operator: ComparisonOperator = ComparisonOperator.LESS_EQUAL
    
    # Context
    scope_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: ensure_utc(datetime.now()))
    
    # Resolution
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    # Actions
    recommended_action: Optional[LimitAction] = None
    action_taken: Optional[LimitAction] = None
    action_timestamp: Optional[datetime] = None
    
    # Metrics
    breach_magnitude: Optional[float] = None  # How much the limit was exceeded
    duration_seconds: Optional[float] = None  # How long the violation lasted
    
    def __post_init__(self):
        """Calculate additional metrics after initialization."""
        self.calculate_breach_magnitude()
    
    def calculate_breach_magnitude(self):
        """Calculate how much the limit was breached."""
        if self.operator == ComparisonOperator.LESS_EQUAL:
            self.breach_magnitude = (self.current_value - self.effective_threshold) / self.effective_threshold
        elif self.operator == ComparisonOperator.GREATER_EQUAL:
            self.breach_magnitude = (self.effective_threshold - self.current_value) / self.effective_threshold
        elif self.operator == ComparisonOperator.LESS_THAN:
            self.breach_magnitude = (self.current_value - self.effective_threshold) / self.effective_threshold
        elif self.operator == ComparisonOperator.GREATER_THAN:
            self.breach_magnitude = (self.effective_threshold - self.current_value) / self.effective_threshold
        else:
            # For other operators, default to simple difference
            self.breach_magnitude = abs(self.current_value - self.effective_threshold)
    
    def resolve(self, notes: Optional[str] = None):
        """Mark the violation as resolved."""
        self.resolved = True
        self.resolution_timestamp = ensure_utc(datetime.now())
        self.resolution_notes = notes
        
        if self.timestamp and self.resolution_timestamp:
            self.duration_seconds = (self.resolution_timestamp - self.timestamp).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'violation_id': self.violation_id,
            'limit_id': self.limit_id,
            'limit_name': self.limit_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'effective_threshold': self.effective_threshold,
            'severity': self.severity.value,
            'operator': self.operator.value,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'breach_magnitude': self.breach_magnitude,
            'duration_seconds': self.duration_seconds,
            'scope_context': self.scope_context,
            'recommended_action': self.recommended_action.value if self.recommended_action else None,
            'action_taken': self.action_taken.value if self.action_taken else None
        }


@dataclass
class LimitCheckResult:
    """
    Result of a limit check operation.
    """
    # Core result
    limit_id: str
    passed: bool
    current_value: float
    threshold_value: float
    
    # Optional violation details
    violation: Optional[LimitViolation] = None
    
    # Additional information
    message: str = ""
    check_timestamp: datetime = field(default_factory=lambda: ensure_utc(datetime.now()))
    check_duration_ms: Optional[float] = None
    
    # Context that was used
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Warnings or info messages
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        """String representation of the result."""
        status = "PASSED" if self.passed else "FAILED"
        base_msg = f"[{status}] Limit {self.limit_id}: {self.current_value:.2f} vs {self.threshold_value:.2f}"
        
        if self.message:
            base_msg += f" - {self.message}"
        
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'limit_id': self.limit_id,
            'passed': self.passed,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'message': self.message,
            'check_timestamp': self.check_timestamp.isoformat(),
            'check_duration_ms': self.check_duration_ms,
            'warnings': self.warnings
        }
        
        if self.violation:
            result['violation'] = self.violation.to_dict()
        
        return result


@dataclass
class AggregatedCheckResult:
    """
    Aggregated results from multiple limit checks.
    """
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    
    results: List[LimitCheckResult] = field(default_factory=list)
    violations: List[LimitViolation] = field(default_factory=list)
    
    total_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: ensure_utc(datetime.now()))
    
    # Summary by severity
    violations_by_severity: Dict[ViolationSeverity, int] = field(default_factory=dict)
    
    def add_result(self, result: LimitCheckResult):
        """Add a check result to the aggregation."""
        self.results.append(result)
        self.total_checks += 1
        
        if result.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1
            
            if result.violation:
                self.violations.append(result.violation)
                severity = result.violation.severity
                self.violations_by_severity[severity] = \
                    self.violations_by_severity.get(severity, 0) + 1
        
        if result.check_duration_ms:
            self.total_duration_ms += result.check_duration_ms
    
    @property
    def all_passed(self) -> bool:
        """Check if all limits passed."""
        return self.failed_checks == 0
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        if self.total_checks == 0:
            return 100.0
        return (self.passed_checks / self.total_checks) * 100
    
    def get_most_severe_violation(self) -> Optional[ViolationSeverity]:
        """Get the most severe violation level."""
        severity_order = [
            ViolationSeverity.CRITICAL,
            ViolationSeverity.HARD_BREACH,
            ViolationSeverity.SOFT_BREACH,
            ViolationSeverity.WARNING,
            ViolationSeverity.INFO
        ]
        
        for severity in severity_order:
            if severity in self.violations_by_severity:
                return severity
        
        return None
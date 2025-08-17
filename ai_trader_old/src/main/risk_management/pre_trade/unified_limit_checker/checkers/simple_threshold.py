"""
Simple Threshold Checker

This module provides a simple threshold-based limit checker that handles
most basic limit types with standard comparison operations.
"""

# Standard library imports
from datetime import datetime
import logging
from typing import Any

from ..models import LimitCheckResult, LimitDefinition, LimitViolation
from ..registry import LimitChecker
from ..types import ComparisonOperator, LimitType, ViolationSeverity

logger = logging.getLogger(__name__)


class SimpleThresholdChecker(LimitChecker):
    """Simple threshold-based limit checker."""

    def check_limit(
        self, limit: LimitDefinition, current_value: float, context: dict[str, Any]
    ) -> LimitCheckResult:
        """Check simple threshold limit."""

        start_time = datetime.now()

        # Get effective threshold
        threshold = limit.get_effective_threshold()

        # Perform comparison
        passed = self._evaluate_condition(
            current_value, limit.operator, threshold, limit.upper_threshold
        )

        # Determine severity
        severity = ViolationSeverity.INFO
        violation = None

        if not passed:
            # Check if it's a soft violation
            if limit.soft_threshold is not None:
                soft_passed = self._evaluate_condition(
                    current_value, limit.operator, limit.soft_threshold
                )
                if not soft_passed:
                    severity = ViolationSeverity.HARD_BREACH
                else:
                    severity = ViolationSeverity.SOFT_BREACH
            else:
                severity = ViolationSeverity.HARD_BREACH

            # Create violation record
            violation = LimitViolation(
                violation_id=f"{limit.limit_id}_{int(datetime.now().timestamp())}",
                limit_id=limit.limit_id,
                limit_name=limit.name,
                current_value=current_value,
                threshold_value=limit.threshold_value,
                effective_threshold=threshold,
                severity=severity,
                scope_context=context,
                operator=limit.operator,
                recommended_action=(
                    limit.violation_action
                    if severity == ViolationSeverity.HARD_BREACH
                    else limit.soft_violation_action
                ),
            )
            violation.calculate_breach_magnitude()

        # Calculate check duration
        check_duration = (datetime.now() - start_time).total_seconds() * 1000

        # Generate message
        message = self._generate_message(limit, current_value, threshold, passed)

        return LimitCheckResult(
            limit_id=limit.limit_id,
            passed=passed,
            current_value=current_value,
            threshold_value=threshold,
            violation=violation,
            severity=severity,
            check_duration_ms=check_duration,
            message=message,
        )

    def supported_limit_types(self) -> list[LimitType]:
        """Return supported limit types."""
        return [
            LimitType.POSITION_SIZE,
            LimitType.PORTFOLIO_EXPOSURE,
            LimitType.RISK_METRIC,
            LimitType.CONCENTRATION,
            LimitType.LEVERAGE,
            LimitType.VAR_UTILIZATION,
        ]

    def _evaluate_condition(
        self,
        value: float,
        operator: ComparisonOperator,
        threshold: float,
        upper_threshold: float | None = None,
    ) -> bool:
        """Evaluate comparison condition."""

        if operator == ComparisonOperator.GREATER_THAN:
            return value > threshold
        elif operator == ComparisonOperator.GREATER_EQUAL:
            return value >= threshold
        elif operator == ComparisonOperator.LESS_THAN:
            return value < threshold
        elif operator == ComparisonOperator.LESS_EQUAL:
            return value <= threshold
        elif operator == ComparisonOperator.EQUAL:
            return abs(value - threshold) < 1e-10  # Handle floating point precision
        elif operator == ComparisonOperator.NOT_EQUAL:
            return abs(value - threshold) >= 1e-10
        elif operator == ComparisonOperator.BETWEEN:
            if upper_threshold is None:
                raise ValueError("Upper threshold required for BETWEEN operator")
            return threshold <= value <= upper_threshold
        elif operator == ComparisonOperator.OUTSIDE:
            if upper_threshold is None:
                raise ValueError("Upper threshold required for OUTSIDE operator")
            return value < threshold or value > upper_threshold
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def _generate_message(
        self, limit: LimitDefinition, current_value: float, threshold: float, passed: bool
    ) -> str:
        """Generate descriptive message for check result."""

        status = "PASSED" if passed else "VIOLATED"
        op_str = limit.operator.value

        if limit.operator in [ComparisonOperator.BETWEEN, ComparisonOperator.OUTSIDE]:
            return (
                f"Limit '{limit.name}' {status}: {current_value:.4f} {op_str} "
                f"[{threshold:.4f}, {limit.upper_threshold:.4f}]"
            )
        else:
            return f"Limit '{limit.name}' {status}: {current_value:.4f} {op_str} {threshold:.4f}"

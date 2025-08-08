"""
Position Size Checker

This module provides a specialized checker for position size limits
that handles both relative (percentage) and absolute position size constraints.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

from ..registry import LimitChecker
from ..types import LimitType, ViolationSeverity
from ..models import LimitDefinition, LimitViolation, LimitCheckResult

logger = logging.getLogger(__name__)


class PositionSizeChecker(LimitChecker):
    """Specialized checker for position size limits."""
    
    def check_limit(self, limit: LimitDefinition, 
                   current_value: float, 
                   context: Dict[str, Any]) -> LimitCheckResult:
        """Check position size limit with portfolio context."""
        
        # Get portfolio value for relative calculations
        portfolio_value = context.get('portfolio_value', 1.0)
        position_value = current_value
        
        # Convert to percentage if threshold is relative
        if limit.scope_filter.get('relative', False):
            current_pct = (position_value / portfolio_value) * 100
            threshold_pct = limit.threshold_value
            
            passed = current_pct <= threshold_pct
            
            violation = None
            if not passed:
                violation = LimitViolation(
                    violation_id=f"{limit.limit_id}_{int(datetime.now().timestamp())}",
                    limit_id=limit.limit_id,
                    limit_name=limit.name,
                    current_value=current_pct,
                    threshold_value=threshold_pct,
                    effective_threshold=threshold_pct,
                    severity=ViolationSeverity.HARD_BREACH,
                    scope_context=context,
                    recommended_action=limit.violation_action
                )
                violation.calculate_breach_magnitude()
            
            return LimitCheckResult(
                limit_id=limit.limit_id,
                passed=passed,
                current_value=current_pct,
                threshold_value=threshold_pct,
                violation=violation,
                message=f"Position size: {current_pct:.2f}% of portfolio (limit: {threshold_pct:.2f}%)"
            )
        
        else:
            # Absolute value check
            threshold = limit.get_effective_threshold()
            passed = position_value <= threshold
            
            violation = None
            if not passed:
                violation = LimitViolation(
                    violation_id=f"{limit.limit_id}_{int(datetime.now().timestamp())}",
                    limit_id=limit.limit_id,
                    limit_name=limit.name,
                    current_value=position_value,
                    threshold_value=threshold,
                    effective_threshold=threshold,
                    severity=ViolationSeverity.HARD_BREACH,
                    scope_context=context,
                    recommended_action=limit.violation_action
                )
                violation.calculate_breach_magnitude()
            
            return LimitCheckResult(
                limit_id=limit.limit_id,
                passed=passed,
                current_value=position_value,
                threshold_value=threshold,
                violation=violation,
                message=f"Position value: ${position_value:,.2f} (limit: ${threshold:,.2f})"
            )
    
    def supported_limit_types(self) -> List[LimitType]:
        """Return supported limit types."""
        return [LimitType.POSITION_SIZE]
    
    def supports_limit_type(self, limit_type: LimitType) -> bool:
        """Check if this checker supports the given limit type."""
        return limit_type == LimitType.POSITION_SIZE
    
    def calculate_current_value(self, limit: LimitDefinition, context: Dict[str, Any]) -> float:
        """
        Calculate the current position value from context.
        
        Args:
            limit: The limit definition
            context: Context containing position and portfolio information
            
        Returns:
            Current position value
        """
        # Get position value from context
        position_value = context.get('position_value', 0.0)
        
        # If relative limit, calculate as percentage of portfolio
        if limit.scope_filter.get('relative', False):
            portfolio_value = context.get('portfolio_value', 1.0)
            if portfolio_value > 0:
                return (position_value / portfolio_value) * 100
            return 0.0
        
        # Return absolute position value
        return position_value
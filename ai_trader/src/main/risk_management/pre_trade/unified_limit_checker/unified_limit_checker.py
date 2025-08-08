"""
Unified Limit Checker - Main Orchestrator

This module provides the main UnifiedLimitChecker class that orchestrates
all limit checking functionality using the modular components.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable

from .config import LimitConfig, get_default_config
from .registry import CheckerRegistry, create_default_registry
from .events import EventManager, create_event_manager_with_defaults
from .types import LimitType
from .models import LimitDefinition, LimitViolation, LimitCheckResult
from .checkers import SimpleThresholdChecker, PositionSizeChecker, DrawdownChecker

logger = logging.getLogger(__name__)


class UnifiedLimitChecker:
    """
    Unified limit checking system for comprehensive threshold validation.
    
    Features:
    - Multiple limit types and specialized checkers
    - Configurable violation actions and severity levels
    - Historical violation tracking and analytics
    - Event-driven architecture with customizable handlers
    - Real-time monitoring and alerting
    - Comprehensive reporting and audit trails
    """
    
    def __init__(self, config: Optional[LimitConfig] = None,
                 registry: Optional[CheckerRegistry] = None,
                 event_manager: Optional[EventManager] = None):
        """Initialize unified limit checker."""
        
        # Initialize components
        self.config = config or get_default_config()
        self.registry = registry or self._create_default_registry()
        self.event_manager = event_manager or create_event_manager_with_defaults()
        
        # Limit definitions storage
        self.limits: Dict[str, LimitDefinition] = {}
        
        # Violation tracking
        self.active_violations: Dict[str, LimitViolation] = {}
        self.violation_history: List[LimitViolation] = []
        
        # Statistics
        self.check_count = 0
        self.violation_count = 0
        self.last_check_time: Optional[datetime] = None
        
        logger.info("UnifiedLimitChecker initialized")
    
    def _create_default_registry(self) -> CheckerRegistry:
        """Create default registry with standard checkers."""
        registry = create_default_registry()
        
        # Register specialized checkers
        registry.register_checker(LimitType.POSITION_SIZE, PositionSizeChecker())
        registry.register_checker(LimitType.DRAWDOWN, DrawdownChecker())
        
        # Register default checker for other types
        registry.register_default_checker(SimpleThresholdChecker())
        
        return registry
    
    def add_limit(self, limit: LimitDefinition) -> None:
        """Add a limit definition."""
        # Validate limit
        if not self.registry.get_checker_for_limit(limit):
            logger.warning(f"No checker available for limit {limit.limit_id}")
        
        self.limits[limit.limit_id] = limit
        logger.info(f"Added limit: {limit.name} ({limit.limit_id})")
    
    def remove_limit(self, limit_id: str) -> bool:
        """Remove a limit definition."""
        if limit_id in self.limits:
            del self.limits[limit_id]
            # Also remove any active violations for this limit
            self._resolve_violations_for_limit(limit_id, "Limit removed")
            logger.info(f"Removed limit: {limit_id}")
            return True
        return False
    
    def get_limit(self, limit_id: str) -> Optional[LimitDefinition]:
        """Get a limit definition by ID."""
        return self.limits.get(limit_id)
    
    def list_limits(self) -> List[LimitDefinition]:
        """List all limit definitions."""
        return list(self.limits.values())
    
    def update_limit(self, limit_id: str, **updates) -> bool:
        """Update a limit definition."""
        if limit_id not in self.limits:
            return False
        
        limit = self.limits[limit_id]
        for key, value in updates.items():
            if hasattr(limit, key):
                setattr(limit, key, value)
        
        limit.updated_at = datetime.now(timezone.utc)
        logger.info(f"Updated limit: {limit_id}")
        return True
    
    def check_limit(self, limit_id: str, current_value: float, 
                   context: Optional[Dict[str, Any]] = None) -> LimitCheckResult:
        """Check a specific limit."""
        
        context = context or {}
        self.check_count += 1
        self.last_check_time = datetime.now(timezone.utc)
        
        if limit_id not in self.limits:
            return LimitCheckResult(
                limit_id=limit_id,
                passed=False,
                current_value=current_value,
                threshold_value=0.0,
                message=f"Limit {limit_id} not found"
            )
        
        limit = self.limits[limit_id]
        
        # Check if limit is active
        if not limit.is_active():
            return LimitCheckResult(
                limit_id=limit_id,
                passed=True,
                current_value=current_value,
                threshold_value=limit.threshold_value,
                message=f"Limit {limit_id} is not active"
            )
        
        # Get appropriate checker
        checker = self.registry.get_checker_for_limit(limit)
        if not checker:
            return LimitCheckResult(
                limit_id=limit_id,
                passed=False,
                current_value=current_value,
                threshold_value=limit.threshold_value,
                message=f"No checker available for limit {limit_id}"
            )
        
        # Perform check
        result = checker.check_limit(limit, current_value, context)
        
        # Fire check event
        self.event_manager.fire_check_event(result, context)
        
        # Handle violations
        if result.violation:
            self._handle_violation(result.violation)
        
        return result
    
    def check_all_limits(self, values: Dict[str, float], 
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, LimitCheckResult]:
        """Check all limits against provided values."""
        
        results = {}
        context = context or {}
        
        for limit_id, limit in self.limits.items():
            if limit_id in values:
                result = self.check_limit(limit_id, values[limit_id], context)
                results[limit_id] = result
        
        return results
    
    def get_active_violations(self) -> List[LimitViolation]:
        """Get all active violations."""
        return list(self.active_violations.values())
    
    def get_violation_history(self, limit_id: Optional[str] = None, 
                            limit: Optional[int] = None) -> List[LimitViolation]:
        """Get violation history with optional filtering."""
        history = self.violation_history
        
        if limit_id:
            history = [v for v in history if v.limit_id == limit_id]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def resolve_violation(self, violation_id: str, note: str = "") -> bool:
        """Resolve a specific violation."""
        if violation_id in self.active_violations:
            violation = self.active_violations[violation_id]
            violation.resolve(note)
            
            # Move to history
            self.violation_history.append(violation)
            del self.active_violations[violation_id]
            
            # Fire resolution event
            self.event_manager.fire_resolution_event(violation)
            
            logger.info(f"Resolved violation: {violation_id}")
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'total_limits': len(self.limits),
            'active_limits': len([l for l in self.limits.values() if l.is_active()]),
            'total_checks': self.check_count,
            'active_violations': len(self.active_violations),
            'total_violations': self.violation_count,
            'violation_history_size': len(self.violation_history),
            'last_check_time': self.last_check_time,
            'registered_checkers': len(self.registry.list_registered_types())
        }
    
    def _handle_violation(self, violation: LimitViolation) -> None:
        """Handle a new violation."""
        self.violation_count += 1
        
        # Check if we already have an active violation for this limit
        existing_violation = None
        for v in self.active_violations.values():
            if v.limit_id == violation.limit_id:
                existing_violation = v
                break
        
        if existing_violation:
            # Update existing violation
            existing_violation.last_detected = violation.first_detected
            existing_violation.current_value = violation.current_value
            existing_violation.calculate_breach_magnitude()
            existing_violation.update_duration()
        else:
            # Add new violation
            self.active_violations[violation.violation_id] = violation
            
            # Fire violation event
            self.event_manager.fire_violation_event(violation)
    
    def _resolve_violations_for_limit(self, limit_id: str, note: str = "") -> None:
        """Resolve all violations for a specific limit."""
        to_resolve = []
        
        for violation_id, violation in self.active_violations.items():
            if violation.limit_id == limit_id:
                to_resolve.append(violation_id)
        
        for violation_id in to_resolve:
            self.resolve_violation(violation_id, note)
    
    def cleanup_old_violations(self, days: int = 30) -> int:
        """Clean up old resolved violations."""
        if not self.violation_history:
            return 0
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        
        original_count = len(self.violation_history)
        self.violation_history = [
            v for v in self.violation_history
            if v.resolution_time and v.resolution_time > cutoff_time
        ]
        
        removed_count = original_count - len(self.violation_history)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old violations")
        
        return removed_count
    
    def add_violation_handler(self, handler: Callable[[LimitViolation], None]) -> None:
        """Add a violation event handler."""
        self.event_manager.add_violation_handler(handler)
    
    def add_resolution_handler(self, handler: Callable[[LimitViolation], None]) -> None:
        """Add a resolution event handler."""
        self.event_manager.add_resolution_handler(handler)
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration."""
        return {
            'limits': [
                {
                    'limit_id': limit.limit_id,
                    'name': limit.name,
                    'description': limit.description,
                    'limit_type': limit.limit_type.value,
                    'scope': limit.scope.value,
                    'threshold_value': limit.threshold_value,
                    'enabled': limit.enabled
                }
                for limit in self.limits.values()
            ],
            'statistics': self.get_statistics()
        }
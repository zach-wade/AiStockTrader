# File: risk_management/pre_trade/unified_limit_checker/registry.py

"""
Registry for limit checkers.

Manages registration and lifecycle of specialized limit checker implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, Set
from datetime import datetime
import time
import uuid

from main.utils.core import get_logger, AsyncCircuitBreaker
from .types import LimitType, ViolationSeverity, ComparisonOperator
from .models import LimitDefinition, LimitCheckResult, LimitViolation
from .config import LimitConfig
from .events import EventManager, CheckEvent

logger = get_logger(__name__)


class LimitChecker(ABC):
    """
    Abstract base class for all limit checkers.
    
    Defines the interface that all checker implementations must follow.
    """
    
    def __init__(self, checker_id: str, config: LimitConfig):
        """
        Initialize base checker.
        
        Args:
            checker_id: Unique identifier for this checker
            config: Limit configuration
        """
        self.checker_id = checker_id
        self.config = config
        self.enabled = True
        self.logger = get_logger(f"{__name__}.{checker_id}")
        
        # Circuit breaker for resilience
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=Exception
        )
    
    @abstractmethod
    def supports_limit_type(self, limit_type: LimitType) -> bool:
        """Check if this checker supports the given limit type."""
        pass
    
    @abstractmethod
    async def check_limit(self, 
                         limit: LimitDefinition, 
                         current_value: float,
                         context: Dict[str, Any]) -> LimitCheckResult:
        """
        Check a limit against current value and context.
        
        Args:
            limit: The limit definition to check
            current_value: Current value to check against limit
            context: Additional context for the check
            
        Returns:
            LimitCheckResult with pass/fail status
        """
        pass
    
    @abstractmethod
    async def calculate_current_value(self,
                                    limit: LimitDefinition,
                                    context: Dict[str, Any]) -> float:
        """
        Calculate the current value for a limit check.
        
        Args:
            limit: The limit definition
            context: Context containing necessary data
            
        Returns:
            The calculated current value
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if this checker is enabled."""
        return self.enabled
    
    def enable(self):
        """Enable this checker."""
        self.enabled = True
        self.logger.info(f"Checker {self.checker_id} enabled")
    
    def disable(self):
        """Disable this checker."""
        self.enabled = False
        self.logger.info(f"Checker {self.checker_id} disabled")
    
    def get_info(self) -> Dict[str, Any]:
        """Get checker information."""
        return {
            'checker_id': self.checker_id,
            'enabled': self.enabled,
            'supported_types': [lt.value for lt in LimitType if self.supports_limit_type(lt)]
        }
    
    async def check_with_circuit_breaker(self,
                                       limit: LimitDefinition,
                                       current_value: float,
                                       context: Dict[str, Any]) -> LimitCheckResult:
        """Check limit with circuit breaker protection."""
        try:
            return await self.circuit_breaker.call(
                self.check_limit,
                limit,
                current_value,
                context
            )
        except Exception as e:
            self.logger.error(f"Circuit breaker open or check failed: {e}")
            # Return a failed check result
            return LimitCheckResult(
                limit_id=limit.limit_id,
                passed=False,
                current_value=current_value,
                threshold_value=limit.threshold_value,
                message=f"Check failed due to error: {str(e)}",
                warnings=[f"Circuit breaker triggered: {str(e)}"]
            )


class CheckerMetrics:
    """Tracks metrics for checker performance."""
    
    def __init__(self):
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0
        self.total_duration_ms = 0.0
        self.checks_by_type: Dict[LimitType, int] = {}
        self.violations_by_severity: Dict[ViolationSeverity, int] = {}
        
    def record_check(self, result: LimitCheckResult, duration_ms: float):
        """Record a check result."""
        self.total_checks += 1
        self.total_duration_ms += duration_ms
        
        if result.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1
            
            if result.violation:
                severity = result.violation.severity
                self.violations_by_severity[severity] = \
                    self.violations_by_severity.get(severity, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        avg_duration = self.total_duration_ms / self.total_checks if self.total_checks > 0 else 0
        
        return {
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'pass_rate': (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0,
            'avg_duration_ms': avg_duration,
            'checks_by_type': dict(self.checks_by_type),
            'violations_by_severity': {
                k.value: v for k, v in self.violations_by_severity.items()
            }
        }


class CheckerRegistry:
    """
    Registry for all limit checker implementations.
    
    Manages lifecycle and routing of limit checks to appropriate checkers.
    """
    
    def __init__(self, config: LimitConfig, event_manager: Optional[EventManager] = None):
        """
        Initialize checker registry.
        
        Args:
            config: Limit configuration
            event_manager: Optional event manager for notifications
        """
        self.config = config
        self.event_manager = event_manager
        self.checkers: Dict[str, LimitChecker] = {}
        self.type_to_checkers: Dict[LimitType, List[str]] = {}
        self.metrics: Dict[str, CheckerMetrics] = {}
        self._lock = asyncio.Lock()
        
        self.logger = get_logger(f"{__name__}.registry")
    
    async def register_checker(self, checker: LimitChecker):
        """
        Register a limit checker.
        
        Args:
            checker: The checker instance to register
        """
        async with self._lock:
            checker_id = checker.checker_id
            
            if checker_id in self.checkers:
                self.logger.warning(f"Replacing existing checker: {checker_id}")
            
            self.checkers[checker_id] = checker
            self.metrics[checker_id] = CheckerMetrics()
            
            # Update type mapping
            for limit_type in LimitType:
                if checker.supports_limit_type(limit_type):
                    if limit_type not in self.type_to_checkers:
                        self.type_to_checkers[limit_type] = []
                    if checker_id not in self.type_to_checkers[limit_type]:
                        self.type_to_checkers[limit_type].append(checker_id)
            
            self.logger.info(f"Registered checker: {checker_id}")
    
    async def unregister_checker(self, checker_id: str):
        """Unregister a checker."""
        async with self._lock:
            if checker_id not in self.checkers:
                self.logger.warning(f"Checker not found: {checker_id}")
                return
            
            # Remove from type mappings
            for limit_type, checker_ids in self.type_to_checkers.items():
                if checker_id in checker_ids:
                    checker_ids.remove(checker_id)
            
            # Remove checker
            del self.checkers[checker_id]
            del self.metrics[checker_id]
            
            self.logger.info(f"Unregistered checker: {checker_id}")
    
    def get_checker_for_limit(self, limit: LimitDefinition) -> Optional[LimitChecker]:
        """
        Get the best checker for a given limit.
        
        Args:
            limit: The limit definition
            
        Returns:
            The most appropriate checker or None
        """
        limit_type = limit.limit_type
        
        if limit_type not in self.type_to_checkers:
            return None
        
        checker_ids = self.type_to_checkers[limit_type]
        if not checker_ids:
            return None
        
        # Return first enabled checker
        for checker_id in checker_ids:
            checker = self.checkers.get(checker_id)
            if checker and checker.is_enabled():
                return checker
        
        return None
    
    async def check_limit(self,
                         limit: LimitDefinition,
                         context: Dict[str, Any]) -> Optional[LimitCheckResult]:
        """
        Check a limit using the appropriate checker.
        
        Args:
            limit: The limit to check
            context: Context for the check
            
        Returns:
            Check result or None if no checker available
        """
        checker = self.get_checker_for_limit(limit)
        if not checker:
            self.logger.warning(f"No checker found for limit type: {limit.limit_type.value}")
            return None
        
        start_time = time.time()
        
        try:
            # Calculate current value
            current_value = await checker.calculate_current_value(limit, context)
            
            # Perform check
            result = await checker.check_with_circuit_breaker(limit, current_value, context)
            
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            result.check_duration_ms = duration_ms
            
            metrics = self.metrics.get(checker.checker_id)
            if metrics:
                metrics.record_check(result, duration_ms)
            
            # Emit event if event manager available
            if self.event_manager:
                event = CheckEvent(
                    check_result=result,
                    check_duration_ms=duration_ms,
                    passed=result.passed,
                    limit_id=limit.limit_id,
                    limit_type=limit.limit_type,
                    context=context
                )
                await self.event_manager.emit(event)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking limit {limit.limit_id}: {e}")
            return None
    
    async def check_multiple_limits(self,
                                  limits: List[LimitDefinition],
                                  context: Dict[str, Any],
                                  parallel: bool = True) -> List[LimitCheckResult]:
        """
        Check multiple limits.
        
        Args:
            limits: List of limits to check
            context: Context for all checks
            parallel: Whether to check in parallel
            
        Returns:
            List of check results
        """
        if not limits:
            return []
        
        if parallel and self.config.parallel_checking:
            # Check limits in parallel
            tasks = [
                self.check_limit(limit, context)
                for limit in limits
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            valid_results = []
            for result in results:
                if isinstance(result, LimitCheckResult):
                    valid_results.append(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Error in parallel check: {result}")
            
            return valid_results
        else:
            # Check limits sequentially
            results = []
            for limit in limits:
                result = await self.check_limit(limit, context)
                if result:
                    results.append(result)
            
            return results
    
    def get_all_checkers(self) -> Dict[str, LimitChecker]:
        """Get all registered checkers."""
        return self.checkers.copy()
    
    def get_enabled_checkers(self) -> Dict[str, LimitChecker]:
        """Get only enabled checkers."""
        return {
            checker_id: checker
            for checker_id, checker in self.checkers.items()
            if checker.is_enabled()
        }
    
    def get_checker_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all checkers."""
        return {
            checker_id: metrics.get_metrics()
            for checker_id, metrics in self.metrics.items()
        }
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get status of the checker registry."""
        return {
            'total_checkers': len(self.checkers),
            'enabled_checkers': len(self.get_enabled_checkers()),
            'registered_checkers': list(self.checkers.keys()),
            'type_coverage': {
                lt.value: len(self.type_to_checkers.get(lt, []))
                for lt in LimitType
            },
            'metrics': self.get_checker_metrics()
        }
    
    async def enable_checker(self, checker_id: str):
        """Enable a specific checker."""
        if checker_id in self.checkers:
            self.checkers[checker_id].enable()
        else:
            self.logger.warning(f"Cannot enable checker: {checker_id} not found")
    
    async def disable_checker(self, checker_id: str):
        """Disable a specific checker."""
        if checker_id in self.checkers:
            self.checkers[checker_id].disable()
        else:
            self.logger.warning(f"Cannot disable checker: {checker_id} not found")
    
    async def shutdown(self):
        """Shutdown all checkers gracefully."""
        async with self._lock:
            for checker_id, checker in self.checkers.items():
                try:
                    # If checker has cleanup method, call it
                    if hasattr(checker, 'cleanup'):
                        await checker.cleanup()
                    self.logger.info(f"Shutdown checker: {checker_id}")
                except Exception as e:
                    self.logger.error(f"Error shutting down checker {checker_id}: {e}")
            
            self.checkers.clear()
            self.type_to_checkers.clear()
            self.metrics.clear()
            
            self.logger.info("All checkers shutdown complete")


class SimpleThresholdChecker(LimitChecker):
    """
    Simple threshold-based limit checker.
    
    Handles basic comparison operations for any limit type.
    """
    
    def __init__(self, config: LimitConfig):
        """Initialize simple threshold checker."""
        super().__init__("simple_threshold", config)
        self.supported_types = set(LimitType)  # Supports all types
    
    def supports_limit_type(self, limit_type: LimitType) -> bool:
        """Check if this checker supports the given limit type."""
        return True  # Supports all types
    
    async def calculate_current_value(self,
                                    limit: LimitDefinition,
                                    context: Dict[str, Any]) -> float:
        """
        Calculate current value from context.
        
        For simple checker, expects 'current_value' in context.
        """
        if 'current_value' not in context:
            raise ValueError("'current_value' not found in context")
        
        return float(context['current_value'])
    
    async def check_limit(self,
                         limit: LimitDefinition,
                         current_value: float,
                         context: Dict[str, Any]) -> LimitCheckResult:
        """Perform simple threshold check."""
        start_time = time.time()
        
        # Get effective threshold
        effective_threshold = limit.get_effective_threshold(context)
        
        # Perform comparison
        passed = self._compare_values(
            current_value,
            effective_threshold,
            limit.operator,
            limit.upper_threshold
        )
        
        # Check soft threshold if applicable
        severity = None
        if not passed:
            severity = ViolationSeverity.HARD_BREACH
        elif limit.soft_threshold is not None:
            soft_passed = self._compare_values(
                current_value,
                limit.soft_threshold,
                limit.operator,
                limit.upper_threshold
            )
            if not soft_passed:
                severity = ViolationSeverity.SOFT_BREACH
        
        # Create result
        result = LimitCheckResult(
            limit_id=limit.limit_id,
            passed=passed and severity != ViolationSeverity.SOFT_BREACH,
            current_value=current_value,
            threshold_value=effective_threshold,
            context=context,
            check_duration_ms=(time.time() - start_time) * 1000
        )
        
        # Add violation if failed
        if severity:
            result.violation = LimitViolation(
                violation_id=str(uuid.uuid4()),
                limit_id=limit.limit_id,
                limit_name=limit.name,
                current_value=current_value,
                threshold_value=limit.threshold_value,
                effective_threshold=effective_threshold,
                severity=severity,
                operator=limit.operator,
                scope_context=context,
                recommended_action=limit.violation_action if severity == ViolationSeverity.HARD_BREACH else limit.soft_violation_action
            )
            
            result.message = f"Limit {limit.name} violated: {current_value} {limit.operator.value} {effective_threshold}"
        else:
            result.message = f"Limit {limit.name} passed: {current_value} {limit.operator.value} {effective_threshold}"
        
        return result
    
    def _compare_values(self,
                       current: float,
                       threshold: float,
                       operator: ComparisonOperator,
                       upper_threshold: Optional[float] = None) -> bool:
        """Perform comparison based on operator."""
        if operator == ComparisonOperator.LESS_THAN:
            return current < threshold
        elif operator == ComparisonOperator.LESS_EQUAL:
            return current <= threshold
        elif operator == ComparisonOperator.GREATER_THAN:
            return current > threshold
        elif operator == ComparisonOperator.GREATER_EQUAL:
            return current >= threshold
        elif operator == ComparisonOperator.EQUAL:
            return current == threshold
        elif operator == ComparisonOperator.NOT_EQUAL:
            return current != threshold
        elif operator == ComparisonOperator.BETWEEN:
            if upper_threshold is None:
                raise ValueError("BETWEEN operator requires upper_threshold")
            return threshold <= current <= upper_threshold
        elif operator == ComparisonOperator.OUTSIDE:
            if upper_threshold is None:
                raise ValueError("OUTSIDE operator requires upper_threshold")
            return current < threshold or current > upper_threshold
        else:
            raise ValueError(f"Unknown operator: {operator}")


def create_default_registry(config: LimitConfig, event_manager: Optional[EventManager] = None) -> CheckerRegistry:
    """
    Create a registry with default checkers.
    
    Args:
        config: Limit configuration
        event_manager: Optional event manager
        
    Returns:
        Configured CheckerRegistry
    """
    registry = CheckerRegistry(config, event_manager)
    
    # Register default checkers
    simple_checker = SimpleThresholdChecker(config)
    asyncio.create_task(registry.register_checker(simple_checker))
    
    return registry
"""
Integration with Existing Resilience Infrastructure

Integrates monitoring components with the existing resilience infrastructure:
- Circuit breaker monitoring
- Retry operation tracing
- Fallback mechanism observability
- Health check coordination
- Error handling integration
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ..resilience.health import HealthCheck, HealthCheckResult, HealthStatus

if TYPE_CHECKING:
    from ..resilience.circuit_breaker import CircuitBreaker, CircuitState
    from ..resilience.fallback import FallbackStrategy
    from ..resilience.retry import RetryConfig
from ..resilience.error_handling import ErrorHandler, ErrorSeverity
from .health import TradingHealthChecker, create_trading_health_checker
from .logging import get_correlation_id, get_trading_logger
from .metrics import get_trading_metrics
from .performance import get_performance_monitor
from .telemetry import (
    add_trading_attributes,
    get_current_span,
    trace_trading_operation,
    trading_tracer,
)

logger = logging.getLogger(__name__)


@dataclass
class ResilienceEvent:
    """Event from resilience infrastructure."""

    timestamp: float
    component: str  # circuit_breaker, retry, fallback, error_handler
    event_type: str  # state_change, operation_start, operation_end, error
    operation: str
    details: dict[str, Any]
    correlation_id: str | None = None


class CircuitBreakerMonitor:
    """Monitor circuit breaker operations."""

    def __init__(self) -> None:
        self.tracer = trading_tracer()
        self.logger = get_trading_logger(__name__)
        self.metrics = get_trading_metrics()

        # Create custom metrics for circuit breaker
        self.state_changes = self.metrics.create_counter(
            "circuit_breaker_state_changes_total", "Total circuit breaker state changes"
        )

        self.operations_total = self.metrics.create_counter(
            "circuit_breaker_operations_total", "Total operations through circuit breaker"
        )

        self.failures_total = self.metrics.create_counter(
            "circuit_breaker_failures_total", "Total circuit breaker failures"
        )

        self.state_duration = self.metrics.create_histogram(
            "circuit_breaker_state_duration_seconds", "Time spent in each circuit breaker state"
        )

    def instrument_circuit_breaker(self, circuit_breaker: "CircuitBreaker") -> "CircuitBreaker":
        """Instrument circuit breaker with monitoring."""
        # Store original methods
        original_call = circuit_breaker.call
        original_on_state_change = getattr(circuit_breaker, "_on_state_change", None)

        def monitored_call(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            """Monitored circuit breaker call."""
            operation_name = getattr(operation, "__name__", "unknown")

            with self.tracer.start_as_current_span(f"circuit_breaker.{operation_name}") as span:
                add_trading_attributes(
                    span,
                    operation_type="circuit_breaker_operation",
                    circuit_breaker_name=circuit_breaker.name,
                    current_state=circuit_breaker.state.value,
                )

                start_time = time.perf_counter()

                try:
                    # Record operation attempt
                    self.operations_total.increment(
                        1.0,
                        {
                            "circuit_breaker": circuit_breaker.name,
                            "operation": operation_name,
                            "state": circuit_breaker.state.value,
                        },
                    )

                    # Call original method
                    result = original_call(operation, *args, **kwargs)

                    # Record success
                    span.set_attribute("circuit_breaker.result", "success")

                    logger.info(
                        f"Circuit breaker operation succeeded: {operation_name}",
                        extra={
                            "operation_type": "circuit_breaker_operation",
                            "circuit_breaker_name": circuit_breaker.name,
                            "state": circuit_breaker.state.value,
                            "duration_ms": (time.perf_counter() - start_time) * 1000,
                        },
                    )

                    return result

                except Exception as e:
                    # Record failure
                    self.failures_total.increment(
                        1.0,
                        {
                            "circuit_breaker": circuit_breaker.name,
                            "operation": operation_name,
                            "error_type": type(e).__name__,
                        },
                    )

                    span.set_attribute("circuit_breaker.result", "failure")
                    span.set_attribute("circuit_breaker.error_type", type(e).__name__)
                    span.record_exception(e)

                    logger.error(
                        f"Circuit breaker operation failed: {operation_name} - {e}",
                        extra={
                            "operation_type": "circuit_breaker_operation",
                            "circuit_breaker_name": circuit_breaker.name,
                            "state": circuit_breaker.state.value,
                            "error_type": type(e).__name__,
                            "duration_ms": (time.perf_counter() - start_time) * 1000,
                        },
                    )

                    raise

        def monitored_state_change(old_state: "CircuitState", new_state: "CircuitState") -> None:
            """Monitor state changes."""
            self.state_changes.increment(
                1.0,
                {
                    "circuit_breaker": circuit_breaker.name,
                    "from_state": old_state.value,
                    "to_state": new_state.value,
                },
            )

            logger.warning(
                f"Circuit breaker state change: {old_state.value} -> {new_state.value}",
                extra={
                    "operation_type": "circuit_breaker_state_change",
                    "circuit_breaker_name": circuit_breaker.name,
                    "from_state": old_state.value,
                    "to_state": new_state.value,
                },
            )

            # Call original handler if it exists
            if original_on_state_change:
                original_on_state_change(old_state, new_state)

        # Replace methods
        circuit_breaker.call = monitored_call
        circuit_breaker.on_state_change = monitored_state_change

        return circuit_breaker


class RetryMonitor:
    """Monitor retry operations."""

    def __init__(self) -> None:
        self.tracer = trading_tracer()
        self.logger = get_trading_logger(__name__)
        self.metrics = get_trading_metrics()

        # Create custom metrics for retry operations
        self.retry_attempts = self.metrics.create_counter(
            "retry_attempts_total", "Total retry attempts"
        )

        self.retry_successes = self.metrics.create_counter(
            "retry_successes_total", "Total successful retries"
        )

        self.retry_failures = self.metrics.create_counter(
            "retry_failures_total", "Total failed retries after all attempts"
        )

        self.retry_duration = self.metrics.create_histogram(
            "retry_operation_duration_seconds", "Duration of retry operations"
        )

    @trace_trading_operation(operation_name="retry_operation")
    def monitor_retry_operation(
        self,
        operation: Callable[..., Any],
        retry_policy: "RetryConfig",
        operation_name: str | None = None,
        **operation_kwargs: Any,
    ) -> Any:
        """Monitor a retry operation."""
        operation_name = operation_name or getattr(operation, "__name__", "unknown")
        assert operation_name is not None  # Type narrowing for mypy

        span = get_current_span()
        add_trading_attributes(
            span,
            operation_type="retry_operation",
            max_attempts=retry_policy.max_retries,
            initial_delay=retry_policy.initial_delay,
        )

        start_time = time.perf_counter()
        attempt = 0
        last_exception = None

        while attempt < retry_policy.max_retries:
            attempt += 1

            # Record attempt
            self.retry_attempts.increment(
                1.0, {"operation": operation_name, "attempt": str(attempt)}
            )

            try:
                with self.tracer.start_as_current_span(f"retry_attempt_{attempt}") as attempt_span:
                    attempt_span.set_attribute("retry.attempt_number", attempt)

                    result = operation(**operation_kwargs)

                    # Success - record and return
                    duration = time.perf_counter() - start_time

                    self.retry_successes.increment(
                        1.0, {"operation": operation_name, "attempts_used": str(attempt)}
                    )

                    self.retry_duration.observe(
                        duration, {"operation": operation_name, "result": "success"}
                    )

                    span.set_attribute("retry.final_attempt", attempt)
                    span.set_attribute("retry.result", "success")

                    logger.info(
                        f"Retry operation succeeded after {attempt} attempts: {operation_name}",
                        extra={
                            "operation_type": "retry_success",
                            "operation_name": operation_name,
                            "attempts_used": attempt,
                            "duration_ms": duration * 1000,
                        },
                    )

                    return result

            except Exception as e:
                last_exception = e

                with self.tracer.start_as_current_span("retry_attempt_failed") as fail_span:
                    fail_span.set_attribute("retry.attempt_number", attempt)
                    fail_span.set_attribute("retry.error_type", type(e).__name__)
                    fail_span.record_exception(e)

                logger.warning(
                    f"Retry attempt {attempt} failed: {operation_name} - {e}",
                    extra={
                        "operation_type": "retry_attempt_failed",
                        "operation_name": operation_name,
                        "attempt_number": attempt,
                        "error_type": type(e).__name__,
                    },
                )

                # Check if we should retry
                from ..resilience.retry import ExponentialBackoff, is_retryable_exception

                if attempt < retry_policy.max_retries and is_retryable_exception(e, retry_policy):
                    backoff = ExponentialBackoff(retry_policy)
                    delay = backoff.get_delay(attempt - 1)  # 0-based
                    time.sleep(delay)
                else:
                    break

        # All attempts failed
        duration = time.perf_counter() - start_time

        self.retry_failures.increment(
            1.0, {"operation": operation_name, "attempts_exhausted": str(retry_policy.max_retries)}
        )

        self.retry_duration.observe(duration, {"operation": operation_name, "result": "failure"})

        span.set_attribute("retry.final_attempt", attempt)
        span.set_attribute("retry.result", "failure")
        if last_exception:
            span.set_attribute("retry.final_error_type", type(last_exception).__name__)
            span.record_exception(last_exception)

        logger.error(
            f"Retry operation failed after {attempt} attempts: {operation_name}",
            extra={
                "operation_type": "retry_exhausted",
                "operation_name": operation_name,
                "attempts_used": attempt,
                "duration_ms": duration * 1000,
                "final_error": str(last_exception),
            },
        )

        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Retry operation failed after {attempt} attempts")


class FallbackMonitor:
    """Monitor fallback operations."""

    def __init__(self) -> None:
        self.tracer = trading_tracer()
        self.logger = get_trading_logger(__name__)
        self.metrics = get_trading_metrics()

        # Create custom metrics for fallback operations
        self.fallback_triggers = self.metrics.create_counter(
            "fallback_triggers_total", "Total fallback triggers"
        )

        self.fallback_successes = self.metrics.create_counter(
            "fallback_successes_total", "Total successful fallback operations"
        )

        self.fallback_failures = self.metrics.create_counter(
            "fallback_failures_total", "Total failed fallback operations"
        )

    @trace_trading_operation(operation_name="fallback_operation")
    def monitor_fallback(
        self,
        primary_operation: Callable[..., Any],
        fallback_strategy: "FallbackStrategy[Any]",
        operation_name: str | None = None,
        **operation_kwargs: Any,
    ) -> Any:
        """Monitor fallback operation."""
        operation_name = operation_name or getattr(primary_operation, "__name__", "unknown")
        assert operation_name is not None  # Type narrowing for mypy

        span = get_current_span()
        add_trading_attributes(
            span, operation_type="fallback_operation", primary_operation=operation_name
        )

        try:
            # Try primary operation first
            with self.tracer.start_as_current_span("primary_operation") as primary_span:
                result = primary_operation(**operation_kwargs)

                primary_span.set_attribute("fallback.used", False)
                span.set_attribute("fallback.result", "primary_success")

                logger.debug(
                    f"Primary operation succeeded: {operation_name}",
                    extra={
                        "operation_type": "primary_operation_success",
                        "operation_name": operation_name,
                    },
                )

                return result

        except Exception as primary_error:
            # Primary failed, trigger fallback
            self.fallback_triggers.increment(
                1.0,
                {"operation": operation_name, "primary_error_type": type(primary_error).__name__},
            )

            logger.warning(
                f"Primary operation failed, triggering fallback: {operation_name} - {primary_error}",
                extra={
                    "operation_type": "fallback_triggered",
                    "operation_name": operation_name,
                    "primary_error_type": type(primary_error).__name__,
                },
            )

            try:
                with self.tracer.start_as_current_span("fallback_operation") as fallback_span:
                    fallback_result = fallback_strategy.execute(  # type: ignore[attr-defined]
                        primary_error, operation_name, **operation_kwargs
                    )

                    fallback_span.set_attribute("fallback.used", True)
                    fallback_span.set_attribute(
                        "fallback.strategy", fallback_strategy.__class__.__name__
                    )
                    span.set_attribute("fallback.result", "fallback_success")

                    self.fallback_successes.increment(
                        1.0,
                        {
                            "operation": operation_name,
                            "strategy": fallback_strategy.__class__.__name__,
                        },
                    )

                    logger.info(
                        f"Fallback operation succeeded: {operation_name}",
                        extra={
                            "operation_type": "fallback_success",
                            "operation_name": operation_name,
                            "fallback_strategy": fallback_strategy.__class__.__name__,
                        },
                    )

                    return fallback_result

            except Exception as fallback_error:
                self.fallback_failures.increment(
                    1.0,
                    {
                        "operation": operation_name,
                        "strategy": fallback_strategy.__class__.__name__,
                        "fallback_error_type": type(fallback_error).__name__,
                    },
                )

                span.set_attribute("fallback.result", "fallback_failure")
                span.set_attribute("fallback.error_type", type(fallback_error).__name__)
                span.record_exception(fallback_error)

                logger.error(
                    f"Fallback operation also failed: {operation_name} - {fallback_error}",
                    extra={
                        "operation_type": "fallback_failure",
                        "operation_name": operation_name,
                        "fallback_strategy": fallback_strategy.__class__.__name__,
                        "fallback_error_type": type(fallback_error).__name__,
                    },
                )

                # Re-raise the fallback error
                raise fallback_error


class ErrorHandlingMonitor:
    """Monitor error handling operations."""

    def __init__(self) -> None:
        self.logger = get_trading_logger(__name__)
        self.metrics = get_trading_metrics()

        # Create custom metrics for error handling
        self.errors_handled = self.metrics.create_counter(
            "errors_handled_total", "Total errors handled by error handler"
        )

        self.error_severity = self.metrics.create_counter(
            "errors_by_severity_total", "Total errors by severity level"
        )

    def instrument_error_handler(self, error_handler: "ErrorHandler") -> "ErrorHandler":
        """Instrument error handler with monitoring."""
        original_handle_error = error_handler.handle_error  # type: ignore[attr-defined]

        def monitored_handle_error(error: Exception, context: dict[str, Any] | None = None) -> Any:
            """Monitor error handling."""
            context = context or {}

            # Determine severity
            severity = error_handler.classify_error(error)  # type: ignore[attr-defined]

            # Record metrics
            self.errors_handled.increment(
                1.0,
                {
                    "error_type": type(error).__name__,
                    "severity": severity.value,
                    "operation": context.get("operation", "unknown"),
                },
            )

            self.error_severity.increment(1.0, {"severity": severity.value})

            # Log the error handling
            log_level = self._severity_to_log_level(severity)
            logger.log(
                log_level,
                f"Error handled: {type(error).__name__} - {error}",
                extra={
                    "operation_type": "error_handling",
                    "error_type": type(error).__name__,
                    "severity": severity.value,
                    "context": context,
                },
            )

            # Call original handler
            return original_handle_error(error, context)

        error_handler.handle_error = monitored_handle_error  # type: ignore[attr-defined]
        return error_handler

    def _severity_to_log_level(self, severity: "ErrorSeverity") -> int:
        """Convert error severity to log level."""
        severity_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        return severity_map.get(severity, logging.ERROR)


class ResilienceIntegration:
    """
    Main integration class for resilience infrastructure monitoring.

    Coordinates monitoring across all resilience components and provides
    unified observability for the trading system's resilience patterns.
    """

    def __init__(self) -> None:
        self.logger = get_trading_logger(__name__)
        self.metrics = get_trading_metrics()
        self.performance_monitor = get_performance_monitor()

        # Component monitors
        self.circuit_breaker_monitor = CircuitBreakerMonitor()
        self.retry_monitor = RetryMonitor()
        self.fallback_monitor = FallbackMonitor()
        self.error_handling_monitor = ErrorHandlingMonitor()

        # Integrated health checker
        self.health_checker: TradingHealthChecker | None = None

        # Event tracking
        self.resilience_events: list[ResilienceEvent] = []

        logger.info("Resilience integration initialized")

    def setup_integrated_health_checking(
        self,
        database_connection_factory: Callable[..., Any] | None = None,
        broker_client: Any | None = None,
        market_data_client: Any | None = None,
        **health_checker_kwargs: Any,
    ) -> "TradingHealthChecker":
        """Setup integrated health checking with resilience awareness."""

        self.health_checker = create_trading_health_checker(
            database_connection_factory=database_connection_factory,
            broker_client=broker_client,
            market_data_client=market_data_client,
            **health_checker_kwargs,
        )

        # Add resilience-specific health checks
        self._add_resilience_health_checks()

        return self.health_checker

    def _add_resilience_health_checks(self) -> None:
        """Add health checks for resilience components."""
        if not self.health_checker:
            return

        # Create custom health checks for resilience monitoring
        class ResilienceMetricsHealthCheck(HealthCheck):
            def __init__(self, integration: "ResilienceIntegration") -> None:
                super().__init__("resilience_metrics")
                self.integration = integration

            async def check_health(self) -> HealthCheckResult:
                start_time = time.time()

                try:
                    # Check recent error rates
                    metrics = self.integration.get_resilience_summary()

                    # Determine health based on error rates
                    circuit_breaker_failures = metrics.get("circuit_breaker_failures_1h", 0)
                    retry_failures = metrics.get("retry_failures_1h", 0)
                    fallback_triggers = metrics.get("fallback_triggers_1h", 0)

                    if circuit_breaker_failures > 10 or retry_failures > 20:
                        status = HealthStatus.DEGRADED
                    elif circuit_breaker_failures > 20 or retry_failures > 50:
                        status = HealthStatus.UNHEALTHY
                    else:
                        status = HealthStatus.HEALTHY

                    return HealthCheckResult(
                        service_name=self.name,
                        status=status,
                        response_time=time.time() - start_time,
                        timestamp=start_time,
                        details=metrics,
                    )

                except Exception as e:
                    return HealthCheckResult(
                        service_name=self.name,
                        status=HealthStatus.UNHEALTHY,
                        response_time=time.time() - start_time,
                        timestamp=start_time,
                        error=str(e),
                    )

        # Register the resilience health check
        resilience_health_check = ResilienceMetricsHealthCheck(self)
        self.health_checker.register_health_check(resilience_health_check)

    def instrument_circuit_breaker(self, circuit_breaker: "CircuitBreaker") -> "CircuitBreaker":
        """Instrument circuit breaker with full monitoring."""
        return self.circuit_breaker_monitor.instrument_circuit_breaker(circuit_breaker)

    def monitor_retry_operation(
        self, operation: Callable[..., Any], retry_policy: "RetryConfig", **kwargs: Any
    ) -> Any:
        """Monitor retry operation with full observability."""
        return self.retry_monitor.monitor_retry_operation(operation, retry_policy, **kwargs)

    def monitor_fallback(
        self,
        primary_operation: Callable[..., Any],
        fallback_strategy: "FallbackStrategy[Any]",
        **kwargs: Any,
    ) -> Any:
        """Monitor fallback operation with full observability."""
        return self.fallback_monitor.monitor_fallback(
            primary_operation, fallback_strategy, **kwargs
        )

    def instrument_error_handler(self, error_handler: "ErrorHandler") -> "ErrorHandler":
        """Instrument error handler with monitoring."""
        return self.error_handling_monitor.instrument_error_handler(error_handler)

    def record_resilience_event(
        self, component: str, event_type: str, operation: str, details: dict[str, Any]
    ) -> None:
        """Record resilience event for analysis."""
        event = ResilienceEvent(
            timestamp=time.time(),
            component=component,
            event_type=event_type,
            operation=operation,
            details=details,
            correlation_id=get_correlation_id(),
        )

        self.resilience_events.append(event)

        # Keep only last 1000 events
        if len(self.resilience_events) > 1000:
            self.resilience_events = self.resilience_events[-1000:]

    def get_resilience_summary(self, time_window_hours: float = 1.0) -> dict[str, Any]:
        """Get comprehensive resilience summary."""
        cutoff_time = time.time() - (time_window_hours * 3600)

        recent_events = [event for event in self.resilience_events if event.timestamp > cutoff_time]

        summary: dict[str, Any] = {
            "time_window_hours": time_window_hours,
            "total_events": len(recent_events),
            "events_by_component": {},
            "events_by_type": {},
            "circuit_breaker_failures_1h": 0,
            "retry_failures_1h": 0,
            "fallback_triggers_1h": 0,
            "error_handling_events_1h": 0,
        }

        # Analyze events
        for event in recent_events:
            # Count by component
            component = event.component
            events_by_component = summary["events_by_component"]
            if isinstance(events_by_component, dict):
                events_by_component[component] = events_by_component.get(component, 0) + 1

            # Count by type
            event_type = event.event_type
            events_by_type = summary["events_by_type"]
            if isinstance(events_by_type, dict):
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1

            # Specific counters
            if component == "circuit_breaker" and event_type == "failure":
                summary["circuit_breaker_failures_1h"] = summary["circuit_breaker_failures_1h"] + 1
            elif component == "retry" and event_type == "exhausted":
                summary["retry_failures_1h"] = summary["retry_failures_1h"] + 1
            elif component == "fallback" and event_type == "triggered":
                summary["fallback_triggers_1h"] = summary["fallback_triggers_1h"] + 1
            elif component == "error_handler":
                summary["error_handling_events_1h"] = summary["error_handling_events_1h"] + 1

        # Add current metrics
        try:
            current_metrics = self.metrics.collect_all_metrics()
            summary["current_metrics"] = {
                k: v
                for k, v in current_metrics.items()
                if any(
                    pattern in k for pattern in ["circuit_breaker", "retry", "fallback", "error"]
                )
            }
        except Exception as e:
            summary["metrics_error"] = str(e)

        return summary

    async def start_integrated_monitoring(self) -> None:
        """Start all integrated monitoring components."""
        tasks = []

        # Start health checking
        if self.health_checker:
            tasks.append(self.health_checker.start_monitoring())

        # Start metrics collection
        tasks.append(self.metrics.start_background_collection())

        # Start performance monitoring
        tasks.append(self.performance_monitor.start_background_monitoring())

        # Execute all startup tasks
        if tasks:
            await asyncio.gather(*tasks)

        self.logger.info("Integrated monitoring started")

    async def stop_integrated_monitoring(self) -> None:
        """Stop all integrated monitoring components."""
        tasks = []

        # Stop health checking
        if self.health_checker:
            tasks.append(self.health_checker.stop_monitoring())

        # Stop metrics collection
        tasks.append(self.metrics.stop_background_collection())

        # Stop performance monitoring
        tasks.append(self.performance_monitor.stop_background_monitoring())

        # Execute all shutdown tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("Integrated monitoring stopped")

    def generate_resilience_report(self) -> dict[str, Any]:
        """Generate comprehensive resilience monitoring report."""
        return {
            "report_timestamp": time.time(),
            "resilience_summary": self.get_resilience_summary(time_window_hours=24),
            "health_status": (
                self.health_checker.get_health_summary() if self.health_checker else None
            ),
            "performance_metrics": (
                self.performance_monitor.get_bottlenecks() if self.performance_monitor else []
            ),
            "recommendations": self._generate_resilience_recommendations(),
        }

    def _generate_resilience_recommendations(self) -> list[str]:
        """Generate recommendations based on resilience monitoring data."""
        recommendations = []

        summary = self.get_resilience_summary(time_window_hours=24)

        # Check circuit breaker failures
        cb_failures = summary.get("circuit_breaker_failures_1h", 0)
        if cb_failures > 5:
            recommendations.append(
                f"High circuit breaker failure rate ({cb_failures}/hour). "
                "Consider reviewing service dependencies and increasing timeouts."
            )

        # Check retry failures
        retry_failures = summary.get("retry_failures_1h", 0)
        if retry_failures > 10:
            recommendations.append(
                f"High retry failure rate ({retry_failures}/hour). "
                "Consider reviewing retry policies and identifying root causes."
            )

        # Check fallback usage
        fallback_triggers = summary.get("fallback_triggers_1h", 0)
        if fallback_triggers > 20:
            recommendations.append(
                f"High fallback trigger rate ({fallback_triggers}/hour). "
                "Primary operations may be unreliable. Consider system health review."
            )

        # Check error handling volume
        error_events = summary.get("error_handling_events_1h", 0)
        if error_events > 50:
            recommendations.append(
                f"High error rate ({error_events}/hour). "
                "Consider investigating underlying causes and improving error prevention."
            )

        if not recommendations:
            recommendations.append("Resilience metrics are within normal ranges.")

        return recommendations


# Global resilience integration instance
_resilience_integration: ResilienceIntegration | None = None


def initialize_resilience_integration() -> ResilienceIntegration:
    """Initialize global resilience integration."""
    global _resilience_integration
    _resilience_integration = ResilienceIntegration()
    return _resilience_integration


def get_resilience_integration() -> ResilienceIntegration:
    """Get global resilience integration instance."""
    if not _resilience_integration:
        raise RuntimeError(
            "Resilience integration not initialized. Call initialize_resilience_integration() first."
        )
    return _resilience_integration


# Convenience decorator for integrated monitoring
def monitor_with_resilience(
    circuit_breaker: Optional["CircuitBreaker"] = None,
    retry_policy: Optional["RetryConfig"] = None,
    fallback_strategy: Optional["FallbackStrategy[Any]"] = None,
    error_handler: Optional["ErrorHandler"] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that applies resilience patterns with full monitoring."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @trace_trading_operation()
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            integration = get_resilience_integration()

            # Build the resilient operation
            operation = func

            # Apply error handling
            if error_handler:
                instrumented_handler = integration.instrument_error_handler(error_handler)

                def error_handled_operation(*args: Any, **kwargs: Any) -> Any:
                    try:
                        return operation(*args, **kwargs)
                    except Exception as e:
                        instrumented_handler.handle_error(
                            e,
                            {  # type: ignore[attr-defined]
                                "operation": func.__name__,
                                "args": str(args)[:100],  # Truncate for logging
                                "kwargs": str(kwargs)[:100],
                            },
                        )
                        raise

                operation = error_handled_operation

            # Apply fallback
            if fallback_strategy:

                def fallback_operation(*args: Any, **kwargs: Any) -> Any:
                    return integration.monitor_fallback(
                        operation, fallback_strategy, operation_name=func.__name__, *args, **kwargs
                    )

                operation = fallback_operation

            # Apply retry
            if retry_policy:

                def retry_operation(*args: Any, **kwargs: Any) -> Any:
                    return integration.monitor_retry_operation(
                        operation, retry_policy, operation_name=func.__name__, *args, **kwargs
                    )

                operation = retry_operation

            # Apply circuit breaker
            if circuit_breaker:
                instrumented_cb = integration.instrument_circuit_breaker(circuit_breaker)

                def circuit_breaker_operation(*args: Any, **kwargs: Any) -> Any:
                    return instrumented_cb.call(lambda: operation(*args, **kwargs))

                operation = circuit_breaker_operation

            return operation(*args, **kwargs)

        return wrapper  # type: ignore[no-any-return]

    return decorator

"""
Resilience Policy Service - Domain Layer

This service contains all business logic for system resilience policies,
including retry strategies, circuit breaker configurations, and error handling rules.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCategory(Enum):
    """Categories of errors for handling decisions."""

    TRANSIENT = "transient"  # Temporary, can retry
    CLIENT_ERROR = "client_error"  # Bad request, don't retry
    SERVER_ERROR = "server_error"  # Server issue, might retry
    NETWORK_ERROR = "network_error"  # Network issue, should retry
    TIMEOUT = "timeout"  # Operation timeout, might retry
    RATE_LIMIT = "rate_limit"  # Rate limited, retry with backoff
    BUSINESS_ERROR = "business_error"  # Business rule violation, don't retry
    CRITICAL = "critical"  # Critical error, alert and don't retry


class ServicePriority(Enum):
    """Service priority levels for resilience decisions."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RetryPolicy:
    """Retry policy configuration."""

    max_attempts: int
    initial_delay: float  # seconds
    max_delay: float  # seconds
    exponential_base: float
    jitter: bool
    retryable_errors: list[ErrorCategory]


@dataclass
class CircuitBreakerPolicy:
    """Circuit breaker configuration."""

    failure_threshold: int  # Number of failures to open circuit
    success_threshold: int  # Number of successes to close circuit
    timeout: float  # Seconds before attempting to close
    half_open_max_calls: int  # Max calls in half-open state
    error_categories: list[ErrorCategory]  # Errors that count as failures


@dataclass
class FallbackStrategy:
    """Fallback strategy for failures."""

    strategy_type: str  # 'cache', 'default', 'degrade', 'queue'
    cache_ttl: float | None = None
    default_value: Any | None = None
    degraded_service: str | None = None
    queue_name: str | None = None


class ResiliencePolicyService:
    """
    Domain service for resilience policies.

    Contains all business logic for determining retry strategies,
    circuit breaker configurations, and error handling rules.
    """

    # Service-specific retry policies
    SERVICE_RETRY_POLICIES = {
        "market_data": RetryPolicy(
            max_attempts=5,
            initial_delay=0.1,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True,
            retryable_errors=[
                ErrorCategory.TRANSIENT,
                ErrorCategory.NETWORK_ERROR,
                ErrorCategory.TIMEOUT,
            ],
        ),
        "order_execution": RetryPolicy(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=5.0,
            exponential_base=1.5,
            jitter=False,
            retryable_errors=[ErrorCategory.TRANSIENT, ErrorCategory.TIMEOUT],
        ),
        "database": RetryPolicy(
            max_attempts=3,
            initial_delay=0.2,
            max_delay=2.0,
            exponential_base=2.0,
            jitter=True,
            retryable_errors=[ErrorCategory.TRANSIENT, ErrorCategory.NETWORK_ERROR],
        ),
        "external_api": RetryPolicy(
            max_attempts=4,
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
            retryable_errors=[
                ErrorCategory.TRANSIENT,
                ErrorCategory.SERVER_ERROR,
                ErrorCategory.RATE_LIMIT,
            ],
        ),
    }

    # Circuit breaker configurations by service
    CIRCUIT_BREAKER_POLICIES = {
        "market_data": CircuitBreakerPolicy(
            failure_threshold=5,
            success_threshold=2,
            timeout=30.0,
            half_open_max_calls=3,
            error_categories=[
                ErrorCategory.SERVER_ERROR,
                ErrorCategory.CRITICAL,
                ErrorCategory.TIMEOUT,
            ],
        ),
        "order_execution": CircuitBreakerPolicy(
            failure_threshold=3,
            success_threshold=3,
            timeout=60.0,
            half_open_max_calls=1,
            error_categories=[ErrorCategory.CRITICAL, ErrorCategory.SERVER_ERROR],
        ),
        "database": CircuitBreakerPolicy(
            failure_threshold=10,
            success_threshold=5,
            timeout=20.0,
            half_open_max_calls=5,
            error_categories=[ErrorCategory.CRITICAL, ErrorCategory.SERVER_ERROR],
        ),
    }

    # Error type mapping
    ERROR_MAPPINGS = {
        "ConnectionError": ErrorCategory.NETWORK_ERROR,
        "TimeoutError": ErrorCategory.TIMEOUT,
        "RateLimitError": ErrorCategory.RATE_LIMIT,
        "ValidationError": ErrorCategory.CLIENT_ERROR,
        "BusinessRuleViolation": ErrorCategory.BUSINESS_ERROR,
        "InternalServerError": ErrorCategory.SERVER_ERROR,
        "ServiceUnavailable": ErrorCategory.TRANSIENT,
        "DatabaseError": ErrorCategory.SERVER_ERROR,
        "AuthenticationError": ErrorCategory.CLIENT_ERROR,
        "AuthorizationError": ErrorCategory.CLIENT_ERROR,
        "ResourceNotFound": ErrorCategory.CLIENT_ERROR,
        "DuplicateEntry": ErrorCategory.BUSINESS_ERROR,
        "InsufficientFunds": ErrorCategory.BUSINESS_ERROR,
        "MarketClosed": ErrorCategory.BUSINESS_ERROR,
    }

    def calculate_retry_delay(
        self, attempt: int, error_type: str, service: str = "default"
    ) -> float:
        """
        Calculate retry delay based on business rules.

        Args:
            attempt: Current attempt number (1-based)
            error_type: Type of error that occurred
            service: Service name for specific policies

        Returns:
            Delay in seconds before next retry
        """
        policy = self.SERVICE_RETRY_POLICIES.get(service, self._get_default_retry_policy())

        if attempt >= policy.max_attempts:
            return -1  # No more retries

        # Check if error is retryable
        error_category = self.categorize_error(error_type)
        if error_category not in policy.retryable_errors:
            return -1  # Error not retryable

        # Base delay calculation
        delay = min(
            policy.initial_delay * (policy.exponential_base ** (attempt - 1)), policy.max_delay
        )

        # Special handling for rate limit errors
        if error_category == ErrorCategory.RATE_LIMIT:
            # Longer delay for rate limits
            delay = max(delay * 2, 60.0)

        # Add jitter if configured
        if policy.jitter:
            import random

            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(delay, 0)  # Ensure non-negative

    def determine_circuit_breaker_threshold(self, service: str) -> CircuitBreakerPolicy:
        """
        Determine circuit breaker thresholds for a service.

        Args:
            service: Service name

        Returns:
            CircuitBreakerPolicy with configured thresholds
        """
        return self.CIRCUIT_BREAKER_POLICIES.get(
            service, self._get_default_circuit_breaker_policy()
        )

    def categorize_error(self, error: Any) -> ErrorCategory:
        """
        Categorize errors based on business rules.

        Args:
            error: Exception or error string

        Returns:
            ErrorCategory enum
        """
        # Handle string error types
        if isinstance(error, str):
            return self.ERROR_MAPPINGS.get(error, ErrorCategory.TRANSIENT)

        # Handle exception objects
        error_name = type(error).__name__

        # Check explicit mappings first
        if error_name in self.ERROR_MAPPINGS:
            return self.ERROR_MAPPINGS[error_name]

        # Analyze error attributes and messages
        error_msg = str(error).lower()

        if "timeout" in error_msg:
            return ErrorCategory.TIMEOUT
        elif "rate limit" in error_msg or "429" in error_msg:
            return ErrorCategory.RATE_LIMIT
        elif "connection" in error_msg or "network" in error_msg:
            return ErrorCategory.NETWORK_ERROR
        elif (
            "validation" in error_msg
            or "invalid" in error_msg
            or ("unauthorized" in error_msg or "forbidden" in error_msg)
            or ("not found" in error_msg or "404" in error_msg)
        ):
            return ErrorCategory.CLIENT_ERROR
        elif "server error" in error_msg or "500" in error_msg:
            return ErrorCategory.SERVER_ERROR
        elif "insufficient" in error_msg or "exceeded" in error_msg:
            return ErrorCategory.BUSINESS_ERROR
        elif "critical" in error_msg or "fatal" in error_msg:
            return ErrorCategory.CRITICAL

        # Default to transient for unknown errors
        return ErrorCategory.TRANSIENT

    def should_circuit_break(self, service: str, failure_count: int, time_window: float) -> bool:
        """
        Determine if circuit breaker should open.

        Args:
            service: Service name
            failure_count: Number of failures in time window
            time_window: Time window in seconds

        Returns:
            True if circuit should break
        """
        policy = self.determine_circuit_breaker_threshold(service)

        # Quick failure check
        if failure_count >= policy.failure_threshold:
            return True

        # Rate-based check (failures per minute)
        if time_window > 0:
            failure_rate = (failure_count / time_window) * 60
            # Break if failure rate exceeds threshold per minute
            if failure_rate > policy.failure_threshold * 2:
                return True

        return False

    def get_fallback_strategy(self, service: str, operation: str) -> FallbackStrategy:
        """
        Determine fallback strategy for a failed operation.

        Args:
            service: Service that failed
            operation: Operation that failed

        Returns:
            FallbackStrategy to use
        """
        # Market data fallbacks
        if service == "market_data":
            if operation == "get_quote":
                return FallbackStrategy(
                    strategy_type="cache",
                    cache_ttl=60.0,  # Use 1-minute old data
                )
            elif operation == "get_historical":
                return FallbackStrategy(
                    strategy_type="degrade", degraded_service="backup_market_data"
                )

        # Order execution fallbacks
        elif service == "order_execution":
            if operation == "place_order":
                return FallbackStrategy(strategy_type="queue", queue_name="pending_orders")
            elif operation == "cancel_order":
                return FallbackStrategy(strategy_type="queue", queue_name="pending_cancellations")

        # Database fallbacks
        elif service == "database":
            if operation.startswith("read"):
                return FallbackStrategy(
                    strategy_type="cache",
                    cache_ttl=300.0,  # Use 5-minute cache
                )
            elif operation.startswith("write"):
                return FallbackStrategy(strategy_type="queue", queue_name="write_queue")

        # Default fallback
        return FallbackStrategy(strategy_type="default", default_value=None)

    def determine_timeout(self, service: str, operation: str) -> float:
        """
        Determine timeout for an operation.

        Args:
            service: Service name
            operation: Operation name

        Returns:
            Timeout in seconds
        """
        # Base timeouts by service
        base_timeouts = {
            "market_data": 5.0,
            "order_execution": 10.0,
            "database": 3.0,
            "external_api": 30.0,
            "calculation": 2.0,
        }

        base = base_timeouts.get(service, 10.0)

        # Adjust for specific operations
        if "bulk" in operation or "batch" in operation:
            return base * 3
        elif "real_time" in operation or "quote" in operation:
            return base * 0.5
        elif "historical" in operation or "report" in operation:
            return base * 5

        return base

    def get_service_priority(self, service: str) -> ServicePriority:
        """
        Get priority level for a service.

        Args:
            service: Service name

        Returns:
            ServicePriority enum
        """
        priority_map = {
            "order_execution": ServicePriority.CRITICAL,
            "risk_management": ServicePriority.CRITICAL,
            "market_data": ServicePriority.HIGH,
            "portfolio_management": ServicePriority.HIGH,
            "database": ServicePriority.HIGH,
            "reporting": ServicePriority.MEDIUM,
            "analytics": ServicePriority.MEDIUM,
            "logging": ServicePriority.LOW,
            "metrics": ServicePriority.LOW,
        }

        return priority_map.get(service, ServicePriority.MEDIUM)

    def _get_default_retry_policy(self) -> RetryPolicy:
        """Get default retry policy."""
        return RetryPolicy(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True,
            retryable_errors=[ErrorCategory.TRANSIENT, ErrorCategory.NETWORK_ERROR],
        )

    def _get_default_circuit_breaker_policy(self) -> CircuitBreakerPolicy:
        """Get default circuit breaker policy."""
        return CircuitBreakerPolicy(
            failure_threshold=5,
            success_threshold=3,
            timeout=30.0,
            half_open_max_calls=2,
            error_categories=[ErrorCategory.CRITICAL, ErrorCategory.SERVER_ERROR],
        )

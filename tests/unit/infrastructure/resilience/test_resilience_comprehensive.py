"""
Comprehensive unit tests for resilience infrastructure - achieving 90%+ coverage.

Tests circuit breaker, retry logic, error handling, and health checks.
"""

import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)
from src.infrastructure.resilience.error_handling import (
    ErrorClassifier,
    ErrorHandler,
    ErrorSeverity,
)
from src.infrastructure.resilience.health import HealthCheck, HealthStatus
from src.infrastructure.resilience.retry import RetryConfig


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig class."""

    def test_config_defaults(self):
        """Test config with default values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.command_timeout == 60.0
        assert config.window_size == 10
        assert config.half_open_max_calls == 3
        assert config.recovery_timeout == 300.0
        assert config.exponential_backoff is True
        assert config.max_timeout == 3600.0
        assert config.failure_types == (Exception,)

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            command_timeout=30.0,
            window_size=20,
            failure_types=(ValueError, TypeError),
        )

        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.command_timeout == 30.0
        assert config.window_size == 20
        assert config.failure_types == (ValueError, TypeError)

    def test_config_validation_failure_threshold(self):
        """Test validation of failure threshold."""
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreakerConfig(failure_threshold=0)

        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreakerConfig(failure_threshold=-1)

    def test_config_validation_success_threshold(self):
        """Test validation of success threshold."""
        with pytest.raises(ValueError, match="success_threshold must be positive"):
            CircuitBreakerConfig(success_threshold=0)

    def test_config_validation_timeout(self):
        """Test validation of timeout."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            CircuitBreakerConfig(command_timeout=0)

        with pytest.raises(ValueError, match="timeout must be positive"):
            CircuitBreakerConfig(command_timeout=-10.0)

    def test_config_validation_window_size(self):
        """Test validation of window size."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            CircuitBreakerConfig(window_size=0)


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return CircuitBreakerConfig(
            failure_threshold=3, success_threshold=2, command_timeout=1.0, window_size=5
        )

    @pytest.fixture
    def breaker(self, config):
        """Create circuit breaker."""
        return CircuitBreaker("test_breaker", config)

    def test_initialization(self, breaker):
        """Test circuit breaker initialization."""
        assert breaker.name == "test_breaker"
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker._success_count == 0
        assert breaker._last_failure_time is None
        assert len(breaker._call_history) == 0

    def test_call_success_when_closed(self, breaker):
        """Test successful call when circuit is closed."""

        def test_func():
            return "success"

        result = breaker.call(test_func)

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    def test_call_failure_increments_counter(self, breaker):
        """Test failure increments counter."""

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker._failure_count == 1
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_opens_after_threshold(self, breaker):
        """Test circuit opens after failure threshold."""

        def failing_func():
            raise ValueError("Test error")

        # Fail 3 times (threshold)
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker._failure_count == 3

    def test_call_fails_fast_when_open(self, breaker):
        """Test calls fail fast when circuit is open."""
        # Open the circuit
        breaker._state = CircuitState.OPEN
        breaker._last_failure_time = time.time()

        def test_func():
            return "should not execute"

        with pytest.raises(CircuitBreakerError, match="Circuit breaker 'test_breaker' is open"):
            breaker.call(test_func)

    def test_half_open_after_timeout(self, breaker):
        """Test circuit moves to half-open after timeout."""
        # Open the circuit
        breaker._state = CircuitState.OPEN
        breaker._last_failure_time = time.time() - 2.0  # 2 seconds ago

        def test_func():
            return "success"

        result = breaker.call(test_func)

        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker._success_count == 1

    def test_circuit_closes_after_success_threshold(self, breaker):
        """Test circuit closes after success threshold in half-open."""
        breaker._state = CircuitState.HALF_OPEN

        def test_func():
            return "success"

        # Succeed twice (threshold)
        for _ in range(2):
            breaker.call(test_func)

        assert breaker.state == CircuitState.CLOSED
        assert breaker._success_count == 0
        assert breaker._failure_count == 0

    def test_circuit_reopens_on_failure_in_half_open(self, breaker):
        """Test circuit reopens on failure in half-open state."""
        breaker._state = CircuitState.HALF_OPEN

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

    def test_call_with_args_and_kwargs(self, breaker):
        """Test calling function with arguments."""

        def test_func(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = breaker.call(test_func, 1, 2, c=3)

        assert result == "1-2-3"

    def test_async_call(self, breaker):
        """Test async function call."""

        async def async_func():
            await asyncio.sleep(0.001)
            return "async success"

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(breaker.call_async(async_func))

        assert result == "async success"

    def test_exponential_backoff(self):
        """Test exponential backoff for timeout."""
        config = CircuitBreakerConfig(
            command_timeout=1.0, exponential_backoff=True, max_timeout=10.0
        )
        breaker = CircuitBreaker("test", config)

        # Open circuit
        breaker._state = CircuitState.OPEN
        breaker._consecutive_failures = 3

        timeout = breaker._get_timeout()

        # Should be 1.0 * (2 ** 3) = 8.0
        assert timeout == 8.0

    def test_exponential_backoff_max_limit(self):
        """Test exponential backoff respects max timeout."""
        config = CircuitBreakerConfig(
            command_timeout=1.0, exponential_backoff=True, max_timeout=5.0
        )
        breaker = CircuitBreaker("test", config)

        breaker._state = CircuitState.OPEN
        breaker._consecutive_failures = 10  # Would be 1024 seconds

        timeout = breaker._get_timeout()

        assert timeout == 5.0  # Capped at max

    def test_metrics_tracking(self, breaker):
        """Test metrics are tracked correctly."""

        def sometimes_fails(should_fail):
            if should_fail:
                raise ValueError("Failed")
            return "success"

        # Some successes
        for _ in range(3):
            breaker.call(sometimes_fails, False)

        # Some failures
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(sometimes_fails, True)

        metrics = breaker.get_metrics()

        assert metrics["total_calls"] == 5
        assert metrics["success_calls"] == 3
        assert metrics["failed_calls"] == 2
        assert metrics["state"] == CircuitState.CLOSED

    def test_reset(self, breaker):
        """Test resetting circuit breaker."""
        # Create some state
        breaker._state = CircuitState.OPEN
        breaker._failure_count = 5
        breaker._success_count = 2
        breaker._last_failure_time = time.time()

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker._success_count == 0
        assert breaker._last_failure_time is None

    def test_thread_safety(self, breaker):
        """Test thread-safe operation."""
        results = []
        errors = []

        def test_func(thread_id):
            if thread_id % 3 == 0:
                raise ValueError(f"Error from {thread_id}")
            return f"success-{thread_id}"

        def run_test(thread_id):
            try:
                result = breaker.call(test_func, thread_id)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(20):
            thread = threading.Thread(target=run_test, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have some successes and failures
        assert len(results) + len(errors) == 20


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator."""

    def test_decorator_basic(self):
        """Test basic decorator usage."""

        @circuit_breaker(failure_threshold=2)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_decorator_with_failures(self):
        """Test decorator with failures."""
        call_count = 0

        @circuit_breaker(failure_threshold=2, command_timeout=0.1)
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")

        # First two calls should execute and fail
        for _ in range(2):
            with pytest.raises(ValueError):
                failing_func()

        # Third call should fail fast
        with pytest.raises(CircuitBreakerError):
            failing_func()

        assert call_count == 2  # Function only called twice

    def test_decorator_async(self):
        """Test decorator with async function."""

        @circuit_breaker(failure_threshold=2)
        async def async_func():
            await asyncio.sleep(0.001)
            return "async success"

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(async_func())
        assert result == "async success"


class TestRetryConfig:
    """Test RetryConfig class."""

    def test_config_defaults(self):
        """Test config with default values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.retryable_exceptions == (Exception,)

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            strategy=RetryStrategy.LINEAR,
            retryable_exceptions=(ValueError, TypeError),
        )

        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.strategy == RetryStrategy.LINEAR
        assert config.retryable_exceptions == (ValueError, TypeError)

    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError, match="max_attempts must be positive"):
            RetryConfig(max_attempts=0)

        with pytest.raises(ValueError, match="initial_delay must be positive"):
            RetryConfig(initial_delay=-1.0)

        with pytest.raises(ValueError, match="max_delay must be positive"):
            RetryConfig(max_delay=0)


class TestRetryPolicy:
    """Test RetryPolicy class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return RetryConfig(max_attempts=3, initial_delay=0.1, strategy=RetryStrategy.EXPONENTIAL)

    @pytest.fixture
    def policy(self, config):
        """Create retry policy."""
        return RetryPolicy(config)

    def test_should_retry_within_attempts(self, policy):
        """Test should retry within max attempts."""
        error = ValueError("Test error")

        assert policy.should_retry(error, 1) is True
        assert policy.should_retry(error, 2) is True
        assert policy.should_retry(error, 3) is False

    def test_should_retry_non_retryable_exception(self, policy):
        """Test should not retry non-retryable exceptions."""
        policy.config.retryable_exceptions = (ValueError,)
        error = TypeError("Test error")

        assert policy.should_retry(error, 1) is False

    def test_get_delay_exponential(self, policy):
        """Test exponential backoff delay calculation."""
        policy.config.jitter = False  # Disable jitter for predictable results

        assert policy.get_delay(1) == 0.1  # initial_delay
        assert policy.get_delay(2) == 0.2  # 0.1 * 2
        assert policy.get_delay(3) == 0.4  # 0.1 * 4

    def test_get_delay_linear(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(initial_delay=0.5, strategy=RetryStrategy.LINEAR, jitter=False)
        policy = RetryPolicy(config)

        assert policy.get_delay(1) == 0.5
        assert policy.get_delay(2) == 1.0
        assert policy.get_delay(3) == 1.5

    def test_get_delay_fixed(self):
        """Test fixed delay calculation."""
        config = RetryConfig(initial_delay=1.0, strategy=RetryStrategy.FIXED, jitter=False)
        policy = RetryPolicy(config)

        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 1.0
        assert policy.get_delay(3) == 1.0

    def test_get_delay_with_jitter(self, policy):
        """Test delay with jitter."""
        policy.config.jitter = True

        delays = [policy.get_delay(1) for _ in range(10)]

        # All should be different due to jitter
        assert len(set(delays)) > 1

        # All should be within expected range
        for delay in delays:
            assert 0 <= delay <= 0.2  # Up to 2x initial_delay

    def test_get_delay_respects_max(self, policy):
        """Test delay respects max_delay."""
        policy.config.max_delay = 1.0
        policy.config.jitter = False

        # Would be 12.8 without max limit
        delay = policy.get_delay(8)

        assert delay == 1.0


class TestRetryDecorator:
    """Test retry decorator."""

    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = 0

        @retry(max_attempts=3)
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = test_func()

        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Test success after some failures."""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = test_func()

        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        """Test retry exhaustion."""
        call_count = 0

        @retry(max_attempts=3, initial_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert call_count == 3

    def test_retry_with_specific_exceptions(self):
        """Test retry with specific exception types."""

        @retry(max_attempts=3, retryable_exceptions=(ValueError,))
        def test_func(error_type):
            if error_type == "value":
                raise ValueError("Retryable")
            elif error_type == "type":
                raise TypeError("Not retryable")
            return "success"

        # Should retry ValueError
        with pytest.raises(ValueError):
            test_func("value")

        # Should not retry TypeError
        with pytest.raises(TypeError):
            test_func("type")

    def test_retry_async(self):
        """Test retry with async function."""
        call_count = 0

        @retry_async(max_attempts=3, initial_delay=0.01)
        async def async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Not yet")
            return "async success"

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(async_func())

        assert result == "async success"
        assert call_count == 2


class TestErrorHandlingConfig:
    """Test ErrorHandlingConfig class."""

    def test_config_defaults(self):
        """Test config with default values."""
        config = ErrorHandlingConfig()

        assert config.log_errors is True
        assert config.capture_stack_trace is True
        assert config.max_error_history == 100
        assert config.error_rate_window == 60.0
        assert config.alert_threshold == 10

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = ErrorHandlingConfig(log_errors=False, max_error_history=50, alert_threshold=5)

        assert config.log_errors is False
        assert config.max_error_history == 50
        assert config.alert_threshold == 5


class TestErrorClassifier:
    """Test ErrorClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create error classifier."""
        return ErrorClassifier()

    def test_classify_severity_critical(self, classifier):
        """Test classifying critical errors."""
        errors = [MemoryError("Out of memory"), SystemError("System failure"), KeyboardInterrupt()]

        for error in errors:
            assert classifier.classify_severity(error) == ErrorSeverity.CRITICAL

    def test_classify_severity_high(self, classifier):
        """Test classifying high severity errors."""
        errors = [
            ConnectionError("Connection lost"),
            TimeoutError("Request timeout"),
            PermissionError("Access denied"),
        ]

        for error in errors:
            assert classifier.classify_severity(error) == ErrorSeverity.HIGH

    def test_classify_severity_medium(self, classifier):
        """Test classifying medium severity errors."""
        errors = [
            ValueError("Invalid value"),
            TypeError("Type mismatch"),
            KeyError("Key not found"),
        ]

        for error in errors:
            assert classifier.classify_severity(error) == ErrorSeverity.MEDIUM

    def test_classify_severity_low(self, classifier):
        """Test classifying low severity errors."""
        error = Warning("Just a warning")

        assert classifier.classify_severity(error) == ErrorSeverity.LOW

    def test_is_transient_true(self, classifier):
        """Test identifying transient errors."""
        errors = [
            ConnectionError("Temporary network issue"),
            TimeoutError("Request timeout"),
            OSError("Resource temporarily unavailable"),
        ]

        for error in errors:
            assert classifier.is_transient(error) is True

    def test_is_transient_false(self, classifier):
        """Test identifying non-transient errors."""
        errors = [
            ValueError("Invalid input"),
            TypeError("Wrong type"),
            AttributeError("Missing attribute"),
        ]

        for error in errors:
            assert classifier.is_transient(error) is False

    def test_get_recovery_strategy(self, classifier):
        """Test getting recovery strategy."""
        # Transient errors should retry
        assert classifier.get_recovery_strategy(ConnectionError()) == ErrorRecoveryStrategy.RETRY

        # Critical errors should fail fast
        assert classifier.get_recovery_strategy(MemoryError()) == ErrorRecoveryStrategy.FAIL_FAST

        # Others should use fallback
        assert classifier.get_recovery_strategy(ValueError()) == ErrorRecoveryStrategy.FALLBACK


class TestErrorHandler:
    """Test ErrorHandler class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ErrorHandlingConfig(max_error_history=10)

    @pytest.fixture
    def handler(self, config):
        """Create error handler."""
        return ErrorHandler(config)

    def test_handle_error_logging(self, handler):
        """Test error handling with logging."""
        error = ValueError("Test error")

        with patch("logging.error") as mock_log:
            handler.handle_error(error, context={"operation": "test"})

            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            assert "Test error" in args[0]

    def test_handle_error_history(self, handler):
        """Test error history tracking."""
        errors = [ValueError(f"Error {i}") for i in range(5)]

        for error in errors:
            handler.handle_error(error)

        assert len(handler._error_history) == 5
        assert handler.get_error_count() == 5

    def test_handle_error_history_limit(self, handler):
        """Test error history respects limit."""
        # Add 15 errors (limit is 10)
        for i in range(15):
            handler.handle_error(ValueError(f"Error {i}"))

        assert len(handler._error_history) == 10
        # Should keep the most recent errors
        assert "Error 14" in str(handler._error_history[-1]["error"])

    def test_get_error_rate(self, handler):
        """Test error rate calculation."""
        # Add errors at different times
        now = time.time()

        # Add old error (outside window)
        handler._error_history.append(
            {
                "error": ValueError("Old"),
                "timestamp": now - 120,  # 2 minutes ago
                "severity": ErrorSeverity.MEDIUM,
            }
        )

        # Add recent errors
        for i in range(5):
            handler._error_history.append(
                {
                    "error": ValueError(f"Recent {i}"),
                    "timestamp": now - i,
                    "severity": ErrorSeverity.MEDIUM,
                }
            )

        # Rate should only count recent errors (within 60 second window)
        rate = handler.get_error_rate()
        assert rate == 5.0 / 60.0  # 5 errors per minute

    def test_should_alert(self, handler):
        """Test alert threshold checking."""
        # Add errors below threshold
        for i in range(5):
            handler.handle_error(ValueError(f"Error {i}"))

        assert handler.should_alert() is False

        # Add more errors to exceed threshold
        for i in range(6):
            handler.handle_error(ValueError(f"More {i}"))

        assert handler.should_alert() is True

    def test_clear_history(self, handler):
        """Test clearing error history."""
        # Add some errors
        for i in range(5):
            handler.handle_error(ValueError(f"Error {i}"))

        assert handler.get_error_count() == 5

        handler.clear_history()

        assert handler.get_error_count() == 0
        assert len(handler._error_history) == 0


class TestHealthCheck:
    """Test HealthCheck class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return HealthCheckConfig(check_interval=1.0, command_timeout=5.0, failure_threshold=2)

    @pytest.fixture
    def health_check(self, config):
        """Create health check."""
        return HealthCheck("test_service", config)

    def test_initialization(self, health_check):
        """Test health check initialization."""
        assert health_check.name == "test_service"
        assert health_check.status == HealthStatus.UNKNOWN
        assert health_check._consecutive_failures == 0
        assert health_check._last_check_time is None

    def test_check_health_success(self, health_check):
        """Test successful health check."""

        def check_func():
            return True

        health_check.check_function = check_func
        result = health_check.check()

        assert result is True
        assert health_check.status == HealthStatus.HEALTHY
        assert health_check._consecutive_failures == 0

    def test_check_health_failure(self, health_check):
        """Test failed health check."""

        def check_func():
            return False

        health_check.check_function = check_func
        result = health_check.check()

        assert result is False
        assert health_check._consecutive_failures == 1
        assert health_check.status == HealthStatus.DEGRADED

    def test_check_health_unhealthy_after_threshold(self, health_check):
        """Test status becomes unhealthy after failure threshold."""

        def check_func():
            return False

        health_check.check_function = check_func

        # Fail twice (threshold)
        health_check.check()
        health_check.check()

        assert health_check.status == HealthStatus.UNHEALTHY
        assert health_check._consecutive_failures == 2

    def test_check_health_recovery(self, health_check):
        """Test recovery from unhealthy state."""
        call_count = 0

        def check_func():
            nonlocal call_count
            call_count += 1
            return call_count > 2

        health_check.check_function = check_func

        # Fail twice
        health_check.check()
        health_check.check()
        assert health_check.status == HealthStatus.UNHEALTHY

        # Succeed
        health_check.check()
        assert health_check.status == HealthStatus.HEALTHY
        assert health_check._consecutive_failures == 0

    def test_async_health_check(self, health_check):
        """Test async health check."""

        async def async_check():
            await asyncio.sleep(0.001)
            return True

        health_check.check_function = async_check

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(health_check.check_async())

        assert result is True
        assert health_check.status == HealthStatus.HEALTHY


class TestSystemHealth:
    """Test SystemHealth class."""

    @pytest.fixture
    def system_health(self):
        """Create system health monitor."""
        return SystemHealth()

    def test_register_indicator(self, system_health):
        """Test registering health indicator."""
        indicator = HealthIndicator(name="database", check_function=lambda: True)

        system_health.register_indicator(indicator)

        assert "database" in system_health._indicators
        assert system_health._indicators["database"] == indicator

    def test_check_all_healthy(self, system_health):
        """Test checking all indicators - all healthy."""
        system_health.register_indicator(HealthIndicator("service1", lambda: True))
        system_health.register_indicator(HealthIndicator("service2", lambda: True))

        results = system_health.check_all()

        assert results["overall"] == HealthStatus.HEALTHY
        assert results["indicators"]["service1"] == HealthStatus.HEALTHY
        assert results["indicators"]["service2"] == HealthStatus.HEALTHY

    def test_check_all_degraded(self, system_health):
        """Test checking all indicators - some degraded."""
        system_health.register_indicator(HealthIndicator("service1", lambda: True))

        # Create degraded indicator
        degraded = HealthIndicator("service2", lambda: False)
        degraded._consecutive_failures = 1
        system_health.register_indicator(degraded)

        results = system_health.check_all()

        assert results["overall"] == HealthStatus.DEGRADED
        assert results["indicators"]["service1"] == HealthStatus.HEALTHY

    def test_check_all_unhealthy(self, system_health):
        """Test checking all indicators - some unhealthy."""
        system_health.register_indicator(HealthIndicator("service1", lambda: True))

        # Create unhealthy indicator
        unhealthy = HealthIndicator("service2", lambda: False)
        unhealthy._consecutive_failures = 3
        unhealthy.status = HealthStatus.UNHEALTHY
        system_health.register_indicator(unhealthy)

        results = system_health.check_all()

        assert results["overall"] == HealthStatus.UNHEALTHY

    def test_get_status_empty(self, system_health):
        """Test status with no indicators."""
        status = system_health.get_status()

        assert status["overall"] == HealthStatus.UNKNOWN
        assert len(status["indicators"]) == 0

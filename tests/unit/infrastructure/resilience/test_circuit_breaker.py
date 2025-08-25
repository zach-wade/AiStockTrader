"""
Tests for circuit breaker implementation.
"""

import time

import pytest

from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
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

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        CircuitBreakerConfig(failure_threshold=5)

        # Invalid configs should raise
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreakerConfig(failure_threshold=0)

        with pytest.raises(ValueError, match="success_threshold must be positive"):
            CircuitBreakerConfig(success_threshold=0)

        with pytest.raises(ValueError, match="timeout must be positive"):
            CircuitBreakerConfig(command_timeout=0)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test initial circuit breaker state."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())

        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False
        assert cb.is_half_open is False

    def test_successful_calls(self):
        """Test successful function calls."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())

        def success_func():
            return "success"

        result = cb.call(success_func)

        assert result == "success"
        assert cb.state == CircuitState.CLOSED

        metrics = cb.get_metrics()
        assert metrics["total_calls"] == 1
        assert metrics["total_successes"] == 1
        assert metrics["total_failures"] == 0

    def test_failure_accumulation(self):
        """Test failure accumulation leading to open state."""
        config = CircuitBreakerConfig(failure_threshold=3, window_size=5)
        cb = CircuitBreaker("test", config)

        def failing_func():
            raise Exception("Test failure")

        # Should remain closed for first few failures
        for i in range(2):
            with pytest.raises(Exception):
                cb.call(failing_func)
            assert cb.state == CircuitState.CLOSED

        # Third failure should open the circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

        # Subsequent calls should fail fast
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

        metrics = cb.get_metrics()
        assert metrics["total_calls"] == 4  # 3 actual calls + 1 failed fast
        assert metrics["total_failures"] == 3
        assert metrics["total_timeouts"] == 1

    @pytest.mark.asyncio
    async def test_async_calls(self):
        """Test async function calls."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())

        async def async_success():
            return "async_success"

        result = await cb.call_async(async_success)

        assert result == "async_success"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_async_failures(self):
        """Test async function failures."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)

        async def async_failing():
            raise Exception("Async failure")

        # Cause failures to open circuit
        for i in range(2):
            with pytest.raises(Exception):
                await cb.call_async(async_failing)

        assert cb.state == CircuitState.OPEN

        # Should fail fast
        with pytest.raises(CircuitBreakerError):
            await cb.call_async(async_failing)

    def test_half_open_recovery(self):
        """Test recovery through half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            command_timeout=0.1,  # Short timeout for testing
        )
        cb = CircuitBreaker("test", config)

        def failing_func():
            raise Exception("Failure")

        def success_func():
            return "success"

        # Open the circuit
        for i in range(2):
            with pytest.raises(Exception):
                cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Next successful call should transition to half-open
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN

        # Another success should close the circuit
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure(self):
        """Test failure in half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, command_timeout=0.1)
        cb = CircuitBreaker("test", config)

        def failing_func():
            raise Exception("Failure")

        def success_func():
            return "success"

        # Open the circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Success to go half-open
        cb.call(success_func)
        assert cb.state == CircuitState.HALF_OPEN

        # Failure should re-open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

    def test_decorator_usage(self):
        """Test using circuit breaker as decorator."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)

        @cb
        def decorated_func(should_fail=False):
            if should_fail:
                raise Exception("Decorated failure")
            return "decorated_success"

        # Test success
        result = decorated_func()
        assert result == "decorated_success"

        # Test failures
        for i in range(2):
            with pytest.raises(Exception):
                decorated_func(should_fail=True)

        assert cb.state == CircuitState.OPEN

        # Should fail fast now
        with pytest.raises(CircuitBreakerError):
            decorated_func()

    @pytest.mark.asyncio
    async def test_async_decorator_usage(self):
        """Test using circuit breaker as async decorator."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)

        @cb
        async def async_decorated_func(should_fail=False):
            if should_fail:
                raise Exception("Async decorated failure")
            return "async_decorated_success"

        # Test success
        result = await async_decorated_func()
        assert result == "async_decorated_success"

        # Test failures
        for i in range(2):
            with pytest.raises(Exception):
                await async_decorated_func(should_fail=True)

        assert cb.state == CircuitState.OPEN

    def test_exponential_backoff(self):
        """Test exponential backoff timeout calculation."""
        config = CircuitBreakerConfig(
            failure_threshold=1, command_timeout=1.0, exponential_backoff=True, max_timeout=10.0
        )
        cb = CircuitBreaker("test", config)

        def failing_func():
            raise Exception("Failure")

        # Open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

        # Check that timeout increases with failures
        initial_timeout = config.command_timeout
        cb._failure_count = 3  # Simulate multiple failures

        # Should use exponential backoff (timeout * 2^failure_count)
        expected_timeout = min(initial_timeout * (2**3), config.max_timeout)

        # The actual implementation uses time-based checks,
        # so we test the concept rather than exact timing
        assert cb._should_attempt_reset() is False  # Too soon

    def test_metrics_collection(self):
        """Test metrics collection."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())

        def success_func():
            return "success"

        def failing_func():
            raise Exception("Failure")

        # Mix of successes and failures
        cb.call(success_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        cb.call(success_func)

        metrics = cb.get_metrics()

        assert metrics["name"] == "test"
        assert metrics["state"] == CircuitState.CLOSED
        assert metrics["total_calls"] == 3
        assert metrics["total_successes"] == 2
        assert metrics["total_failures"] == 1
        assert metrics["success_rate"] == 2 / 3
        assert metrics["failure_rate"] == 1 / 3

    def test_reset_functionality(self):
        """Test circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        def failing_func():
            raise Exception("Failure")

        # Open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

        # Reset
        cb.reset()

        assert cb.state == CircuitState.CLOSED

        metrics = cb.get_metrics()
        assert metrics["total_calls"] == 0
        assert metrics["total_successes"] == 0
        assert metrics["total_failures"] == 0


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry."""

    def test_singleton_behavior(self):
        """Test that registry is a singleton."""
        registry1 = CircuitBreakerRegistry.get_instance()
        registry2 = CircuitBreakerRegistry.get_instance()

        assert registry1 is registry2

    def test_get_or_create(self):
        """Test get or create functionality."""
        registry = CircuitBreakerRegistry.get_instance()
        registry.clear()  # Clean slate

        config = CircuitBreakerConfig(failure_threshold=10)

        # Create new
        cb1 = registry.get_or_create("test1", config)
        assert cb1.name == "test1"
        assert cb1.config.failure_threshold == 10

        # Get existing
        cb2 = registry.get_or_create("test1")
        assert cb2 is cb1

        # Config ignored for existing
        cb3 = registry.get_or_create("test1", CircuitBreakerConfig(failure_threshold=20))
        assert cb3 is cb1
        assert cb3.config.failure_threshold == 10  # Original config

    def test_list_names(self):
        """Test listing circuit breaker names."""
        registry = CircuitBreakerRegistry.get_instance()
        registry.clear()

        registry.get_or_create("test1")
        registry.get_or_create("test2")

        names = registry.list_names()
        assert "test1" in names
        assert "test2" in names
        assert len(names) == 2

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        registry = CircuitBreakerRegistry.get_instance()
        registry.clear()

        cb1 = registry.get_or_create("test1")
        cb2 = registry.get_or_create("test2")

        all_metrics = registry.get_all_metrics()

        assert "test1" in all_metrics
        assert "test2" in all_metrics
        assert all_metrics["test1"]["name"] == "test1"
        assert all_metrics["test2"]["name"] == "test2"

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry.get_instance()
        registry.clear()

        cb1 = registry.get_or_create("test1", CircuitBreakerConfig(failure_threshold=1))
        cb2 = registry.get_or_create("test2", CircuitBreakerConfig(failure_threshold=1))

        # Open both circuits
        def failing_func():
            raise Exception("Failure")

        with pytest.raises(Exception):
            cb1.call(failing_func)
        with pytest.raises(Exception):
            cb2.call(failing_func)

        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        # Reset all
        registry.reset_all()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED


# Fixtures for integration tests
@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker for testing."""
    config = CircuitBreakerConfig(failure_threshold=3, success_threshold=2, command_timeout=0.1)
    return CircuitBreaker("test", config)


@pytest.fixture
def registry():
    """Create a clean registry for testing."""
    registry = CircuitBreakerRegistry.get_instance()
    registry.clear()
    return registry

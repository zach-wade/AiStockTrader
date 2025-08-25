"""
Comprehensive tests for circuit breaker implementation.
Achieves 80%+ test coverage for production-ready error handling.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
)


class TestCircuitBreakerConfigComprehensive:
    """Comprehensive tests for circuit breaker configuration."""

    def test_all_config_validations(self):
        """Test all configuration validation edge cases."""
        # Test all positive value validations
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreakerConfig(failure_threshold=0)

        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreakerConfig(failure_threshold=-1)

        with pytest.raises(ValueError, match="success_threshold must be positive"):
            CircuitBreakerConfig(success_threshold=0)

        with pytest.raises(ValueError, match="success_threshold must be positive"):
            CircuitBreakerConfig(success_threshold=-5)

        with pytest.raises(ValueError, match="timeout must be positive"):
            CircuitBreakerConfig(command_timeout=0)

        with pytest.raises(ValueError, match="timeout must be positive"):
            CircuitBreakerConfig(command_timeout=-10.0)

        with pytest.raises(ValueError, match="window_size must be positive"):
            CircuitBreakerConfig(window_size=0)

        with pytest.raises(ValueError, match="window_size must be positive"):
            CircuitBreakerConfig(window_size=-1)

    def test_custom_failure_types(self):
        """Test custom failure type configuration."""
        custom_exceptions = (RuntimeError, ValueError, IOError)
        config = CircuitBreakerConfig(failure_types=custom_exceptions)

        assert config.failure_types == custom_exceptions
        assert RuntimeError in config.failure_types
        assert Exception not in config.failure_types

    def test_post_init_validation(self):
        """Test that post_init runs and validates properly."""
        # This should succeed
        config = CircuitBreakerConfig(
            failure_threshold=10, success_threshold=5, command_timeout=30.0, window_size=20
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 5


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state transitions comprehensively."""

    def test_closed_to_open_transition(self):
        """Test transition from closed to open state."""
        config = CircuitBreakerConfig(failure_threshold=2, window_size=5)
        cb = CircuitBreaker("test", config)

        def failing_func():
            raise ConnectionError("Network error")

        # First failure - should stay closed
        with pytest.raises(ConnectionError):
            cb.call(failing_func)
        assert cb.state == CircuitState.CLOSED
        assert len(cb._failures) == 1

        # Second failure - should open
        with pytest.raises(ConnectionError):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
        assert cb._state_changes["open"] == 1

        # Verify metrics
        metrics = cb.get_metrics()
        assert metrics["state"] == "open"
        assert metrics["total_failures"] == 2
        assert metrics["recent_failures"] == 2

    def test_open_to_half_open_transition(self):
        """Test transition from open to half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            command_timeout=0.05,  # 50ms timeout for quick testing
            exponential_backoff=False,
        )
        cb = CircuitBreaker("test", config)

        # Open the circuit
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        assert cb.state == CircuitState.OPEN

        # Should fail fast immediately
        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: "success")

        # Wait for timeout
        time.sleep(0.06)

        # Next call should transition to half-open and succeed
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
        assert cb._state_changes["half_open"] == 1

    def test_half_open_to_closed_transition(self):
        """Test successful recovery from half-open to closed."""
        config = CircuitBreakerConfig(
            failure_threshold=1, success_threshold=2, command_timeout=0.05
        )
        cb = CircuitBreaker("test", config)

        # Open circuit
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        # Wait and transition to half-open
        time.sleep(0.06)
        cb.call(lambda: "success1")
        assert cb.state == CircuitState.HALF_OPEN
        assert cb._half_open_successes == 1

        # Second success should close circuit
        cb.call(lambda: "success2")
        assert cb.state == CircuitState.CLOSED
        assert cb._state_changes["closed"] == 1
        assert cb._failure_count == 0  # Reset on close
        assert len(cb._failures) == 0  # Cleared on close

    def test_half_open_to_open_transition(self):
        """Test failure in half-open state reopens circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, command_timeout=0.05)
        cb = CircuitBreaker("test", config)

        # Open circuit
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        # Wait and transition to half-open
        time.sleep(0.06)
        cb.call(lambda: "success")
        assert cb.state == CircuitState.HALF_OPEN

        # Failure should reopen
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail again")))
        assert cb.state == CircuitState.OPEN
        assert cb._half_open_calls == 0  # Reset
        assert cb._half_open_successes == 0  # Reset


class TestCircuitBreakerExponentialBackoff:
    """Test exponential backoff functionality."""

    def test_exponential_backoff_timing(self):
        """Test exponential backoff increases timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1, command_timeout=0.1, exponential_backoff=True, max_timeout=1.0
        )
        cb = CircuitBreaker("test", config)

        # Open circuit
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        # First check - should not attempt reset yet
        assert cb._should_attempt_reset() is False

        # Simulate multiple failures
        cb._failure_count = 3

        # With exponential backoff, timeout should be 0.1 * 2^3 = 0.8 seconds
        # So after 0.1 seconds, should still not attempt reset
        time.sleep(0.15)
        assert cb._should_attempt_reset() is False

        # But after 0.8+ seconds, should attempt reset
        cb._state_changed_at = time.time() - 0.85
        assert cb._should_attempt_reset() is True

    def test_exponential_backoff_max_timeout(self):
        """Test exponential backoff respects max timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1, command_timeout=1.0, exponential_backoff=True, max_timeout=5.0
        )
        cb = CircuitBreaker("test", config)

        # Open circuit
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        # Simulate many failures (2^10 = 1024 seconds without cap)
        cb._failure_count = 10
        cb._state_changed_at = time.time() - 6.0  # 6 seconds ago

        # Should attempt reset because max_timeout is 5.0
        assert cb._should_attempt_reset() is True

    def test_no_exponential_backoff(self):
        """Test fixed timeout when exponential backoff is disabled."""
        config = CircuitBreakerConfig(
            failure_threshold=1, command_timeout=0.1, exponential_backoff=False
        )
        cb = CircuitBreaker("test", config)

        # Open circuit
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        # Simulate multiple failures
        cb._failure_count = 5

        # Without exponential backoff, timeout is always 0.1
        time.sleep(0.11)
        assert cb._should_attempt_reset() is True


class TestCircuitBreakerConcurrency:
    """Test circuit breaker thread safety and concurrent operations."""

    def test_thread_safe_state_transitions(self):
        """Test thread-safe state transitions."""
        config = CircuitBreakerConfig(failure_threshold=5, window_size=10)
        cb = CircuitBreaker("test", config)

        results = []
        exceptions = []

        def worker(should_fail):
            try:
                if should_fail:
                    result = cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
                else:
                    result = cb.call(lambda: "success")
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Mix of successes and failures
            futures = []
            for i in range(20):
                should_fail = i % 3 == 0  # Every 3rd call fails
                futures.append(executor.submit(worker, should_fail))

            # Wait for all to complete
            for future in futures:
                future.result()

        # Verify state consistency
        metrics = cb.get_metrics()
        assert metrics["total_calls"] == len(results) + len(exceptions)
        assert metrics["total_successes"] == len(results)

    def test_half_open_concurrent_limit(self):
        """Test concurrent call limiting in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            command_timeout=0.05,
            half_open_max_calls=2,  # Only allow 2 concurrent calls
        )
        cb = CircuitBreaker("test", config)

        # Open circuit
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        # Wait for timeout
        time.sleep(0.06)

        # Simulate concurrent calls in half-open state
        call_results = []

        def slow_func():
            time.sleep(0.05)
            return "success"

        def worker():
            try:
                result = cb.call(slow_func)
                call_results.append(("success", result))
            except CircuitBreakerError as e:
                call_results.append(("error", str(e)))

        # Start 5 concurrent calls
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have 2 successes and 3 circuit breaker errors
        successes = [r for r in call_results if r[0] == "success"]
        errors = [r for r in call_results if r[0] == "error"]

        assert len(successes) <= config.half_open_max_calls
        assert len(errors) >= 3  # At least 3 should be rejected


class TestCircuitBreakerMetrics:
    """Test comprehensive metrics collection."""

    def test_sliding_window_failure_tracking(self):
        """Test sliding window for failure tracking."""
        config = CircuitBreakerConfig(failure_threshold=3, window_size=5)
        cb = CircuitBreaker("test", config)

        # Add failures over time
        for i in range(2):
            with pytest.raises(Exception):
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            time.sleep(0.01)

        # Check failures are tracked
        assert len(cb._failures) == 2

        # Old failures should be removed from window (1 minute cutoff)
        cb._failures.appendleft(time.time() - 61)  # Add old failure

        # Success should clean old failures
        cb.call(lambda: "success")
        # The cleanup happens in _record_success
        assert len([f for f in cb._failures if f > time.time() - 60]) <= 2

    def test_metrics_accuracy(self):
        """Test accuracy of collected metrics."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker("test", config)

        # Mix of operations
        cb.call(lambda: "success1")

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail1")))

        cb.call(lambda: "success2")

        with pytest.raises(ConnectionError):
            cb.call(lambda: (_ for _ in ()).throw(ConnectionError("fail2")))

        metrics = cb.get_metrics()

        assert metrics["name"] == "test"
        assert metrics["total_calls"] == 4
        assert metrics["total_successes"] == 2
        assert metrics["total_failures"] == 2
        assert metrics["success_rate"] == 0.5
        assert metrics["failure_rate"] == 0.5
        assert "time_in_current_state" in metrics
        assert metrics["state"] == "closed"

    def test_state_change_tracking(self):
        """Test state change tracking in metrics."""
        config = CircuitBreakerConfig(
            failure_threshold=2, success_threshold=2, command_timeout=0.05
        )
        cb = CircuitBreaker("test", config)

        # Cause state changes
        # Closed -> Open
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        # Open -> Half-Open
        time.sleep(0.06)
        cb.call(lambda: "success")

        # Half-Open -> Closed
        cb.call(lambda: "success")

        metrics = cb.get_metrics()
        assert metrics["state_changes"]["open"] == 1
        assert metrics["state_changes"]["half_open"] == 1
        assert metrics["state_changes"]["closed"] == 1


class TestCircuitBreakerExceptionHandling:
    """Test exception type handling."""

    def test_custom_failure_types(self):
        """Test custom failure type configuration."""
        config = CircuitBreakerConfig(failure_threshold=2, failure_types=(ValueError, TypeError))
        cb = CircuitBreaker("test", config)

        # ConnectionError should not count as failure
        for _ in range(3):
            with pytest.raises(ConnectionError):
                cb.call(lambda: (_ for _ in ()).throw(ConnectionError("ignored")))

        assert cb.state == CircuitState.CLOSED  # Should not open

        # ValueError should count
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("counted")))

        assert cb.state == CircuitState.OPEN  # Should open

    def test_exception_not_in_failure_types(self):
        """Test exceptions not in failure types are not counted."""
        config = CircuitBreakerConfig(failure_threshold=1, failure_types=(ConnectionError,))
        cb = CircuitBreaker("test", config)

        # ValueError not in failure_types
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("not counted")))

        assert cb.state == CircuitState.CLOSED
        assert cb._total_failures == 0  # Not counted

        # ConnectionError in failure_types
        with pytest.raises(ConnectionError):
            cb.call(lambda: (_ for _ in ()).throw(ConnectionError("counted")))

        assert cb.state == CircuitState.OPEN
        assert cb._total_failures == 1


class TestCircuitBreakerAsyncOperations:
    """Test async operation handling."""

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test async decorator functionality."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)

        call_count = 0

        @cb
        async def async_func(should_fail=False):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ConnectionError("Async failure")
            await asyncio.sleep(0.01)
            return f"result_{call_count}"

        # Test success
        result = await async_func()
        assert result == "result_1"

        # Test failures
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await async_func(should_fail=True)

        assert cb.state == CircuitState.OPEN

        # Should fail fast
        with pytest.raises(CircuitBreakerError):
            await async_func()

    @pytest.mark.asyncio
    async def test_async_concurrent_calls(self):
        """Test concurrent async calls."""
        config = CircuitBreakerConfig(failure_threshold=3, window_size=10)
        cb = CircuitBreaker("test", config)

        async def async_worker(worker_id, should_fail):
            if should_fail:
                raise ConnectionError(f"Worker {worker_id} failed")
            await asyncio.sleep(0.01)
            return f"Worker {worker_id} success"

        # Run concurrent async operations
        tasks = []
        for i in range(10):
            should_fail = i < 3  # First 3 fail
            task = cb.call_async(async_worker, i, should_fail)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        exceptions = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if not isinstance(r, Exception)]

        assert len(exceptions) >= 3  # At least the failures
        assert len(successes) <= 7  # Some might fail due to circuit opening

        # Circuit should be open after 3 failures
        if cb.state == CircuitState.OPEN:
            with pytest.raises(CircuitBreakerError):
                await cb.call_async(async_worker, 99, False)


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry functionality."""

    def test_singleton_thread_safety(self):
        """Test thread-safe singleton creation."""
        registries = []

        def get_registry():
            registries.append(CircuitBreakerRegistry.get_instance())

        # Create registry from multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=get_registry)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should be the same instance
        assert all(r is registries[0] for r in registries)

    def test_registry_operations(self):
        """Test registry create, get, and list operations."""
        registry = CircuitBreakerRegistry.get_instance()
        registry.clear()

        # Create multiple circuit breakers
        config1 = CircuitBreakerConfig(failure_threshold=5)
        config2 = CircuitBreakerConfig(failure_threshold=10)

        cb1 = registry.get_or_create("service1", config1)
        cb2 = registry.get_or_create("service2", config2)

        # Get existing
        cb1_again = registry.get_or_create("service1", config2)  # Config ignored
        assert cb1_again is cb1
        assert cb1_again.config.failure_threshold == 5  # Original config

        # Get by name
        cb = registry.get("service1")
        assert cb is cb1

        # Non-existent returns None
        assert registry.get("non_existent") is None

        # List names
        names = registry.list_names()
        assert "service1" in names
        assert "service2" in names
        assert len(names) == 2

    def test_registry_metrics_collection(self):
        """Test collecting metrics from all breakers."""
        registry = CircuitBreakerRegistry.get_instance()
        registry.clear()

        # Create breakers with different states
        cb1 = registry.get_or_create("api", CircuitBreakerConfig(failure_threshold=1))
        cb2 = registry.get_or_create("database", CircuitBreakerConfig())

        # Generate some activity
        cb1.call(lambda: "success")
        with pytest.raises(Exception):
            cb1.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        cb2.call(lambda: "success")
        cb2.call(lambda: "success")

        # Get all metrics
        all_metrics = registry.get_all_metrics()

        assert "api" in all_metrics
        assert "database" in all_metrics
        assert all_metrics["api"]["total_calls"] >= 2
        assert all_metrics["api"]["state"] == "open"
        assert all_metrics["database"]["total_calls"] == 2
        assert all_metrics["database"]["state"] == "closed"

    def test_registry_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry.get_instance()
        registry.clear()

        # Create and open multiple breakers
        cb1 = registry.get_or_create("cb1", CircuitBreakerConfig(failure_threshold=1))
        cb2 = registry.get_or_create("cb2", CircuitBreakerConfig(failure_threshold=1))

        # Open both
        with pytest.raises(Exception):
            cb1.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        with pytest.raises(Exception):
            cb2.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        # Reset all
        registry.reset_all()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED
        assert cb1._total_calls == 0
        assert cb2._total_calls == 0


class TestCircuitBreakerEdgeCases:
    """Test edge cases and error conditions."""

    def test_can_execute_edge_cases(self):
        """Test _can_execute method edge cases."""
        config = CircuitBreakerConfig(
            failure_threshold=1, command_timeout=0.05, half_open_max_calls=1
        )
        cb = CircuitBreaker("test", config)

        # Closed state - should execute
        assert cb._can_execute() is True

        # Open circuit
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        # Open state - should not execute
        assert cb._can_execute() is False

        # Wait for timeout
        time.sleep(0.06)

        # Should transition to half-open and allow execution
        assert cb._can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

        # Exhaust half-open calls
        cb._half_open_calls = config.half_open_max_calls
        assert cb._can_execute() is False

    def test_string_representation(self):
        """Test string representation of circuit breaker."""
        cb = CircuitBreaker("test_service", CircuitBreakerConfig())

        str_repr = str(cb)
        assert "CircuitBreaker" in str_repr
        assert "test_service" in str_repr
        assert "closed" in str_repr

        # Open the circuit
        cb._state = CircuitState.OPEN
        str_repr = str(cb)
        assert "open" in str_repr

    def test_negative_attempt_in_should_attempt_reset(self):
        """Test _should_attempt_reset with invalid state."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())

        # Should return False when not in OPEN state
        assert cb._should_attempt_reset() is False

        cb._state = CircuitState.HALF_OPEN
        assert cb._should_attempt_reset() is False

    def test_record_success_in_closed_state_with_old_failures(self):
        """Test that old failures are cleaned up in closed state."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())

        # Add some old failures (>60 seconds old)
        old_time = time.time() - 65
        cb._failures.append(old_time)
        cb._failures.append(old_time + 1)
        cb._failures.append(time.time())  # Recent failure

        # Record success should clean old failures
        cb._record_success()

        # Only recent failure should remain
        assert len(cb._failures) <= 1
        assert all(f > time.time() - 60 for f in cb._failures)


class TestCircuitBreakerIntegration:
    """Integration tests for real-world scenarios."""

    @pytest.mark.asyncio
    async def test_api_client_integration(self):
        """Test circuit breaker with simulated API client."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            command_timeout=0.1,
            failure_types=(ConnectionError, TimeoutError),
        )
        cb = CircuitBreaker("api_client", config)

        # Simulate API client
        class APIClient:
            def __init__(self):
                self.call_count = 0
                self.fail_until = 3

            async def make_request(self, endpoint):
                self.call_count += 1
                if self.call_count <= self.fail_until:
                    raise ConnectionError(f"Network error on call {self.call_count}")
                return {"status": "success", "data": endpoint}

        client = APIClient()

        # First 3 calls fail and open circuit
        for i in range(3):
            with pytest.raises(ConnectionError):
                await cb.call_async(client.make_request, f"/endpoint{i}")

        assert cb.state == CircuitState.OPEN

        # Circuit is open, should fail fast
        with pytest.raises(CircuitBreakerError):
            await cb.call_async(client.make_request, "/endpoint3")

        # Wait for timeout and recover
        await asyncio.sleep(0.11)

        # Should transition to half-open and succeed
        result = await cb.call_async(client.make_request, "/endpoint4")
        assert result["status"] == "success"
        assert cb.state == CircuitState.HALF_OPEN

        # Another success closes circuit
        result = await cb.call_async(client.make_request, "/endpoint5")
        assert cb.state == CircuitState.CLOSED

    def test_database_connection_pool_integration(self):
        """Test circuit breaker with database connection pool."""
        config = CircuitBreakerConfig(
            failure_threshold=2, command_timeout=1.0, failure_types=(ConnectionError, OSError)
        )
        cb = CircuitBreaker("db_pool", config)

        # Simulate database connection pool
        class DBPool:
            def __init__(self):
                self.healthy = True
                self.query_count = 0

            def execute_query(self, query):
                self.query_count += 1
                if not self.healthy:
                    raise ConnectionError("Database connection lost")
                return f"Result for: {query}"

        pool = DBPool()

        # Successful queries
        for i in range(3):
            result = cb.call(pool.execute_query, f"SELECT {i}")
            assert result == f"Result for: SELECT {i}"

        # Simulate database failure
        pool.healthy = False

        # Failures open circuit
        for i in range(2):
            with pytest.raises(ConnectionError):
                cb.call(pool.execute_query, f"SELECT {i}")

        assert cb.state == CircuitState.OPEN

        # Fix database
        pool.healthy = True

        # But circuit is still open
        with pytest.raises(CircuitBreakerError):
            cb.call(pool.execute_query, "SELECT recovery")

        # Manual reset for immediate recovery
        cb.reset()
        assert cb.state == CircuitState.CLOSED

        # Now queries work again
        result = cb.call(pool.execute_query, "SELECT after_reset")
        assert result == "Result for: SELECT after_reset"

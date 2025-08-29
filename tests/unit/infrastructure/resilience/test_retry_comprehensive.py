"""
Comprehensive tests for retry logic with exponential backoff.
Achieves 80%+ test coverage for production-ready retry mechanisms.
"""

import asyncio
import time

import pytest

from src.infrastructure.resilience.retry import (
    API_RETRY_CONFIG,
    BROKER_RETRY_CONFIG,
    DATABASE_RETRY_CONFIG,
    MARKET_DATA_RETRY_CONFIG,
    ExponentialBackoff,
    RetryConfig,
    RetryExhaustedException,
    is_retryable_exception,
    retry,
    retry_with_backoff,
    retry_with_backoff_sync,
)


class TestRetryConfigValidation:
    """Comprehensive tests for retry configuration validation."""

    def test_all_validation_errors(self):
        """Test all configuration validation error cases."""
        # Test max_retries validation
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RetryConfig(max_retries=-1)

        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RetryConfig(max_retries=-10)

        # Test initial_delay validation
        with pytest.raises(ValueError, match="initial_delay must be positive"):
            RetryConfig(initial_delay=0)

        with pytest.raises(ValueError, match="initial_delay must be positive"):
            RetryConfig(initial_delay=-1.0)

        # Test max_delay validation
        with pytest.raises(ValueError, match="max_delay must be positive"):
            RetryConfig(max_delay=0)

        with pytest.raises(ValueError, match="max_delay must be positive"):
            RetryConfig(max_delay=-5.0)

        # Test backoff_multiplier validation
        with pytest.raises(ValueError, match="backoff_multiplier must be greater than 1.0"):
            RetryConfig(backoff_multiplier=1.0)

        with pytest.raises(ValueError, match="backoff_multiplier must be greater than 1.0"):
            RetryConfig(backoff_multiplier=0.5)

        # Test jitter_range validation
        with pytest.raises(ValueError, match="jitter_range must be between 0 and 1"):
            RetryConfig(jitter_range=-0.1)

        with pytest.raises(ValueError, match="jitter_range must be between 0 and 1"):
            RetryConfig(jitter_range=1.1)

        with pytest.raises(ValueError, match="jitter_range must be between 0 and 1"):
            RetryConfig(jitter_range=2.0)

    def test_valid_edge_case_configs(self):
        """Test valid edge case configurations."""
        # Zero retries is valid (means no retry)
        config = RetryConfig(max_retries=0)
        assert config.max_retries == 0

        # Very small delays are valid
        config = RetryConfig(initial_delay=0.001, max_delay=0.001)
        assert config.initial_delay == 0.001

        # Backoff multiplier just above 1.0 is valid
        config = RetryConfig(backoff_multiplier=1.001)
        assert config.backoff_multiplier == 1.001

        # Jitter range at boundaries
        config = RetryConfig(jitter_range=0.0)
        assert config.jitter_range == 0.0

        config = RetryConfig(jitter_range=1.0)
        assert config.jitter_range == 1.0


class TestExponentialBackoffCalculation:
    """Comprehensive tests for exponential backoff calculation."""

    def test_get_delay_calculation(self):
        """Test delay calculation for various attempts."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=100.0, jitter=False)
        backoff = ExponentialBackoff(config)

        # Test negative attempt (edge case)
        assert backoff.get_delay(-1) == 0.0
        assert backoff.get_delay(-10) == 0.0

        # Test exponential progression
        assert backoff.get_delay(0) == 1.0  # 1 * 2^0 = 1
        assert backoff.get_delay(1) == 2.0  # 1 * 2^1 = 2
        assert backoff.get_delay(2) == 4.0  # 1 * 2^2 = 4
        assert backoff.get_delay(3) == 8.0  # 1 * 2^3 = 8
        assert backoff.get_delay(4) == 16.0  # 1 * 2^4 = 16

        # Test max_delay capping
        assert backoff.get_delay(10) == 100.0  # Would be 1024, but capped
        assert backoff.get_delay(20) == 100.0  # Still capped

    def test_different_exponential_bases(self):
        """Test different exponential base values."""
        # Base 3
        config = RetryConfig(
            initial_delay=1.0, exponential_base=3.0, max_delay=1000.0, jitter=False
        )
        backoff = ExponentialBackoff(config)

        assert backoff.get_delay(0) == 1.0  # 1 * 3^0 = 1
        assert backoff.get_delay(1) == 3.0  # 1 * 3^1 = 3
        assert backoff.get_delay(2) == 9.0  # 1 * 3^2 = 9
        assert backoff.get_delay(3) == 27.0  # 1 * 3^3 = 27

        # Base 1.5 (smaller multiplier)
        config = RetryConfig(initial_delay=2.0, exponential_base=1.5, max_delay=50.0, jitter=False)
        backoff = ExponentialBackoff(config)

        assert backoff.get_delay(0) == 2.0  # 2 * 1.5^0 = 2
        assert backoff.get_delay(1) == 3.0  # 2 * 1.5^1 = 3
        assert backoff.get_delay(2) == 4.5  # 2 * 1.5^2 = 4.5
        assert backoff.get_delay(3) == 6.75  # 2 * 1.5^3 = 6.75

    def test_jitter_application(self):
        """Test jitter is correctly applied to delays."""
        config = RetryConfig(
            initial_delay=10.0,
            exponential_base=2.0,
            max_delay=100.0,
            jitter=True,
            jitter_range=0.2,  # ±20%
        )
        backoff = ExponentialBackoff(config)

        # Generate multiple delays to test jitter
        delays = []
        for _ in range(100):
            delay = backoff.get_delay(1)  # Should be 20.0 base
            delays.append(delay)

        # All delays should be within 20 ± 20% (16 to 24)
        assert all(16.0 <= d <= 24.0 for d in delays)

        # Should have variation due to jitter
        assert len(set(delays)) > 50  # Should be many different values

        # Minimum delay should be respected
        assert all(d >= 0.1 for d in delays)

    def test_jitter_with_small_delays(self):
        """Test jitter with very small delays."""
        config = RetryConfig(
            initial_delay=0.01,
            exponential_base=2.0,
            max_delay=1.0,
            jitter=True,
            jitter_range=0.5,  # ±50%
        )
        backoff = ExponentialBackoff(config)

        # Even with jitter reducing delay, should maintain minimum
        delays = [backoff.get_delay(0) for _ in range(100)]

        # All should be at least 0.1 (minimum enforced)
        assert all(d >= 0.1 for d in delays)

    def test_get_delays_sequence(self):
        """Test getting a sequence of delays."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=10.0, jitter=False)
        backoff = ExponentialBackoff(config)

        # Get sequence of delays
        delays = backoff.get_delays(6)

        assert len(delays) == 6
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0
        assert delays[3] == 8.0
        assert delays[4] == 10.0  # Capped
        assert delays[5] == 10.0  # Capped

        # Empty sequence
        assert backoff.get_delays(0) == []

        # Single delay
        assert backoff.get_delays(1) == [1.0]


class TestRetryExceptionHandling:
    """Comprehensive tests for exception handling in retry logic."""

    def test_retryable_exception_classification(self):
        """Test classification of retryable vs non-retryable exceptions."""
        config = RetryConfig()

        # Default retryable exceptions
        assert is_retryable_exception(ConnectionError("test"), config) is True
        assert is_retryable_exception(TimeoutError("test"), config) is True
        assert is_retryable_exception(OSError("test"), config) is True

        # Default non-retryable exceptions
        assert is_retryable_exception(ValueError("test"), config) is False
        assert is_retryable_exception(TypeError("test"), config) is False
        assert is_retryable_exception(KeyError("test"), config) is False
        assert is_retryable_exception(AttributeError("test"), config) is False

        # Unknown exceptions default to non-retryable
        assert is_retryable_exception(RuntimeError("test"), config) is False
        assert is_retryable_exception(Exception("test"), config) is False

    def test_non_retryable_takes_precedence(self):
        """Test that non-retryable exceptions take precedence."""
        config = RetryConfig(
            retryable_exceptions=(Exception,),  # All exceptions retryable
            non_retryable_exceptions=(ValueError, TypeError),  # Except these
        )

        # Non-retryable takes precedence
        assert is_retryable_exception(ValueError("test"), config) is False
        assert is_retryable_exception(TypeError("test"), config) is False

        # Other exceptions are retryable
        assert is_retryable_exception(ConnectionError("test"), config) is True
        assert is_retryable_exception(RuntimeError("test"), config) is True

    def test_custom_exception_hierarchies(self):
        """Test exception hierarchies with custom exceptions."""

        class CustomError(Exception):
            pass

        class CustomNetworkError(CustomError):
            pass

        class CustomValueError(CustomError):
            pass

        config = RetryConfig(
            retryable_exceptions=(CustomNetworkError,), non_retryable_exceptions=(CustomValueError,)
        )

        assert is_retryable_exception(CustomNetworkError("test"), config) is True
        assert is_retryable_exception(CustomValueError("test"), config) is False
        assert is_retryable_exception(CustomError("test"), config) is False  # Base not listed


class TestAsyncRetryWithBackoff:
    """Comprehensive tests for async retry functionality."""

    @pytest.mark.asyncio
    async def test_successful_retry_after_failures(self):
        """Test successful retry after transient failures."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError(f"Failure {call_count}")
            return f"Success on attempt {call_count}"

        config = RetryConfig(max_retries=3, initial_delay=0.01, jitter=False)

        result = await retry_with_backoff(flaky_func, config=config)

        assert result == "Success on attempt 3"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry exhaustion after max attempts."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"Failure {call_count}")

        config = RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)

        with pytest.raises(RetryExhaustedException) as exc_info:
            await retry_with_backoff(always_fails, config=config)

        exc = exc_info
        assert exc.attempts == 3  # Initial + 2 retries
        assert isinstance(exc.last_exception, ConnectionError)
        assert exc.total_time > 0
        assert "Retry exhausted after 3 attempts" in str(exc)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test that non-retryable exceptions fail immediately."""
        call_count = 0

        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        config = RetryConfig(max_retries=3, initial_delay=0.01)

        with pytest.raises(RetryExhaustedException) as exc_info:
            await retry_with_backoff(raises_value_error, config=config)

        # Should only try once (no retries for non-retryable)
        assert call_count == 1
        assert exc_info.attempts == 1
        assert isinstance(exc_info.last_exception, ValueError)

    @pytest.mark.asyncio
    async def test_timeout_per_attempt(self):
        """Test timeout per individual attempt."""
        call_count = 0

        async def slow_func():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.5)  # Longer than timeout
            return "Should not reach here"

        config = RetryConfig(
            max_retries=2,
            initial_delay=0.01,
            timeout_per_attempt=0.1,  # 100ms timeout
        )

        with pytest.raises(RetryExhaustedException) as exc_info:
            await retry_with_backoff(slow_func, config=config)

        # Should timeout on each attempt
        assert call_count == 3
        assert isinstance(exc_info.last_exception, asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_total_timeout(self):
        """Test total timeout across all attempts."""
        call_count = 0

        async def slow_with_retries():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            raise ConnectionError("Retry me")

        config = RetryConfig(
            max_retries=10,  # Many retries
            initial_delay=0.01,
            total_timeout=0.15,  # But short total timeout
        )

        start = time.time()
        with pytest.raises(RetryExhaustedException):
            await retry_with_backoff(slow_with_retries, config=config)
        elapsed = time.time() - start

        # Should stop due to total timeout, not max retries
        assert call_count < 10
        assert elapsed < 0.3  # Should not take full retry time

    @pytest.mark.asyncio
    async def test_delay_adjustment_for_total_timeout(self):
        """Test that delays are adjusted to fit within total timeout."""
        call_count = 0
        delays_used = []

        async def track_delays():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                # Track time since last call
                delays_used.append(time.time())
            raise ConnectionError("Retry")

        config = RetryConfig(
            max_retries=5,
            initial_delay=0.1,
            total_timeout=0.25,  # Very short total timeout
            jitter=False,
        )

        with pytest.raises(RetryExhaustedException):
            await retry_with_backoff(track_delays, config=config)

        # Should have adjusted delays to fit timeout
        assert call_count <= 3  # Can't fit many attempts


class TestSyncRetryWithBackoff:
    """Comprehensive tests for sync retry functionality."""

    def test_successful_sync_retry(self):
        """Test successful sync retry after failures."""
        call_count = 0

        def flaky_sync_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError(f"Timeout {call_count}")
            return f"Success {call_count}"

        config = RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)

        result = retry_with_backoff_sync(flaky_sync_func, config=config)

        assert result == "Success 2"
        assert call_count == 2

    def test_sync_retry_exhausted(self):
        """Test sync retry exhaustion."""
        call_count = 0

        def always_fails_sync():
            nonlocal call_count
            call_count += 1
            raise OSError(f"OS Error {call_count}")

        config = RetryConfig(max_retries=1, initial_delay=0.01)

        with pytest.raises(RetryExhaustedException) as exc_info:
            retry_with_backoff_sync(always_fails_sync, config=config)

        assert exc_info.attempts == 2
        assert call_count == 2

    def test_sync_total_timeout(self):
        """Test sync retry with total timeout."""
        call_count = 0

        def slow_sync_func():
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)
            raise ConnectionError("Retry")

        config = RetryConfig(max_retries=10, initial_delay=0.01, total_timeout=0.15)

        start = time.time()
        with pytest.raises(RetryExhaustedException):
            retry_with_backoff_sync(slow_sync_func, config=config)
        elapsed = time.time() - start

        assert call_count < 10  # Stopped by timeout
        assert elapsed < 0.3

    def test_sync_with_arguments(self):
        """Test sync retry with function arguments."""

        def func_with_args(x, y, z=None):
            if z is None:
                raise ConnectionError("Need z parameter")
            return x + y + z

        config = RetryConfig(max_retries=2, initial_delay=0.01)

        # First attempt fails, then we fix it
        with pytest.raises(RetryExhaustedException):
            retry_with_backoff_sync(func_with_args, 1, 2, config=config)

        # With z parameter, should work
        result = retry_with_backoff_sync(func_with_args, 1, 2, config=config, z=3)
        assert result == 6


class TestRetryDecorator:
    """Comprehensive tests for retry decorator."""

    def test_sync_decorator(self):
        """Test retry decorator on sync function."""
        call_count = 0

        @retry(max_retries=2, initial_delay=0.01)
        def decorated_sync_func(should_fail=True):
            nonlocal call_count
            call_count += 1
            if should_fail and call_count < 2:
                raise ConnectionError("Fail")
            return f"Success {call_count}"

        result = decorated_sync_func()
        assert result == "Success 2"
        assert call_count == 2

        # Reset and test without failure
        call_count = 0
        result = decorated_sync_func(should_fail=False)
        assert result == "Success 1"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test retry decorator on async function."""
        call_count = 0

        @retry(max_retries=3, initial_delay=0.01, max_delay=0.05)
        async def decorated_async_func(fail_times=0):
            nonlocal call_count
            call_count += 1
            if call_count <= fail_times:
                raise TimeoutError(f"Timeout {call_count}")
            await asyncio.sleep(0.01)
            return f"Async success {call_count}"

        result = await decorated_async_func(fail_times=2)
        assert result == "Async success 3"
        assert call_count == 3

    def test_decorator_with_config(self):
        """Test decorator with RetryConfig object."""
        config = RetryConfig(
            max_retries=1, initial_delay=0.01, retryable_exceptions=(RuntimeError,)
        )

        @retry(config)
        def func_with_config():
            raise RuntimeError("Should retry")

        with pytest.raises(RetryExhaustedException) as exc_info:
            func_with_config()

        assert exc_info.attempts == 2  # 1 initial + 1 retry

    def test_decorator_config_override(self):
        """Test decorator overriding config parameters."""
        base_config = RetryConfig(max_retries=5, initial_delay=1.0, max_delay=60.0)

        @retry(base_config, max_retries=2, initial_delay=0.01)
        def overridden_func():
            raise ConnectionError("Fail")

        with pytest.raises(RetryExhaustedException) as exc_info:
            overridden_func()

        # Should use overridden values
        assert exc_info.attempts == 3  # 1 + 2 retries (overridden)

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @retry(max_retries=1)
        async def documented_func(x, y):
            """This function adds two numbers."""
            return x + y

        # Check metadata preserved
        assert documented_func.__name__ == "documented_func"
        assert "adds two numbers" in documented_func.__doc__

        # Function still works
        result = await documented_func(2, 3)
        assert result == 5


class TestPredefinedConfigurations:
    """Test predefined retry configurations."""

    def test_database_retry_config(self):
        """Test DATABASE_RETRY_CONFIG settings."""
        config = DATABASE_RETRY_CONFIG

        assert config.max_retries == 3
        assert config.initial_delay == 0.5
        assert config.max_delay == 10.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
        assert config.total_timeout == 30.0
        assert ConnectionError in config.retryable_exceptions

    def test_api_retry_config(self):
        """Test API_RETRY_CONFIG settings."""
        config = API_RETRY_CONFIG

        assert config.max_retries == 5
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.total_timeout == 120.0
        assert TimeoutError in config.retryable_exceptions

    def test_broker_retry_config(self):
        """Test BROKER_RETRY_CONFIG settings."""
        config = BROKER_RETRY_CONFIG

        assert config.max_retries == 2  # Conservative for trading
        assert config.initial_delay == 0.5
        assert config.max_delay == 5.0
        assert config.total_timeout == 15.0  # Fast timeout

    def test_market_data_retry_config(self):
        """Test MARKET_DATA_RETRY_CONFIG settings."""
        config = MARKET_DATA_RETRY_CONFIG

        assert config.max_retries == 4
        assert config.initial_delay == 0.25
        assert config.max_delay == 30.0
        assert config.total_timeout == 60.0

    @pytest.mark.asyncio
    async def test_using_predefined_configs(self):
        """Test using predefined configs in practice."""
        call_count = 0

        async def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("API timeout")
            return {"status": "success"}

        result = await retry_with_backoff(api_call, config=API_RETRY_CONFIG)
        assert result["status"] == "success"
        assert call_count == 3


class TestRetryIntegrationScenarios:
    """Integration tests for real-world retry scenarios."""

    @pytest.mark.asyncio
    async def test_database_connection_retry(self):
        """Test database connection retry scenario."""

        class DatabaseConnection:
            def __init__(self):
                self.connection_attempts = 0
                self.connected = False

            async def connect(self):
                self.connection_attempts += 1
                if self.connection_attempts < 3:
                    raise ConnectionError("Database unavailable")
                self.connected = True
                return "Connected to database"

            async def execute_query(self, query):
                if not self.connected:
                    raise RuntimeError("Not connected")
                return f"Result: {query}"

        db = DatabaseConnection()

        # Use database retry config
        result = await retry_with_backoff(db.connect, config=DATABASE_RETRY_CONFIG)

        assert result == "Connected to database"
        assert db.connection_attempts == 3
        assert db.connected is True

        # Queries should work now
        query_result = await db.execute_query("SELECT 1")
        assert query_result == "Result: SELECT 1"

    @pytest.mark.asyncio
    async def test_api_rate_limiting_retry(self):
        """Test API rate limiting retry scenario."""

        class RateLimitedAPI:
            def __init__(self):
                self.call_count = 0
                self.rate_limit_until = 3

            async def make_request(self, endpoint):
                self.call_count += 1

                if self.call_count < self.rate_limit_until:
                    # Simulate rate limiting
                    raise TimeoutError(f"Rate limited (attempt {self.call_count})")

                return {"endpoint": endpoint, "data": "response data", "attempt": self.call_count}

        api = RateLimitedAPI()

        # Configure retry for rate limiting
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.02,  # Start with small delay
            max_delay=0.1,
            exponential_base=2.0,  # Exponential backoff for rate limits
            jitter=True,
        )

        result = await retry_with_backoff(api.make_request, "/users", config=config)

        assert result["endpoint"] == "/users"
        assert result["attempt"] == 3
        assert api.call_count == 3

    def test_network_partition_recovery(self):
        """Test recovery from network partition."""

        class NetworkService:
            def __init__(self):
                self.partition_duration = 2
                self.call_count = 0

            def check_health(self):
                self.call_count += 1

                if self.call_count <= self.partition_duration:
                    raise OSError(f"Network unreachable (call {self.call_count})")

                return {"status": "healthy", "recovered_at": self.call_count}

        service = NetworkService()

        # Retry with appropriate config
        config = RetryConfig(
            max_retries=3, initial_delay=0.01, retryable_exceptions=(OSError, ConnectionError)
        )

        result = retry_with_backoff_sync(service.check_health, config=config)

        assert result["status"] == "healthy"
        assert result["recovered_at"] == 3
        assert service.call_count == 3

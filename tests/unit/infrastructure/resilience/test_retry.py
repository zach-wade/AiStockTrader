"""
Tests for retry logic with exponential backoff.
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


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
        assert config.jitter_range == 0.1
        assert config.exponential_base == 2.0

        assert ConnectionError in config.retryable_exceptions
        assert TimeoutError in config.retryable_exceptions
        assert OSError in config.retryable_exceptions

        assert ValueError in config.non_retryable_exceptions
        assert TypeError in config.non_retryable_exceptions

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        RetryConfig(max_retries=5)

        # Invalid configs should raise
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RetryConfig(max_retries=-1)

        with pytest.raises(ValueError, match="initial_delay must be positive"):
            RetryConfig(initial_delay=0)

        with pytest.raises(ValueError, match="max_delay must be positive"):
            RetryConfig(max_delay=0)

        with pytest.raises(ValueError, match="backoff_multiplier must be greater than 1.0"):
            RetryConfig(backoff_multiplier=1.0)

        with pytest.raises(ValueError, match="jitter_range must be between 0 and 1"):
            RetryConfig(jitter_range=-0.1)

        with pytest.raises(ValueError, match="jitter_range must be between 0 and 1"):
            RetryConfig(jitter_range=1.1)


class TestExponentialBackoff:
    """Test exponential backoff calculator."""

    def test_delay_calculation(self):
        """Test delay calculation for different attempts."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False,  # Disable jitter for predictable results
        )
        backoff = ExponentialBackoff(config)

        # Test exponential progression
        assert backoff.calculate_delay(0) == 1.0  # 1 * 2^0 = 1
        assert backoff.calculate_delay(1) == 2.0  # 1 * 2^1 = 2
        assert backoff.calculate_delay(2) == 4.0  # 1 * 2^2 = 4
        assert backoff.calculate_delay(3) == 8.0  # 1 * 2^3 = 8

        # Test max delay cap
        assert backoff.calculate_delay(4) == 10.0  # Capped at max_delay
        assert backoff.calculate_delay(10) == 10.0  # Still capped

    def test_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(
            initial_delay=10.0, exponential_base=2.0, max_delay=100.0, jitter=True, jitter_range=0.1
        )
        backoff = ExponentialBackoff(config)

        delays = [backoff.calculate_delay(1) for _ in range(10)]

        # All delays should be around 20.0 (10 * 2^1) with ±10% jitter
        for delay in delays:
            assert 18.0 <= delay <= 22.0  # 20 ± 2 (10%)
            assert delay >= 0.1  # Minimum delay

        # Should have some variation due to jitter
        assert len(set(delays)) > 1

    def test_get_delays(self):
        """Test getting all delays for a sequence."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=10.0, jitter=False)
        backoff = ExponentialBackoff(config)

        delays = backoff.get_delays(4)

        assert len(delays) == 4
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0
        assert delays[3] == 8.0


class TestRetryExceptions:
    """Test retry exception handling."""

    def test_is_retryable_exception(self):
        """Test exception retryability classification."""
        config = RetryConfig()

        # Retryable exceptions
        assert is_retryable_exception(ConnectionError("test"), config) is True
        assert is_retryable_exception(TimeoutError("test"), config) is True
        assert is_retryable_exception(OSError("test"), config) is True

        # Non-retryable exceptions (takes precedence)
        assert is_retryable_exception(ValueError("test"), config) is False
        assert is_retryable_exception(TypeError("test"), config) is False
        assert is_retryable_exception(KeyError("test"), config) is False

        # Unknown exceptions (default to non-retryable)
        assert is_retryable_exception(RuntimeError("test"), config) is False

    def test_custom_exception_config(self):
        """Test custom exception configuration."""
        config = RetryConfig(
            retryable_exceptions=(RuntimeError, ValueError),
            non_retryable_exceptions=(ConnectionError,),
        )

        # Custom retryable
        assert is_retryable_exception(RuntimeError("test"), config) is True
        assert is_retryable_exception(ValueError("test"), config) is True

        # Custom non-retryable (takes precedence)
        assert is_retryable_exception(ConnectionError("test"), config) is False

        # Default behavior for others
        assert is_retryable_exception(TimeoutError("test"), config) is False


class TestAsyncRetry:
    """Test async retry functionality."""

    @pytest.mark.asyncio
    async def test_successful_first_attempt(self):
        """Test successful function on first attempt."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        config = RetryConfig(max_retries=3)
        result = await retry_with_backoff(success_func, config=config)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_exception(self):
        """Test retry on retryable exceptions."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        config = RetryConfig(max_retries=3, initial_delay=0.01, max_delay=0.1)  # Fast for testing

        start_time = time.time()
        result = await retry_with_backoff(flaky_func, config=config)
        end_time = time.time()

        assert result == "success"
        assert call_count == 3

        # Should have taken some time due to delays
        assert end_time - start_time > 0.01

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_exception(self):
        """Test no retry on non-retryable exceptions."""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        config = RetryConfig(max_retries=3)

        with pytest.raises(RetryExhaustedException) as exc_info:
            await retry_with_backoff(failing_func, config=config)

        assert call_count == 1  # Only called once
        assert exc_info.attempts == 1
        assert isinstance(exc_info.last_exception, ValueError)

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test retry exhaustion with retryable exceptions."""
        call_count = 0

        async def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        config = RetryConfig(max_retries=2, initial_delay=0.01, max_delay=0.1)

        start_time = time.time()
        with pytest.raises(RetryExhaustedException) as exc_info:
            await retry_with_backoff(always_failing_func, config=config)
        end_time = time.time()

        assert call_count == 3  # Initial attempt + 2 retries
        assert exc_info.attempts == 3
        assert isinstance(exc_info.last_exception, ConnectionError)
        assert exc_info.total_time > 0.01

        # Should have taken some time due to delays
        assert end_time - start_time > 0.01

    @pytest.mark.asyncio
    async def test_timeout_per_attempt(self):
        """Test timeout per attempt."""
        call_count = 0

        async def slow_func():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Longer than timeout
            return "success"

        config = RetryConfig(
            max_retries=2,
            timeout_per_attempt=0.05,  # Shorter than function duration
            initial_delay=0.01,
        )

        with pytest.raises(RetryExhaustedException) as exc_info:
            await retry_with_backoff(slow_func, config=config)

        # Should have timed out and retried
        assert call_count >= 1
        assert isinstance(exc_info.last_exception, asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_total_timeout(self):
        """Test total timeout across all attempts."""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Failure")

        config = RetryConfig(
            max_retries=10,  # Would normally retry many times
            total_timeout=0.1,  # But total timeout is short
            initial_delay=0.02,
            max_delay=0.05,
        )

        start_time = time.time()
        with pytest.raises(RetryExhaustedException):
            await retry_with_backoff(failing_func, config=config)
        end_time = time.time()

        # Should have stopped due to total timeout
        assert end_time - start_time <= 0.15  # Some tolerance
        assert call_count < 10  # Shouldn't have done all retries


class TestSyncRetry:
    """Test synchronous retry functionality."""

    def test_sync_successful_retry(self):
        """Test successful sync retry."""
        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        config = RetryConfig(max_retries=3, initial_delay=0.01, max_delay=0.1)

        result = retry_with_backoff_sync(flaky_func, config=config)

        assert result == "success"
        assert call_count == 2

    def test_sync_retry_exhaustion(self):
        """Test sync retry exhaustion."""
        call_count = 0

        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        config = RetryConfig(max_retries=2, initial_delay=0.01)

        with pytest.raises(RetryExhaustedException):
            retry_with_backoff_sync(always_failing_func, config=config)

        assert call_count == 3  # Initial + 2 retries


class TestRetryDecorator:
    """Test retry decorator functionality."""

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test async retry decorator."""
        call_count = 0

        @retry(RetryConfig(max_retries=2, initial_delay=0.01))
        async def decorated_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "async_success"

        result = await decorated_async_func()

        assert result == "async_success"
        assert call_count == 2

    def test_sync_decorator(self):
        """Test sync retry decorator."""
        call_count = 0

        @retry(RetryConfig(max_retries=2, initial_delay=0.01))
        def decorated_sync_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "sync_success"

        result = decorated_sync_func()

        assert result == "sync_success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_with_overrides(self):
        """Test decorator with parameter overrides."""
        call_count = 0

        @retry(max_retries=1, initial_delay=0.01)
        async def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Failure")
            return "success"

        # Should only retry once due to override
        with pytest.raises(RetryExhaustedException):
            await decorated_func()

        assert call_count == 2  # Initial + 1 retry


class TestPredefinedConfigs:
    """Test predefined retry configurations."""

    def test_database_config(self):
        """Test database retry configuration."""
        config = DATABASE_RETRY_CONFIG

        assert config.max_retries == 3
        assert config.initial_delay == 0.5
        assert config.max_delay == 10.0
        assert config.total_timeout == 30.0

        assert ConnectionError in config.retryable_exceptions
        assert TimeoutError in config.retryable_exceptions

    def test_api_config(self):
        """Test API retry configuration."""
        config = API_RETRY_CONFIG

        assert config.max_retries == 5
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.total_timeout == 120.0

    def test_broker_config(self):
        """Test broker retry configuration (conservative)."""
        config = BROKER_RETRY_CONFIG

        assert config.max_retries == 2  # Conservative for trading
        assert config.initial_delay == 0.5
        assert config.max_delay == 5.0
        assert config.total_timeout == 15.0  # Fast for trading

    def test_market_data_config(self):
        """Test market data retry configuration."""
        config = MARKET_DATA_RETRY_CONFIG

        assert config.max_retries == 4
        assert config.initial_delay == 0.25
        assert config.max_delay == 30.0
        assert config.total_timeout == 60.0


# Fixtures
@pytest.fixture
def retry_config():
    """Create a retry config for testing."""
    return RetryConfig(
        max_retries=3,
        initial_delay=0.01,  # Fast for testing
        max_delay=0.1,
        jitter=False,  # Predictable for testing
    )

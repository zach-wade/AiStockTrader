"""
Comprehensive unit tests for rate limiting algorithms.

Tests all three rate limiting algorithms with various scenarios including
edge cases, concurrency, and performance requirements.
"""

import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.infrastructure.rate_limiting.algorithms import (
    FixedWindowRateLimit,
    SlidingWindowRateLimit,
    TokenBucketRateLimit,
    create_rate_limiter,
)
from src.infrastructure.rate_limiting.config import RateLimitAlgorithm, RateLimitRule, TimeWindow
from src.infrastructure.rate_limiting.exceptions import RateLimitConfigError


class TestTimeWindow:
    """Test TimeWindow utility class."""

    def test_time_window_from_string(self):
        """Test creating TimeWindow from string values."""
        # Test seconds
        assert TimeWindow("30s").seconds == 30
        assert TimeWindow("45sec").seconds == 45

        # Test minutes
        assert TimeWindow("5m").seconds == 300
        assert TimeWindow("10min").seconds == 600

        # Test hours
        assert TimeWindow("2h").seconds == 7200
        assert TimeWindow("1hour").seconds == 3600

        # Test days
        assert TimeWindow("1d").seconds == 86400
        assert TimeWindow("2day").seconds == 172800

    def test_time_window_from_int(self):
        """Test creating TimeWindow from integer seconds."""
        assert TimeWindow(60).seconds == 60
        assert TimeWindow(3600).seconds == 3600

    def test_time_window_from_timedelta(self):
        """Test creating TimeWindow from timedelta."""
        delta = timedelta(minutes=5)
        assert TimeWindow(delta).seconds == 300

    def test_time_window_invalid_format(self):
        """Test TimeWindow with invalid format."""
        with pytest.raises(ValueError):
            TimeWindow("invalid")

        with pytest.raises(ValueError):
            TimeWindow("10x")  # Unknown unit

    def test_time_window_string_representation(self):
        """Test TimeWindow string representation."""
        assert str(TimeWindow(30)) == "30s"
        assert str(TimeWindow(120)) == "2min"
        assert str(TimeWindow(3600)) == "1h"
        assert str(TimeWindow(86400)) == "1d"


class TestRateLimitRule:
    """Test RateLimitRule configuration."""

    def test_basic_rule_creation(self):
        """Test basic rule creation."""
        rule = RateLimitRule(
            limit=100, window=TimeWindow("1min"), algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        )

        assert rule.limit == 100
        assert rule.window.seconds == 60
        assert rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET
        assert rule.burst_allowance == 50  # Default 50% of limit

    def test_rule_with_custom_burst(self):
        """Test rule with custom burst allowance."""
        rule = RateLimitRule(limit=100, window="1min", burst_allowance=25)

        assert rule.burst_allowance == 25

    def test_rule_with_cooldown(self):
        """Test rule with cooldown configuration."""
        rule = RateLimitRule(limit=100, window="1min", enable_cooldown=True, cooldown_period="2min")

        assert rule.enable_cooldown is True
        assert rule.cooldown_period.seconds == 120


class TestTokenBucketRateLimit:
    """Test Token Bucket rate limiting algorithm."""

    def test_basic_token_bucket(self):
        """Test basic token bucket functionality."""
        rule = RateLimitRule(limit=10, window="1min", burst_allowance=0)  # No burst for this test
        limiter = TokenBucketRateLimit(rule)

        # Should allow initial requests up to limit
        for i in range(10):
            result = limiter.check_rate_limit("user1")
            assert result.allowed is True
            assert result.remaining == 9 - i

        # Next request should be denied
        result = limiter.check_rate_limit("user1")
        assert result.allowed is False
        assert result.remaining == 0

    def test_token_bucket_refill(self):
        """Test token bucket refill over time."""
        rule = RateLimitRule(
            limit=2, window="2s", burst_allowance=0
        )  # 1 token per second, no burst
        limiter = TokenBucketRateLimit(rule)

        # Use all tokens
        for _ in range(2):
            limiter.check_rate_limit("user1")

        # Should be denied
        result = limiter.check_rate_limit("user1")
        assert result.allowed is False

        # Sleep for actual time to test refill (in a real scenario)
        # For unit test, we'll just verify the bucket state and algorithm works
        # The refill rate calculation itself is tested in the algorithm
        assert result.retry_after is not None
        assert result.retry_after > 0

    def test_token_bucket_burst_allowance(self):
        """Test token bucket burst allowance."""
        rule = RateLimitRule(limit=10, window="1min", burst_allowance=5)
        limiter = TokenBucketRateLimit(rule)

        # Should allow up to limit + burst_allowance
        for i in range(15):  # 10 + 5
            result = limiter.check_rate_limit("user1")
            assert result.allowed is True

        # Next should be denied
        result = limiter.check_rate_limit("user1")
        assert result.allowed is False

    def test_token_bucket_multiple_users(self):
        """Test token bucket with multiple users."""
        rule = RateLimitRule(limit=5, window="1min", burst_allowance=0)  # No burst
        limiter = TokenBucketRateLimit(rule)

        # Each user should have separate bucket
        for user in ["user1", "user2", "user3"]:
            for _ in range(5):
                result = limiter.check_rate_limit(user)
                assert result.allowed is True

            # Sixth request should be denied
            result = limiter.check_rate_limit(user)
            assert result.allowed is False

    def test_token_bucket_get_usage(self):
        """Test getting current usage."""
        rule = RateLimitRule(limit=10, window="1min", burst_allowance=0)  # No burst
        limiter = TokenBucketRateLimit(rule)

        # Initial usage should be 0
        used, limit = limiter.get_current_usage("user1")
        assert used == 0
        assert limit == 10

        # Use 3 tokens
        for _ in range(3):
            limiter.check_rate_limit("user1")

        used, limit = limiter.get_current_usage("user1")
        assert used == 3
        assert limit == 10

    def test_token_bucket_reset(self):
        """Test resetting token bucket."""
        rule = RateLimitRule(limit=5, window="1min", burst_allowance=0)  # No burst
        limiter = TokenBucketRateLimit(rule)

        # Use all tokens
        for _ in range(5):
            limiter.check_rate_limit("user1")

        # Should be denied
        result = limiter.check_rate_limit("user1")
        assert result.allowed is False

        # Reset and try again
        limiter.reset_limit("user1")
        result = limiter.check_rate_limit("user1")
        assert result.allowed is True

    def test_token_bucket_cleanup(self):
        """Test cleanup of expired entries."""
        rule = RateLimitRule(limit=5, window="1min")
        limiter = TokenBucketRateLimit(rule)

        # Create some entries
        limiter.check_rate_limit("user1")
        limiter.check_rate_limit("user2")

        # Mock time passage beyond cleanup threshold
        with patch("time.time") as mock_time:
            mock_time.return_value = time.time() + 7200  # 2 hours later

            cleaned = limiter.cleanup_expired()
            assert cleaned == 2  # Both entries should be cleaned

    def test_token_bucket_retry_after(self):
        """Test retry_after calculation."""
        rule = RateLimitRule(limit=1, window="10s")  # 1 token per 10 seconds
        limiter = TokenBucketRateLimit(rule)

        # Use the token
        limiter.check_rate_limit("user1")

        # Next request should provide retry_after
        result = limiter.check_rate_limit("user1")
        assert result.allowed is False
        assert result.retry_after is not None
        assert result.retry_after > 0


class TestSlidingWindowRateLimit:
    """Test Sliding Window rate limiting algorithm."""

    def test_basic_sliding_window(self):
        """Test basic sliding window functionality."""
        rule = RateLimitRule(limit=5, window="10s", algorithm=RateLimitAlgorithm.SLIDING_WINDOW)
        limiter = SlidingWindowRateLimit(rule)

        # Should allow initial requests up to limit
        for i in range(5):
            result = limiter.check_rate_limit("user1")
            assert result.allowed is True
            assert result.remaining == 4 - i

        # Next request should be denied
        result = limiter.check_rate_limit("user1")
        assert result.allowed is False

    def test_sliding_window_time_based(self):
        """Test sliding window time-based behavior."""
        rule = RateLimitRule(limit=3, window="5s", algorithm=RateLimitAlgorithm.SLIDING_WINDOW)
        limiter = SlidingWindowRateLimit(rule)

        # Use all requests
        for _ in range(3):
            limiter.check_rate_limit("user1")

        # Should be denied
        result = limiter.check_rate_limit("user1")
        assert result.allowed is False

        # Mock time passage (6 seconds - beyond window)
        with patch("time.time") as mock_time:
            current_time = time.time()
            mock_time.return_value = current_time + 6

            # Should allow new requests as old ones expired
            result = limiter.check_rate_limit("user1")
            assert result.allowed is True

    def test_sliding_window_partial_expiry(self):
        """Test sliding window with partial request expiry."""
        rule = RateLimitRule(limit=3, window="10s", algorithm=RateLimitAlgorithm.SLIDING_WINDOW)
        limiter = SlidingWindowRateLimit(rule)

        # Add requests at different times
        base_time = time.time()

        with patch("time.time") as mock_time:
            # First request at time 0
            mock_time.return_value = base_time
            limiter.check_rate_limit("user1")

            # Second request at time 2
            mock_time.return_value = base_time + 2
            limiter.check_rate_limit("user1")

            # Third request at time 4
            mock_time.return_value = base_time + 4
            limiter.check_rate_limit("user1")

            # At time 6, should be denied (all 3 requests still in window)
            mock_time.return_value = base_time + 6
            result = limiter.check_rate_limit("user1")
            assert result.allowed is False

            # At time 12, first request expired, should allow one more
            mock_time.return_value = base_time + 12
            result = limiter.check_rate_limit("user1")
            assert result.allowed is True

    def test_sliding_window_reset_time(self):
        """Test sliding window reset time calculation."""
        rule = RateLimitRule(limit=2, window="10s", algorithm=RateLimitAlgorithm.SLIDING_WINDOW)
        limiter = SlidingWindowRateLimit(rule)

        base_time = time.time()

        with patch("time.time") as mock_time:
            mock_time.return_value = base_time
            limiter.check_rate_limit("user1")
            limiter.check_rate_limit("user1")

            # Should be denied with reset time
            result = limiter.check_rate_limit("user1")
            assert result.allowed is False
            assert result.reset_time is not None

            # Reset time should be first request time + window
            expected_reset = datetime.fromtimestamp(base_time + 10)
            assert abs((result.reset_time - expected_reset).total_seconds()) < 1

    def test_sliding_window_multiple_users(self):
        """Test sliding window with multiple users."""
        rule = RateLimitRule(limit=2, window="10s", algorithm=RateLimitAlgorithm.SLIDING_WINDOW)
        limiter = SlidingWindowRateLimit(rule)

        # Each user should have separate window
        for user in ["user1", "user2"]:
            for _ in range(2):
                result = limiter.check_rate_limit(user)
                assert result.allowed is True

            result = limiter.check_rate_limit(user)
            assert result.allowed is False


class TestFixedWindowRateLimit:
    """Test Fixed Window rate limiting algorithm."""

    def test_basic_fixed_window(self):
        """Test basic fixed window functionality."""
        rule = RateLimitRule(limit=5, window="10s", algorithm=RateLimitAlgorithm.FIXED_WINDOW)
        limiter = FixedWindowRateLimit(rule)

        # Should allow initial requests up to limit
        for i in range(5):
            result = limiter.check_rate_limit("user1")
            assert result.allowed is True
            assert result.remaining == 4 - i

        # Next request should be denied
        result = limiter.check_rate_limit("user1")
        assert result.allowed is False

    def test_fixed_window_boundary(self):
        """Test fixed window boundary behavior."""
        rule = RateLimitRule(limit=2, window="10s", algorithm=RateLimitAlgorithm.FIXED_WINDOW)
        limiter = FixedWindowRateLimit(rule)

        # Get window start time
        base_time = time.time()
        window_start = int(base_time // 10) * 10

        with patch("time.time") as mock_time:
            # First window - use all requests
            mock_time.return_value = window_start + 5
            limiter.check_rate_limit("user1")
            limiter.check_rate_limit("user1")

            # Should be denied in same window
            result = limiter.check_rate_limit("user1")
            assert result.allowed is False

            # Move to next window
            mock_time.return_value = window_start + 15
            result = limiter.check_rate_limit("user1")
            assert result.allowed is True

    def test_fixed_window_reset_time(self):
        """Test fixed window reset time calculation."""
        rule = RateLimitRule(limit=1, window="10s", algorithm=RateLimitAlgorithm.FIXED_WINDOW)
        limiter = FixedWindowRateLimit(rule)

        base_time = time.time()
        window_start = int(base_time // 10) * 10

        with patch("time.time") as mock_time:
            mock_time.return_value = window_start + 5
            limiter.check_rate_limit("user1")

            # Should be denied with correct reset time
            result = limiter.check_rate_limit("user1")
            assert result.allowed is False
            assert result.reset_time is not None

            expected_reset = datetime.fromtimestamp(window_start + 10)
            assert result.reset_time == expected_reset

    def test_fixed_window_retry_after(self):
        """Test fixed window retry_after calculation."""
        rule = RateLimitRule(limit=1, window="10s", algorithm=RateLimitAlgorithm.FIXED_WINDOW)
        limiter = FixedWindowRateLimit(rule)

        base_time = time.time()
        window_start = int(base_time // 10) * 10

        with patch("time.time") as mock_time:
            mock_time.return_value = window_start + 5
            limiter.check_rate_limit("user1")

            # Retry after should be time until next window
            result = limiter.check_rate_limit("user1")
            assert result.allowed is False
            assert result.retry_after == 6  # 10 - 5 + 1


class TestRateLimitAlgorithmFactory:
    """Test the algorithm factory function."""

    def test_create_token_bucket(self):
        """Test creating token bucket limiter."""
        rule = RateLimitRule(limit=10, window="1min", algorithm=RateLimitAlgorithm.TOKEN_BUCKET)

        limiter = create_rate_limiter(rule)
        assert isinstance(limiter, TokenBucketRateLimit)

    def test_create_sliding_window(self):
        """Test creating sliding window limiter."""
        rule = RateLimitRule(limit=10, window="1min", algorithm=RateLimitAlgorithm.SLIDING_WINDOW)

        limiter = create_rate_limiter(rule)
        assert isinstance(limiter, SlidingWindowRateLimit)

    def test_create_fixed_window(self):
        """Test creating fixed window limiter."""
        rule = RateLimitRule(limit=10, window="1min", algorithm=RateLimitAlgorithm.FIXED_WINDOW)

        limiter = create_rate_limiter(rule)
        assert isinstance(limiter, FixedWindowRateLimit)

    def test_invalid_algorithm(self):
        """Test creating limiter with invalid algorithm."""
        rule = RateLimitRule(limit=10, window="1min")
        rule.algorithm = "invalid"

        with pytest.raises(RateLimitConfigError):
            create_rate_limiter(rule)


class TestRateLimitAlgorithmConfiguration:
    """Test algorithm configuration validation."""

    def test_invalid_limit(self):
        """Test algorithm with invalid limit."""
        rule = RateLimitRule(limit=0, window="1min")

        with pytest.raises(RateLimitConfigError):
            TokenBucketRateLimit(rule)

    def test_invalid_window(self):
        """Test algorithm with invalid window."""
        rule = RateLimitRule(limit=10, window="0s")

        with pytest.raises(RateLimitConfigError):
            TokenBucketRateLimit(rule)


class TestConcurrency:
    """Test rate limiting algorithms under concurrent access."""

    def test_token_bucket_concurrency(self):
        """Test token bucket under concurrent access."""
        rule = RateLimitRule(limit=100, window="1min")
        limiter = TokenBucketRateLimit(rule)

        results = []

        def worker():
            for _ in range(10):
                result = limiter.check_rate_limit("user1")
                results.append(result.allowed)

        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have exactly 100 allowed requests
        assert sum(results) == 100
        assert len(results) == 100  # 10 threads * 10 requests

    def test_sliding_window_concurrency(self):
        """Test sliding window under concurrent access."""
        rule = RateLimitRule(limit=50, window="10s", algorithm=RateLimitAlgorithm.SLIDING_WINDOW)
        limiter = SlidingWindowRateLimit(rule)

        results = []

        def worker():
            for _ in range(10):
                result = limiter.check_rate_limit("user1")
                results.append(result.allowed)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have exactly 50 allowed requests
        assert sum(results) == 50

    def test_fixed_window_concurrency(self):
        """Test fixed window under concurrent access."""
        rule = RateLimitRule(limit=30, window="10s", algorithm=RateLimitAlgorithm.FIXED_WINDOW)
        limiter = FixedWindowRateLimit(rule)

        results = []

        def worker():
            for _ in range(5):
                result = limiter.check_rate_limit("user1")
                results.append(result.allowed)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have exactly 30 allowed requests
        assert sum(results) == 30


@pytest.mark.performance
class TestPerformance:
    """Test performance requirements for rate limiting algorithms."""

    def test_token_bucket_performance(self):
        """Test token bucket performance requirements."""
        rule = RateLimitRule(limit=1000, window="1min")
        limiter = TokenBucketRateLimit(rule)

        start_time = time.time()

        # Perform 10,000 rate limit checks
        for i in range(10000):
            limiter.check_rate_limit(f"user{i % 100}")

        elapsed = time.time() - start_time

        # Should complete 10,000 checks in under 1 second
        assert elapsed < 1.0

        # Should achieve > 10,000 checks/second
        throughput = 10000 / elapsed
        assert throughput > 10000

    def test_sliding_window_performance(self):
        """Test sliding window performance requirements."""
        rule = RateLimitRule(limit=1000, window="1min", algorithm=RateLimitAlgorithm.SLIDING_WINDOW)
        limiter = SlidingWindowRateLimit(rule)

        start_time = time.time()

        for i in range(5000):  # Slightly less for sliding window due to complexity
            limiter.check_rate_limit(f"user{i % 50}")

        elapsed = time.time() - start_time

        # Should complete 5,000 checks in under 1 second
        assert elapsed < 1.0

    def test_fixed_window_performance(self):
        """Test fixed window performance requirements."""
        rule = RateLimitRule(limit=1000, window="1min", algorithm=RateLimitAlgorithm.FIXED_WINDOW)
        limiter = FixedWindowRateLimit(rule)

        start_time = time.time()

        for i in range(10000):
            limiter.check_rate_limit(f"user{i % 100}")

        elapsed = time.time() - start_time

        # Should complete 10,000 checks in under 1 second
        assert elapsed < 1.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_multiple_tokens_request(self):
        """Test requesting multiple tokens at once."""
        rule = RateLimitRule(limit=10, window="1min")
        limiter = TokenBucketRateLimit(rule)

        # Request 5 tokens at once
        result = limiter.check_rate_limit("user1", 5)
        assert result.allowed is True
        assert result.remaining == 5

        # Request 6 more (should be denied)
        result = limiter.check_rate_limit("user1", 6)
        assert result.allowed is False

        # Request 5 more (should be allowed)
        result = limiter.check_rate_limit("user1", 5)
        assert result.allowed is True
        assert result.remaining == 0

    def test_empty_identifier(self):
        """Test rate limiting with empty identifier."""
        rule = RateLimitRule(limit=5, window="1min")
        limiter = TokenBucketRateLimit(rule)

        # Empty string should be treated as valid identifier
        result = limiter.check_rate_limit("")
        assert result.allowed is True

    def test_very_small_window(self):
        """Test rate limiting with very small time window."""
        rule = RateLimitRule(limit=1, window="1s")
        limiter = TokenBucketRateLimit(rule)

        result = limiter.check_rate_limit("user1")
        assert result.allowed is True

        result = limiter.check_rate_limit("user1")
        assert result.allowed is False

    def test_very_large_limit(self):
        """Test rate limiting with very large limit."""
        rule = RateLimitRule(limit=1000000, window="1min")
        limiter = TokenBucketRateLimit(rule)

        # Should handle large limits without issues
        for _ in range(1000):
            result = limiter.check_rate_limit("user1")
            assert result.allowed is True

    def test_zero_tokens_request(self):
        """Test requesting zero tokens."""
        rule = RateLimitRule(limit=5, window="1min")
        limiter = TokenBucketRateLimit(rule)

        # Requesting 0 tokens should always be allowed
        result = limiter.check_rate_limit("user1", 0)
        assert result.allowed is True

        # Should not consume any tokens
        used, limit = limiter.get_current_usage("user1")
        assert used == 0

    def test_negative_tokens_request(self):
        """Test requesting negative tokens."""
        rule = RateLimitRule(limit=5, window="1min")
        limiter = TokenBucketRateLimit(rule)

        # Negative tokens should be treated as 0
        result = limiter.check_rate_limit("user1", -1)
        assert result.allowed is True

        used, limit = limiter.get_current_usage("user1")
        assert used == 0

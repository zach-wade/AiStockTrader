"""
Core rate limiting algorithms for the AI Trading System.

Implements Token Bucket, Sliding Window, and Fixed Window algorithms
with high performance and accuracy.
"""

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from .config import RateLimitRule
from .exceptions import RateLimitConfigError


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    current_count: int
    limit: int
    remaining: int
    reset_time: datetime | None
    retry_after: int | None  # Seconds to wait before retry


class RateLimitAlgorithm(ABC):
    """Abstract base class for rate limiting algorithms."""

    def __init__(self, rule: RateLimitRule) -> None:
        self.rule = rule
        self.lock = threading.RLock()

        if rule.limit <= 0:
            raise RateLimitConfigError("Rate limit must be positive")
        if rule.window.seconds <= 0:
            raise RateLimitConfigError("Time window must be positive")

    @abstractmethod
    def check_rate_limit(self, identifier: str, tokens: int = 1) -> RateLimitResult:
        """Check if request is within rate limit."""
        pass

    @abstractmethod
    def get_current_usage(self, identifier: str) -> tuple[int, int]:
        """Get current usage count and limit."""
        pass

    @abstractmethod
    def reset_limit(self, identifier: str) -> None:
        """Reset rate limit for identifier."""
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Clean up expired entries and return number removed."""
        pass


class TokenBucketRateLimit(RateLimitAlgorithm):
    """
    Token Bucket rate limiting algorithm.

    Allows burst traffic up to bucket capacity while maintaining
    average rate over time. Most suitable for APIs that need to
    handle occasional traffic spikes.
    """

    def __init__(self, rule: RateLimitRule) -> None:
        super().__init__(rule)

        # Each identifier has: (tokens, last_refill_time)
        self._buckets: dict[str, tuple[float, float]] = {}

        # Calculate refill rate (tokens per second)
        self._refill_rate = rule.limit / rule.window.seconds
        self._bucket_capacity = rule.limit + (rule.burst_allowance or 0)

    def check_rate_limit(self, identifier: str, tokens: int = 1) -> RateLimitResult:
        """Check rate limit using token bucket algorithm."""
        with self.lock:
            current_time = time.time()

            # Get or create bucket
            if identifier not in self._buckets:
                self._buckets[identifier] = (self._bucket_capacity, current_time)

            current_tokens, last_refill = self._buckets[identifier]

            # Calculate tokens to add based on time elapsed
            time_elapsed = current_time - last_refill
            tokens_to_add = time_elapsed * self._refill_rate

            # Update bucket
            new_tokens = min(self._bucket_capacity, current_tokens + tokens_to_add)

            # Check if we have enough tokens
            if new_tokens >= tokens:
                # Allow request
                new_tokens -= tokens
                self._buckets[identifier] = (new_tokens, current_time)

                return RateLimitResult(
                    allowed=True,
                    current_count=int(self._bucket_capacity - new_tokens),
                    limit=self.rule.limit,
                    remaining=int(new_tokens),
                    reset_time=None,  # Token bucket doesn't have fixed reset time
                    retry_after=None,
                )
            else:
                # Deny request
                self._buckets[identifier] = (new_tokens, current_time)

                # Calculate retry after (time to get enough tokens)
                tokens_needed = tokens - new_tokens
                retry_after = int(tokens_needed / self._refill_rate) + 1

                return RateLimitResult(
                    allowed=False,
                    current_count=int(self._bucket_capacity - new_tokens),
                    limit=self.rule.limit,
                    remaining=int(new_tokens),
                    reset_time=datetime.fromtimestamp(current_time + retry_after),
                    retry_after=retry_after,
                )

    def get_current_usage(self, identifier: str) -> tuple[int, int]:
        """Get current token usage."""
        with self.lock:
            if identifier not in self._buckets:
                return 0, self.rule.limit

            current_tokens, last_refill = self._buckets[identifier]
            current_time = time.time()

            # Update tokens
            time_elapsed = current_time - last_refill
            tokens_to_add = time_elapsed * self._refill_rate
            updated_tokens = min(self._bucket_capacity, current_tokens + tokens_to_add)

            used_tokens = int(self._bucket_capacity - updated_tokens)
            return used_tokens, self.rule.limit

    def reset_limit(self, identifier: str) -> None:
        """Reset token bucket for identifier."""
        with self.lock:
            if identifier in self._buckets:
                self._buckets[identifier] = (self._bucket_capacity, time.time())

    def cleanup_expired(self) -> int:
        """Clean up buckets that haven't been used recently."""
        with self.lock:
            current_time = time.time()
            expired_threshold = current_time - (self.rule.window.seconds * 2)

            expired_keys = [
                key
                for key, (_, last_refill) in self._buckets.items()
                if last_refill < expired_threshold
            ]

            for key in expired_keys:
                del self._buckets[key]

            return len(expired_keys)


class SlidingWindowRateLimit(RateLimitAlgorithm):
    """
    Sliding Window rate limiting algorithm.

    Maintains precise tracking of requests within a sliding time window.
    Provides accurate rate limiting but uses more memory than other algorithms.
    """

    def __init__(self, rule: RateLimitRule) -> None:
        super().__init__(rule)

        # Each identifier has a deque of request timestamps
        self._windows: dict[str, deque[float]] = {}

    def check_rate_limit(self, identifier: str, tokens: int = 1) -> RateLimitResult:
        """Check rate limit using sliding window algorithm."""
        with self.lock:
            current_time = time.time()
            window_start = current_time - self.rule.window.seconds

            # Get or create window
            if identifier not in self._windows:
                self._windows[identifier] = deque()

            window = self._windows[identifier]

            # Remove expired requests
            while window and window[0] <= window_start:
                window.popleft()

            # Check if adding tokens would exceed limit
            if len(window) + tokens <= self.rule.limit:
                # Allow request
                for _ in range(tokens):
                    window.append(current_time)

                reset_time = (
                    datetime.fromtimestamp(window[0] + self.rule.window.seconds) if window else None
                )

                return RateLimitResult(
                    allowed=True,
                    current_count=len(window),
                    limit=self.rule.limit,
                    remaining=self.rule.limit - len(window),
                    reset_time=reset_time,
                    retry_after=None,
                )
            else:
                # Deny request
                retry_after = None
                if window:
                    # Calculate when oldest request will expire
                    oldest_request = window[0]
                    retry_after = int(oldest_request + self.rule.window.seconds - current_time) + 1

                reset_time = (
                    datetime.fromtimestamp(window[0] + self.rule.window.seconds)
                    if window
                    else datetime.fromtimestamp(current_time + self.rule.window.seconds)
                )

                return RateLimitResult(
                    allowed=False,
                    current_count=len(window),
                    limit=self.rule.limit,
                    remaining=self.rule.limit - len(window),
                    reset_time=reset_time,
                    retry_after=retry_after,
                )

    def get_current_usage(self, identifier: str) -> tuple[int, int]:
        """Get current window usage."""
        with self.lock:
            if identifier not in self._windows:
                return 0, self.rule.limit

            current_time = time.time()
            window_start = current_time - self.rule.window.seconds
            window = self._windows[identifier]

            # Remove expired requests
            while window and window[0] <= window_start:
                window.popleft()

            return len(window), self.rule.limit

    def reset_limit(self, identifier: str) -> None:
        """Reset sliding window for identifier."""
        with self.lock:
            if identifier in self._windows:
                self._windows[identifier].clear()

    def cleanup_expired(self) -> int:
        """Clean up expired entries from all windows."""
        with self.lock:
            current_time = time.time()
            window_start = current_time - self.rule.window.seconds
            cleaned_count = 0

            # Clean expired requests from all windows
            for identifier, window in list(self._windows.items()):
                original_size = len(window)

                # Remove expired requests
                while window and window[0] <= window_start:
                    window.popleft()

                cleaned_count += original_size - len(window)

                # Remove empty windows that haven't been used recently
                if not window:
                    del self._windows[identifier]

            return cleaned_count


class FixedWindowRateLimit(RateLimitAlgorithm):
    """
    Fixed Window rate limiting algorithm.

    Divides time into fixed windows and tracks requests per window.
    Memory efficient but can allow traffic bursts at window boundaries.
    """

    def __init__(self, rule: RateLimitRule) -> None:
        super().__init__(rule)

        # Each identifier has: (window_start, request_count)
        self._windows: dict[str, tuple[float, int]] = {}

    def _get_window_start(self, current_time: float) -> float:
        """Get the start time of the current window."""
        return int(current_time // self.rule.window.seconds) * self.rule.window.seconds

    def check_rate_limit(self, identifier: str, tokens: int = 1) -> RateLimitResult:
        """Check rate limit using fixed window algorithm."""
        with self.lock:
            current_time = time.time()
            window_start = self._get_window_start(current_time)

            # Get or create window
            if identifier not in self._windows:
                self._windows[identifier] = (window_start, 0)

            stored_window_start, request_count = self._windows[identifier]

            # Check if we're in a new window
            if stored_window_start < window_start:
                # New window, reset count
                request_count = 0
                stored_window_start = window_start

            # Check if adding tokens would exceed limit
            if request_count + tokens <= self.rule.limit:
                # Allow request
                new_count = request_count + tokens
                self._windows[identifier] = (stored_window_start, new_count)

                reset_time = datetime.fromtimestamp(window_start + self.rule.window.seconds)

                return RateLimitResult(
                    allowed=True,
                    current_count=new_count,
                    limit=self.rule.limit,
                    remaining=self.rule.limit - new_count,
                    reset_time=reset_time,
                    retry_after=None,
                )
            else:
                # Deny request
                retry_after = int(window_start + self.rule.window.seconds - current_time) + 1
                reset_time = datetime.fromtimestamp(window_start + self.rule.window.seconds)

                return RateLimitResult(
                    allowed=False,
                    current_count=request_count,
                    limit=self.rule.limit,
                    remaining=self.rule.limit - request_count,
                    reset_time=reset_time,
                    retry_after=retry_after,
                )

    def get_current_usage(self, identifier: str) -> tuple[int, int]:
        """Get current window usage."""
        with self.lock:
            if identifier not in self._windows:
                return 0, self.rule.limit

            current_time = time.time()
            window_start = self._get_window_start(current_time)
            stored_window_start, request_count = self._windows[identifier]

            # Check if we're in a new window
            if stored_window_start < window_start:
                return 0, self.rule.limit

            return request_count, self.rule.limit

    def reset_limit(self, identifier: str) -> None:
        """Reset fixed window for identifier."""
        with self.lock:
            if identifier in self._windows:
                current_time = time.time()
                window_start = self._get_window_start(current_time)
                self._windows[identifier] = (window_start, 0)

    def cleanup_expired(self) -> int:
        """Clean up expired windows."""
        with self.lock:
            current_time = time.time()
            current_window = self._get_window_start(current_time)

            expired_keys = [
                key
                for key, (window_start, _) in self._windows.items()
                if window_start < current_window - self.rule.window.seconds
            ]

            for key in expired_keys:
                del self._windows[key]

            return len(expired_keys)


def create_rate_limiter(rule: RateLimitRule) -> RateLimitAlgorithm:
    """Factory function to create appropriate rate limiter."""
    from .config import RateLimitAlgorithm as AlgorithmType

    if rule.algorithm == AlgorithmType.TOKEN_BUCKET:
        return TokenBucketRateLimit(rule)
    elif rule.algorithm == AlgorithmType.SLIDING_WINDOW:
        return SlidingWindowRateLimit(rule)
    elif rule.algorithm == AlgorithmType.FIXED_WINDOW:
        return FixedWindowRateLimit(rule)
    else:
        raise RateLimitConfigError(f"Unknown algorithm: {rule.algorithm}")

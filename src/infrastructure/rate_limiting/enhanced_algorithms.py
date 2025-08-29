"""
Enhanced rate limiting algorithms with backoff mechanisms and adaptive limiting.

Provides advanced rate limiting capabilities for high-throughput trading systems.
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .algorithms import RateLimitAlgorithm, RateLimitResult
from .config import RateLimitRule

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for rate limiting."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    JITTERED_EXPONENTIAL = "jittered_exponential"
    POLYNOMIAL = "polynomial"


@dataclass
class BackoffConfig:
    """Configuration for backoff mechanisms."""

    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    base_delay: int = 1  # seconds
    max_delay: int = 300  # 5 minutes max
    multiplier: float = 2.0
    jitter_factor: float = 0.1  # 10% jitter
    reset_after: int = 3600  # Reset after 1 hour


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive rate limiting."""

    enabled: bool = False
    base_limit: int = 100
    min_limit: int = 10
    max_limit: int = 1000
    adjustment_factor: float = 0.1
    success_threshold: float = 0.95  # 95% success rate
    error_threshold: float = 0.05  # 5% error rate
    measurement_window: int = 300  # 5 minutes


class EnhancedRateLimitAlgorithm(RateLimitAlgorithm, ABC):
    """Base class for enhanced rate limiting algorithms."""

    def __init__(
        self,
        rule: RateLimitRule,
        backoff_config: BackoffConfig | None = None,
        adaptive_config: AdaptiveConfig | None = None,
    ):
        super().__init__(rule)
        self.backoff_config = backoff_config or BackoffConfig()
        self.adaptive_config = adaptive_config or AdaptiveConfig()

        # Track backoff state per identifier
        self._backoff_state: dict[str, dict[str, Any]] = {}

        # Track adaptive state
        self._adaptive_state: dict[str, dict[str, Any]] = {}

    @abstractmethod
    def _base_check_rate_limit(self, identifier: str, tokens: int = 1) -> RateLimitResult:
        """Base rate limit check without backoff or adaptive logic."""
        pass

    def check_rate_limit(self, identifier: str, tokens: int = 1) -> RateLimitResult:
        """Enhanced rate limit check with backoff and adaptive features."""
        current_time = time.time()

        # Check if identifier is in backoff period
        if self._is_in_backoff(identifier, current_time):
            return self._create_backoff_result(identifier, current_time)

        # Get current limit (may be adjusted by adaptive algorithm)
        current_limit = self._get_adaptive_limit(identifier, current_time)

        # Update rule with adaptive limit if different
        original_limit = self.rule.limit
        if current_limit != original_limit:
            self.rule = RateLimitRule(
                limit=current_limit,
                window=self.rule.window,
                algorithm=self.rule.algorithm,
                burst_allowance=self.rule.burst_allowance,
                action=self.rule.action,
            )

        try:
            # Perform base rate limit check
            result = self._base_check_rate_limit(identifier, tokens)

            # Update adaptive state
            self._update_adaptive_state(identifier, current_time, result.allowed)

            # Handle rate limit exceeded
            if not result.allowed:
                self._handle_rate_limit_exceeded(identifier, current_time)
            else:
                # Reset backoff on successful request
                self._reset_backoff(identifier)

            return result

        finally:
            # Restore original rule
            if current_limit != original_limit:
                self.rule = RateLimitRule(
                    limit=original_limit,
                    window=self.rule.window,
                    algorithm=self.rule.algorithm,
                    burst_allowance=self.rule.burst_allowance,
                    action=self.rule.action,
                )

    def _is_in_backoff(self, identifier: str, current_time: float) -> bool:
        """Check if identifier is currently in backoff period."""
        if identifier not in self._backoff_state:
            return False

        state = self._backoff_state[identifier]
        backoff_until = state.get("backoff_until", 0)

        return bool(current_time < backoff_until)

    def _create_backoff_result(self, identifier: str, current_time: float) -> RateLimitResult:
        """Create rate limit result for backoff period."""
        state = self._backoff_state[identifier]
        backoff_until = state.get("backoff_until", current_time)
        retry_after = int(backoff_until - current_time)

        return RateLimitResult(
            allowed=False,
            limit=self.rule.limit,
            remaining=0,
            current_count=self.rule.limit,
            reset_time=datetime.fromtimestamp(backoff_until),
            retry_after=retry_after,
        )

    def _handle_rate_limit_exceeded(self, identifier: str, current_time: float) -> None:
        """Handle rate limit exceeded by updating backoff state."""
        if identifier not in self._backoff_state:
            self._backoff_state[identifier] = {
                "consecutive_violations": 0,
                "first_violation": current_time,
                "last_violation": current_time,
            }

        state = self._backoff_state[identifier]
        state["consecutive_violations"] += 1
        state["last_violation"] = current_time

        # Calculate backoff delay
        delay = self._calculate_backoff_delay(state["consecutive_violations"])
        state["backoff_until"] = current_time + delay

        # Reset if too much time has passed
        if current_time - state["first_violation"] > self.backoff_config.reset_after:
            state["consecutive_violations"] = 1
            state["first_violation"] = current_time

        logger.debug(
            f"Rate limit exceeded for {identifier}, "
            f"backoff until {datetime.fromtimestamp(state['backoff_until'])}"
        )

    def _reset_backoff(self, identifier: str) -> None:
        """Reset backoff state on successful request."""
        if identifier in self._backoff_state:
            # Gradually reduce consecutive violations
            state = self._backoff_state[identifier]
            if state["consecutive_violations"] > 0:
                state["consecutive_violations"] = max(0, state["consecutive_violations"] - 1)

            # Clear backoff if no violations
            if state["consecutive_violations"] == 0:
                state.pop("backoff_until", None)

    def _calculate_backoff_delay(self, consecutive_violations: int) -> float:
        """Calculate backoff delay based on strategy."""
        base_delay = self.backoff_config.base_delay
        max_delay = self.backoff_config.max_delay
        multiplier = self.backoff_config.multiplier

        delay: float
        if self.backoff_config.strategy == BackoffStrategy.FIXED:
            delay = float(base_delay)
        elif self.backoff_config.strategy == BackoffStrategy.LINEAR:
            delay = float(base_delay * consecutive_violations)
        elif self.backoff_config.strategy == BackoffStrategy.EXPONENTIAL:
            delay = float(base_delay * (multiplier ** (consecutive_violations - 1)))
        elif self.backoff_config.strategy == BackoffStrategy.JITTERED_EXPONENTIAL:
            base_exp_delay = float(base_delay * (multiplier ** (consecutive_violations - 1)))
            jitter = float(base_exp_delay * self.backoff_config.jitter_factor)
            delay = float(base_exp_delay + random.uniform(-jitter, jitter))
        elif self.backoff_config.strategy == BackoffStrategy.POLYNOMIAL:
            delay = float(base_delay * (consecutive_violations**2))

        return min(delay, max_delay)

    def _get_adaptive_limit(self, identifier: str, current_time: float) -> int:
        """Get current limit with adaptive adjustments."""
        if not self.adaptive_config.enabled:
            return self.rule.limit

        if identifier not in self._adaptive_state:
            self._adaptive_state[identifier] = {
                "current_limit": self.adaptive_config.base_limit,
                "success_count": 0,
                "total_count": 0,
                "last_adjustment": current_time,
                "measurement_start": current_time,
            }

        state = self._adaptive_state[identifier]

        # Check if measurement window has passed
        if current_time - state["measurement_start"] >= self.adaptive_config.measurement_window:
            self._adjust_adaptive_limit(identifier, current_time)

        return int(state["current_limit"])

    def _update_adaptive_state(self, identifier: str, current_time: float, success: bool) -> None:
        """Update adaptive state based on request result."""
        if not self.adaptive_config.enabled:
            return

        if identifier not in self._adaptive_state:
            return

        state = self._adaptive_state[identifier]
        state["total_count"] += 1
        if success:
            state["success_count"] += 1

    def _adjust_adaptive_limit(self, identifier: str, current_time: float) -> None:
        """Adjust rate limit based on success/error rates."""
        state = self._adaptive_state[identifier]

        if state["total_count"] == 0:
            return

        success_rate = state["success_count"] / state["total_count"]
        error_rate = 1 - success_rate

        current_limit = state["current_limit"]
        adjustment_factor = self.adaptive_config.adjustment_factor

        # Adjust limit based on performance
        if success_rate >= self.adaptive_config.success_threshold:
            # High success rate - increase limit
            new_limit = int(current_limit * (1 + adjustment_factor))
        elif error_rate >= self.adaptive_config.error_threshold:
            # High error rate - decrease limit
            new_limit = int(current_limit * (1 - adjustment_factor))
        else:
            # Within acceptable range - no change
            new_limit = current_limit

        # Apply min/max constraints
        new_limit = max(self.adaptive_config.min_limit, new_limit)
        new_limit = min(self.adaptive_config.max_limit, new_limit)

        # Update state
        state["current_limit"] = new_limit
        state["success_count"] = 0
        state["total_count"] = 0
        state["last_adjustment"] = current_time
        state["measurement_start"] = current_time

        logger.info(
            f"Adaptive limit adjusted for {identifier}: "
            f"{current_limit} -> {new_limit} (success_rate: {success_rate:.2%})"
        )

    def cleanup_expired(self) -> int:
        """Clean up expired backoff and adaptive state."""
        current_time = time.time()
        cleaned = 0

        # Clean up backoff state
        expired_backoff = []
        for identifier, state in self._backoff_state.items():
            backoff_until = state.get("backoff_until", 0)
            last_violation = state.get("last_violation", 0)

            # Remove if backoff expired and no recent violations
            if (
                backoff_until < current_time
                and current_time - last_violation > self.backoff_config.reset_after
            ):
                expired_backoff.append(identifier)

        for identifier in expired_backoff:
            del self._backoff_state[identifier]
            cleaned += 1

        # Clean up adaptive state
        expired_adaptive = []
        for identifier, state in self._adaptive_state.items():
            last_adjustment = state.get("last_adjustment", 0)

            # Remove if no activity for a long time
            if current_time - last_adjustment > self.adaptive_config.measurement_window * 4:
                expired_adaptive.append(identifier)

        for identifier in expired_adaptive:
            del self._adaptive_state[identifier]
            cleaned += 1

        # Note: Base algorithm cleanup should be implemented by concrete classes

        return cleaned


class EnhancedTokenBucket(EnhancedRateLimitAlgorithm):
    """Enhanced token bucket with backoff and adaptive features."""

    def __init__(
        self,
        rule: RateLimitRule,
        backoff_config: BackoffConfig | None = None,
        adaptive_config: AdaptiveConfig | None = None,
    ):
        super().__init__(rule, backoff_config, adaptive_config)
        self._buckets: dict[str, dict[str, Any]] = {}

    def _base_check_rate_limit(self, identifier: str, tokens: int = 1) -> RateLimitResult:
        """Token bucket rate limiting without enhancements."""
        current_time = time.time()

        if identifier not in self._buckets:
            self._buckets[identifier] = {
                "tokens": float(self.rule.limit),
                "last_refill": current_time,
            }

        bucket = self._buckets[identifier]

        # Calculate refill
        time_passed = current_time - bucket["last_refill"]
        refill_rate = self.rule.limit / self.rule.window.seconds
        tokens_to_add = time_passed * refill_rate

        # Update bucket
        bucket["tokens"] = min(self.rule.limit, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = current_time

        # Check if enough tokens
        if bucket["tokens"] >= tokens:
            bucket["tokens"] -= tokens
            allowed = True
            remaining = int(bucket["tokens"])
        else:
            allowed = False
            remaining = int(bucket["tokens"])

        # Calculate reset time
        if not allowed:
            tokens_needed = tokens - bucket["tokens"]
            time_to_refill = tokens_needed / refill_rate
            reset_time = datetime.fromtimestamp(current_time + time_to_refill)
            retry_after = int(time_to_refill) + 1
        else:
            reset_time = None
            retry_after = None

        return RateLimitResult(
            allowed=allowed,
            limit=self.rule.limit,
            remaining=remaining,
            current_count=self.rule.limit - remaining,
            reset_time=reset_time,
            retry_after=retry_after,
        )

    def get_current_usage(self, identifier: str) -> tuple[int, int]:
        """Get current usage count and limit for token bucket."""
        with self.lock:
            if identifier in self._buckets:
                bucket = self._buckets[identifier]
                used = int(self.rule.limit - bucket["tokens"])
                return used, self.rule.limit
            return 0, self.rule.limit

    def reset_limit(self, identifier: str) -> None:
        """Reset rate limit for identifier in token bucket."""
        with self.lock:
            if identifier in self._buckets:
                self._buckets[identifier]["tokens"] = float(self.rule.limit)
                self._buckets[identifier]["last_refill"] = time.time()
            # Also reset enhancement states
            self._reset_backoff(identifier)


class EnhancedSlidingWindow(EnhancedRateLimitAlgorithm):
    """Enhanced sliding window with backoff and adaptive features."""

    def __init__(
        self,
        rule: RateLimitRule,
        backoff_config: BackoffConfig | None = None,
        adaptive_config: AdaptiveConfig | None = None,
    ):
        super().__init__(rule, backoff_config, adaptive_config)
        self._windows: dict[str, list[float]] = {}

    def _base_check_rate_limit(self, identifier: str, tokens: int = 1) -> RateLimitResult:
        """Sliding window rate limiting without enhancements."""
        current_time = time.time()
        window_start = current_time - self.rule.window.seconds

        if identifier not in self._windows:
            self._windows[identifier] = []

        window = self._windows[identifier]

        # Remove old entries
        window[:] = [t for t in window if t > window_start]

        # Check if we can add more requests
        if len(window) + tokens <= self.rule.limit:
            # Add current requests
            for _ in range(tokens):
                window.append(current_time)
            allowed = True
            remaining = self.rule.limit - len(window)
        else:
            allowed = False
            remaining = max(0, self.rule.limit - len(window))

        # Calculate reset time
        if not allowed and window:
            oldest_request = min(window)
            reset_time = datetime.fromtimestamp(oldest_request + self.rule.window.seconds)
            retry_after = int(oldest_request + self.rule.window.seconds - current_time)
        else:
            reset_time = None
            retry_after = None

        return RateLimitResult(
            allowed=allowed,
            limit=self.rule.limit,
            remaining=remaining,
            current_count=len(window),
            reset_time=reset_time,
            retry_after=retry_after,
        )

    def get_current_usage(self, identifier: str) -> tuple[int, int]:
        """Get current usage count and limit for sliding window."""
        with self.lock:
            if identifier in self._windows:
                current_time = time.time()
                window_start = current_time - self.rule.window.seconds
                # Clean old entries
                self._windows[identifier] = [
                    t for t in self._windows[identifier] if t > window_start
                ]
                return len(self._windows[identifier]), self.rule.limit
            return 0, self.rule.limit

    def reset_limit(self, identifier: str) -> None:
        """Reset rate limit for identifier in sliding window."""
        with self.lock:
            if identifier in self._windows:
                self._windows[identifier] = []
            # Also reset enhancement states
            self._reset_backoff(identifier)


def create_enhanced_rate_limiter(
    rule: RateLimitRule,
    backoff_config: BackoffConfig | None = None,
    adaptive_config: AdaptiveConfig | None = None,
) -> EnhancedRateLimitAlgorithm:
    """Factory function to create enhanced rate limiter."""
    if rule.algorithm.value == "token_bucket":
        return EnhancedTokenBucket(rule, backoff_config, adaptive_config)
    elif rule.algorithm.value == "sliding_window":
        return EnhancedSlidingWindow(rule, backoff_config, adaptive_config)
    else:
        # Fall back to token bucket for unsupported algorithms
        return EnhancedTokenBucket(rule, backoff_config, adaptive_config)


def create_trading_backoff_config() -> BackoffConfig:
    """Create backoff configuration optimized for trading systems."""
    return BackoffConfig(
        strategy=BackoffStrategy.JITTERED_EXPONENTIAL,
        base_delay=1,
        max_delay=60,  # 1 minute max for trading
        multiplier=1.5,  # Gentler exponential backoff
        jitter_factor=0.2,  # 20% jitter
        reset_after=1800,  # Reset after 30 minutes
    )


def create_trading_adaptive_config() -> AdaptiveConfig:
    """Create adaptive configuration optimized for trading systems."""
    return AdaptiveConfig(
        enabled=True,
        base_limit=100,
        min_limit=50,  # Don't throttle too much
        max_limit=500,  # Allow bursts up to 500
        adjustment_factor=0.05,  # Small adjustments
        success_threshold=0.98,  # High success rate required
        error_threshold=0.02,  # Low error tolerance
        measurement_window=120,  # 2 minutes
    )

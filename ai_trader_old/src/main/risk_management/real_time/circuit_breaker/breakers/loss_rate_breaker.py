"""
Loss Rate Circuit Breaker

Monitors the velocity of losses and triggers protection when losses
occur too rapidly within a specified time window.

Created: 2025-07-15
"""

# Standard library imports
from collections import deque
from datetime import datetime
import logging
from typing import Any

# Third-party imports
import numpy as np

from ..config import BreakerConfig
from ..registry import BaseBreaker
from ..types import BreakerMetrics, BreakerType, MarketConditions

logger = logging.getLogger(__name__)


class LossRateBreaker(BaseBreaker):
    """
    Circuit breaker for rapid loss protection.

    Monitors:
    - Loss velocity within time windows
    - Consecutive loss periods
    - Loss acceleration patterns
    """

    def __init__(self, breaker_type: BreakerType, config: BreakerConfig):
        """Initialize loss rate breaker."""
        super().__init__(breaker_type, config)

        # Loss tracking
        self.loss_history: deque = deque(maxlen=200)
        self.loss_rate_threshold = config.loss_rate_threshold
        self.loss_rate_window = config.loss_rate_window

        # Configuration
        self.warning_threshold = self.loss_rate_threshold * 0.75  # 75% of threshold
        self.consecutive_loss_limit = 5  # Number of consecutive loss periods
        self.severe_loss_threshold = self.loss_rate_threshold * 2  # 2x threshold for severe losses

        # State tracking
        self.consecutive_losses = 0
        self.last_profit_time = datetime.now()
        self.max_loss_rate_seen = 0.0

        logger.info(
            f"Loss rate breaker initialized - threshold: {self.loss_rate_threshold:.2%} in {self.loss_rate_window}"
        )

    async def check(
        self, portfolio_value: float, positions: dict[str, Any], market_conditions: MarketConditions
    ) -> bool:
        """
        Check if loss rate breaker should trip.

        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            market_conditions: Current market conditions

        Returns:
            True if breaker should trip
        """
        if not self.is_enabled():
            return False

        # Update loss history
        now = datetime.now()
        self.loss_history.append((now, portfolio_value))

        # Clean old entries outside the time window
        cutoff_time = now - self.loss_rate_window
        self.loss_history = deque((t, v) for t, v in self.loss_history if t > cutoff_time)

        # Check rapid loss rate
        current_loss_rate = self._calculate_loss_rate()
        if current_loss_rate > self.loss_rate_threshold:
            self.logger.error(
                f"Rapid loss rate detected: {current_loss_rate:.2%} > {self.loss_rate_threshold:.2%}"
            )
            return True

        # Check for severe single-period losses
        if current_loss_rate > self.severe_loss_threshold:
            self.logger.error(f"Severe loss rate detected: {current_loss_rate:.2%}")
            return True

        # Check consecutive losses
        if await self._check_consecutive_losses(portfolio_value):
            return True

        # Check loss acceleration
        if await self._check_loss_acceleration():
            return True

        return False

    async def check_warning_conditions(
        self, portfolio_value: float, positions: dict[str, Any], market_conditions: MarketConditions
    ) -> bool:
        """
        Check if loss rate breaker should be in warning state.

        Returns:
            True if breaker should be in warning state
        """
        if not self.is_enabled():
            return False

        current_loss_rate = self._calculate_loss_rate()

        # Warning if approaching loss rate threshold
        if current_loss_rate > self.warning_threshold:
            self.logger.warning(f"Approaching loss rate threshold: {current_loss_rate:.2%}")
            return True

        # Warning if multiple consecutive losses
        if self.consecutive_losses >= 3:
            self.logger.warning(f"Multiple consecutive losses: {self.consecutive_losses}")
            return True

        return False

    def _calculate_loss_rate(self) -> float:
        """Calculate current loss rate over the time window."""
        if len(self.loss_history) < 2:
            return 0.0

        # Get start and end values
        start_value = self.loss_history[0][1]
        current_value = self.loss_history[-1][1]

        if start_value <= 0:
            return 0.0

        # Calculate loss rate
        loss_rate = (start_value - current_value) / start_value

        # Update max loss rate seen
        if loss_rate > self.max_loss_rate_seen:
            self.max_loss_rate_seen = loss_rate

        return max(0.0, loss_rate)  # Only positive loss rates

    async def _check_consecutive_losses(self, current_value: float) -> bool:
        """Check for consecutive loss periods."""
        if len(self.loss_history) < 2:
            return False

        # Check if this is a loss period
        previous_value = self.loss_history[-2][1]

        if current_value < previous_value:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            self.last_profit_time = datetime.now()

        # Trip if too many consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit:
            self.logger.warning(f"Too many consecutive losses: {self.consecutive_losses}")
            return True

        return False

    async def _check_loss_acceleration(self) -> bool:
        """Check if loss rate is accelerating."""
        if len(self.loss_history) < 6:  # Need sufficient data
            return False

        # Calculate loss rates for different periods
        values = [v for _, v in self.loss_history]

        # Recent period (last 1/3 of window)
        recent_period = len(values) // 3
        recent_values = values[-recent_period:]
        recent_loss_rate = (
            (recent_values[0] - recent_values[-1]) / recent_values[0] if recent_values[0] > 0 else 0
        )

        # Earlier period (first 1/3 of window)
        earlier_values = values[:recent_period]
        earlier_loss_rate = (
            (earlier_values[0] - earlier_values[-1]) / earlier_values[0]
            if earlier_values[0] > 0
            else 0
        )

        # Check if loss rate is accelerating
        if recent_loss_rate > earlier_loss_rate * 1.5:  # 50% acceleration
            self.logger.warning(
                f"Loss rate accelerating: {recent_loss_rate:.2%} vs {earlier_loss_rate:.2%}"
            )
            return True

        return False

    def get_metrics(self) -> BreakerMetrics:
        """Get current loss rate metrics."""
        metrics = BreakerMetrics()

        current_loss_rate = self._calculate_loss_rate()
        metrics.loss_rate = current_loss_rate

        return metrics

    def get_loss_statistics(self) -> dict[str, float]:
        """Get detailed loss statistics."""
        if len(self.loss_history) < 2:
            return {
                "current_loss_rate": 0.0,
                "max_loss_rate": self.max_loss_rate_seen,
                "consecutive_losses": self.consecutive_losses,
                "time_since_profit_hours": 0.0,
                "loss_frequency": 0.0,
                "avg_loss_magnitude": 0.0,
                "loss_volatility": 0.0,
            }

        current_loss_rate = self._calculate_loss_rate()

        # Calculate time since last profit
        time_since_profit = (datetime.now() - self.last_profit_time).total_seconds() / 3600

        # Calculate loss frequency and magnitude
        values = [v for _, v in self.loss_history]
        losses = []

        for i in range(1, len(values)):
            if values[i] < values[i - 1]:
                loss_pct = (values[i - 1] - values[i]) / values[i - 1] if values[i - 1] > 0 else 0
                losses.append(loss_pct)

        loss_frequency = len(losses) / len(values) if values else 0
        avg_loss_magnitude = np.mean(losses) if losses else 0
        loss_volatility = np.std(losses) if len(losses) > 1 else 0

        return {
            "current_loss_rate": float(current_loss_rate),
            "max_loss_rate": float(self.max_loss_rate_seen),
            "consecutive_losses": int(self.consecutive_losses),
            "time_since_profit_hours": float(time_since_profit),
            "loss_frequency": float(loss_frequency),
            "avg_loss_magnitude": float(avg_loss_magnitude),
            "loss_volatility": float(loss_volatility),
        }

    def get_loss_pattern_analysis(self) -> dict[str, Any]:
        """Analyze loss patterns and trends."""
        if len(self.loss_history) < 10:
            return {"insufficient_data": True}

        values = [v for _, v in self.loss_history]
        times = [t for t, _ in self.loss_history]

        # Calculate loss events
        loss_events = []
        for i in range(1, len(values)):
            if values[i] < values[i - 1]:
                loss_pct = (values[i - 1] - values[i]) / values[i - 1] if values[i - 1] > 0 else 0
                duration = (times[i] - times[i - 1]).total_seconds() / 60  # minutes
                loss_events.append(
                    {"loss_pct": loss_pct, "duration_minutes": duration, "timestamp": times[i]}
                )

        if not loss_events:
            return {"no_loss_events": True}

        # Analyze patterns
        loss_magnitudes = [le["loss_pct"] for le in loss_events]
        loss_durations = [le["duration_minutes"] for le in loss_events]

        # Time-based analysis
        recent_events = [
            le for le in loss_events if (datetime.now() - le["timestamp"]).total_seconds() < 3600
        ]  # Last hour

        return {
            "total_loss_events": len(loss_events),
            "avg_loss_magnitude": np.mean(loss_magnitudes),
            "max_loss_magnitude": max(loss_magnitudes),
            "avg_loss_duration": np.mean(loss_durations),
            "recent_loss_events": len(recent_events),
            "loss_clustering": len(recent_events) / len(loss_events) if loss_events else 0,
            "loss_trend": self._calculate_loss_trend(loss_events),
        }

    def _calculate_loss_trend(self, loss_events: list) -> str:
        """Calculate the trend in loss patterns."""
        if len(loss_events) < 5:
            return "insufficient_data"

        recent_events = loss_events[-5:]
        earlier_events = loss_events[-10:-5] if len(loss_events) >= 10 else loss_events[:-5]

        if not earlier_events:
            return "insufficient_data"

        recent_avg = np.mean([le["loss_pct"] for le in recent_events])
        earlier_avg = np.mean([le["loss_pct"] for le in earlier_events])

        if recent_avg > earlier_avg * 1.2:
            return "worsening"
        elif recent_avg < earlier_avg * 0.8:
            return "improving"
        else:
            return "stable"

    def reset_consecutive_losses(self):
        """Reset consecutive loss counter."""
        self.consecutive_losses = 0
        self.last_profit_time = datetime.now()
        self.logger.info("Consecutive loss counter reset")

    def get_info(self) -> dict[str, Any]:
        """Get breaker information including loss rate-specific details."""
        base_info = super().get_info()

        loss_stats = self.get_loss_statistics()
        pattern_analysis = self.get_loss_pattern_analysis()

        base_info.update(
            {
                "loss_rate_threshold": self.loss_rate_threshold,
                "warning_threshold": self.warning_threshold,
                "time_window_minutes": self.loss_rate_window.total_seconds() / 60,
                "consecutive_loss_limit": self.consecutive_loss_limit,
                "severe_loss_threshold": self.severe_loss_threshold,
                "max_loss_rate_seen": self.max_loss_rate_seen,
                "current_stats": loss_stats,
                "pattern_analysis": pattern_analysis,
            }
        )

        return base_info

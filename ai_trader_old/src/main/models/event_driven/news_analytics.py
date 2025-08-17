"""
News analytics event-driven trading strategy.

This module implements a trading strategy based on news sentiment analysis,
article volume, and event detection from financial news sources.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.models.common import SignalType, StrategySignal
from main.models.event_driven.base_event_strategy import BaseEventStrategy
from main.utils.core import ensure_utc, get_logger, timer
from main.utils.monitoring import record_metric

logger = get_logger(__name__)


@dataclass
class NewsEvent:
    """Represents a news event that could trigger trading."""

    timestamp: datetime
    symbol: str
    headline: str
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1
    event_type: str  # earnings, merger, regulatory, etc.
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def impact_score(self) -> float:
        """Calculate overall impact score."""
        return abs(self.sentiment_score) * self.relevance_score


@dataclass
class NewsAnalyticsConfig:
    """Configuration for news analytics strategy."""

    # Sentiment thresholds
    bullish_threshold: float = 0.6
    bearish_threshold: float = -0.6

    # Volume thresholds
    min_article_volume: int = 3  # Min articles for signal
    volume_spike_multiplier: float = 2.0  # Multiple of average

    # Time windows
    sentiment_window_hours: int = 24
    volume_window_days: int = 30
    signal_decay_hours: int = 48

    # Event weights
    event_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "earnings": 2.0,
            "merger": 1.8,
            "regulatory": 1.5,
            "product": 1.3,
            "partnership": 1.2,
            "general": 1.0,
        }
    )

    # Risk controls
    max_positions: int = 10
    position_size_pct: float = 0.05
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.05


class NewsAnalyticsStrategy(BaseEventStrategy):
    """
    Event-driven strategy based on news analytics.

    Features:
    - Real-time news sentiment analysis
    - Event type classification and weighting
    - Article volume spike detection
    - Sentiment momentum tracking
    - Multi-source aggregation
    - Signal decay over time
    """

    def __init__(
        self,
        config: Optional[NewsAnalyticsConfig] = None,
        data_providers: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize news analytics strategy.

        Args:
            config: Strategy configuration
            data_providers: Data provider instances
        """
        super().__init__("NewsAnalytics")
        self.config = config or NewsAnalyticsConfig()
        self.data_providers = data_providers or {}

        # Tracking
        self._sentiment_history: Dict[str, List[NewsEvent]] = {}
        self._volume_baseline: Dict[str, float] = {}
        self._active_signals: Dict[str, StrategySignal] = {}

    @timer
    async def process_event(self, event: NewsEvent) -> Optional[StrategySignal]:
        """
        Process a news event and generate trading signal if appropriate.

        Args:
            event: News event to process

        Returns:
            Trading signal or None
        """
        logger.debug(f"Processing news event for {event.symbol}: {event.headline[:50]}...")

        # Update history
        self._update_event_history(event)

        # Check if event is significant
        if not self._is_significant_event(event):
            return None

        # Calculate aggregate sentiment
        aggregate_sentiment = await self._calculate_aggregate_sentiment(event.symbol)

        # Check for volume spike
        has_volume_spike = await self._check_volume_spike(event.symbol)

        # Generate signal if conditions met
        signal = self._generate_signal(event, aggregate_sentiment, has_volume_spike)

        if signal:
            # Store active signal
            self._active_signals[event.symbol] = signal

            # Record metrics
            record_metric(
                "news_analytics.signal_generated",
                1,
                tags={
                    "symbol": event.symbol,
                    "signal_type": signal.signal_type.value,
                    "event_type": event.event_type,
                },
            )

            logger.info(
                f"Generated {signal.signal_type.value} signal for {event.symbol} "
                f"based on news event: {event.headline[:50]}"
            )

        return signal

    async def get_active_signals(self) -> List[StrategySignal]:
        """Get all active signals, removing expired ones."""
        current_time = datetime.utcnow()
        active = []

        # Check each signal for expiration
        expired = []
        for symbol, signal in self._active_signals.items():
            age_hours = (current_time - signal.timestamp).total_seconds() / 3600

            if age_hours < self.config.signal_decay_hours:
                # Adjust confidence based on age
                decay_factor = 1.0 - (age_hours / self.config.signal_decay_hours)
                signal.confidence *= decay_factor
                active.append(signal)
            else:
                expired.append(symbol)

        # Remove expired signals
        for symbol in expired:
            del self._active_signals[symbol]

        return active

    def _update_event_history(self, event: NewsEvent) -> None:
        """Update event history for symbol."""
        if event.symbol not in self._sentiment_history:
            self._sentiment_history[event.symbol] = []

        # Add new event
        self._sentiment_history[event.symbol].append(event)

        # Remove old events
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.sentiment_window_hours)
        self._sentiment_history[event.symbol] = [
            e for e in self._sentiment_history[event.symbol] if e.timestamp > cutoff_time
        ]

    def _is_significant_event(self, event: NewsEvent) -> bool:
        """Check if event is significant enough to consider."""
        # Check relevance
        if event.relevance_score < 0.5:
            return False

        # Check sentiment strength
        if abs(event.sentiment_score) < 0.3:
            return False

        # Check event type importance
        event_weight = self.config.event_weights.get(
            event.event_type, self.config.event_weights["general"]
        )

        # Calculate weighted impact
        weighted_impact = event.impact_score * event_weight

        return weighted_impact > 0.5

    async def _calculate_aggregate_sentiment(self, symbol: str) -> float:
        """Calculate aggregate sentiment for symbol."""
        events = self._sentiment_history.get(symbol, [])

        if not events:
            return 0.0

        # Weight by recency and relevance
        current_time = datetime.utcnow()
        weighted_sum = 0.0
        weight_total = 0.0

        for event in events:
            # Calculate time decay
            age_hours = (current_time - event.timestamp).total_seconds() / 3600
            time_weight = np.exp(-age_hours / 12)  # 12-hour half-life

            # Calculate event weight
            event_weight = self.config.event_weights.get(
                event.event_type, self.config.event_weights["general"]
            )

            # Combined weight
            weight = time_weight * event.relevance_score * event_weight

            weighted_sum += event.sentiment_score * weight
            weight_total += weight

        if weight_total > 0:
            return weighted_sum / weight_total

        return 0.0

    async def _check_volume_spike(self, symbol: str) -> bool:
        """Check if there's a news volume spike."""
        events = self._sentiment_history.get(symbol, [])

        # Count recent articles
        recent_cutoff = datetime.utcnow() - timedelta(hours=6)
        recent_count = sum(1 for e in events if e.timestamp > recent_cutoff)

        # Check against minimum threshold
        if recent_count < self.config.min_article_volume:
            return False

        # Get baseline if available
        baseline = self._volume_baseline.get(symbol, 1.0)

        # Check for spike
        return recent_count > baseline * self.config.volume_spike_multiplier

    def _generate_signal(
        self, event: NewsEvent, aggregate_sentiment: float, has_volume_spike: bool
    ) -> Optional[StrategySignal]:
        """Generate trading signal based on analysis."""
        # Determine signal type
        if aggregate_sentiment >= self.config.bullish_threshold:
            signal_type = SignalType.BUY
        elif aggregate_sentiment <= self.config.bearish_threshold:
            signal_type = SignalType.SELL
        else:
            return None

        # Calculate confidence
        confidence = min(abs(aggregate_sentiment), 1.0)

        # Boost confidence for volume spikes
        if has_volume_spike:
            confidence = min(confidence * 1.2, 1.0)

        # Boost confidence for high-impact events
        event_weight = self.config.event_weights.get(
            event.event_type, self.config.event_weights["general"]
        )
        if event_weight > 1.5:
            confidence = min(confidence * 1.1, 1.0)

        # Create signal
        signal = StrategySignal(
            symbol=event.symbol,
            signal_type=signal_type,
            confidence=confidence,
            strategy_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "trigger_event": event.headline,
                "event_type": event.event_type,
                "aggregate_sentiment": aggregate_sentiment,
                "has_volume_spike": has_volume_spike,
                "event_count": len(self._sentiment_history.get(event.symbol, [])),
                "stop_loss": self.config.stop_loss_pct,
                "take_profit": self.config.take_profit_pct,
            },
        )

        return signal

    async def update_volume_baseline(self, symbol: str, article_count: float) -> None:
        """Update volume baseline for symbol."""
        if symbol not in self._volume_baseline:
            self._volume_baseline[symbol] = article_count
        else:
            # Exponential moving average
            alpha = 0.1
            self._volume_baseline[symbol] = (
                alpha * article_count + (1 - alpha) * self._volume_baseline[symbol]
            )

    def get_sentiment_summary(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment summary for symbol."""
        events = self._sentiment_history.get(symbol, [])

        if not events:
            return {
                "symbol": symbol,
                "event_count": 0,
                "aggregate_sentiment": 0.0,
                "latest_event": None,
            }

        # Calculate metrics
        sentiments = [e.sentiment_score for e in events]

        return {
            "symbol": symbol,
            "event_count": len(events),
            "aggregate_sentiment": np.mean(sentiments),
            "sentiment_std": np.std(sentiments),
            "positive_ratio": sum(1 for s in sentiments if s > 0) / len(sentiments),
            "latest_event": events[-1].headline,
            "latest_timestamp": events[-1].timestamp,
            "event_types": list(set(e.event_type for e in events)),
        }

    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for monitoring."""
        return {
            "active_signals": len(self._active_signals),
            "tracked_symbols": len(self._sentiment_history),
            "total_events": sum(len(events) for events in self._sentiment_history.values()),
            "volume_baselines": dict(self._volume_baseline),
            "config": {
                "bullish_threshold": self.config.bullish_threshold,
                "bearish_threshold": self.config.bearish_threshold,
                "signal_decay_hours": self.config.signal_decay_hours,
            },
        }

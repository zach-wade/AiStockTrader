"""
Circuit breaker system for automated trading risk control.

This module provides circuit breakers that automatically halt trading
when certain risk conditions are met to prevent catastrophic losses.
"""

# Import specific breaker implementations
from .breakers import DrawdownBreaker, LossRateBreaker, PositionLimitBreaker, VolatilityBreaker
from .config import BreakerConfig
from .events import BreakerResetEvent, BreakerTrippedEvent, BreakerWarningEvent, CircuitBreakerEvent
from .facade import CircuitBreakerFacade, SystemStatus
from .registry import BaseBreaker, BreakerRegistry
from .types import (
    BreakerEvent,
    BreakerMetrics,
    BreakerPriority,
    BreakerStatus,
    BreakerType,
    MarketConditions,
)

__all__ = [
    # Main facade
    "CircuitBreakerFacade",
    "SystemStatus",
    # Configuration
    "BreakerConfig",
    # Types
    "BreakerType",
    "BreakerStatus",
    "BreakerEvent",
    "MarketConditions",
    "BreakerMetrics",
    "BreakerPriority",
    # Events
    "CircuitBreakerEvent",
    "BreakerTrippedEvent",
    "BreakerResetEvent",
    "BreakerWarningEvent",
    # Registry
    "BreakerRegistry",
    "BaseBreaker",
    # Breaker implementations
    "DrawdownBreaker",
    "VolatilityBreaker",
    "LossRateBreaker",
    "PositionLimitBreaker",
]

"""
Circuit breaker system for automated trading risk control.

This module provides circuit breakers that automatically halt trading
when certain risk conditions are met to prevent catastrophic losses.
"""

from .facade import (
    CircuitBreakerFacade,
    SystemStatus
)

from .config import (
    BreakerConfig
)

from .types import (
    BreakerType,
    BreakerStatus,
    BreakerEvent,
    MarketConditions,
    BreakerMetrics,
    BreakerPriority
)

from .events import (
    CircuitBreakerEvent,
    BreakerTrippedEvent,
    BreakerResetEvent,
    BreakerWarningEvent
)

from .registry import (
    BreakerRegistry,
    BaseBreaker
)

# Import specific breaker implementations
from .breakers import (
    DrawdownBreaker,
    VolatilityBreaker,
    LossRateBreaker,
    PositionLimitBreaker
)

__all__ = [
    # Main facade
    'CircuitBreakerFacade',
    'SystemStatus',
    
    # Configuration
    'BreakerConfig',
    
    # Types
    'BreakerType',
    'BreakerStatus',
    'BreakerEvent',
    'MarketConditions',
    'BreakerMetrics',
    'BreakerPriority',
    
    # Events
    'CircuitBreakerEvent',
    'BreakerTrippedEvent',
    'BreakerResetEvent',
    'BreakerWarningEvent',
    
    # Registry
    'BreakerRegistry',
    'BaseBreaker',
    
    # Breaker implementations
    'DrawdownBreaker',
    'VolatilityBreaker',
    'LossRateBreaker',
    'PositionLimitBreaker'
]
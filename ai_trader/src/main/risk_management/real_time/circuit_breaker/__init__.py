"""
Circuit breaker system for automated trading risk control.

This module provides circuit breakers that automatically halt trading
when certain risk conditions are met to prevent catastrophic losses.
"""

from .facade import (
    CircuitBreakerFacade,
    CircuitBreakerSystem
)

from .config import (
    CircuitBreakerConfig,
    BreakerThresholds,
    BreakerAction
)

from .types import (
    CircuitBreakerType,
    BreakerStatus,
    BreakerPriority,
    TripReason
)

from .events import (
    CircuitBreakerEvent,
    BreakerTrippedEvent,
    BreakerResetEvent,
    BreakerWarningEvent
)

from .registry import (
    CircuitBreakerRegistry,
    BreakerRegistration
)

# Import specific breaker implementations
from .breakers import (
    BaseCircuitBreaker,
    DrawdownBreaker,
    VolatilityBreaker,
    LossRateBreaker,
    PositionLimitBreaker
)

__all__ = [
    # Main facade
    'CircuitBreakerFacade',
    'CircuitBreakerSystem',
    
    # Configuration
    'CircuitBreakerConfig',
    'BreakerThresholds',
    'BreakerAction',
    
    # Types
    'CircuitBreakerType',
    'BreakerStatus',
    'BreakerPriority',
    'TripReason',
    
    # Events
    'CircuitBreakerEvent',
    'BreakerTrippedEvent',
    'BreakerResetEvent',
    'BreakerWarningEvent',
    
    # Registry
    'CircuitBreakerRegistry',
    'BreakerRegistration',
    
    # Breaker implementations
    'BaseCircuitBreaker',
    'DrawdownBreaker',
    'VolatilityBreaker',
    'LossRateBreaker',
    'PositionLimitBreaker'
]
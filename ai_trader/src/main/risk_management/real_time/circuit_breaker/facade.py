"""
Circuit breaker facade for unified interface to all breakers.

This module provides a unified interface to manage multiple circuit breakers,
coordinating their actions and providing a single point of control.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

from .types import (
    BreakerType, BreakerStatus, BreakerMetrics,
    MarketConditions, BreakerEvent
)
from .events import (
    CircuitBreakerEvent, BreakerTrippedEvent, BreakerResetEvent,
    BreakerWarningEvent, CircuitBreakerEventBuilder
)
from .config import BreakerConfig
from .registry import BreakerRegistry
from .registry import BaseBreaker
from main.utils.core import ErrorHandlingMixin
from main.utils.monitoring import timer

logger = logging.getLogger(__name__)

# These mappings are no longer needed - all references should use correct names
# Keeping as comment for reference:
# CircuitBreakerType was renamed to BreakerType
# BreakerPriority is now properly defined in types.py


@dataclass
class SystemStatus:
    """Overall system trading status."""
    can_trade: bool
    active_breakers: List[str]
    tripped_breakers: List[str]
    warning_breakers: List[str]
    overall_risk_score: float  # 0-100
    last_update: datetime
    message: str


class CircuitBreakerFacade(ErrorHandlingMixin):
    """
    Unified interface for circuit breaker system.
    
    Manages multiple circuit breakers and provides coordinated
    control over trading system safety mechanisms.
    """
    
    def __init__(self, config: Optional[BreakerConfig] = None):
        """Initialize circuit breaker facade."""
        super().__init__()
        self.config = config or BreakerConfig({})
        
        # Circuit breaker registry
        self.registry = BreakerRegistry(self.config)
        
        # Event callbacks
        self._event_callbacks: List[Callable] = []
        
        # System state
        self._system_enabled = True
        self._emergency_stop = False
        self._tripped_breakers: Set[str] = set()
        self._cooldown_timers: Dict[str, asyncio.Task] = {}
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._check_interval = 1.0  # Seconds between checks
        
        # Statistics
        self._trip_count = 0
        self._reset_count = 0
        self._warning_count = 0
        
        logger.info("Circuit breaker facade initialized")
    
    async def initialize(self):
        """Initialize the circuit breaker system."""
        with self._handle_error("initializing circuit breaker system"):
            # Register default breakers based on config
            await self._register_default_breakers()
            
            # Start monitoring
            await self.start_monitoring()
            
            logger.info("Circuit breaker system initialized")
    
    async def start_monitoring(self):
        """Start continuous monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Circuit breaker monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Circuit breaker monitoring stopped")
    
    @timer
    async def check_conditions(self,
                             metrics: BreakerMetrics,
                             market_conditions: MarketConditions) -> SystemStatus:
        """Check all circuit breakers and return system status."""
        with self._handle_error("checking circuit breaker conditions"):
            if self._emergency_stop:
                return SystemStatus(
                    can_trade=False,
                    active_breakers=[],
                    tripped_breakers=["EMERGENCY_STOP"],
                    warning_breakers=[],
                    overall_risk_score=100.0,
                    last_update=datetime.utcnow(),
                    message="Emergency stop activated"
                )
            
            if not self._system_enabled:
                return SystemStatus(
                    can_trade=False,
                    active_breakers=[],
                    tripped_breakers=["SYSTEM_DISABLED"],
                    warning_breakers=[],
                    overall_risk_score=0.0,
                    last_update=datetime.utcnow(),
                    message="Circuit breaker system disabled"
                )
            
            # Check each registered breaker
            active_breakers = []
            warning_breakers = []
            risk_scores = []
            
            for breaker_name, breaker in self.registry.get_all_breakers().items():
                if breaker_name in self._tripped_breakers:
                    continue  # Skip if already tripped
                
                # Check breaker status
                status = await breaker.check(metrics, market_conditions)
                
                if status == BreakerStatus.TRIPPED:
                    await self._handle_breaker_trip(breaker_name, breaker, metrics)
                elif status == BreakerStatus.WARNING:
                    warning_breakers.append(breaker_name)
                    self._warning_count += 1
                    # Emit warning event
                    await self._emit_warning(breaker_name, breaker, metrics)
                else:
                    active_breakers.append(breaker_name)
                
                # Get risk score from breaker
                risk_score = getattr(breaker, 'get_risk_score', lambda: 0.0)()
                risk_scores.append(risk_score)
            
            # Calculate overall risk score
            overall_risk = max(risk_scores) if risk_scores else 0.0
            
            # Determine if trading is allowed
            can_trade = len(self._tripped_breakers) == 0 and not self._emergency_stop
            
            status = SystemStatus(
                can_trade=can_trade,
                active_breakers=active_breakers,
                tripped_breakers=list(self._tripped_breakers),
                warning_breakers=warning_breakers,
                overall_risk_score=overall_risk,
                last_update=datetime.utcnow(),
                message=self._generate_status_message(can_trade, self._tripped_breakers)
            )
            
            return status
    
    async def trip_breaker(self, breaker_name: str, reason: str):
        """Manually trip a circuit breaker."""
        with self._handle_error(f"manually tripping breaker {breaker_name}"):
            if breaker_name in self._tripped_breakers:
                logger.warning(f"Breaker {breaker_name} already tripped")
                return
            
            self._tripped_breakers.add(breaker_name)
            self._trip_count += 1
            
            # Get breaker instance
            breaker = self.registry.get_breaker(breaker_name)
            if breaker:
                breaker_type = breaker.breaker_type
            else:
                breaker_type = BreakerType.MANUAL
            
            # Emit trip event
            event = CircuitBreakerEventBuilder.build_trip_event(
                breaker_name=breaker_name,
                breaker_type=breaker_type,
                trip_reason=f"Manual trip: {reason}",
                current_value=0.0,
                threshold_value=0.0,
                actions_taken=["Trading halted", "Positions frozen"],
                cooldown_seconds=self.config.default_cooldown_seconds,
                auto_reset=self.config.auto_reset_enabled
            )
            
            await self._emit_event(event)
            
            # Start cooldown if auto-reset enabled
            if self.config.auto_reset_enabled:
                await self._start_cooldown(breaker_name)
            
            logger.warning(f"Circuit breaker {breaker_name} manually tripped: {reason}")
    
    async def reset_breaker(self, breaker_name: str, reason: str = "Manual reset"):
        """Reset a tripped circuit breaker."""
        with self._handle_error(f"resetting breaker {breaker_name}"):
            if breaker_name not in self._tripped_breakers:
                logger.info(f"Breaker {breaker_name} not tripped")
                return
            
            self._tripped_breakers.remove(breaker_name)
            self._reset_count += 1
            
            # Cancel cooldown if active
            if breaker_name in self._cooldown_timers:
                self._cooldown_timers[breaker_name].cancel()
                del self._cooldown_timers[breaker_name]
            
            # Get breaker and reset its state
            breaker = self.registry.get_breaker(breaker_name)
            if breaker:
                await breaker.reset()
                breaker_type = breaker.breaker_type
            else:
                breaker_type = BreakerType.MANUAL
            
            # Emit reset event
            event = CircuitBreakerEventBuilder.build_reset_event(
                breaker_name=breaker_name,
                breaker_type=breaker_type,
                reset_reason=reason,
                was_manual=True
            )
            
            await self._emit_event(event)
            
            logger.info(f"Circuit breaker {breaker_name} reset: {reason}")
    
    async def emergency_stop(self, reason: str):
        """Activate emergency stop - halt all trading immediately."""
        with self._handle_error("activating emergency stop"):
            self._emergency_stop = True
            
            # Trip all breakers
            for breaker_name in self.registry.get_all_breakers():
                if breaker_name not in self._tripped_breakers:
                    await self.trip_breaker(breaker_name, f"Emergency stop: {reason}")
            
            # Emit emergency event
            event = CircuitBreakerEventBuilder.build_trip_event(
                breaker_name="EMERGENCY_STOP",
                breaker_type=BreakerType.KILL_SWITCH,
                trip_reason=reason,
                current_value=0.0,
                threshold_value=0.0,
                actions_taken=["All trading halted", "Emergency liquidation initiated"],
                cooldown_seconds=0,
                auto_reset=False
            )
            event.priority = BreakerType.KILL_SWITCH  # Highest priority
            
            await self._emit_event(event)
            
            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
    
    async def clear_emergency_stop(self):
        """Clear emergency stop status."""
        self._emergency_stop = False
        logger.warning("Emergency stop cleared")
    
    # Event management
    
    def add_event_callback(self, callback: Callable):
        """Add callback for circuit breaker events."""
        self._event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable):
        """Remove event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
    
    async def _emit_event(self, event: CircuitBreakerEvent):
        """Emit event to all callbacks."""
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    # Private methods
    
    async def _register_default_breakers(self):
        """Register default circuit breakers based on config."""
        # This would create and register the default breakers
        # Implementation depends on breaker implementations
        pass
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while True:
            try:
                # Periodic checks could go here
                await asyncio.sleep(self._check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _handle_breaker_trip(self,
                                  breaker_name: str,
                                  breaker: BaseBreaker,
                                  metrics: BreakerMetrics):
        """Handle a breaker trip."""
        self._tripped_breakers.add(breaker_name)
        self._trip_count += 1
        
        # Get trip details from breaker
        trip_reason = breaker.get_trip_reason()
        current_value = breaker.get_current_value()
        threshold_value = breaker.get_threshold_value()
        
        # Emit trip event
        event = CircuitBreakerEventBuilder.build_trip_event(
            breaker_name=breaker_name,
            breaker_type=breaker.breaker_type,
            trip_reason=trip_reason,
            current_value=current_value,
            threshold_value=threshold_value,
            cooldown_seconds=self.config.default_cooldown_seconds,
            auto_reset=self.config.auto_reset_enabled
        )
        
        await self._emit_event(event)
        
        # Start cooldown if enabled
        if self.config.auto_reset_enabled:
            await self._start_cooldown(breaker_name)
    
    async def _emit_warning(self,
                          breaker_name: str,
                          breaker: BaseBreaker,
                          metrics: BreakerMetrics):
        """Emit warning event for breaker."""
        warning_level = breaker.get_warning_level()
        current_value = breaker.get_current_value()
        threshold_value = breaker.get_threshold_value()
        
        event = CircuitBreakerEventBuilder.build_warning_event(
            breaker_name=breaker_name,
            breaker_type=breaker.breaker_type,
            warning_level=warning_level,
            current_value=current_value,
            threshold_value=threshold_value,
            trend="increasing"  # Could be determined from breaker
        )
        
        await self._emit_event(event)
    
    async def _start_cooldown(self, breaker_name: str):
        """Start cooldown timer for breaker."""
        cooldown_seconds = self.config.default_cooldown_seconds
        
        async def cooldown_task():
            await asyncio.sleep(cooldown_seconds)
            await self.reset_breaker(breaker_name, "Automatic reset after cooldown")
        
        self._cooldown_timers[breaker_name] = asyncio.create_task(cooldown_task())
    
    def _generate_status_message(self, can_trade: bool, tripped_breakers: Set[str]) -> str:
        """Generate status message."""
        if can_trade:
            return "System operational - all breakers active"
        elif self._emergency_stop:
            return "Emergency stop - all trading halted"
        elif tripped_breakers:
            return f"Trading halted - breakers tripped: {', '.join(tripped_breakers)}"
        else:
            return "System disabled"
    
    # Status and statistics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'system_enabled': self._system_enabled,
            'emergency_stop': self._emergency_stop,
            'total_breakers': len(self.registry.get_all_breakers()),
            'tripped_breakers': len(self._tripped_breakers),
            'trip_count': self._trip_count,
            'reset_count': self._reset_count,
            'warning_count': self._warning_count,
            'active_cooldowns': len(self._cooldown_timers)
        }
    
    def is_trading_allowed(self) -> bool:
        """Quick check if trading is allowed."""
        return (self._system_enabled and 
                not self._emergency_stop and 
                len(self._tripped_breakers) == 0)
    
    def get_tripped_breakers(self) -> List[str]:
        """Get list of currently tripped breakers."""
        return list(self._tripped_breakers)
    
    def enable_system(self):
        """Enable circuit breaker system."""
        self._system_enabled = True
        logger.info("Circuit breaker system enabled")
    
    def disable_system(self):
        """Disable circuit breaker system (for testing/maintenance)."""
        self._system_enabled = False
        logger.warning("Circuit breaker system disabled")


# Alias for backward compatibility
CircuitBreakerSystem = CircuitBreakerFacade
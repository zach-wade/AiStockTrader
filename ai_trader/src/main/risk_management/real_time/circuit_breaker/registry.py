"""
Circuit Breaker Registry

Manages registration and lifecycle of individual circuit breaker components.
Extracted from monolithic circuit_breaker.py for better organization.

Created: 2025-07-15
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod
from datetime import datetime

from .types import BreakerType, BreakerStatus, MarketConditions, BreakerMetrics
from .config import BreakerConfig
# TODO: BreakerEventManager and BreakerStateManager need to be implemented
# from .events import BreakerEventManager, BreakerStateManager

logger = logging.getLogger(__name__)


class BaseBreaker(ABC):
    """
    Base class for all circuit breaker components.
    
    Defines the interface that all breaker implementations must follow.
    """
    
    def __init__(self, breaker_type: BreakerType, config: BreakerConfig):
        """Initialize base breaker."""
        self.breaker_type = breaker_type
        self.config = config
        self.breaker_config = config.get_breaker_config(breaker_type)
        self.enabled = self.breaker_config.enabled
        self.threshold = self.breaker_config.threshold
        self.logger = logging.getLogger(f"{__name__}.{breaker_type.value}")
    
    @abstractmethod
    async def check(self, 
                   portfolio_value: float,
                   positions: Dict[str, Any],
                   market_conditions: MarketConditions) -> bool:
        """
        Check if this breaker should trip.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            market_conditions: Current market conditions
            
        Returns:
            True if breaker should trip, False otherwise
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> BreakerMetrics:
        """Get current metrics for this breaker."""
        pass
    
    async def check_warning_conditions(self, 
                                     portfolio_value: float,
                                     positions: Dict[str, Any],
                                     market_conditions: MarketConditions) -> bool:
        """
        Check if this breaker should be in warning state.
        
        Default implementation returns False. Override in subclasses.
        """
        return False
    
    def is_enabled(self) -> bool:
        """Check if this breaker is enabled."""
        return self.enabled
    
    def enable(self):
        """Enable this breaker."""
        self.enabled = True
        self.logger.info(f"{self.breaker_type.value} breaker enabled")
    
    def disable(self):
        """Disable this breaker."""
        self.enabled = False
        self.logger.info(f"{self.breaker_type.value} breaker disabled")
    
    def update_threshold(self, new_threshold: float):
        """Update the threshold for this breaker."""
        old_threshold = self.threshold
        self.threshold = new_threshold
        self.logger.info(f"{self.breaker_type.value} threshold updated: {old_threshold} -> {new_threshold}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get breaker information."""
        return {
            'type': self.breaker_type.value,
            'enabled': self.enabled,
            'threshold': self.threshold,
            'auto_reset': self.breaker_config.auto_reset,
            'cooldown_minutes': self.breaker_config.cooldown_minutes,
            'severity_weight': self.breaker_config.severity_weight
        }


class BreakerRegistry:
    """
    Registry for all circuit breaker components.
    
    Manages the lifecycle and coordination of individual breaker instances.
    """
    
    def __init__(self, config: BreakerConfig, event_manager: Optional[Any] = None, state_manager: Optional[Any] = None):
        """Initialize breaker registry."""
        self.config = config
        self.event_manager = event_manager  # TODO: Implement BreakerEventManager
        self.state_manager = state_manager  # TODO: Implement BreakerStateManager
        self.breakers: Dict[BreakerType, BaseBreaker] = {}
        self.breaker_classes: Dict[BreakerType, Type[BaseBreaker]] = {}
        self._lock = asyncio.Lock()
    
    def register_breaker_class(self, breaker_type: BreakerType, breaker_class: Type[BaseBreaker]):
        """Register a breaker class for a specific type."""
        self.breaker_classes[breaker_type] = breaker_class
        self.logger.info(f"Registered breaker class: {breaker_type.value} -> {breaker_class.__name__}")
    
    async def initialize_breakers(self):
        """Initialize all registered breaker instances."""
        async with self._lock:
            for breaker_type, breaker_class in self.breaker_classes.items():
                try:
                    breaker_instance = breaker_class(breaker_type, self.config)
                    self.breakers[breaker_type] = breaker_instance
                    self.logger.info(f"Initialized {breaker_type.value} breaker")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {breaker_type.value} breaker: {e}")
    
    async def check_all_breakers(self, 
                               portfolio_value: float,
                               positions: Dict[str, Any],
                               market_conditions: MarketConditions) -> Dict[BreakerType, bool]:
        """
        Check all registered breakers.
        
        Returns:
            Dictionary mapping breaker types to their trip status
        """
        results = {}
        
        for breaker_type, breaker in self.breakers.items():
            if not breaker.is_enabled():
                results[breaker_type] = False
                continue
            
            try:
                is_tripped = await breaker.check(portfolio_value, positions, market_conditions)
                results[breaker_type] = is_tripped
                
                # Check for warning conditions
                if not is_tripped:
                    is_warning = await breaker.check_warning_conditions(portfolio_value, positions, market_conditions)
                    if is_warning:
                        await self.state_manager.set_warning_state(breaker_type)
                
                # Update breaker state
                cooldown_time = None
                if is_tripped:
                    cooldown_time = datetime.now() + self.config.cooldown_period
                
                await self.state_manager.update_breaker_state(
                    breaker_type, is_tripped, market_conditions, cooldown_time
                )
                
            except Exception as e:
                self.logger.error(f"Error checking {breaker_type.value} breaker: {e}")
                results[breaker_type] = False
        
        return results
    
    def get_breaker(self, breaker_type: BreakerType) -> Optional[BaseBreaker]:
        """Get a specific breaker instance."""
        return self.breakers.get(breaker_type)
    
    def get_all_breakers(self) -> Dict[BreakerType, BaseBreaker]:
        """Get all breaker instances."""
        return self.breakers.copy()
    
    def get_enabled_breakers(self) -> Dict[BreakerType, BaseBreaker]:
        """Get only enabled breakers."""
        return {
            breaker_type: breaker
            for breaker_type, breaker in self.breakers.items()
            if breaker.is_enabled()
        }
    
    def get_breaker_metrics(self) -> Dict[BreakerType, BreakerMetrics]:
        """Get metrics from all breakers."""
        metrics = {}
        for breaker_type, breaker in self.breakers.items():
            try:
                metrics[breaker_type] = breaker.get_metrics()
            except Exception as e:
                self.logger.error(f"Error getting metrics from {breaker_type.value} breaker: {e}")
        return metrics
    
    def get_breaker_info(self) -> Dict[BreakerType, Dict[str, Any]]:
        """Get information about all breakers."""
        info = {}
        for breaker_type, breaker in self.breakers.items():
            info[breaker_type] = breaker.get_info()
        return info
    
    async def enable_breaker(self, breaker_type: BreakerType):
        """Enable a specific breaker."""
        if breaker_type in self.breakers:
            self.breakers[breaker_type].enable()
        else:
            self.logger.warning(f"Cannot enable {breaker_type.value} breaker: not found")
    
    async def disable_breaker(self, breaker_type: BreakerType):
        """Disable a specific breaker."""
        if breaker_type in self.breakers:
            self.breakers[breaker_type].disable()
        else:
            self.logger.warning(f"Cannot disable {breaker_type.value} breaker: not found")
    
    async def update_breaker_threshold(self, breaker_type: BreakerType, new_threshold: float):
        """Update threshold for a specific breaker."""
        if breaker_type in self.breakers:
            self.breakers[breaker_type].update_threshold(new_threshold)
        else:
            self.logger.warning(f"Cannot update threshold for {breaker_type.value} breaker: not found")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get status of the breaker registry."""
        return {
            'total_breakers': len(self.breakers),
            'enabled_breakers': len(self.get_enabled_breakers()),
            'registered_types': [bt.value for bt in self.breakers.keys()],
            'breaker_states': {
                bt.value: self.state_manager.get_breaker_status(bt).value
                for bt in self.breakers.keys()
            }
        }
    
    async def shutdown(self):
        """Shutdown all breakers gracefully."""
        async with self._lock:
            for breaker_type, breaker in self.breakers.items():
                try:
                    # If breaker has cleanup method, call it
                    if hasattr(breaker, 'cleanup'):
                        await breaker.cleanup()
                    self.logger.info(f"Shutdown {breaker_type.value} breaker")
                except Exception as e:
                    self.logger.error(f"Error shutting down {breaker_type.value} breaker: {e}")
            
            self.breakers.clear()
            self.logger.info("All breakers shutdown complete")
    
    @property
    def logger(self):
        """Get logger instance."""
        return logging.getLogger(f"{__name__}.registry")
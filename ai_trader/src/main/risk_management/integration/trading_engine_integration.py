"""
Trading engine integration for risk management.

This module provides integration between the risk management system
and the trading engine, ensuring all trades pass through risk checks.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from main.risk_management.types import (
    RiskCheckResult, RiskEvent, RiskLevel,
    RiskEventType, PortfolioRisk
)
from main.risk_management.pre_trade import UnifiedLimitChecker
from main.risk_management.real_time import (
    LiveRiskMonitor, CircuitBreakerFacade,
    DynamicStopLossManager, DrawdownController
)
# from main.risk_management.position_sizing import BasePositionSizer  # TODO: Need to implement
from main.utils.core import ErrorHandlingMixin
from main.utils.monitoring import record_metric, timer

logger = logging.getLogger(__name__)


class OrderAction(Enum):
    """Actions for order handling."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    HOLD = "hold"


@dataclass
class OrderRiskCheck:
    """Result of order risk check."""
    order_id: str
    action: OrderAction
    checks_passed: List[str]
    checks_failed: List[str]
    risk_score: float  # 0-100
    modified_quantity: Optional[float] = None
    modified_price: Optional[float] = None
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingEngineConfig:
    """Configuration for trading engine integration."""
    enable_pre_trade_checks: bool = True
    enable_real_time_monitoring: bool = True
    enable_position_sizing: bool = True
    enable_stop_loss: bool = True
    enable_circuit_breakers: bool = True
    
    # Risk limits
    max_order_value: float = 100000
    max_position_value: float = 500000
    max_daily_loss: float = 50000
    max_portfolio_var: float = 100000
    
    # Behavior
    reject_on_warning: bool = False
    auto_reduce_position_size: bool = True
    emergency_liquidation_enabled: bool = True
    
    # Monitoring
    risk_check_timeout: float = 1.0  # Seconds
    monitoring_interval: float = 1.0


class TradingEngineRiskIntegration(ErrorHandlingMixin):
    """
    Integration layer between trading engine and risk management.
    
    Provides comprehensive risk checks for all trading activities
    and ensures compliance with risk limits.
    """
    
    def __init__(self, config: Optional[TradingEngineConfig] = None):
        """Initialize trading engine risk integration."""
        super().__init__()
        self.config = config or TradingEngineConfig()
        
        # Risk components
        self.limit_checker: Optional[UnifiedLimitChecker] = None
        self.risk_monitor: Optional[LiveRiskMonitor] = None
        self.circuit_breakers: Optional[CircuitBreakerFacade] = None
        self.stop_loss_manager: Optional[DynamicStopLossManager] = None
        self.drawdown_controller: Optional[DrawdownController] = None
        self.position_sizer: Optional[Any] = None  # BasePositionSizer
        
        # State tracking
        self._is_initialized = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._risk_events: List[RiskEvent] = []
        
        # Callbacks
        self._order_callbacks: List[Callable] = []
        self._risk_event_callbacks: List[Callable] = []
        
        # Statistics
        self._checks_performed = 0
        self._orders_approved = 0
        self._orders_rejected = 0
        self._orders_modified = 0
        
        logger.info("Trading engine risk integration initialized")
    
    async def initialize(self,
                        limit_checker: UnifiedLimitChecker,
                        risk_monitor: LiveRiskMonitor,
                        circuit_breakers: CircuitBreakerFacade,
                        stop_loss_manager: DynamicStopLossManager,
                        position_sizer: Any):  # BasePositionSizer
        """Initialize with risk management components."""
        with self._handle_error("initializing risk integration"):
            self.limit_checker = limit_checker
            self.risk_monitor = risk_monitor
            self.circuit_breakers = circuit_breakers
            self.stop_loss_manager = stop_loss_manager
            self.position_sizer = position_sizer
            
            # Start monitoring if enabled
            if self.config.enable_real_time_monitoring:
                await self.start_monitoring()
            
            self._is_initialized = True
            logger.info("Trading engine risk integration initialized")
    
    @timer
    async def check_order(self,
                         order_id: str,
                         symbol: str,
                         side: str,
                         quantity: float,
                         price: float,
                         order_type: str,
                         portfolio_value: float,
                         current_positions: Dict[str, Any]) -> OrderRiskCheck:
        """
        Perform comprehensive risk check on order.
        
        Returns OrderRiskCheck with action to take.
        """
        with self._handle_error("checking order risk"):
            self._checks_performed += 1
            
            checks_passed = []
            checks_failed = []
            warnings = []
            risk_scores = []
            
            # Check if system is operational
            if not await self._check_system_operational():
                return OrderRiskCheck(
                    order_id=order_id,
                    action=OrderAction.REJECT,
                    checks_passed=[],
                    checks_failed=["system_operational"],
                    risk_score=100.0,
                    rejection_reason="Trading system halted by circuit breakers"
                )
            
            order_value = quantity * price
            
            # 1. Pre-trade limit checks
            if self.config.enable_pre_trade_checks and self.limit_checker:
                limit_results = await self.limit_checker.check_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    side=side,
                    portfolio_value=portfolio_value,
                    current_positions=current_positions
                )
                
                for result in limit_results:
                    if result.passed:
                        checks_passed.append(result.check_name)
                    else:
                        checks_failed.append(result.check_name)
                        if result.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                            warnings.append(f"{result.check_name}: {result.message}")
                    
                    risk_scores.append(result.utilization)
            
            # 2. Position sizing check
            modified_quantity = quantity
            if self.config.enable_position_sizing and self.position_sizer:
                size_result = await self.position_sizer.calculate_position_size(
                    symbol=symbol,
                    signal_strength=1.0,  # Assume full signal
                    portfolio_value=portfolio_value,
                    current_positions=current_positions
                )
                
                if size_result.recommended_size < quantity:
                    if self.config.auto_reduce_position_size:
                        modified_quantity = size_result.recommended_size
                        warnings.append(
                            f"Position size reduced from {quantity} to {modified_quantity}"
                        )
                        checks_passed.append("position_sizing")
                    else:
                        checks_failed.append("position_sizing")
                else:
                    checks_passed.append("position_sizing")
            
            # 3. Order value limits
            if order_value > self.config.max_order_value:
                checks_failed.append("max_order_value")
            else:
                checks_passed.append("max_order_value")
            
            # 4. Portfolio VaR check
            if self.risk_monitor:
                portfolio_risk = await self.risk_monitor.get_portfolio_risk()
                if portfolio_risk.var_95 > self.config.max_portfolio_var:
                    warnings.append("Portfolio VaR approaching limit")
                    risk_scores.append(
                        portfolio_risk.var_95 / self.config.max_portfolio_var * 100
                    )
            
            # Calculate overall risk score
            risk_score = max(risk_scores) if risk_scores else 0.0
            
            # Determine action
            if checks_failed:
                if any(check in ["system_operational", "max_order_value"] 
                      for check in checks_failed):
                    action = OrderAction.REJECT
                    rejection_reason = f"Failed checks: {', '.join(checks_failed)}"
                elif self.config.reject_on_warning:
                    action = OrderAction.REJECT
                    rejection_reason = "Risk warnings present"
                else:
                    action = OrderAction.MODIFY if modified_quantity != quantity else OrderAction.HOLD
                    rejection_reason = None
            else:
                action = OrderAction.APPROVE
                rejection_reason = None
            
            # Update statistics
            if action == OrderAction.APPROVE:
                self._orders_approved += 1
            elif action == OrderAction.REJECT:
                self._orders_rejected += 1
            elif action == OrderAction.MODIFY:
                self._orders_modified += 1
            
            # Record metrics
            record_metric(
                'risk.order_checks',
                1,
                tags={
                    'action': action.value,
                    'symbol': symbol,
                    'risk_score': f"{risk_score:.0f}"
                }
            )
            
            result = OrderRiskCheck(
                order_id=order_id,
                action=action,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                risk_score=risk_score,
                modified_quantity=modified_quantity if action == OrderAction.MODIFY else None,
                rejection_reason=rejection_reason,
                warnings=warnings
            )
            
            # Notify callbacks
            await self._notify_order_callbacks(result)
            
            return result
    
    async def register_trade_execution(self,
                                     trade_id: str,
                                     symbol: str,
                                     side: str,
                                     quantity: float,
                                     price: float,
                                     timestamp: datetime):
        """Register executed trade with risk system."""
        with self._handle_error("registering trade execution"):
            # Update stop loss manager
            if self.config.enable_stop_loss and self.stop_loss_manager:
                await self.stop_loss_manager.register_position(
                    symbol=symbol,
                    entry_price=price,
                    quantity=quantity,
                    side=side
                )
            
            # Update risk monitor
            if self.risk_monitor:
                await self.risk_monitor.update_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    side=side
                )
            
            # Record metric
            record_metric(
                'risk.trade_registered',
                1,
                tags={'symbol': symbol, 'side': side}
            )
    
    async def check_portfolio_risk(self) -> PortfolioRisk:
        """Get current portfolio risk metrics."""
        if self.risk_monitor:
            return await self.risk_monitor.get_portfolio_risk()
        else:
            # Return default if monitor not available
            return PortfolioRisk(
                total_value=0,
                var_95=0,
                var_99=0,
                cvar_95=0,
                cvar_99=0,
                portfolio_beta=0,
                portfolio_volatility=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                current_drawdown=0,
                gross_exposure=0,
                net_exposure=0,
                leverage=0,
                concentration_score=0,
                liquidity_score=1,
                position_count=0
            )
    
    async def handle_risk_event(self, event: RiskEvent):
        """Handle risk event from risk management system."""
        with self._handle_error("handling risk event"):
            self._risk_events.append(event)
            
            # Take action based on event type and severity
            if event.severity in [RiskLevel.CRITICAL, RiskLevel.EXTREME]:
                if event.event_type == RiskEventType.DRAWDOWN_ALERT:
                    await self._handle_drawdown_alert(event)
                elif event.event_type == RiskEventType.VAR_BREACH:
                    await self._handle_var_breach(event)
                elif event.event_type == RiskEventType.CIRCUIT_BREAKER_TRIGGERED:
                    await self._handle_circuit_breaker(event)
            
            # Notify callbacks
            await self._notify_risk_event_callbacks(event)
    
    # Monitoring
    
    async def start_monitoring(self):
        """Start real-time risk monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Risk monitoring started")
    
    async def stop_monitoring(self):
        """Stop risk monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Risk monitoring stopped")
    
    async def _monitoring_loop(self):
        """Continuous risk monitoring loop."""
        while True:
            try:
                # Check stop losses
                if self.stop_loss_manager:
                    triggered_stops = await self.stop_loss_manager.check_all()
                    for stop in triggered_stops:
                        await self._handle_stop_loss_trigger(stop)
                
                # Check drawdowns
                if self.drawdown_controller:
                    drawdown_action = await self.drawdown_controller.check()
                    if drawdown_action:
                        await self._handle_drawdown_action(drawdown_action)
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    # Private methods
    
    async def _check_system_operational(self) -> bool:
        """Check if trading system is operational."""
        if self.circuit_breakers:
            return self.circuit_breakers.is_trading_allowed()
        return True
    
    async def _handle_drawdown_alert(self, event: RiskEvent):
        """Handle drawdown alert."""
        logger.warning(f"Drawdown alert: {event.description}")
        # Could trigger position reduction or halt new trades
    
    async def _handle_var_breach(self, event: RiskEvent):
        """Handle VaR breach."""
        logger.error(f"VaR breach: {event.description}")
        # Could trigger risk reduction measures
    
    async def _handle_circuit_breaker(self, event: RiskEvent):
        """Handle circuit breaker trigger."""
        logger.critical(f"Circuit breaker triggered: {event.description}")
        # System should already be halted by circuit breaker
    
    async def _handle_stop_loss_trigger(self, stop_order: Any):
        """Handle triggered stop loss."""
        logger.info(f"Stop loss triggered: {stop_order}")
        # Would send order to trading engine
    
    async def _handle_drawdown_action(self, action: Any):
        """Handle drawdown control action."""
        logger.warning(f"Drawdown action required: {action}")
        # Would implement drawdown control measures
    
    async def _notify_order_callbacks(self, result: OrderRiskCheck):
        """Notify order check callbacks."""
        for callback in self._order_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    async def _notify_risk_event_callbacks(self, event: RiskEvent):
        """Notify risk event callbacks."""
        for callback in self._risk_event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in risk event callback: {e}")
    
    # Public API
    
    def add_order_callback(self, callback: Callable):
        """Add callback for order risk checks."""
        self._order_callbacks.append(callback)
    
    def add_risk_event_callback(self, callback: Callable):
        """Add callback for risk events."""
        self._risk_event_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            'is_initialized': self._is_initialized,
            'checks_performed': self._checks_performed,
            'orders_approved': self._orders_approved,
            'orders_rejected': self._orders_rejected,
            'orders_modified': self._orders_modified,
            'rejection_rate': (self._orders_rejected / self._checks_performed 
                             if self._checks_performed > 0 else 0),
            'risk_events_count': len(self._risk_events),
            'monitoring_active': (self._monitoring_task is not None and 
                                not self._monitoring_task.done())
        }
    
    def get_recent_risk_events(self, limit: int = 10) -> List[RiskEvent]:
        """Get recent risk events."""
        return self._risk_events[-limit:]


class RiskEventBridge:
    """
    Bridge for routing risk events between components.
    
    Facilitates communication between risk management components
    and external systems like trading engine and monitoring.
    """
    
    def __init__(self):
        """Initialize risk event bridge."""
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._is_running = False
        
    async def start(self):
        """Start event processing."""
        self._is_running = True
        self._processing_task = asyncio.create_task(self._process_events())
        
    async def stop(self):
        """Stop event processing."""
        self._is_running = False
        if self._processing_task:
            await self._event_queue.put(None)  # Sentinel
            await self._processing_task
    
    async def publish_event(self, event_type: str, event_data: Any):
        """Publish risk event."""
        await self._event_queue.put((event_type, event_data))
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)
    
    async def _process_events(self):
        """Process events from queue."""
        while self._is_running:
            try:
                item = await self._event_queue.get()
                if item is None:  # Sentinel
                    break
                    
                event_type, event_data = item
                
                # Notify subscribers
                if event_type in self._subscribers:
                    for callback in self._subscribers[event_type]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(event_data)
                            else:
                                callback(event_data)
                        except Exception as e:
                            logger.error(f"Error in event callback: {e}")
                            
            except Exception as e:
                logger.error(f"Error processing event: {e}")


class RiskDashboardIntegration:
    """Integration with risk monitoring dashboards."""
    
    def __init__(self, risk_integration: TradingEngineRiskIntegration):
        """Initialize dashboard integration."""
        self.risk_integration = risk_integration
        
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary for dashboard."""
        portfolio_risk = await self.risk_integration.check_portfolio_risk()
        stats = self.risk_integration.get_statistics()
        
        return {
            'portfolio_risk': {
                'var_95': portfolio_risk.var_95,
                'var_99': portfolio_risk.var_99,
                'current_drawdown': portfolio_risk.current_drawdown,
                'risk_score': portfolio_risk.risk_score
            },
            'order_statistics': {
                'total_checks': stats['checks_performed'],
                'approved': stats['orders_approved'],
                'rejected': stats['orders_rejected'],
                'modified': stats['orders_modified'],
                'rejection_rate': stats['rejection_rate']
            },
            'recent_events': [
                {
                    'type': event.event_type.value,
                    'severity': event.severity.value,
                    'title': event.title,
                    'timestamp': event.timestamp.isoformat()
                }
                for event in self.risk_integration.get_recent_risk_events()
            ],
            'system_status': {
                'operational': await self.risk_integration._check_system_operational(),
                'monitoring_active': stats['monitoring_active']
            }
        }
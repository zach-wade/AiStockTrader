# File: trading_engine/core/risk_integrated_order_manager.py

"""
Risk-Integrated Order Manager

This module extends the base OrderManager with comprehensive real-time risk
integration, providing pre-trade risk validation, order blocking capabilities,
and real-time risk monitoring throughout the order lifecycle.

Key Features:
- Pre-trade risk validation with automatic order blocking
- Real-time risk monitoring during order execution
- VaR-based position sizing validation
- Circuit breaker integration
- Comprehensive risk logging and audit trail
- Risk-adjusted order modifications
- Emergency risk controls
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from decimal import Decimal
from enum import Enum
import uuid

# Import base components
from main.trading_engine.core.order_manager import OrderManager
from main.trading_engine.core.unified_position_manager import UnifiedPositionManager, OrderSide
from main.risk_management.real_time.live_risk_monitor import (
    LiveRiskMonitor, RiskDecision, RiskDecisionType, RiskProfileType
)

# Import common models
from main.models.common import Order, OrderStatus, OrderType, TimeInForce

# Import config
from main.config.config_manager import get_config

logger = logging.getLogger(__name__)


class RiskOrderStatus(Enum):
    """Enhanced order status with risk states."""
    PENDING_RISK_CHECK = "pending_risk_check"
    RISK_APPROVED = "risk_approved"
    RISK_BLOCKED = "risk_blocked"
    RISK_MODIFIED = "risk_modified"
    RISK_EMERGENCY_CANCELLED = "risk_emergency_cancelled"


class RiskIntegratedOrderManager:
    """
    Order manager with comprehensive real-time risk integration.
    
    This class wraps the base OrderManager and adds risk validation
    at every step of the order lifecycle.
    """
    
    def __init__(self,
                 base_order_manager: OrderManager,
                 position_manager: UnifiedPositionManager,
                 risk_monitor: LiveRiskMonitor,
                 config: Any):
        """
        Initialize risk-integrated order manager.
        
        Args:
            base_order_manager: Base order manager for actual order operations
            position_manager: Position manager for portfolio state
            risk_monitor: Live risk monitor for risk decisions
            config: Trading configuration
        """
        self.base_order_manager = base_order_manager
        self.position_manager = position_manager
        self.risk_monitor = risk_monitor
        self.config = config
        
        # Risk decision tracking
        self.risk_decisions: Dict[str, RiskDecision] = {}  # order_id -> RiskDecision
        self.blocked_orders: Dict[str, Order] = {}         # order_id -> Order
        self.risk_modified_orders: Dict[str, Dict] = {}    # order_id -> modification_log
        
        # Risk statistics
        self.total_order_requests = 0
        self.risk_blocked_count = 0
        self.risk_modified_count = 0
        self.emergency_cancelled_count = 0
        
        # Risk callbacks
        self.risk_block_handlers: List[callable] = []
        self.risk_approval_handlers: List[callable] = []
        
        # Emergency controls
        self.emergency_halt_active = False
        self.emergency_halt_reason = ""
        
        # Subscribe to risk monitor events
        self.risk_monitor.add_risk_decision_handler(self._handle_risk_decision)
        self.risk_monitor.add_emergency_handler(self._handle_emergency_event)
        
        logger.info("RiskIntegratedOrderManager initialized with comprehensive risk controls")
    
    async def submit_order_with_risk_check(self,
                                         symbol: str,
                                         side: OrderSide,
                                         quantity: Union[float, Decimal],
                                         order_type: OrderType = OrderType.MARKET,
                                         limit_price: Optional[float] = None,
                                         stop_price: Optional[float] = None,
                                         time_in_force: TimeInForce = TimeInForce.DAY,
                                         strategy: Optional[str] = None,
                                         metadata: Optional[Dict[str, Any]] = None,
                                         force_approval: bool = False) -> Optional[str]:
        """
        Submit order with comprehensive risk validation.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force
            strategy: Strategy name
            metadata: Additional order metadata
            force_approval: Force approval (admin override)
            
        Returns:
            Order ID if successful, None if blocked by risk
        """
        self.total_order_requests += 1
        quantity = Decimal(str(quantity))
        
        # Check emergency halt
        if self.emergency_halt_active and not force_approval:
            logger.error(f"Order submission blocked due to emergency halt: {self.emergency_halt_reason}")
            return None
        
        # Generate order ID for tracking
        order_id = str(uuid.uuid4())
        
        try:
            # 1. Determine price for risk calculation
            price = self._get_order_price_for_risk_calc(
                symbol, order_type, limit_price, stop_price
            )
            
            if price <= 0:
                logger.error(f"Cannot determine valid price for risk calculation: {symbol}")
                return None
            
            # 2. Perform comprehensive risk evaluation
            risk_decision = await self.risk_monitor.evaluate_trade_risk(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=Decimal(str(price))
            )
            
            # Store risk decision
            self.risk_decisions[order_id] = risk_decision
            
            # 3. Handle risk decision
            if not risk_decision.allowed and not force_approval:
                logger.warning(
                    f"Order blocked by risk system: {symbol} {side.value} {quantity} @ {price} | "
                    f"Violations: {risk_decision.risk_violations}"
                )
                
                # Store blocked order for analysis
                blocked_order = Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=float(quantity),
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    time_in_force=time_in_force,
                    status=OrderStatus.REJECTED,
                    reject_reason=f"Risk blocked: {'; '.join(risk_decision.risk_violations)}",
                    strategy=strategy,
                    metadata={**(metadata or {}), 'risk_decision_id': risk_decision.decision_id},
                    created_at=datetime.now()
                )
                
                self.blocked_orders[order_id] = blocked_order
                self.risk_blocked_count += 1
                
                # Emit block event
                await self._emit_risk_block_event(blocked_order, risk_decision)
                
                return None
            
            # 4. Check for risk-based modifications
            modified_params = {}
            if risk_decision.alternative_sizing and not force_approval:
                # Apply risk-based position sizing
                alt_quantity = risk_decision.alternative_sizing.get('quantity')
                if alt_quantity and alt_quantity < quantity:
                    logger.info(
                        f"Applying risk-based position sizing: {quantity} -> {alt_quantity} for {symbol}"
                    )
                    modified_params['quantity'] = float(alt_quantity)
                    quantity = alt_quantity
                    self.risk_modified_count += 1
            
            # 5. Create order with risk metadata
            enhanced_metadata = {
                **(metadata or {}),
                'risk_decision_id': risk_decision.decision_id,
                'risk_approved': True,
                'risk_confidence_score': risk_decision.confidence_score,
                'risk_var_utilization': risk_decision.var_utilization,
                'risk_profile': self.risk_monitor.risk_profile.value
            }
            
            if modified_params:
                enhanced_metadata['risk_modifications'] = modified_params
                self.risk_modified_orders[order_id] = {
                    'original_quantity': float(Decimal(str(quantity)) / Decimal(str(modified_params.get('quantity', 1)))),
                    'modified_quantity': modified_params.get('quantity'),
                    'modification_reason': 'risk_position_sizing'
                }
            
            # 6. Submit order through base order manager
            submitted_order_id = await self.base_order_manager.submit_new_order(
                symbol=symbol,
                side=side,
                quantity=float(quantity),
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                strategy=strategy,
                metadata=enhanced_metadata
            )
            
            if submitted_order_id:
                # Update risk decision with actual order ID
                risk_decision.metadata = {
                    **risk_decision.metadata,
                    'submitted_order_id': submitted_order_id
                }
                
                # Move risk decision to submitted order ID
                self.risk_decisions[submitted_order_id] = self.risk_decisions.pop(order_id, risk_decision)
                
                # Emit approval event
                await self._emit_risk_approval_event(submitted_order_id, risk_decision)
                
                logger.info(
                    f"Order submitted with risk approval: {submitted_order_id} | "
                    f"Risk confidence: {risk_decision.confidence_score:.1f}%"
                )
                
                return submitted_order_id
            else:
                logger.error("Order submission failed after risk approval")
                return None
        
        except Exception as e:
            logger.error(f"Error in risk-integrated order submission: {e}", exc_info=True)
            return None
    
    async def modify_order_with_risk_check(self,
                                         order_id: str,
                                         quantity: Optional[float] = None,
                                         limit_price: Optional[float] = None,
                                         stop_price: Optional[float] = None) -> bool:
        """
        Modify order with risk validation of changes.
        
        Args:
            order_id: Order ID to modify
            quantity: New quantity
            limit_price: New limit price
            stop_price: New stop price
            
        Returns:
            True if modification successful and risk-approved
        """
        try:
            # Get current order
            current_order = await self.base_order_manager.get_order(order_id)
            if not current_order:
                logger.error(f"Order not found for modification: {order_id}")
                return False
            
            # Calculate modification impact
            if quantity and quantity != current_order.quantity:
                quantity_change = quantity - current_order.quantity
                
                # Determine effective price for risk calculation
                price = limit_price or current_order.limit_price or stop_price or current_order.stop_price
                if not price:
                    logger.error("Cannot determine price for risk calculation in order modification")
                    return False
                
                # Evaluate risk of the quantity change
                if quantity_change > 0:  # Increasing position
                    risk_decision = await self.risk_monitor.evaluate_trade_risk(
                        symbol=current_order.symbol,
                        side=current_order.side,
                        quantity=Decimal(str(quantity_change)),
                        price=Decimal(str(price))
                    )
                    
                    if not risk_decision.allowed:
                        logger.warning(
                            f"Order modification blocked by risk system: {order_id} | "
                            f"Quantity change: {quantity_change} | "
                            f"Violations: {risk_decision.risk_violations}"
                        )
                        return False
            
            # Proceed with modification if risk-approved
            success = await self.base_order_manager.modify_order(
                order_id=order_id,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            if success and quantity:
                # Log modification
                self.risk_modified_orders[order_id] = {
                    **self.risk_modified_orders.get(order_id, {}),
                    'modified_at': datetime.now().isoformat(),
                    'new_quantity': quantity,
                    'previous_quantity': current_order.quantity,
                    'modification_type': 'user_requested'
                }
            
            return success
        
        except Exception as e:
            logger.error(f"Error in risk-integrated order modification: {e}", exc_info=True)
            return False
    
    async def cancel_order_with_risk_log(self, order_id: str, reason: str = "User cancel") -> bool:
        """
        Cancel order with risk logging.
        
        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason
            
        Returns:
            True if cancellation successful
        """
        try:
            success = await self.base_order_manager.cancel_order(order_id)
            
            if success:
                # Log cancellation in risk tracking
                if order_id in self.risk_decisions:
                    risk_decision = self.risk_decisions[order_id]
                    risk_decision.metadata = {
                        **risk_decision.metadata,
                        'cancelled_at': datetime.now().isoformat(),
                        'cancellation_reason': reason
                    }
                
                logger.info(f"Order cancelled with risk logging: {order_id} | Reason: {reason}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error in risk-integrated order cancellation: {e}", exc_info=True)
            return False
    
    async def emergency_cancel_all_orders(self, reason: str = "Emergency risk halt") -> int:
        """
        Emergency cancellation of all active orders.
        
        Args:
            reason: Emergency cancellation reason
            
        Returns:
            Number of orders cancelled
        """
        logger.critical(f"EMERGENCY ORDER CANCELLATION: {reason}")
        
        try:
            # Set emergency halt
            self.emergency_halt_active = True
            self.emergency_halt_reason = reason
            
            # Cancel all active orders
            cancelled_count = await self.base_order_manager.cancel_all_orders()
            self.emergency_cancelled_count += cancelled_count
            
            logger.critical(f"Emergency cancelled {cancelled_count} orders due to: {reason}")
            
            return cancelled_count
        
        except Exception as e:
            logger.error(f"Error in emergency order cancellation: {e}", exc_info=True)
            return 0
    
    def _get_order_price_for_risk_calc(self,
                                     symbol: str,
                                     order_type: OrderType,
                                     limit_price: Optional[float],
                                     stop_price: Optional[float]) -> float:
        """Get appropriate price for risk calculation based on order type."""
        
        if order_type == OrderType.LIMIT and limit_price:
            return limit_price
        elif order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price:
            return stop_price
        elif order_type == OrderType.MARKET:
            # For market orders, try to get current market price
            current_position = self.position_manager.get_position(symbol)
            if current_position:
                return float(current_position.current_price)
            else:
                # Fallback - would need market data integration
                logger.warning(f"No current price available for {symbol}, using default estimation")
                return 100.0  # Placeholder
        else:
            return limit_price or stop_price or 100.0
    
    async def _emit_risk_block_event(self, order: Order, risk_decision: RiskDecision):
        """Emit risk block event to handlers."""
        
        for handler in self.risk_block_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(order, risk_decision)
                else:
                    handler(order, risk_decision)
            except Exception as e:
                logger.error(f"Error in risk block handler: {e}")
    
    async def _emit_risk_approval_event(self, order_id: str, risk_decision: RiskDecision):
        """Emit risk approval event to handlers."""
        
        for handler in self.risk_approval_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(order_id, risk_decision)
                else:
                    handler(order_id, risk_decision)
            except Exception as e:
                logger.error(f"Error in risk approval handler: {e}")
    
    def _handle_risk_decision(self, decision: RiskDecision):
        """Handle risk decision from monitor."""
        
        # Log decision for audit trail
        logger.info(
            f"Risk decision recorded: {decision.decision_id} | "
            f"Symbol: {decision.symbol} | "
            f"Allowed: {decision.allowed} | "
            f"Confidence: {decision.confidence_score:.1f}%"
        )
    
    async def _handle_emergency_event(self, event_type: str, event_data: Dict):
        """Handle emergency events from risk monitor."""
        
        logger.critical(f"EMERGENCY EVENT: {event_type} | Data: {event_data}")
        
        # Auto-cancel all orders on critical events
        if "circuit_breaker" in event_type.lower():
            await self.emergency_cancel_all_orders(f"Circuit breaker: {event_type}")
    
    # Risk management specific methods
    
    def add_risk_block_handler(self, handler: callable):
        """Add handler for risk block events."""
        self.risk_block_handlers.append(handler)
    
    def add_risk_approval_handler(self, handler: callable):
        """Add handler for risk approval events."""
        self.risk_approval_handlers.append(handler)
    
    def get_risk_decision(self, order_id: str) -> Optional[RiskDecision]:
        """Get risk decision for an order."""
        return self.risk_decisions.get(order_id)
    
    def get_blocked_orders(self) -> List[Order]:
        """Get all blocked orders."""
        return list(self.blocked_orders.values())
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get comprehensive risk statistics."""
        
        return {
            'total_order_requests': self.total_order_requests,
            'risk_blocked_count': self.risk_blocked_count,
            'risk_modified_count': self.risk_modified_count,
            'emergency_cancelled_count': self.emergency_cancelled_count,
            'block_rate_pct': (self.risk_blocked_count / self.total_order_requests * 100) 
                             if self.total_order_requests > 0 else 0,
            'modification_rate_pct': (self.risk_modified_count / self.total_order_requests * 100) 
                                   if self.total_order_requests > 0 else 0,
            'emergency_halt_active': self.emergency_halt_active,
            'emergency_halt_reason': self.emergency_halt_reason,
            'risk_decisions_tracked': len(self.risk_decisions),
            'blocked_orders_count': len(self.blocked_orders),
            'risk_profile': self.risk_monitor.risk_profile.value
        }
    
    def clear_emergency_halt(self, reason: str = "Manual override"):
        """Clear emergency halt status."""
        
        self.emergency_halt_active = False
        self.emergency_halt_reason = ""
        logger.info(f"Emergency halt cleared: {reason}")
    
    def get_risk_audit_trail(self, order_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get risk audit trail for analysis."""
        
        audit_trail = []
        
        if order_id:
            # Single order audit
            if order_id in self.risk_decisions:
                decision = self.risk_decisions[order_id]
                audit_trail.append({
                    'order_id': order_id,
                    'decision_id': decision.decision_id,
                    'timestamp': decision.timestamp.isoformat(),
                    'symbol': decision.symbol,
                    'side': decision.side.value,
                    'quantity': float(decision.quantity),
                    'price': float(decision.price),
                    'allowed': decision.allowed,
                    'violations': decision.risk_violations,
                    'warnings': decision.risk_warnings,
                    'confidence_score': decision.confidence_score,
                    'processing_time_ms': decision.processing_time_ms
                })
        else:
            # All decisions audit
            for oid, decision in self.risk_decisions.items():
                audit_trail.append({
                    'order_id': oid,
                    'decision_id': decision.decision_id,
                    'timestamp': decision.timestamp.isoformat(),
                    'symbol': decision.symbol,
                    'allowed': decision.allowed,
                    'confidence_score': decision.confidence_score
                })
        
        return sorted(audit_trail, key=lambda x: x['timestamp'], reverse=True)
    
    # Delegate methods to base order manager
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order from base manager."""
        return await self.base_order_manager.get_order(order_id)
    
    async def get_orders(self, **kwargs) -> List[Order]:
        """Get orders from base manager."""
        return await self.base_order_manager.get_orders(**kwargs)
    
    def get_active_orders(self) -> List[Order]:
        """Get active orders from base manager."""
        return self.base_order_manager.get_active_orders()
    
    async def get_order_summary(self) -> Dict[str, Any]:
        """Get enhanced order summary with risk metrics."""
        
        base_summary = await self.base_order_manager.get_order_summary()
        risk_stats = self.get_risk_statistics()
        
        return {
            **base_summary,
            'risk_integration': risk_stats
        }
    
    def register_callback(self, event: str, callback: callable):
        """Register callback with base manager."""
        self.base_order_manager.register_callback(event, callback)


# Convenience function for creating risk-integrated order manager
def create_risk_integrated_order_manager(
    base_order_manager: OrderManager,
    position_manager: UnifiedPositionManager,
    risk_profile: RiskProfileType = RiskProfileType.PAPER_TRADING,
    config: Optional[Config] = None
) -> RiskIntegratedOrderManager:
    """
    Create a risk-integrated order manager with proper configuration.
    
    Args:
        base_order_manager: Base order manager
        position_manager: Position manager
        risk_profile: Risk profile for trading mode
        config: Trading configuration
        
    Returns:
        Configured RiskIntegratedOrderManager
    """
    # Create risk monitor
    risk_monitor = LiveRiskMonitor(
        position_manager=position_manager,
        risk_profile=risk_profile
    )
    
    # Create integrated order manager
    integrated_manager = RiskIntegratedOrderManager(
        base_order_manager=base_order_manager,
        position_manager=position_manager,
        risk_monitor=risk_monitor,
        config=config or Config()
    )
    
    return integrated_manager
# File: src/main/trading_engine/core/fast_execution_path.py
"""
Fast Execution Path for Hunter-Killer Strategy

Provides optimized order execution for high-confidence, time-sensitive opportunities.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass
import time

from main.models.common import (
    Order, OrderStatus, OrderType, OrderSide, TimeInForce
)
from main.trading_engine.brokers.broker_interface import BrokerInterface
from main.utils.cache import MemoryBackend, CacheType, get_global_cache

logger = logging.getLogger(__name__)


@dataclass
class FastExecutionMetrics:
    """Metrics for fast execution performance."""
    signal_to_order_ms: float
    order_to_fill_ms: float
    total_execution_ms: float
    slippage_bps: float
    execution_price: float
    intended_price: float


class FastExecutionPath:
    """
    Optimized execution path for rapid order placement and monitoring.
    
    Features:
    - Pre-allocated order objects
    - Minimal validation for trusted signals
    - Direct broker API calls
    - Aggressive order monitoring (sub-second)
    - Smart order type selection
    """
    
    def __init__(self, broker: BrokerInterface, config: Dict[str, Any]):
        """
        Initialize fast execution path.
        
        Args:
            broker: Broker interface for order placement
            config: Configuration dictionary
        """
        self.broker = broker
        self.config = config
        self.cache = get_global_cache()
        
        # Fast path configuration
        self.max_position_size = config.get('execution', {}).get('fast_path_max_position_size', 10000)
        self.max_slippage_bps = config.get('hunter_killer', {}).get('max_slippage_bps', 10)
        self.monitoring_interval = 0.1  # 100ms order monitoring
        
        # Pre-allocated order pool
        self.order_pool: List[Order] = []
        self._initialize_order_pool()
        
        # Performance tracking
        self.execution_metrics: Dict[str, FastExecutionMetrics] = {}
        
        logger.info("Fast execution path initialized")
    
    def _initialize_order_pool(self):
        """Pre-allocate order objects for performance."""
        pool_size = 50  # Pre-allocate 50 orders
        
        for i in range(pool_size):
            order = Order(
                order_id=f"FAST_{i}",
                symbol="",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0,
                status=OrderStatus.PENDING,
                time_in_force=TimeInForce.DAY,
                created_at=datetime.now(timezone.utc)
            )
            self.order_pool.append(order)
        
        logger.debug(f"Pre-allocated {pool_size} orders")
    
    def _get_order_from_pool(self) -> Optional[Order]:
        """Get a pre-allocated order from pool."""
        for order in self.order_pool:
            if order.status == OrderStatus.PENDING and not order.symbol:
                return order
        
        # Pool exhausted, create new order
        logger.warning("Order pool exhausted, creating new order")
        return Order(
            order_id=f"FAST_NEW_{int(time.time() * 1000)}",
            symbol="",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0,
            status=OrderStatus.PENDING,
            time_in_force=TimeInForce.DAY,
            created_at=datetime.now(timezone.utc)
        )
    
    async def execute_fast(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        signal_timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Order]:
        """
        Execute order through fast path with minimal latency.
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            signal_timestamp: When signal was generated
            metadata: Optional metadata
            
        Returns:
            Executed order or None
        """
        start_time = time.time()
        
        try:
            # Get market data from cache (should be fresh from WebSocket)
            market_data = await self.cache.get(CacheType.QUOTES, f"market:{symbol}")
            if not market_data:
                logger.error(f"No market data for {symbol}")
                return None
            
            current_price = market_data['price']
            spread = market_data['spread']
            
            # Quick position size check
            position_value = quantity * current_price
            if position_value > self.max_position_size:
                logger.warning(f"Position size ${position_value} exceeds fast path limit")
                return None
            
            # Get order from pool
            order = self._get_order_from_pool()
            if not order:
                return None
            
            # Configure order
            order.symbol = symbol
            order.side = side
            order.quantity = quantity
            order.created_at = datetime.now(timezone.utc)
            
            # Smart order type selection based on spread
            spread_bps = (spread / current_price) * 10000 if current_price > 0 else 0
            
            if spread_bps < 5:  # Tight spread - use market order
                order.order_type = OrderType.MARKET
                order.limit_price = None
                logger.debug(f"Using MARKET order for {symbol} (spread: {spread_bps:.1f} bps)")
            else:  # Wide spread - use aggressive limit
                order.order_type = OrderType.LIMIT
                if side == OrderSide.BUY:
                    # Place limit at ask + small buffer
                    order.limit_price = market_data['ask'] * 1.0001
                else:
                    # Place limit at bid - small buffer
                    order.limit_price = market_data['bid'] * 0.9999
                logger.debug(f"Using LIMIT order for {symbol} @ {order.limit_price} (spread: {spread_bps:.1f} bps)")
            
            # Submit order directly to broker
            order_submit_time = time.time()
            broker_order_id = await self.broker.submit_order(order)
            
            if not broker_order_id:
                logger.error(f"Failed to submit order for {symbol}")
                order.status = OrderStatus.FAILED
                return None
            
            order.broker_order_id = broker_order_id
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now(timezone.utc)
            
            # Log submission latency
            signal_to_order_ms = (order_submit_time - signal_timestamp.timestamp()) * 1000
            logger.info(f"Order submitted in {signal_to_order_ms:.1f}ms: {symbol} {side.value} {quantity}")
            
            # Aggressive monitoring for fill
            filled_order = await self._monitor_order_aggressive(order, current_price)
            
            # Calculate execution metrics
            end_time = time.time()
            total_ms = (end_time - start_time) * 1000
            
            if filled_order and filled_order.status == OrderStatus.FILLED:
                metrics = FastExecutionMetrics(
                    signal_to_order_ms=signal_to_order_ms,
                    order_to_fill_ms=(filled_order.filled_at.timestamp() - order_submit_time) * 1000,
                    total_execution_ms=total_ms,
                    slippage_bps=self._calculate_slippage_bps(current_price, filled_order.avg_fill_price),
                    execution_price=filled_order.avg_fill_price,
                    intended_price=current_price
                )
                
                self.execution_metrics[order.order_id] = metrics
                
                logger.info(
                    f"Fast execution completed: {symbol} in {total_ms:.1f}ms "
                    f"(slippage: {metrics.slippage_bps:.1f} bps)"
                )
                
                # Cache execution metrics
                await self.cache.set(
                    CacheType.METRICS,
                    f"exec_metrics:{symbol}:{order.order_id}",
                    {
                        'signal_to_order_ms': metrics.signal_to_order_ms,
                        'order_to_fill_ms': metrics.order_to_fill_ms,
                        'total_ms': metrics.total_execution_ms,
                        'slippage_bps': metrics.slippage_bps
                    },
                    3600
                )
            
            return filled_order
            
        except Exception as e:
            logger.error(f"Error in fast execution for {symbol}: {e}")
            return None
    
    async def _monitor_order_aggressive(self, order: Order, 
                                      intended_price: float) -> Optional[Order]:
        """
        Aggressively monitor order for rapid fill detection.
        
        Args:
            order: Order to monitor
            intended_price: Intended execution price
            
        Returns:
            Updated order
        """
        max_attempts = 50  # 5 seconds at 100ms intervals
        attempts = 0
        
        while attempts < max_attempts:
            try:
                # Get order status from broker
                broker_order = await self.broker.get_order(order.broker_order_id)
                
                if not broker_order:
                    logger.error(f"Order {order.broker_order_id} not found")
                    break
                
                # Update order status
                order.status = broker_order.status
                order.filled_qty = broker_order.filled_qty
                order.avg_fill_price = broker_order.avg_fill_price
                
                if broker_order.filled_at:
                    order.filled_at = broker_order.filled_at
                
                # Check if filled
                if order.status == OrderStatus.FILLED:
                    logger.debug(f"Order filled after {attempts * 100}ms")
                    return order
                
                # Check if failed
                if order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.FAILED]:
                    logger.warning(f"Order {order.status.value}: {order.order_id}")
                    return order
                
                # Check slippage on partial fills
                if order.filled_qty > 0 and order.avg_fill_price:
                    slippage_bps = self._calculate_slippage_bps(intended_price, order.avg_fill_price)
                    if slippage_bps > self.max_slippage_bps:
                        logger.warning(f"Excessive slippage {slippage_bps:.1f} bps, cancelling order")
                        await self.broker.cancel_order(order.broker_order_id)
                        order.status = OrderStatus.CANCELLED
                        return order
                
                # For limit orders, adjust price if not filling quickly
                if (order.order_type == OrderType.LIMIT and 
                    attempts > 10 and  # After 1 second
                    order.filled_qty == 0):
                    
                    # Get fresh market data
                    market_data = await self.cache.get(CacheType.QUOTES, f"market:{order.symbol}")
                    if market_data:
                        # Adjust limit price more aggressively
                        if order.side == OrderSide.BUY:
                            new_price = market_data['ask'] * 1.001  # 10 bps above ask
                        else:
                            new_price = market_data['bid'] * 0.999  # 10 bps below bid
                        
                        logger.debug(f"Adjusting limit price from {order.limit_price} to {new_price}")
                        
                        # Cancel and replace
                        await self.broker.cancel_order(order.broker_order_id)
                        order.limit_price = new_price
                        new_broker_id = await self.broker.submit_order(order)
                        
                        if new_broker_id:
                            order.broker_order_id = new_broker_id
                            attempts = 0  # Reset counter
                
                # Wait before next check
                await asyncio.sleep(self.monitoring_interval)
                attempts += 1
                
            except Exception as e:
                logger.error(f"Error monitoring order {order.order_id}: {e}")
                attempts += 1
                await asyncio.sleep(self.monitoring_interval)
        
        # Timeout - cancel order
        logger.warning(f"Order monitoring timeout for {order.order_id}")
        try:
            await self.broker.cancel_order(order.broker_order_id)
            order.status = OrderStatus.CANCELLED
        except Exception as e:
            logger.debug(f"Could not cancel order {order.broker_order_id}: {e}")
            pass
        
        return order
    
    def _calculate_slippage_bps(self, intended_price: float, 
                               execution_price: float) -> float:
        """Calculate slippage in basis points."""
        if intended_price <= 0:
            return 0.0
        
        return abs(execution_price - intended_price) / intended_price * 10000
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get fast execution statistics."""
        if not self.execution_metrics:
            return {}
        
        metrics_list = list(self.execution_metrics.values())
        
        return {
            'total_executions': len(metrics_list),
            'avg_signal_to_order_ms': sum(m.signal_to_order_ms for m in metrics_list) / len(metrics_list),
            'avg_order_to_fill_ms': sum(m.order_to_fill_ms for m in metrics_list) / len(metrics_list),
            'avg_total_execution_ms': sum(m.total_execution_ms for m in metrics_list) / len(metrics_list),
            'avg_slippage_bps': sum(m.slippage_bps for m in metrics_list) / len(metrics_list),
            'max_slippage_bps': max(m.slippage_bps for m in metrics_list),
            'min_execution_ms': min(m.total_execution_ms for m in metrics_list),
            'max_execution_ms': max(m.total_execution_ms for m in metrics_list)
        }
    
    def reset_order_pool(self):
        """Reset order pool for reuse."""
        for order in self.order_pool:
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                               OrderStatus.REJECTED, OrderStatus.FAILED]:
                # Reset order for reuse
                order.symbol = ""
                order.status = OrderStatus.PENDING
                order.filled_qty = 0
                order.avg_fill_price = None
                order.broker_order_id = None
                order.submitted_at = None
                order.filled_at = None
                order.cancelled_at = None
        
        logger.debug("Order pool reset")
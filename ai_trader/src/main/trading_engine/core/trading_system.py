"""
Trading System - Main Coordinator

Central orchestrator for all trading operations, integrating existing components:
- PortfolioManager (existing)
- PositionManager (new)  
- FillProcessor (existing)
- BrokerReconciler (existing)
- FastExecutionPath (existing)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
from contextlib import asynccontextmanager

# Import existing common models
from main.models.common import (
    Order, OrderStatus, OrderSide, OrderType, Position
)

# Import existing core components
from main.trading_engine.core.portfolio_manager import PortfolioManager
from main.trading_engine.core.position_manager import PositionManager
from main.trading_engine.core.fill_processor import FillProcessor
from main.trading_engine.core.broker_reconciler import BrokerReconciler
from main.trading_engine.core.fast_execution_path import FastExecutionPath
from main.trading_engine.core.position_events import PositionEvent

# Import configuration and cache
from main.config.config_manager import get_config
from main.utils.cache import get_global_cache, CacheType

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading system operational modes."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


class TradingSystemStatus(Enum):
    """Trading system status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TradingSystem:
    """
    Main trading system coordinator that orchestrates all trading operations.
    
    Integrates existing components:
    - Portfolio management (PortfolioManager)
    - Position tracking (PositionManager)  
    - Order fill processing (FillProcessor)
    - Broker reconciliation (BrokerReconciler)
    - Fast execution (FastExecutionPath)
    """
    
    def __init__(self, 
                 broker_interface=None,
                 config: Optional[Dict[str, Any]] = None,
                 mode: TradingMode = TradingMode.PAPER):
        """
        Initialize trading system.
        
        Args:
            broker_interface: Broker interface implementation
            config: System configuration
            mode: Trading mode (live, paper, backtest)
        """
        self.broker = broker_interface
        self.config = config or get_config()
        self.mode = mode
        self.status = TradingSystemStatus.STOPPED
        
        # Core components - using existing implementations
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.position_manager: Optional[PositionManager] = None
        self.fill_processor: Optional[FillProcessor] = None
        self.broker_reconciler: Optional[BrokerReconciler] = None
        self.fast_execution: Optional[FastExecutionPath] = None
        
        # Order management
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Event handling
        self.event_handlers: List[Callable] = []
        
        # System state
        self.cache = get_global_cache()
        self.is_trading_enabled = False
        self.start_time: Optional[datetime] = None
        
        # Performance metrics
        self.metrics = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'total_pnl': 0.0,
            'positions_opened': 0,
            'positions_closed': 0
        }
        
        # System locks for thread safety
        self._system_lock = asyncio.Lock()
        self._order_lock = asyncio.Lock()
        
        logger.info(f"Trading system initialized in {mode.value} mode")
    
    async def initialize(self):
        """Initialize all trading system components."""
        async with self._system_lock:
            try:
                self.status = TradingSystemStatus.STARTING
                logger.info("Initializing trading system components...")
                
                # Initialize portfolio manager with broker
                if self.broker:
                    self.portfolio_manager = PortfolioManager(self.broker, self.config)
                    await self.portfolio_manager.initialize_portfolio_from_broker()
                    logger.info("✅ Portfolio manager initialized")
                
                # Initialize position manager
                self.position_manager = PositionManager(self.portfolio_manager)
                
                # Add position event handler
                self.position_manager.add_event_handler(self._handle_position_event)
                logger.info("✅ Position manager initialized")
                
                # Initialize fill processor
                if self.position_manager:
                    self.fill_processor = FillProcessor(
                        self.position_manager.position_tracker
                    )
                    logger.info("✅ Fill processor initialized")
                
                # Initialize broker reconciler if we have a broker
                if self.broker and self.position_manager:
                    self.broker_reconciler = BrokerReconciler(
                        self.broker,
                        self.position_manager.position_tracker
                    )
                    logger.info("✅ Broker reconciler initialized")
                
                # Initialize fast execution path if we have a broker
                if self.broker:
                    self.fast_execution = FastExecutionPath(self.broker, self.config)
                    logger.info("✅ Fast execution path initialized")
                
                # Start background tasks
                await self._start_background_tasks()
                
                self.status = TradingSystemStatus.RUNNING
                self.start_time = datetime.now(timezone.utc)
                
                logger.info("✅ Trading system initialization complete")
                
            except Exception as e:
                self.status = TradingSystemStatus.ERROR
                logger.error(f"Failed to initialize trading system: {e}")
                raise
    
    async def _start_background_tasks(self):
        """Start background tasks for system maintenance."""
        try:
            # Start broker reconciliation if available
            if self.broker_reconciler:
                await self.broker_reconciler.start_continuous_reconciliation()
            
            # Cache system status
            await self.cache.set(
                CacheType.METRICS,
                "trading_system_status",
                {
                    'status': self.status.value,
                    'mode': self.mode.value,
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'components_initialized': {
                        'portfolio_manager': self.portfolio_manager is not None,
                        'position_manager': self.position_manager is not None,
                        'fill_processor': self.fill_processor is not None,
                        'broker_reconciler': self.broker_reconciler is not None,
                        'fast_execution': self.fast_execution is not None
                    }
                },
                300  # 5 minute cache
            )
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def enable_trading(self):
        """Enable trading operations."""
        if self.status != TradingSystemStatus.RUNNING:
            raise RuntimeError("Trading system must be running to enable trading")
        
        self.is_trading_enabled = True
        logger.info("✅ Trading enabled")
    
    async def disable_trading(self):
        """Disable trading operations."""
        self.is_trading_enabled = False
        logger.info("⚠️ Trading disabled")
    
    async def submit_order(self, order: Order) -> Optional[str]:
        """
        Submit order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            Broker order ID if successful
        """
        if not self.is_trading_enabled:
            logger.warning(f"Trading disabled - cannot submit order {order.order_id}")
            return None
        
        if not self.broker:
            logger.error("No broker interface - cannot submit order")
            return None
        
        async with self._order_lock:
            try:
                # Pre-trade risk checks could go here
                if not await self._pre_trade_checks(order):
                    return None
                
                # Submit to broker
                broker_order_id = await self.broker.submit_order(order)
                
                if broker_order_id:
                    # Update order status
                    order.status = OrderStatus.SUBMITTED
                    order.broker_order_id = broker_order_id
                    order.submitted_at = datetime.now(timezone.utc)
                    
                    # Track order
                    self.active_orders[order.order_id] = order
                    self.metrics['orders_submitted'] += 1
                    
                    logger.info(f"Order submitted: {order.symbol} {order.side.value} {order.quantity}")
                    
                return broker_order_id
                
            except Exception as e:
                logger.error(f"Error submitting order {order.order_id}: {e}")
                order.status = OrderStatus.FAILED
                return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel active order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successfully cancelled
        """
        async with self._order_lock:
            try:
                order = self.active_orders.get(order_id)
                if not order:
                    logger.warning(f"Order {order_id} not found in active orders")
                    return False
                
                if not order.broker_order_id:
                    logger.warning(f"Order {order_id} has no broker order ID")
                    return False
                
                # Cancel with broker
                success = await self.broker.cancel_order(order.broker_order_id)
                
                if success:
                    order.status = OrderStatus.CANCELLED
                    order.cancelled_at = datetime.now(timezone.utc)
                    
                    # Move to history
                    self.order_history.append(order)
                    del self.active_orders[order_id]
                    
                    self.metrics['orders_cancelled'] += 1
                    
                    logger.info(f"Order cancelled: {order_id}")
                
                return success
                
            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
                return False
    
    async def process_fill(self, 
                         order: Order, 
                         fill_price: float, 
                         fill_quantity: float,
                         commission: float = 0.0,
                         fees: float = 0.0):
        """
        Process order fill using existing FillProcessor.
        
        Args:
            order: Order that was filled
            fill_price: Execution price
            fill_quantity: Filled quantity
            commission: Commission paid
            fees: Fees paid
        """
        if not self.fill_processor:
            logger.error("Fill processor not initialized")
            return
        
        try:
            # Process fill using existing FillProcessor
            fill_result = await self.fill_processor.process_fill(
                order=order,
                fill_price=fill_price,
                fill_quantity=fill_quantity,
                commission=commission,
                fees=fees
            )
            
            # Update order status
            order.filled_qty = (order.filled_qty or 0) + fill_quantity
            order.avg_fill_price = fill_price  # Simplified - would need proper averaging
            
            if order.filled_qty >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now(timezone.utc)
                
                # Move to history
                self.order_history.append(order)
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
                
                self.metrics['orders_filled'] += 1
            else:
                order.status = OrderStatus.PARTIAL
            
            # Update metrics
            self.metrics['total_pnl'] += float(fill_result.realized_pnl)
            
            logger.info(f"Fill processed: {order.symbol} {fill_quantity} @ {fill_price}")
            
        except Exception as e:
            logger.error(f"Error processing fill for order {order.order_id}: {e}")
    
    async def fast_execute(self,
                         symbol: str,
                         side: OrderSide,
                         quantity: float,
                         signal_timestamp: datetime,
                         metadata: Optional[Dict[str, Any]] = None) -> Optional[Order]:
        """
        Execute order through fast execution path.
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            signal_timestamp: When signal was generated
            metadata: Optional metadata
            
        Returns:
            Executed order if successful
        """
        if not self.fast_execution:
            logger.error("Fast execution not available")
            return None
        
        if not self.is_trading_enabled:
            logger.warning("Trading disabled - cannot fast execute")
            return None
        
        try:
            # Execute using existing FastExecutionPath
            executed_order = await self.fast_execution.execute_fast(
                symbol=symbol,
                side=side,
                quantity=quantity,
                signal_timestamp=signal_timestamp,
                metadata=metadata
            )
            
            if executed_order:
                logger.info(f"Fast execution completed: {symbol} {side.value} {quantity}")
            
            return executed_order
            
        except Exception as e:
            logger.error(f"Error in fast execution for {symbol}: {e}")
            return None
    
    async def _pre_trade_checks(self, order: Order) -> bool:
        """Perform pre-trade risk and validation checks."""
        try:
            # Basic validation
            if order.quantity <= 0:
                logger.warning(f"Invalid quantity for order {order.order_id}: {order.quantity}")
                return False
            
            # Check if we can open new position
            if self.portfolio_manager:
                can_open = await self.portfolio_manager.can_open_position()
                if not can_open:
                    logger.warning(f"Cannot open new position - limit reached")
                    return False
            
            # Additional risk checks would go here
            
            return True
            
        except Exception as e:
            logger.error(f"Error in pre-trade checks: {e}")
            return False
    
    async def _handle_position_event(self, event: PositionEvent):
        """Handle position events."""
        try:
            # Update metrics based on event type
            if event.event_type.value == "position_opened":
                self.metrics['positions_opened'] += 1
            elif event.event_type.value == "position_closed":
                self.metrics['positions_closed'] += 1
            
            # Forward to external handlers
            for handler in self.event_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in position event handler: {e}")
            
        except Exception as e:
            logger.error(f"Error handling position event: {e}")
    
    def add_event_handler(self, handler: Callable):
        """Add event handler for position events."""
        self.event_handlers.append(handler)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = None
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        # Get position summary
        position_summary = {}
        if self.position_manager:
            position_summary = await self.position_manager.get_position_summary()
        
        # Get portfolio summary  
        portfolio_summary = {}
        if self.portfolio_manager:
            portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
        
        return {
            'status': self.status.value,
            'mode': self.mode.value,
            'trading_enabled': self.is_trading_enabled,
            'uptime_seconds': uptime,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'active_orders': len(self.active_orders),
            'metrics': self.metrics.copy(),
            'position_summary': position_summary,
            'portfolio_summary': portfolio_summary,
            'components': {
                'portfolio_manager': self.portfolio_manager is not None,
                'position_manager': self.position_manager is not None,
                'fill_processor': self.fill_processor is not None,
                'broker_reconciler': self.broker_reconciler is not None,
                'fast_execution': self.fast_execution is not None
            }
        }
    
    async def shutdown(self):
        """Shutdown trading system gracefully."""
        logger.info("Shutting down trading system...")
        
        self.status = TradingSystemStatus.STOPPING
        self.is_trading_enabled = False
        
        try:
            # Stop broker reconciliation
            if self.broker_reconciler:
                await self.broker_reconciler.stop_continuous_reconciliation()
            
            # Cancel any remaining active orders
            if self.active_orders:
                logger.info(f"Cancelling {len(self.active_orders)} active orders...")
                for order_id in list(self.active_orders.keys()):
                    await self.cancel_order(order_id)
            
            # Cleanup components
            if self.position_manager:
                await self.position_manager.cleanup()
            
            if self.broker_reconciler:
                await self.broker_reconciler.cleanup()
            
            self.status = TradingSystemStatus.STOPPED
            logger.info("✅ Trading system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.status = TradingSystemStatus.ERROR
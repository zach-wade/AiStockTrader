"""
Execution Engine - Central Orchestrator

The top-level orchestrator that coordinates all trading operations across
the entire system. Manages the lifecycle of trading sessions and provides
unified interfaces for trade execution, portfolio management, and system control.

This engine serves as the main entry point for trading operations, coordinating:
- TradingSystem (comprehensive coordinator)
- Multiple broker integrations
- Risk management systems
- Performance monitoring
- System health checks
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum
from contextlib import asynccontextmanager

# Core trading components
from main.trading_engine.core.trading_system import TradingSystem, TradingMode, TradingSystemStatus
from main.trading_engine.core.position_manager import PositionManager
from main.trading_engine.core.position_events import PositionEvent, PositionEventType

# Import existing models
from main.models.common import Order, OrderStatus, OrderSide, OrderType, Position

# Configuration and utilities
from main.config.config_manager import get_config
from main.utils.cache import get_global_cache, CacheType

# Import existing risk management
from main.trading_engine.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class SignalStatus(Enum):
    """Status of a trading signal."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TradingSignal:
    """Represents a trading signal to be executed."""
    
    def __init__(self,
                 symbol: str,
                 side: OrderSide,
                 quantity: int,
                 signal_type: str,
                 source: str,
                 confidence: float = 1.0,
                 metadata: Optional[Dict[str, Any]] = None):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.signal_type = signal_type
        self.source = source
        self.confidence = confidence
        self.metadata = metadata or {}
        self.status = SignalStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        self.signal_id = f"{source}_{symbol}_{self.created_at.isoformat()}"


class ExecutionEngineStatus(Enum):
    """Execution engine status."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ExecutionMode(Enum):
    """Execution engine operational modes."""
    MANUAL = "manual"          # Manual trade approval required
    SEMI_AUTO = "semi_auto"    # Some automation with human oversight
    FULL_AUTO = "full_auto"    # Fully automated execution
    RESEARCH = "research"      # Research/analysis mode only
    EMERGENCY = "emergency"    # Emergency shutdown/liquidation mode


class ExecutionEngine:
    """
    Central execution engine that orchestrates all trading operations.
    
    This is the highest-level component that manages multiple TradingSystems,
    coordinates cross-system operations, and provides unified control interfaces.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 trading_mode: TradingMode = TradingMode.PAPER,
                 execution_mode: ExecutionMode = ExecutionMode.SEMI_AUTO):
        """
        Initialize execution engine.
        
        Args:
            config: System configuration
            trading_mode: Trading mode for all systems
            execution_mode: Execution automation level
        """
        self.config = config or get_config()
        self.trading_mode = trading_mode
        self.execution_mode = execution_mode
        self.status = ExecutionEngineStatus.STOPPED
        
        # Core systems
        self.trading_systems: Dict[str, TradingSystem] = {}
        self.active_brokers: Set[str] = set()
        self.risk_manager: Optional[RiskManager] = None
        
        # Global state management
        self.cache = get_global_cache()
        self.global_position_manager: Optional[PositionManager] = None
        
        # Event handling and coordination
        self.event_handlers: List[callable] = []
        self.cross_system_orders: Dict[str, Order] = {}
        
        # Performance and monitoring
        self.session_start_time: Optional[datetime] = None
        self.session_metrics = {
            'total_orders_submitted': 0,
            'total_orders_filled': 0,
            'total_orders_cancelled': 0,
            'total_realized_pnl': 0.0,
            'total_unrealized_pnl': 0.0,
            'active_positions': 0,
            'systems_online': 0,
            'last_heartbeat': None
        }
        
        # System coordination locks
        self._engine_lock = asyncio.Lock()
        self._order_coordination_lock = asyncio.Lock()
        self._position_sync_lock = asyncio.Lock()
        
        # Emergency controls
        self.emergency_stop_active = False
        self.emergency_liquidation_active = False
        
        logger.info(f"Execution engine initialized: {trading_mode.value} mode, {execution_mode.value} execution")
    
    async def initialize(self) -> bool:
        """
        Initialize the execution engine and all subsystems.
        
        Returns:
            True if initialization successful
        """
        async with self._engine_lock:
            try:
                self.status = ExecutionEngineStatus.INITIALIZING
                logger.info("=� Initializing execution engine...")
                
                # Initialize risk management first
                await self._initialize_risk_management()
                
                # Initialize global position tracking
                await self._initialize_global_position_tracking()
                
                # Initialize trading systems based on configuration
                await self._initialize_trading_systems()
                
                # Start system coordination tasks
                await self._start_coordination_tasks()
                
                # Perform system health checks
                if not await self._perform_health_checks():
                    raise RuntimeError("System health checks failed")
                
                self.status = ExecutionEngineStatus.READY
                self.session_start_time = datetime.now(timezone.utc)
                
                logger.info(" Execution engine initialization complete")
                return True
                
            except Exception as e:
                self.status = ExecutionEngineStatus.ERROR
                logger.error(f"L Execution engine initialization failed: {e}")
                return False
    
    async def _initialize_risk_management(self):
        """Initialize risk management systems."""
        try:
            # Initialize with existing RiskManager if available
            self.risk_manager = RiskManager(self.config)
            await self.risk_manager.initialize()
            logger.info(" Risk management initialized")
            
        except Exception as e:
            logger.error(f"Risk management initialization failed: {e}")
            # Continue without risk manager in some modes
            if self.trading_mode in [TradingMode.PAPER, TradingMode.BACKTEST]:
                logger.warning("Continuing without risk manager in test mode")
            else:
                raise
    
    async def _initialize_global_position_tracking(self):
        """Initialize global position tracking across all systems."""
        try:
            self.global_position_manager = PositionManager()
            self.global_position_manager.add_event_handler(self._handle_global_position_event)
            logger.info(" Global position tracking initialized")
            
        except Exception as e:
            logger.error(f"Global position tracking initialization failed: {e}")
            raise
    
    async def _initialize_trading_systems(self):
        """Initialize trading systems for configured brokers."""
        try:
            broker_configs = self.config.get('brokers', {})
            
            for broker_name, broker_config in broker_configs.items():
                if not broker_config.get('enabled', False):
                    logger.info(f"Skipping disabled broker: {broker_name}")
                    continue
                
                # Create trading system for this broker
                trading_system = await self._create_trading_system(broker_name, broker_config)
                
                if trading_system:
                    self.trading_systems[broker_name] = trading_system
                    self.active_brokers.add(broker_name)
                    logger.info(f" Trading system initialized for {broker_name}")
            
            if not self.trading_systems:
                logger.warning("No trading systems initialized")
            else:
                logger.info(f"Initialized {len(self.trading_systems)} trading systems")
                
        except Exception as e:
            logger.error(f"Trading systems initialization failed: {e}")
            raise
    
    async def _create_trading_system(self, broker_name: str, broker_config: Dict[str, Any]) -> Optional[TradingSystem]:
        """Create and initialize a trading system for a specific broker."""
        try:
            # Import broker interface dynamically based on configuration
            broker_interface = await self._create_broker_interface(broker_name, broker_config)
            
            if not broker_interface:
                logger.warning(f"Failed to create broker interface for {broker_name}")
                return None
            
            # Create trading system
            trading_system = TradingSystem(
                broker_interface=broker_interface,
                config=self.config,
                mode=self.trading_mode
            )
            
            # Initialize the system
            await trading_system.initialize()
            
            # Add event handlers for coordination
            trading_system.add_event_handler(self._handle_trading_system_event)
            
            return trading_system
            
        except Exception as e:
            logger.error(f"Failed to create trading system for {broker_name}: {e}")
            return None
    
    async def _create_broker_interface(self, broker_name: str, broker_config: Dict[str, Any]):
        """Create broker interface based on configuration."""
        try:
            # Dynamic import based on broker type
            if broker_name.lower() == 'alpaca':
                from main.trading_engine.brokers.alpaca_broker import AlpacaBroker
                return AlpacaBroker(broker_config)
            elif broker_name.lower() == 'interactive_brokers':
                from main.trading_engine.brokers.ib_broker import IBBroker
                return IBBroker(broker_config)
            else:
                logger.error(f"Unknown broker type: {broker_name}")
                return None
                
        except ImportError as e:
            logger.error(f"Failed to import broker {broker_name}: {e}")
            return None
    
    async def _start_coordination_tasks(self):
        """Start background tasks for system coordination."""
        try:
            # Start system monitoring
            asyncio.create_task(self._system_monitor_task())
            
            # Start position synchronization
            asyncio.create_task(self._position_sync_task())
            
            # Start performance tracking
            asyncio.create_task(self._performance_tracking_task())
            
            logger.info(" Coordination tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start coordination tasks: {e}")
            raise
    
    async def _perform_health_checks(self) -> bool:
        """Perform comprehensive system health checks."""
        try:
            health_status = {
                'trading_systems': 0,
                'brokers_connected': 0,
                'risk_manager': False,
                'global_positions': False
            }
            
            # Check trading systems
            for broker_name, system in self.trading_systems.items():
                status = await system.get_system_status()
                if status['status'] == 'running':
                    health_status['trading_systems'] += 1
                    health_status['brokers_connected'] += 1
            
            # Check risk manager
            if self.risk_manager:
                health_status['risk_manager'] = True
            
            # Check global position tracking
            if self.global_position_manager:
                health_status['global_positions'] = True
            
            # Cache health status
            await self.cache.set(
                CacheType.METRICS,
                "execution_engine_health",
                health_status,
                300  # 5 minute cache
            )
            
            logger.info(f"Health check: {health_status}")
            return health_status['trading_systems'] > 0
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def start_trading(self) -> bool:
        """
        Start active trading operations.
        
        Returns:
            True if trading started successfully
        """
        if self.status != ExecutionEngineStatus.READY:
            logger.error("Engine must be in READY status to start trading")
            return False
        
        try:
            self.status = ExecutionEngineStatus.ACTIVE
            
            # Enable trading on all systems
            for broker_name, system in self.trading_systems.items():
                await system.enable_trading()
                logger.info(f"Trading enabled for {broker_name}")
            
            logger.info("<� Trading operations started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            self.status = ExecutionEngineStatus.ERROR
            return False
    
    async def pause_trading(self):
        """Pause trading operations without shutdown."""
        if self.status != ExecutionEngineStatus.ACTIVE:
            logger.warning("Trading is not currently active")
            return
        
        try:
            self.status = ExecutionEngineStatus.PAUSED
            
            # Disable trading on all systems
            for broker_name, system in self.trading_systems.items():
                await system.disable_trading()
                logger.info(f"Trading paused for {broker_name}")
            
            logger.info("� Trading operations paused")
            
        except Exception as e:
            logger.error(f"Failed to pause trading: {e}")
    
    async def resume_trading(self):
        """Resume paused trading operations."""
        if self.status != ExecutionEngineStatus.PAUSED:
            logger.warning("Trading is not currently paused")
            return
        
        try:
            self.status = ExecutionEngineStatus.ACTIVE
            
            # Re-enable trading on all systems
            for broker_name, system in self.trading_systems.items():
                await system.enable_trading()
                logger.info(f"Trading resumed for {broker_name}")
            
            logger.info("� Trading operations resumed")
            
        except Exception as e:
            logger.error(f"Failed to resume trading: {e}")
    
    async def submit_cross_system_order(self, 
                                      order: Order, 
                                      preferred_broker: Optional[str] = None) -> Optional[str]:
        """
        Submit order with intelligent broker selection and coordination.
        
        Args:
            order: Order to submit
            preferred_broker: Preferred broker for execution
            
        Returns:
            Order ID if successful
        """
        async with self._order_coordination_lock:
            try:
                # Pre-execution risk checks
                if self.risk_manager:
                    risk_approved = await self.risk_manager.evaluate_order_risk(order)
                    if not risk_approved:
                        logger.warning(f"Order rejected by risk management: {order.order_id}")
                        return None
                
                # Select optimal broker
                selected_broker = await self._select_optimal_broker(order, preferred_broker)
                if not selected_broker:
                    logger.error("No suitable broker available for order")
                    return None
                
                # Submit order through selected trading system
                trading_system = self.trading_systems[selected_broker]
                broker_order_id = await trading_system.submit_order(order)
                
                if broker_order_id:
                    # Track cross-system order
                    self.cross_system_orders[order.order_id] = order
                    self.session_metrics['total_orders_submitted'] += 1
                    
                    logger.info(f"Cross-system order submitted: {order.symbol} via {selected_broker}")
                
                return broker_order_id
                
            except Exception as e:
                logger.error(f"Failed to submit cross-system order: {e}")
                return None
    
    async def _select_optimal_broker(self, order: Order, preferred_broker: Optional[str]) -> Optional[str]:
        """Select optimal broker for order execution."""
        try:
            # Use preferred broker if specified and available
            if preferred_broker and preferred_broker in self.active_brokers:
                system = self.trading_systems[preferred_broker]
                status = await system.get_system_status()
                if status['status'] == 'running' and status['trading_enabled']:
                    return preferred_broker
            
            # Find best available broker
            for broker_name in self.active_brokers:
                system = self.trading_systems[broker_name]
                status = await system.get_system_status()
                if status['status'] == 'running' and status['trading_enabled']:
                    return broker_name
            
            return None
            
        except Exception as e:
            logger.error(f"Broker selection failed: {e}")
            return None
    
    async def emergency_stop(self):
        """Execute emergency stop of all trading operations."""
        logger.critical("=� EMERGENCY STOP ACTIVATED")
        
        self.emergency_stop_active = True
        self.status = ExecutionEngineStatus.EMERGENCY
        
        try:
            # Stop all trading immediately
            for broker_name, system in self.trading_systems.items():
                await system.disable_trading()
                logger.info(f"Emergency stop: {broker_name} trading disabled")
            
            # Cancel all active orders
            await self._cancel_all_active_orders()
            
            logger.critical("=� Emergency stop completed")
            
        except Exception as e:
            logger.critical(f"Emergency stop failed: {e}")
    
    async def emergency_liquidate_all(self):
        """Emergency liquidation of all positions."""
        logger.critical("=� EMERGENCY LIQUIDATION ACTIVATED")
        
        self.emergency_liquidation_active = True
        self.status = ExecutionEngineStatus.EMERGENCY
        
        try:
            # Get all positions across all systems
            all_positions = []
            for broker_name, system in self.trading_systems.items():
                positions = await system.position_manager.get_all_positions()
                all_positions.extend(positions)
            
            # Liquidate each position
            for position in all_positions:
                try:
                    # Create market order to close position
                    close_side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                    
                    liquidation_order = Order(
                        symbol=position.symbol,
                        side=close_side,
                        quantity=abs(position.quantity),
                        order_type=OrderType.MARKET
                    )
                    
                    await self.submit_cross_system_order(liquidation_order)
                    logger.info(f"Emergency liquidation order submitted: {position.symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to liquidate position {position.symbol}: {e}")
            
            logger.critical("=� Emergency liquidation orders submitted")
            
        except Exception as e:
            logger.critical(f"Emergency liquidation failed: {e}")
    
    async def _cancel_all_active_orders(self):
        """Cancel all active orders across all systems."""
        try:
            for broker_name, system in self.trading_systems.items():
                status = await system.get_system_status()
                active_orders = status.get('active_orders', 0)
                
                if active_orders > 0:
                    # Cancel orders through trading system
                    for order_id in list(system.active_orders.keys()):
                        await system.cancel_order(order_id)
                        logger.info(f"Cancelled order {order_id} on {broker_name}")
        
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
    
    async def _handle_global_position_event(self, event: PositionEvent):
        """Handle position events from global position manager."""
        try:
            # Update session metrics
            if event.event_type == PositionEventType.POSITION_OPENED:
                self.session_metrics['active_positions'] += 1
            elif event.event_type == PositionEventType.POSITION_CLOSED:
                self.session_metrics['active_positions'] -= 1
                if event.realized_pnl:
                    self.session_metrics['total_realized_pnl'] += float(event.realized_pnl)
            
            # Forward to external handlers
            for handler in self.event_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in global position event handler: {e}")
        
        except Exception as e:
            logger.error(f"Failed to handle global position event: {e}")
    
    async def _handle_trading_system_event(self, event):
        """Handle events from individual trading systems."""
        try:
            # Coordinate cross-system events here
            logger.debug(f"Trading system event: {event}")
            
        except Exception as e:
            logger.error(f"Failed to handle trading system event: {e}")
    
    async def _system_monitor_task(self):
        """Background task for system monitoring."""
        while self.status not in [ExecutionEngineStatus.STOPPED, ExecutionEngineStatus.ERROR]:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Update heartbeat
                self.session_metrics['last_heartbeat'] = datetime.now(timezone.utc).isoformat()
                
                # Check system health
                healthy_systems = 0
                for broker_name, system in self.trading_systems.items():
                    try:
                        status = await system.get_system_status()
                        if status['status'] == 'running':
                            healthy_systems += 1
                    except Exception as e:
                        logger.warning(f"Health check failed for {broker_name}: {e}")
                
                self.session_metrics['systems_online'] = healthy_systems
                
                if healthy_systems == 0 and self.status == ExecutionEngineStatus.ACTIVE:
                    logger.error("All trading systems offline - pausing operations")
                    await self.pause_trading()
                
            except Exception as e:
                logger.error(f"System monitor task error: {e}")
    
    async def _position_sync_task(self):
        """Background task for position synchronization."""
        while self.status not in [ExecutionEngineStatus.STOPPED, ExecutionEngineStatus.ERROR]:
            try:
                await asyncio.sleep(60)  # Sync every minute
                
                # Synchronize positions across systems
                async with self._position_sync_lock:
                    await self._synchronize_positions()
                
            except Exception as e:
                logger.error(f"Position sync task error: {e}")
    
    async def _synchronize_positions(self):
        """Synchronize position data across all systems."""
        try:
            # This would implement cross-broker position reconciliation
            # For now, just log position counts
            total_positions = 0
            
            for broker_name, system in self.trading_systems.items():
                try:
                    summary = await system.position_manager.get_position_summary()
                    total_positions += summary.get('position_count', 0)
                except Exception as e:
                    logger.debug(f"Position sync error for {broker_name}: {e}")
            
            self.session_metrics['active_positions'] = total_positions
            
        except Exception as e:
            logger.error(f"Position synchronization failed: {e}")
    
    async def _performance_tracking_task(self):
        """Background task for performance tracking."""
        while self.status not in [ExecutionEngineStatus.STOPPED, ExecutionEngineStatus.ERROR]:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                # Aggregate performance metrics
                await self._update_performance_metrics()
                
                # Cache performance data
                await self.cache.set(
                    CacheType.METRICS,
                    "execution_engine_performance",
                    self.session_metrics.copy(),
                    600  # 10 minute cache
                )
                
            except Exception as e:
                logger.error(f"Performance tracking task error: {e}")
    
    async def _update_performance_metrics(self):
        """Update aggregated performance metrics."""
        try:
            total_unrealized_pnl = 0.0
            
            for broker_name, system in self.trading_systems.items():
                try:
                    status = await system.get_system_status()
                    metrics = status.get('metrics', {})
                    
                    # Aggregate unrealized P&L from position summaries
                    position_summary = status.get('position_summary', {})
                    total_unrealized_pnl += position_summary.get('total_unrealized_pnl', 0.0)
                    
                except Exception as e:
                    logger.debug(f"Metrics update error for {broker_name}: {e}")
            
            self.session_metrics['total_unrealized_pnl'] = total_unrealized_pnl
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    def add_event_handler(self, handler: callable):
        """Add global event handler."""
        self.event_handlers.append(handler)
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive execution engine status."""
        try:
            # Get individual system statuses
            system_statuses = {}
            for broker_name, system in self.trading_systems.items():
                try:
                    system_statuses[broker_name] = await system.get_system_status()
                except Exception as e:
                    system_statuses[broker_name] = {"error": str(e)}
            
            # Calculate uptime
            uptime = None
            if self.session_start_time:
                uptime = (datetime.now(timezone.utc) - self.session_start_time).total_seconds()
            
            return {
                'engine_status': self.status.value,
                'trading_mode': self.trading_mode.value,
                'execution_mode': self.execution_mode.value,
                'session_uptime_seconds': uptime,
                'emergency_stop_active': self.emergency_stop_active,
                'emergency_liquidation_active': self.emergency_liquidation_active,
                'active_brokers': list(self.active_brokers),
                'session_metrics': self.session_metrics.copy(),
                'trading_systems': system_statuses,
                'risk_manager_active': self.risk_manager is not None,
                'global_position_tracking': self.global_position_manager is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive status: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown execution engine gracefully."""
        logger.info("= Shutting down execution engine...")
        
        self.status = ExecutionEngineStatus.SHUTTING_DOWN
        
        try:
            # Disable all trading first
            for broker_name, system in self.trading_systems.items():
                await system.disable_trading()
                logger.info(f"Trading disabled for {broker_name}")
            
            # Shutdown all trading systems
            for broker_name, system in self.trading_systems.items():
                await system.shutdown()
                logger.info(f"Trading system shutdown complete: {broker_name}")
            
            # Cleanup global resources
            if self.global_position_manager:
                await self.global_position_manager.cleanup()
            
            if self.risk_manager:
                await self.risk_manager.cleanup()
            
            self.status = ExecutionEngineStatus.STOPPED
            logger.info(" Execution engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.status = ExecutionEngineStatus.ERROR


# Convenience function for creating and initializing execution engine
async def create_execution_engine(
    config: Optional[Dict[str, Any]] = None,
    trading_mode: TradingMode = TradingMode.PAPER,
    execution_mode: ExecutionMode = ExecutionMode.SEMI_AUTO
) -> ExecutionEngine:
    """
    Create and initialize execution engine.
    
    Args:
        config: System configuration
        trading_mode: Trading mode
        execution_mode: Execution automation level
        
    Returns:
        Initialized ExecutionEngine
    """
    engine = ExecutionEngine(
        config=config,
        trading_mode=trading_mode,
        execution_mode=execution_mode
    )
    
    success = await engine.initialize()
    if not success:
        raise RuntimeError("Failed to initialize execution engine")
    
    return engine
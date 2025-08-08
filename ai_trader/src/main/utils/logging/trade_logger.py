"""
Trade execution logging for comprehensive trade tracking.

This module provides specialized logging for trade execution, orders,
positions, and related trading activities.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
import logging

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    timer
)
from main.utils.database import DatabasePool
from main.monitoring.metrics.unified_metrics_integration import UnifiedMetricsAdapter

logger = get_logger(__name__)


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TradeLogEntry:
    """Base trade log entry."""
    timestamp: datetime
    log_id: str
    level: LogLevel
    category: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        return data


@dataclass
class OrderLogEntry(TradeLogEntry):
    """Order-specific log entry."""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    quantity: float = 0.0
    price: Optional[float] = None
    status: str = ""
    filled_quantity: float = 0
    avg_fill_price: Optional[float] = None
    commission: float = 0
    slippage: float = 0
    
    def __post_init__(self):
        """Initialize base fields."""
        self.category = "order"


@dataclass
class PositionLogEntry(TradeLogEntry):
    """Position-specific log entry."""
    symbol: str = ""
    position_size: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    market_value: float = 0.0
    
    def __post_init__(self):
        """Initialize base fields."""
        self.category = "position"


@dataclass
class ExecutionLogEntry(TradeLogEntry):
    """Execution-specific log entry."""
    execution_id: str = ""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    venue: str = ""
    liquidity_type: str = ""  # 'maker' or 'taker'
    latency_ms: float = 0.0
    
    def __post_init__(self):
        """Initialize base fields."""
        self.category = "execution"


class TradeLogger(ErrorHandlingMixin):
    """
    Specialized logger for trade execution and order lifecycle.
    
    Features:
    - Order placement and lifecycle tracking
    - Position change logging
    - Execution quality metrics
    - P&L tracking
    - Trade analytics
    """
    
    def __init__(
        self,
        db_pool: DatabasePool,
        log_dir: str = "logs/trades",
        buffer_size: int = 1000,
        flush_interval: int = 5,  # seconds
        metrics_adapter: Optional[UnifiedMetricsAdapter] = None
    ):
        """Initialize trade logger."""
        super().__init__()
        self.db_pool = db_pool
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Initialize metrics adapter if not provided
        if metrics_adapter is None:
            from main.monitoring.metrics.unified_metrics import UnifiedMetrics
            unified_metrics = UnifiedMetrics(db_pool)
            self.metrics_adapter = UnifiedMetricsAdapter(unified_metrics)
        else:
            self.metrics_adapter = metrics_adapter
        
        # Log buffers
        self._log_buffer: List[TradeLogEntry] = []
        self._order_buffer: List[OrderLogEntry] = []
        self._position_buffer: List[PositionLogEntry] = []
        self._execution_buffer: List[ExecutionLogEntry] = []
        
        # File handlers
        self._file_handlers = self._setup_file_handlers()
        
        # State tracking
        self._is_running = False
        self._flush_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._log_count = 0
        self._flush_count = 0
        self._error_count = 0
        
        # Order state tracking
        self._active_orders: Dict[str, OrderLogEntry] = {}
        self._position_states: Dict[str, PositionLogEntry] = {}
    
    def _setup_file_handlers(self) -> Dict[str, logging.Handler]:
        """Setup file handlers for different log types."""
        handlers = {}
        
        # Order log
        order_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "orders.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        order_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        handlers['order'] = order_handler
        
        # Position log
        position_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "positions.log",
            maxBytes=50*1024*1024,
            backupCount=10
        )
        position_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        handlers['position'] = position_handler
        
        # Execution log
        execution_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "executions.log",
            maxBytes=50*1024*1024,
            backupCount=10
        )
        execution_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        handlers['execution'] = execution_handler
        
        # JSON log for analysis
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trades.json",
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5
        )
        handlers['json'] = json_handler
        
        return handlers
    
    async def start(self):
        """Start the trade logger."""
        with self._handle_error("starting trade logger"):
            if self._is_running:
                logger.warning("Trade logger already running")
                return
            
            self._is_running = True
            self._flush_task = asyncio.create_task(self._flush_loop())
            
            logger.info("Started trade logger")
    
    async def stop(self):
        """Stop the trade logger."""
        self._is_running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self._flush_all_buffers()
        
        logger.info("Stopped trade logger")
    
    def log_order_placed(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log order placement."""
        with self._handle_error("logging order placement"):
            entry = OrderLogEntry(
                timestamp=datetime.utcnow(),
                log_id=f"ORD_{self._generate_log_id()}",
                level=LogLevel.INFO,
                message=f"Order placed: {side} {quantity} {symbol}",
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                status="pending",
                metadata=metadata or {}
            )
            
            self._add_to_buffer(entry)
            self._active_orders[order_id] = entry
            
            # Record metric
            self.metrics_adapter.record_metric(
                'trading.orders.placed',
                1,
                tags={'symbol': symbol, 'side': side, 'type': order_type}
            )
    
    def log_order_filled(
        self,
        order_id: str,
        filled_quantity: float,
        fill_price: float,
        commission: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log order fill."""
        with self._handle_error("logging order fill"):
            if order_id in self._active_orders:
                order = self._active_orders[order_id]
                
                # Update order state
                order.filled_quantity += filled_quantity
                order.status = "filled" if order.filled_quantity >= order.quantity else "partial"
                
                # Calculate average fill price
                if order.avg_fill_price:
                    total_value = (order.avg_fill_price * (order.filled_quantity - filled_quantity) +
                                 fill_price * filled_quantity)
                    order.avg_fill_price = total_value / order.filled_quantity
                else:
                    order.avg_fill_price = fill_price
                
                order.commission += commission
                
                # Create fill entry
                entry = OrderLogEntry(
                    timestamp=datetime.utcnow(),
                    log_id=f"FIL_{self._generate_log_id()}",
                    level=LogLevel.INFO,
                    message=f"Order filled: {filled_quantity} @ {fill_price}",
                    order_id=order_id,
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=filled_quantity,
                    price=fill_price,
                    status="fill",
                    commission=commission,
                    metadata=metadata or {}
                )
                
                self._add_to_buffer(entry)
                
                # Record metric
                self.metrics_adapter.record_metric(
                    'trading.orders.filled',
                    filled_quantity,
                    tags={'symbol': order.symbol, 'side': order.side}
                )
    
    def log_order_cancelled(
        self,
        order_id: str,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log order cancellation."""
        with self._handle_error("logging order cancellation"):
            if order_id in self._active_orders:
                order = self._active_orders[order_id]
                order.status = "cancelled"
                
                entry = OrderLogEntry(
                    timestamp=datetime.utcnow(),
                    log_id=f"CAN_{self._generate_log_id()}",
                    level=LogLevel.INFO,
                    message=f"Order cancelled: {reason}",
                    order_id=order_id,
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    price=order.price,
                    status="cancelled",
                    metadata=metadata or {}
                )
                
                self._add_to_buffer(entry)
                del self._active_orders[order_id]
                
                # Record metric
                self.metrics_adapter.record_metric(
                    'trading.orders.cancelled',
                    1,
                    tags={'symbol': order.symbol, 'reason': reason}
                )
    
    def log_position_update(
        self,
        symbol: str,
        position_size: float,
        avg_entry_price: float,
        current_price: float,
        unrealized_pnl: float,
        realized_pnl: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log position update."""
        with self._handle_error("logging position update"):
            total_pnl = unrealized_pnl + realized_pnl
            market_value = abs(position_size * current_price)
            
            entry = PositionLogEntry(
                timestamp=datetime.utcnow(),
                log_id=f"POS_{self._generate_log_id()}",
                level=LogLevel.INFO,
                message=f"Position update: {symbol} {position_size}",
                symbol=symbol,
                position_size=position_size,
                avg_entry_price=avg_entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=total_pnl,
                market_value=market_value,
                metadata=metadata or {}
            )
            
            self._add_to_buffer(entry)
            self._position_states[symbol] = entry
            
            # Record metrics
            self.metrics_adapter.record_metric(
                'trading.position.size',
                abs(position_size),
                tags={'symbol': symbol}
            )
            self.metrics_adapter.record_metric(
                'trading.position.pnl',
                total_pnl,
                tags={'symbol': symbol, 'type': 'total'}
            )
    
    def log_execution(
        self,
        execution_id: str,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        venue: str,
        liquidity_type: str,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log trade execution details."""
        with self._handle_error("logging execution"):
            entry = ExecutionLogEntry(
                timestamp=datetime.utcnow(),
                log_id=f"EXE_{self._generate_log_id()}",
                level=LogLevel.INFO,
                message=f"Execution: {side} {quantity} {symbol} @ {price}",
                execution_id=execution_id,
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                venue=venue,
                liquidity_type=liquidity_type,
                latency_ms=latency_ms,
                metadata=metadata or {}
            )
            
            self._add_to_buffer(entry)
            
            # Record metrics
            self.metrics_adapter.record_metric(
                'trading.execution.latency',
                latency_ms,
                tags={'symbol': symbol, 'venue': venue}
            )
            self.metrics_adapter.record_metric(
                'trading.execution.count',
                1,
                tags={'symbol': symbol, 'liquidity': liquidity_type}
            )
    
    def log_trade_complete(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        duration_seconds: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log completed trade summary."""
        with self._handle_error("logging completed trade"):
            pnl_percent = ((exit_price - entry_price) / entry_price * 100
                          if side == "buy" else
                          (entry_price - exit_price) / entry_price * 100)
            
            entry = TradeLogEntry(
                timestamp=datetime.utcnow(),
                log_id=f"TRD_{self._generate_log_id()}",
                level=LogLevel.INFO,
                category="trade_complete",
                message=(f"Trade complete: {symbol} {side} {quantity} "
                        f"P&L: ${pnl:.2f} ({pnl_percent:.2f}%)"),
                metadata={
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'duration_seconds': duration_seconds,
                    **(metadata or {})
                }
            )
            
            self._add_to_buffer(entry)
            
            # Record metrics
            self.metrics_adapter.record_metric(
                'trading.trades.completed',
                1,
                tags={'symbol': symbol, 'side': side}
            )
            self.metrics_adapter.record_metric(
                'trading.trades.pnl',
                pnl,
                tags={'symbol': symbol}
            )
    
    def log_risk_event(
        self,
        event_type: str,
        symbol: Optional[str] = None,
        message: str = "",
        severity: str = "warning",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log risk management event."""
        with self._handle_error("logging risk event"):
            level = LogLevel.WARNING if severity == "warning" else LogLevel.ERROR
            
            entry = TradeLogEntry(
                timestamp=datetime.utcnow(),
                log_id=f"RSK_{self._generate_log_id()}",
                level=level,
                category="risk",
                message=f"Risk event ({event_type}): {message}",
                metadata={
                    'event_type': event_type,
                    'symbol': symbol,
                    'severity': severity,
                    **(metadata or {})
                }
            )
            
            self._add_to_buffer(entry)
            
            # Record metric
            self.metrics_adapter.record_metric(
                'trading.risk.events',
                1,
                tags={'type': event_type, 'severity': severity}
            )
    
    def _add_to_buffer(self, entry: TradeLogEntry) -> None:
        """Add entry to appropriate buffer."""
        self._log_count += 1
        
        # Add to general buffer
        self._log_buffer.append(entry)
        
        # Add to specific buffer
        if isinstance(entry, OrderLogEntry):
            self._order_buffer.append(entry)
        elif isinstance(entry, PositionLogEntry):
            self._position_buffer.append(entry)
        elif isinstance(entry, ExecutionLogEntry):
            self._execution_buffer.append(entry)
        
        # Check if flush needed
        if len(self._log_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_all_buffers())
    
    async def _flush_loop(self):
        """Periodic buffer flush loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_all_buffers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error in flush loop: {e}")
    
    @timer
    async def _flush_all_buffers(self):
        """Flush all log buffers."""
        if not self._log_buffer:
            return
        
        try:
            # Write to files
            await self._write_to_files()
            
            # Write to database
            await self._write_to_database()
            
            # Clear buffers
            self._log_buffer.clear()
            self._order_buffer.clear()
            self._position_buffer.clear()
            self._execution_buffer.clear()
            
            self._flush_count += 1
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error flushing buffers: {e}")
    
    async def _write_to_files(self):
        """Write buffered logs to files."""
        # Write to JSON file
        json_entries = [entry.to_dict() for entry in self._log_buffer]
        
        with open(self.log_dir / "trades.json", "a") as f:
            for entry in json_entries:
                json.dump(entry, f)
                f.write("\n")
        
        # Write to category-specific files
        for entry in self._order_buffer:
            self._write_entry_to_file(entry, self._file_handlers['order'])
        
        for entry in self._position_buffer:
            self._write_entry_to_file(entry, self._file_handlers['position'])
        
        for entry in self._execution_buffer:
            self._write_entry_to_file(entry, self._file_handlers['execution'])
    
    def _write_entry_to_file(self, entry: TradeLogEntry, handler: logging.Handler):
        """Write single entry to file handler."""
        record = logging.LogRecord(
            name=__name__,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=entry.message,
            args=(),
            exc_info=None
        )
        handler.emit(record)
    
    async def _write_to_database(self):
        """Write buffered logs to database."""
        if not self._log_buffer:
            return
        
        async with self.db_pool.acquire() as conn:
            # Prepare batch insert data
            values = []
            
            for entry in self._log_buffer:
                values.append((
                    entry.log_id,
                    entry.timestamp,
                    entry.level.value,
                    entry.category,
                    entry.message,
                    json.dumps(entry.metadata)
                ))
            
            # Batch insert
            await conn.executemany(
                """
                INSERT INTO trade_logs (
                    log_id, timestamp, level, category, message, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                values
            )
    
    def _generate_log_id(self) -> str:
        """Generate unique log ID."""
        return f"{datetime.utcnow().timestamp():.0f}_{self._log_count}"
    
    async def get_recent_logs(
        self,
        category: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent trade logs."""
        with self._handle_error("getting recent logs"):
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT 
                        log_id,
                        timestamp,
                        level,
                        category,
                        message,
                        metadata
                    FROM trade_logs
                    WHERE 1=1
                """
                params = []
                
                if category:
                    params.append(category)
                    query += f" AND category = ${len(params)}"
                
                if symbol:
                    params.append(symbol)
                    query += f" AND metadata->>'symbol' = ${len(params)}"
                
                query += " ORDER BY timestamp DESC LIMIT $" + str(len(params) + 1)
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        'log_id': row['log_id'],
                        'timestamp': row['timestamp'].isoformat(),
                        'level': row['level'],
                        'category': row['category'],
                        'message': row['message'],
                        'metadata': json.loads(row['metadata'])
                    }
                    for row in rows
                ]
    
    def get_active_orders(self) -> Dict[str, OrderLogEntry]:
        """Get currently active orders."""
        return self._active_orders.copy()
    
    def get_position_states(self) -> Dict[str, PositionLogEntry]:
        """Get current position states."""
        return self._position_states.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        return {
            'log_count': self._log_count,
            'flush_count': self._flush_count,
            'error_count': self._error_count,
            'buffer_size': len(self._log_buffer),
            'active_orders': len(self._active_orders),
            'tracked_positions': len(self._position_states),
            'is_running': self._is_running
        }
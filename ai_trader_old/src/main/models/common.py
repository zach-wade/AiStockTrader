# File: ai_trader/trading_engine/models/common.py

"""
Common data models and enumerations for the trading engine.

Strictly defined, immutable data structures for clarity and consistency.
"""

# Standard library imports
from dataclasses import dataclass, field  # field is needed for default_factory
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np


# --- Enumerations ---
class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "pending"  # Created internally, awaiting submission
    SUBMITTED = "submitted"  # Sent to broker, awaiting confirmation
    PARTIAL = "partial"  # Partially filled by broker
    FILLED = "filled"  # Fully filled by broker
    CANCELLED = "cancelled"  # Cancelled by user or system
    REJECTED = "rejected"  # Rejected by broker
    EXPIRED = "expired"  # Expired due to time-in-force
    FAILED = "failed"  # Internal system failure (not a broker status)


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    """Time in force enumeration."""

    DAY = "day"
    GTC = "gtc"  # Good 'til Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    # Add others as needed by your broker or strategy
    OPG = "opg"  # At the opening
    CLS = "cls"  # At the close


class PositionSide(Enum):
    """Position side enumeration."""

    LONG = "long"
    SHORT = "short"


class VaRMethod(Enum):
    """Value at Risk calculation method."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"


# --- Data Models (Immutable where practical) ---
@dataclass(frozen=True)  # Enforce immutability
class Position:
    """Represents a single stock position."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float  # Cumulative realized PnL for this position
    side: str  # 'long' or 'short' (derived from quantity/direction of initial trade)
    timestamp: datetime  # Last updated time for this position

    @property
    def pnl(self) -> float:
        """Calculates unrealized P&L based on current price."""
        return (self.current_price - self.avg_entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        """Calculates unrealized P&L percentage."""
        return (
            (self.current_price / self.avg_entry_price - 1) * 100 if self.avg_entry_price else 0.0
        )


@dataclass(frozen=True)  # Enforce immutability
class Order:
    """Complete order representation for internal tracking."""

    order_id: str  # Our internal unique ID for the order
    symbol: str
    side: OrderSide  # Enum: BUY or SELL
    quantity: float  # Total quantity intended for the order
    order_type: OrderType  # Enum: MARKET, LIMIT, STOP, etc.
    time_in_force: TimeInForce  # Enum: DAY, GTC, IOC, etc.
    status: OrderStatus  # Enum: PENDING, FILLED, etc.
    created_at: datetime  # Timestamp of internal order creation

    # Optional fields
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Execution details
    filled_qty: float = 0.0  # Cumulative filled quantity
    avg_fill_price: Optional[float] = None  # Cumulative average fill price
    commission: float = 0.0  # Cumulative commission for this order

    # Broker-specific IDs
    broker_order_id: Optional[str] = None  # The ID assigned by the broker
    client_order_id: Optional[str] = None  # Your client-side ID sent to broker

    # Tracking
    strategy: Optional[str] = None
    parent_order_id: Optional[str] = None  # For modified/replaced orders

    # Status timestamps
    submitted_at: Optional[datetime] = None  # Time order was sent to broker
    filled_at: Optional[datetime] = None  # Time order was fully filled
    cancelled_at: Optional[datetime] = None

    reject_reason: Optional[str] = None

    # Metadata for TCA, etc. (immutable dict, but values can be mutable if not careful)
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Use default_factory for mutable defaults

    # Internal history (could be moved to OrderManager for complex tracking)
    fills: List[Dict[str, Any]] = field(default_factory=list)  # Individual fill records
    status_history: List[Dict[str, Any]] = field(default_factory=list)  # Status changes

    @property
    def is_active(self) -> bool:
        """Check if order is still active (pending, submitted, partial)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]

    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled, cancelled, rejected, failed, expired)."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
            OrderStatus.EXPIRED,
        ]

    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to fill."""
        return self.quantity - self.filled_qty

    @property
    def fill_ratio(self) -> float:
        """Calculate fill ratio."""
        return self.filled_qty / self.quantity if self.quantity > 0 else 0.0

    def with_id(self, order_id: str) -> "Order":
        """Create new Order with assigned ID (atomic operation)."""
        return Order(
            order_id=order_id,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            time_in_force=self.time_in_force,
            status=self.status,
            created_at=self.created_at,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            filled_qty=self.filled_qty,
            avg_fill_price=self.avg_fill_price,
            commission=self.commission,
            broker_order_id=self.broker_order_id,
            client_order_id=self.client_order_id,
            strategy=self.strategy,
            parent_order_id=self.parent_order_id,
            submitted_at=self.submitted_at,
            filled_at=self.filled_at,
            cancelled_at=self.cancelled_at,
            reject_reason=self.reject_reason,
            metadata=self.metadata.copy(),
            fills=self.fills.copy(),
            status_history=self.status_history.copy(),
        )

    def with_status(
        self,
        status: OrderStatus,
        timestamp: Optional[datetime] = None,
        message: Optional[str] = None,
    ) -> "Order":
        """Create new Order with updated status and history (atomic operation)."""
        if timestamp is None:
            timestamp = datetime.now()

        new_history = self.status_history.copy()
        new_history.append(
            {
                "status": status.value,
                "timestamp": timestamp,
                "message": message or f"Status changed to {status.value}",
            }
        )

        # Update relevant timestamp fields based on status
        kwargs = {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "time_in_force": self.time_in_force,
            "status": status,
            "created_at": self.created_at,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "strategy": self.strategy,
            "parent_order_id": self.parent_order_id,
            "submitted_at": self.submitted_at,
            "filled_at": self.filled_at,
            "cancelled_at": self.cancelled_at,
            "reject_reason": self.reject_reason,
            "metadata": self.metadata.copy(),
            "fills": self.fills.copy(),
            "status_history": new_history,
        }

        # Update timestamp fields based on status
        if status == OrderStatus.SUBMITTED:
            kwargs["submitted_at"] = timestamp
        elif status == OrderStatus.FILLED:
            kwargs["filled_at"] = timestamp
        elif status == OrderStatus.CANCELLED:
            kwargs["cancelled_at"] = timestamp

        return Order(**kwargs)

    def with_fill(
        self,
        fill_qty: float,
        fill_price: float,
        timestamp: Optional[datetime] = None,
        commission: float = 0.0,
    ) -> "Order":
        """Create new Order with fill information (atomic operation)."""
        if timestamp is None:
            timestamp = datetime.now()

        new_filled_qty = self.filled_qty + fill_qty
        new_commission = self.commission + commission

        # Calculate new average fill price
        if self.filled_qty > 0:
            # Weighted average of existing fills and new fill
            total_value = (self.avg_fill_price * self.filled_qty) + (fill_price * fill_qty)
            new_avg_fill_price = total_value / new_filled_qty
        else:
            new_avg_fill_price = fill_price

        # Add fill record
        new_fills = self.fills.copy()
        new_fills.append(
            {
                "quantity": fill_qty,
                "price": fill_price,
                "timestamp": timestamp,
                "commission": commission,
            }
        )

        # Determine new status
        if new_filled_qty >= self.quantity:
            new_status = OrderStatus.FILLED
        elif new_filled_qty > 0:
            new_status = OrderStatus.PARTIAL
        else:
            new_status = self.status

        return Order(
            order_id=self.order_id,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            time_in_force=self.time_in_force,
            status=new_status,
            created_at=self.created_at,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            filled_qty=new_filled_qty,
            avg_fill_price=new_avg_fill_price,
            commission=new_commission,
            broker_order_id=self.broker_order_id,
            client_order_id=self.client_order_id,
            strategy=self.strategy,
            parent_order_id=self.parent_order_id,
            submitted_at=self.submitted_at,
            filled_at=timestamp if new_status == OrderStatus.FILLED else self.filled_at,
            cancelled_at=self.cancelled_at,
            reject_reason=self.reject_reason,
            metadata=self.metadata.copy(),
            fills=new_fills,
            status_history=self.status_history.copy(),
        ).with_status(new_status, timestamp, f"Fill: {fill_qty} @ {fill_price}")

    def with_reject_reason(self, reason: str, timestamp: Optional[datetime] = None) -> "Order":
        """Create new Order with rejection reason (atomic operation)."""
        if timestamp is None:
            timestamp = datetime.now()

        return Order(
            order_id=self.order_id,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            time_in_force=self.time_in_force,
            status=OrderStatus.REJECTED,
            created_at=self.created_at,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            filled_qty=self.filled_qty,
            avg_fill_price=self.avg_fill_price,
            commission=self.commission,
            broker_order_id=self.broker_order_id,
            client_order_id=self.client_order_id,
            strategy=self.strategy,
            parent_order_id=self.parent_order_id,
            submitted_at=self.submitted_at,
            filled_at=self.filled_at,
            cancelled_at=self.cancelled_at,
            reject_reason=reason,
            metadata=self.metadata.copy(),
            fills=self.fills.copy(),
            status_history=self.status_history.copy(),
        ).with_status(OrderStatus.REJECTED, timestamp, f"Rejected: {reason}")


@dataclass(frozen=True)  # Enforce immutability
class AccountInfo:
    """Account information data structure."""

    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    last_equity: float
    long_market_value: float
    short_market_value: float
    initial_margin: float
    maintenance_margin: float
    sma: float
    daytrade_count: int
    balance_asof: datetime  # Timestamp of when this info was valid
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    trade_suspended_by_user: bool
    currency: str = "USD"


@dataclass(frozen=True)  # Enforce immutability
class MarketData:
    """Market data structure for a single snapshot."""

    symbol: str
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last: Optional[float] = None
    volume: Optional[int] = None  # Volume for a specific period (e.g., daily)
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    vwap: Optional[float] = None
    trade_count: Optional[int] = None  # For bar data


@dataclass
class RiskMetrics:
    """Risk metrics for positions and portfolio."""

    # Position-level metrics
    position_size: float = 0.0
    position_value: float = 0.0
    position_pnl: float = 0.0
    position_pnl_pct: float = 0.0

    # Portfolio-level metrics
    portfolio_value: float = 0.0
    total_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0

    # Risk measures
    volatility: float = 0.0  # Annualized volatility
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0

    # VaR metrics
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    cvar_99: float = 0.0

    # Greeks (for options)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Concentration metrics
    largest_position_pct: float = 0.0
    top_5_concentration: float = 0.0
    sector_concentration: Dict[str, float] = field(default_factory=dict)

    # Correlation metrics
    correlation_to_spy: float = 0.0
    beta_to_spy: float = 0.0

    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Timestamp
    calculated_at: datetime = field(default_factory=datetime.now)


@dataclass
class VaRResult:
    """Value at Risk calculation result."""

    var_value: float  # VaR in currency terms
    var_pct: float  # VaR as percentage
    confidence_level: float  # e.g., 0.95 for 95% confidence
    time_horizon: int  # in days
    method: VaRMethod

    # Additional metrics
    expected_shortfall: float = 0.0  # CVaR/ES
    worst_case_loss: float = 0.0
    best_case_gain: float = 0.0

    # Method-specific details
    historical_observations: Optional[int] = None
    distribution_params: Optional[Dict[str, float]] = None  # For parametric
    num_simulations: Optional[int] = None  # For Monte Carlo

    # Backtesting metrics
    exceptions: Optional[int] = None  # Number of VaR breaches
    exception_rate: Optional[float] = None  # Actual vs expected
    kupiec_test_pvalue: Optional[float] = None  # Backtesting p-value

    # Decomposition
    position_vars: Optional[Dict[str, float]] = None  # VaR by position
    factor_vars: Optional[Dict[str, float]] = None  # VaR by risk factor

    calculated_at: datetime = field(default_factory=datetime.now)

    def is_breach(self, actual_loss: float) -> bool:
        """Check if actual loss exceeds VaR threshold."""
        return actual_loss > self.var_value

    def get_buffer(self, actual_loss: float) -> float:
        """Get remaining buffer before VaR breach."""
        return self.var_value - actual_loss


@dataclass
class Signal:
    """Trading signal representation."""

    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # Signal strength (0-1)
    confidence: float  # Confidence level (0-1)
    quantity: Optional[float] = None
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # Strategy that generated the signal
    reason: str = ""  # Human-readable reason
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Check if signal should trigger a trade."""
        return self.action in ["BUY", "SELL"] and self.strength > 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "strength": self.strength,
            "confidence": self.confidence,
            "quantity": self.quantity,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "reason": self.reason,
            "metadata": self.metadata,
        }


# You might also define a custom Signal model here if it is widely used
# beyond just the ExecutionEngine's internal representation.
# For now, TradingSignal remains in execution_engine.py.


@dataclass
class Fill:
    """Represents an order fill/execution."""

    order_id: str
    client_order_id: Optional[str] = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    commission: float = 0.0
    fees: float = 0.0

    def __post_init__(self):
        """Validate fill data."""
        if self.quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if self.price < 0:
            raise ValueError("Fill price cannot be negative")


@dataclass
class Strategy:
    """
    Base strategy representation for backtesting.

    Provides a complete framework for implementing trading strategies
    with signal generation, risk management, and performance tracking.
    """

    name: str
    symbols: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    # Internal state
    positions: Dict[str, Position] = field(default_factory=dict, init=False)
    pending_orders: List[Order] = field(default_factory=list, init=False)
    performance_metrics: Dict[str, float] = field(default_factory=dict, init=False)
    is_initialized: bool = field(default=False, init=False)

    def __post_init__(self):
        """Validate strategy configuration."""
        if not self.name:
            raise ValueError("Strategy name cannot be empty")

        # Set default parameters if not provided
        self._set_default_parameters()

    def _set_default_parameters(self):
        """Set default strategy parameters."""
        defaults = {
            "max_position_size": 0.1,  # 10% of portfolio
            "stop_loss_pct": 0.02,  # 2% stop loss
            "take_profit_pct": 0.05,  # 5% take profit
            "max_positions": 10,  # Maximum concurrent positions
            "use_trailing_stop": False,
            "rebalance_frequency": "daily",
            "min_holding_period": 1,  # Minimum days to hold
        }

        for key, default_value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = default_value

    def initialize(self, initial_capital: float, data_provider=None):
        """
        Initialize strategy before backtesting.

        Args:
            initial_capital: Starting capital
            data_provider: Optional data provider for historical data
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.data_provider = data_provider
        self.positions.clear()
        self.pending_orders.clear()
        self.performance_metrics.clear()
        self.is_initialized = True

        # Initialize performance tracking
        self.trades = []
        self.daily_returns = []
        self.equity_curve = [initial_capital]

        # Call strategy-specific initialization
        self.on_initialize()

    def on_initialize(self):
        """Override in subclasses for custom initialization."""
        pass

    def on_data(self, timestamp: datetime, market_data: Dict[str, MarketData]):
        """
        Called on each new data point.

        Args:
            timestamp: Current timestamp
            market_data: Dict of symbol -> MarketData

        Returns:
            List of signals
        """
        if not self.is_initialized:
            raise RuntimeError("Strategy not initialized")

        # Update current prices
        self._update_positions(market_data)

        # Check stop losses and take profits
        self._check_exit_conditions(market_data)

        # Generate new signals
        signals = self.generate_signals(timestamp, market_data)

        # Apply risk management
        validated_signals = self._apply_risk_management(signals)

        # Update performance metrics
        self._update_performance_metrics(timestamp, market_data)

        return validated_signals

    def generate_signals(
        self, timestamp: datetime, market_data: Dict[str, MarketData]
    ) -> List[Signal]:
        """
        Generate trading signals. Override in subclasses.

        Args:
            timestamp: Current timestamp
            market_data: Market data by symbol

        Returns:
            List of trading signals
        """
        # Default implementation - override in subclasses
        return []

    def on_order_filled(self, order: Order, fill_price: float, fill_time: datetime):
        """
        Called when an order is filled.

        Args:
            order: The filled order
            fill_price: Actual fill price
            fill_time: Time of fill
        """
        # Update positions
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            if order.side == OrderSide.BUY:
                # Add to position
                new_quantity = position.quantity + order.filled_qty
                new_cost_basis = position.cost_basis + (order.filled_qty * fill_price)
                new_avg_price = new_cost_basis / new_quantity

                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=new_quantity,
                    avg_entry_price=new_avg_price,
                    current_price=fill_price,
                    market_value=new_quantity * fill_price,
                    cost_basis=new_cost_basis,
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0,
                    realized_pnl=position.realized_pnl,
                    side="long" if new_quantity > 0 else "short",
                    timestamp=fill_time,
                )
            else:  # SELL
                # Reduce position
                new_quantity = position.quantity - order.filled_qty
                realized_pnl = order.filled_qty * (fill_price - position.avg_entry_price)

                if abs(new_quantity) < 0.0001:  # Position closed
                    self.trades.append(
                        {
                            "symbol": order.symbol,
                            "entry_time": position.timestamp,
                            "exit_time": fill_time,
                            "entry_price": position.avg_entry_price,
                            "exit_price": fill_price,
                            "quantity": order.filled_qty,
                            "pnl": realized_pnl,
                            "pnl_pct": (fill_price / position.avg_entry_price - 1) * 100,
                        }
                    )
                    del self.positions[order.symbol]
                else:
                    # Update remaining position
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=new_quantity,
                        avg_entry_price=position.avg_entry_price,
                        current_price=fill_price,
                        market_value=new_quantity * fill_price,
                        cost_basis=new_quantity * position.avg_entry_price,
                        unrealized_pnl=0,
                        unrealized_pnl_pct=0,
                        realized_pnl=position.realized_pnl + realized_pnl,
                        side="long" if new_quantity > 0 else "short",
                        timestamp=position.timestamp,
                    )
        else:
            # New position
            if order.side == OrderSide.BUY:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.filled_qty,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    market_value=order.filled_qty * fill_price,
                    cost_basis=order.filled_qty * fill_price,
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0,
                    realized_pnl=0,
                    side="long",
                    timestamp=fill_time,
                )

        # Update capital
        if order.side == OrderSide.BUY:
            self.current_capital -= order.filled_qty * fill_price
        else:
            self.current_capital += order.filled_qty * fill_price

    def on_day_end(self, date: datetime, market_data: Dict[str, MarketData]):
        """
        Called at the end of each trading day.

        Args:
            date: Current date
            market_data: End of day market data
        """
        # Calculate daily return
        portfolio_value = self.get_portfolio_value(market_data)
        daily_return = (portfolio_value / self.equity_curve[-1] - 1) if self.equity_curve else 0
        self.daily_returns.append(daily_return)
        self.equity_curve.append(portfolio_value)

        # Optional rebalancing
        if self.parameters.get("rebalance_frequency") == "daily":
            self._rebalance_portfolio(market_data)

    def calculate_position_size(self, signal: Signal, current_price: float) -> float:
        """
        Calculate position size based on signal and risk parameters.

        Args:
            signal: Trading signal
            current_price: Current price of the asset

        Returns:
            Position size (number of shares)
        """
        # Kelly criterion or fixed fractional sizing
        max_position_value = self.current_capital * self.parameters["max_position_size"]

        # Adjust for signal strength
        position_value = max_position_value * signal.strength

        # Convert to shares
        shares = int(position_value / current_price)

        return shares

    def check_risk_limits(self) -> bool:
        """Check if strategy is within risk limits."""
        # Check maximum positions
        if len(self.positions) >= self.parameters["max_positions"]:
            return False

        # Check drawdown
        if self.equity_curve:
            peak = max(self.equity_curve)
            current = self.equity_curve[-1]
            drawdown = (peak - current) / peak
            if drawdown > 0.2:  # 20% max drawdown
                return False

        return True

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price."""
        stop_loss_pct = self.parameters["stop_loss_pct"]

        if side == "long":
            return entry_price * (1 - stop_loss_pct)
        else:
            return entry_price * (1 + stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price."""
        take_profit_pct = self.parameters["take_profit_pct"]

        if side == "long":
            return entry_price * (1 + take_profit_pct)
        else:
            return entry_price * (1 - take_profit_pct)

    def track_performance(self) -> Dict[str, float]:
        """Calculate and return performance metrics."""
        if not self.daily_returns:
            return {}

        # Calculate metrics
        total_return = (
            (self.equity_curve[-1] / self.initial_capital - 1) if self.equity_curve else 0
        )

        # Win rate
        winning_trades = [t for t in self.trades if t["pnl"] > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        # Average win/loss
        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in self.trades if t["pnl"] <= 0]
        avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0

        # Sharpe ratio
        if len(self.daily_returns) > 1:
            daily_rf = 0.02 / 252  # 2% annual risk-free rate
            excess_returns = [r - daily_rf for r in self.daily_returns]
            sharpe = (
                np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
                if np.std(excess_returns) > 0
                else 0
            )
        else:
            sharpe = 0

        # Max drawdown
        if self.equity_curve:
            peak = self.equity_curve[0]
            max_dd = 0
            for value in self.equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0

        self.performance_metrics = {
            "total_return": total_return,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "current_capital": self.current_capital,
            "portfolio_value": self.equity_curve[-1] if self.equity_curve else self.initial_capital,
        }

        return self.performance_metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics."""
        stats = self.track_performance()

        # Add additional statistics
        stats.update(
            {
                "positions": len(self.positions),
                "position_symbols": list(self.positions.keys()),
                "avg_holding_period": self._calculate_avg_holding_period(),
                "best_trade": max(self.trades, key=lambda x: x["pnl"])["pnl"] if self.trades else 0,
                "worst_trade": (
                    min(self.trades, key=lambda x: x["pnl"])["pnl"] if self.trades else 0
                ),
                "consecutive_wins": self._calculate_consecutive_wins(),
                "consecutive_losses": self._calculate_consecutive_losses(),
            }
        )

        return stats

    def export_results(self, filepath: str = None) -> Dict[str, Any]:
        """Export backtesting results."""
        results = {
            "strategy_name": self.name,
            "parameters": self.parameters,
            "performance": self.get_statistics(),
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "daily_returns": self.daily_returns,
        }

        if filepath:
            # Standard library imports
            import json

            with open(filepath, "w") as f:
                json.dump(results, f, indent=2, default=str)

        return results

    # Helper methods
    def _update_positions(self, market_data: Dict[str, MarketData]):
        """Update position values with current prices."""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol].last
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = position.quantity * (
                    current_price - position.avg_entry_price
                )
                position.unrealized_pnl_pct = (current_price / position.avg_entry_price - 1) * 100

    def _check_exit_conditions(self, market_data: Dict[str, MarketData]):
        """Check stop loss and take profit conditions."""
        positions_to_close = []

        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue

            current_price = market_data[symbol].last

            # Check stop loss
            stop_price = self.calculate_stop_loss(position.avg_entry_price, position.side)
            if (position.side == "long" and current_price <= stop_price) or (
                position.side == "short" and current_price >= stop_price
            ):
                positions_to_close.append((symbol, "stop_loss"))

            # Check take profit
            profit_price = self.calculate_take_profit(position.avg_entry_price, position.side)
            if (position.side == "long" and current_price >= profit_price) or (
                position.side == "short" and current_price <= profit_price
            ):
                positions_to_close.append((symbol, "take_profit"))

        # Generate exit signals
        signals = []
        for symbol, reason in positions_to_close:
            position = self.positions[symbol]
            signal = Signal(
                symbol=symbol,
                action="SELL" if position.side == "long" else "BUY",
                strength=1.0,
                confidence=1.0,
                quantity=abs(position.quantity),
                source=self.name,
                reason=f"Exit: {reason}",
            )
            signals.append(signal)

        return signals

    def _apply_risk_management(self, signals: List[Signal]) -> List[Signal]:
        """Apply risk management rules to signals."""
        if not self.check_risk_limits():
            # Filter out new entry signals if at risk limits
            signals = [
                s
                for s in signals
                if s.action == "SELL"
                or (
                    s.symbol in self.positions
                    and s.action == "BUY"
                    and self.positions[s.symbol].side == "short"
                )
            ]

        # Validate position sizes
        validated_signals = []
        for signal in signals:
            if signal.symbol in self.positions:
                # Existing position - allow exit
                validated_signals.append(signal)
            else:
                # New position - check size
                if signal.quantity is None and signal.action in ["BUY", "SELL"]:
                    # Calculate position size
                    current_price = signal.price or 100  # Default if not provided
                    signal.quantity = self.calculate_position_size(signal, current_price)

                if signal.quantity and signal.quantity > 0:
                    validated_signals.append(signal)

        return validated_signals

    def _update_performance_metrics(self, timestamp: datetime, market_data: Dict[str, MarketData]):
        """Update real-time performance metrics."""
        portfolio_value = self.get_portfolio_value(market_data)

        # Update high water mark
        if not hasattr(self, "_high_water_mark"):
            self._high_water_mark = self.initial_capital

        if portfolio_value > self._high_water_mark:
            self._high_water_mark = portfolio_value

        # Current drawdown
        current_drawdown = (self._high_water_mark - portfolio_value) / self._high_water_mark

        # Update metrics
        self.performance_metrics["current_drawdown"] = current_drawdown
        self.performance_metrics["portfolio_value"] = portfolio_value

    def get_portfolio_value(self, market_data: Dict[str, MarketData]) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(position.market_value for position in self.positions.values())
        return self.current_capital + positions_value

    def _rebalance_portfolio(self, market_data: Dict[str, MarketData]):
        """Rebalance portfolio to target weights."""
        # Override in subclasses for specific rebalancing logic
        pass

    def _calculate_avg_holding_period(self) -> float:
        """Calculate average holding period in days."""
        if not self.trades:
            return 0

        holding_periods = []
        for trade in self.trades:
            period = (trade["exit_time"] - trade["entry_time"]).days
            holding_periods.append(period)

        return np.mean(holding_periods) if holding_periods else 0

    def _calculate_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning trades."""
        if not self.trades:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in self.trades:
            if trade["pnl"] > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades."""
        if not self.trades:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in self.trades:
            if trade["pnl"] <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive


# Alias for backward compatibility
StrategySignal = Signal


class SignalType(Enum):
    """Signal type enumeration."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

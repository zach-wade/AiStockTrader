"""
Paper Trading Broker - Thin adapter for simulated trading
"""

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from src.application.interfaces.broker import (
    AccountInfo,
    BrokerConnectionError,
    IBroker,
    MarketHours,
    OrderNotFoundError,
)
from src.domain.entities.order import Order, OrderStatus, OrderType
from src.domain.entities.position import Position
from src.domain.services.trading_calendar import Exchange, TradingCalendar

logger = logging.getLogger(__name__)


@dataclass
class PaperBrokerState:
    """Minimal state storage for paper trading."""

    initial_capital: Decimal = Decimal("100000")
    cash_balance: Decimal = Decimal("100000")
    orders: dict[UUID, Order] | None = None
    positions: dict[str, Position] | None = None
    market_prices: dict[str, Decimal] | None = None

    def __post_init__(self) -> None:
        if self.orders is None:
            self.orders = {}
        if self.positions is None:
            self.positions = {}
        if self.market_prices is None:
            self.market_prices = {}


class PaperBroker(IBroker):
    """
    Thin paper trading broker adapter with thread safety.

    This is a minimal implementation that only stores state.
    All business logic is delegated to use cases via BrokerCoordinator.
    Thread-safe for concurrent access.
    """

    def __init__(
        self, initial_capital: Decimal = Decimal("100000"), exchange: Exchange = Exchange.NYSE
    ) -> None:
        """Initialize paper broker with minimal state and thread safety."""
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self.state = PaperBrokerState(initial_capital=initial_capital, cash_balance=initial_capital)
        self._connected = False
        self.trading_calendar = TradingCalendar(exchange)
        logger.info(f"Initialized thread-safe paper broker with ${initial_capital} capital")

    def connect(self) -> None:
        """Establish connection (simulated)."""
        self._connected = True
        logger.info("Connected to paper trading broker")

    def disconnect(self) -> None:
        """Close connection (simulated)."""
        self._connected = False
        logger.info("Disconnected from paper trading broker")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def _check_connection(self) -> None:
        """Ensure broker is connected."""
        if not self._connected:
            raise BrokerConnectionError("Not connected to paper broker")

    def submit_order(self, order: Order) -> Order:
        """Store order submission - processing delegated to use cases. Thread-safe."""
        self._check_connection()
        with self._lock:
            broker_id = f"PAPER-{order.id}"
            order.submit(broker_id)
            assert self.state.orders is not None  # Always initialized in __post_init__
            self.state.orders[order.id] = order

            # Auto-fill market orders if we have a market price
            if order.order_type == OrderType.MARKET and self.state.market_prices:
                market_price = self.state.market_prices.get(order.symbol)
                if market_price:
                    order.fill(filled_quantity=order.quantity, fill_price=market_price)
                    logger.info(f"Auto-filled market order: {order.id} at {market_price}")

            logger.info(f"Stored paper order: {order.id}")
        return order

    def cancel_order(self, order_id: UUID) -> bool:
        """Mark order as cancelled - validation delegated to use cases. Thread-safe."""
        self._check_connection()
        with self._lock:
            assert self.state.orders is not None  # Always initialized in __post_init__
            if order_id not in self.state.orders:
                raise OrderNotFoundError(f"Order {order_id} not found")

            order = self.state.orders[order_id]
            if order.is_active():
                order.cancel("User requested")
                logger.info(f"Cancelled order {order_id}")
                return True
            return False

    def get_order(self, order_id: UUID) -> Order:
        """Get order from state. Thread-safe."""
        self._check_connection()
        with self._lock:
            assert self.state.orders is not None  # Always initialized in __post_init__
            if order_id not in self.state.orders:
                raise OrderNotFoundError(f"Order {order_id} not found")
            return self.state.orders[order_id]

    def get_order_status(self, order_id: UUID) -> OrderStatus:
        """Get order status from state. Thread-safe."""
        self._check_connection()
        with self._lock:
            assert self.state.orders is not None  # Always initialized in __post_init__
            if order_id not in self.state.orders:
                raise OrderNotFoundError(f"Order {order_id} not found")
            return self.state.orders[order_id].status

    def update_order(self, order: Order) -> Order:
        """Update order in state. Thread-safe."""
        self._check_connection()
        with self._lock:
            assert self.state.orders is not None  # Always initialized in __post_init__
            if order.id not in self.state.orders:
                raise OrderNotFoundError(f"Order {order.id} not found")
            self.state.orders[order.id] = order
        return order

    def get_recent_orders(self, limit: int = 100) -> list[Order]:
        """Get recent orders from state. Thread-safe."""
        self._check_connection()
        with self._lock:
            assert self.state.orders is not None  # Always initialized in __post_init__
            orders = list(self.state.orders.values())
        orders.sort(key=lambda o: o.created_at, reverse=True)
        return orders[:limit]

    def get_positions(self) -> list[Position]:
        """Get positions from state. Thread-safe."""
        self._check_connection()
        with self._lock:
            assert self.state.positions is not None  # Always initialized in __post_init__
            return list(self.state.positions.values())

    def get_account_info(self) -> AccountInfo:
        """Get account info from state. Thread-safe."""
        self._check_connection()

        with self._lock:
            assert self.state.positions is not None  # Always initialized in __post_init__
            positions_value = (
                Decimal(
                    sum(
                        p.get_position_value() or Decimal("0")
                        for p in self.state.positions.values()
                    )
                )
                if self.state.positions
                else Decimal("0")
            )

            equity = self.state.cash_balance + positions_value
            cash_balance = self.state.cash_balance

        return AccountInfo(
            account_id="PAPER-001",
            account_type="paper",
            equity=equity,
            cash=cash_balance,
            buying_power=cash_balance,
            positions_value=positions_value,
            unrealized_pnl=Decimal("0"),  # Calculated by use cases
            realized_pnl=Decimal("0"),  # Calculated by use cases
            margin_used=Decimal("0"),
            margin_available=Decimal("0"),
            pattern_day_trader=False,
            trades_today=0,
            trades_remaining=None,
            last_updated=datetime.now(UTC),
        )

    def is_market_open(self) -> bool:
        """Check if market is currently open using trading calendar."""
        return self.trading_calendar.is_market_open()

    def get_market_hours(self) -> MarketHours:
        """Get market hours from trading calendar."""
        is_open = self.trading_calendar.is_market_open()
        next_open = self.trading_calendar.next_market_open()
        next_close = self.trading_calendar.next_market_close()

        return MarketHours(
            is_open=is_open,
            next_open=next_open,
            next_close=next_close,
        )

    def update_market_price(self, symbol: str, price: Decimal) -> None:
        """Update market price for a symbol. Thread-safe."""
        with self._lock:
            if self.state.market_prices is None:
                self.state.market_prices = {}
            self.state.market_prices[symbol] = price

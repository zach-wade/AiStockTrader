"""
Alpaca Broker - Thin adapter for Alpaca Trading API
"""

import logging
import os
import threading
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from alpaca.common.exceptions import APIError
from alpaca.trading import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
from alpaca.trading import OrderSide as AlpacaOrderSide
from alpaca.trading import OrderStatus as AlpacaOrderStatus
from alpaca.trading import OrderType as AlpacaOrderType
from alpaca.trading import QueryOrderStatus
from alpaca.trading import TimeInForce as AlpacaTimeInForce
from alpaca.trading import TradingClient
from alpaca.trading.models import Order as AlpacaOrder

from src.application.interfaces.broker import (
    AccountInfo,
    BrokerConnectionError,
    BrokerError,
    IBroker,
    InvalidCredentialsError,
    MarketHours,
    OrderNotFoundError,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from src.domain.entities.position import Position

logger = logging.getLogger(__name__)


class AlpacaBroker(IBroker):
    """
    Thin Alpaca broker adapter with thread safety.

    This is a minimal implementation that only handles API communication.
    All business logic is delegated to use cases via BrokerCoordinator.
    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ):
        """Initialize Alpaca API client with thread safety."""
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        # Strip whitespace and check if credentials are valid
        if self.api_key:
            self.api_key = self.api_key.strip()
        if self.secret_key:
            self.secret_key = self.secret_key.strip()

        if not self.api_key or not self.secret_key:
            raise InvalidCredentialsError("Alpaca API credentials not provided")

        self.paper = paper
        self.client: TradingClient | None = None
        self._order_map: dict[UUID, str] = {}  # Internal UUID to Alpaca ID
        self._order_map_lock = threading.Lock()  # Lock for thread-safe order map access
        self._connected = False

        logger.info(f"Initialized thread-safe Alpaca broker ({'paper' if paper else 'live'})")

    def connect(self) -> None:
        """Connect to Alpaca API."""
        self.client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )
        self._connected = True
        logger.info("Connected to Alpaca API")

    def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        self.client = None
        self._connected = False
        logger.info("Disconnected from Alpaca API")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self.client is not None

    def _check_connection(self) -> None:
        """Ensure broker is connected."""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Alpaca")

    def submit_order(self, order: Order) -> Order:
        """Submit order to Alpaca - validation delegated to use cases."""
        self._check_connection()

        # Map time in force from domain to Alpaca
        time_in_force_map = {
            TimeInForce.DAY: AlpacaTimeInForce.DAY,
            TimeInForce.GTC: AlpacaTimeInForce.GTC,
            TimeInForce.IOC: AlpacaTimeInForce.IOC,
            TimeInForce.FOK: AlpacaTimeInForce.FOK,
        }
        alpaca_tif = time_in_force_map.get(order.time_in_force, AlpacaTimeInForce.DAY)

        # Create Alpaca order request
        # Create order request based on type
        order_request: MarketOrderRequest | LimitOrderRequest
        if order.order_type == OrderType.MARKET:
            order_request = MarketOrderRequest(
                symbol=order.symbol,
                qty=float(order.quantity),
                side=AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL,
                time_in_force=alpaca_tif,
            )
        else:
            order_request = LimitOrderRequest(
                symbol=order.symbol,
                qty=float(order.quantity),
                side=AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL,
                time_in_force=alpaca_tif,
                limit_price=float(order.limit_price) if order.limit_price else None,
            )

        # Submit to Alpaca
        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")

        try:
            alpaca_order = self.client.submit_order(order_data=order_request)
        except APIError as e:
            raise BrokerError(f"Failed to submit order: {e!s}") from e

        # Thread-safe order map update
        alpaca_id = str(alpaca_order.id if hasattr(alpaca_order, "id") else alpaca_order)
        with self._order_map_lock:
            self._order_map[order.id] = alpaca_id

        # Update order with broker ID
        order.submit(alpaca_id)
        logger.info(f"Submitted order {order.id} to Alpaca as {alpaca_id}")
        return order

    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel order on Alpaca. Thread-safe."""
        self._check_connection()

        with self._order_map_lock:
            alpaca_id = self._order_map.get(order_id)
        if not alpaca_id:
            raise OrderNotFoundError(f"Order {order_id} not found")

        try:
            if self.client is None:
                raise RuntimeError("Alpaca client not initialized")
            self.client.cancel_order_by_id(alpaca_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        except APIError:
            return False

    def get_order(self, order_id: UUID) -> Order:
        """Get order from Alpaca. Thread-safe."""
        self._check_connection()

        with self._order_map_lock:
            alpaca_id = self._order_map.get(order_id)
        if not alpaca_id:
            raise OrderNotFoundError(f"Order {order_id} not found")

        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")

        try:
            alpaca_order = self.client.get_order_by_id(alpaca_id)
            return self._map_alpaca_to_domain_order(alpaca_order)
        except APIError as e:
            if "not found" in str(e).lower():
                raise OrderNotFoundError(f"Order {order_id} not found") from e
            raise BrokerError(f"Failed to get order: {e!s}") from e

    def get_order_status(self, order_id: UUID) -> OrderStatus:
        """Get order status from Alpaca. Thread-safe."""
        self._check_connection()

        with self._order_map_lock:
            alpaca_id = self._order_map.get(order_id)
        if not alpaca_id:
            raise OrderNotFoundError(f"Order {order_id} not found")

        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")
        alpaca_order = self.client.get_order_by_id(alpaca_id)
        status = alpaca_order.status if hasattr(alpaca_order, "status") else str(alpaca_order)
        return self._map_status(status)  # type: ignore[arg-type]

    def update_order(self, order: Order) -> Order:
        """Update order from Alpaca status. Thread-safe."""
        self._check_connection()

        with self._order_map_lock:
            alpaca_id = self._order_map.get(order.id)
        if not alpaca_id:
            return order

        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")
        alpaca_order = self.client.get_order_by_id(alpaca_id)
        status = alpaca_order.status if hasattr(alpaca_order, "status") else str(alpaca_order)
        order.status = self._map_status(status)  # type: ignore[arg-type]
        return order

    def get_recent_orders(self, limit: int = 100) -> list[Order]:
        """Get recent orders from Alpaca. Simple mapping from Alpaca to domain orders."""
        self._check_connection()

        # Get recent orders from Alpaca
        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        alpaca_orders = self.client.get_orders(filter=request)

        # Simple mapping to domain orders
        domain_orders = []
        for alpaca_order in alpaca_orders:
            # Map Alpaca order to domain order
            if isinstance(alpaca_order, AlpacaOrder):
                order = self._map_alpaca_to_domain_order(alpaca_order)
                domain_orders.append(order)

        return domain_orders

    def _map_alpaca_to_domain_order(self, alpaca_order: AlpacaOrder) -> Order:
        """Simple direct mapping from Alpaca order to domain order."""
        # Direct field mapping
        order = Order(
            symbol=alpaca_order.symbol if alpaca_order.symbol else "",
            side=self._map_side(alpaca_order.side) if alpaca_order.side else OrderSide.BUY,
            quantity=Decimal(str(alpaca_order.qty)) if alpaca_order.qty else Decimal("0"),
            order_type=(
                self._map_order_type(alpaca_order.order_type)
                if alpaca_order.order_type
                else OrderType.MARKET
            ),
            limit_price=self._safe_decimal(alpaca_order.limit_price),
            stop_price=self._safe_decimal(alpaca_order.stop_price),
        )

        # Direct assignment
        order.status = (
            self._map_status(alpaca_order.status)
            if hasattr(alpaca_order, "status")
            else OrderStatus.PENDING
        )
        order.broker_order_id = str(alpaca_order.id) if hasattr(alpaca_order, "id") else None
        order.created_at = getattr(alpaca_order, "created_at", datetime.now(UTC))
        order.filled_at = getattr(alpaca_order, "filled_at", None)
        filled_qty = self._safe_decimal(getattr(alpaca_order, "filled_qty", None))
        if filled_qty is not None:
            order.filled_quantity = filled_qty
        avg_price = self._safe_decimal(getattr(alpaca_order, "filled_avg_price", None))
        if avg_price is not None:
            order.average_fill_price = avg_price

        return order

    def _map_side(self, alpaca_side: AlpacaOrderSide) -> OrderSide:
        """Simple side mapper."""
        return OrderSide.BUY if alpaca_side == AlpacaOrderSide.BUY else OrderSide.SELL

    def _map_order_type(self, alpaca_type: AlpacaOrderType) -> OrderType:
        """Simple order type mapper."""
        return OrderType.LIMIT if alpaca_type == AlpacaOrderType.LIMIT else OrderType.MARKET

    def _safe_decimal(self, value: Any) -> Decimal | None:
        """Safe conversion to Decimal."""
        return Decimal(str(value)) if value is not None else None

    def get_position(self, symbol: str) -> Position | None:
        """Get position by symbol from Alpaca."""
        self._check_connection()

        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")

        try:
            alpaca_pos = self.client.get_position(symbol)

            # Check for zero quantity positions
            qty = Decimal(str(getattr(alpaca_pos, "qty", 0)))
            if qty == 0:
                return None

            # Create domain position from Alpaca position
            position = Position(
                symbol=getattr(alpaca_pos, "symbol", ""),
                quantity=qty,
                average_entry_price=Decimal(str(getattr(alpaca_pos, "avg_entry_price", 0))),
                current_price=(
                    Decimal(str(getattr(alpaca_pos, "current_price", 0)))
                    if hasattr(alpaca_pos, "current_price")
                    else None
                ),
            )

            # Set P&L if available
            if hasattr(alpaca_pos, "realized_pl") and alpaca_pos.realized_pl is not None:
                position.realized_pnl = Decimal(str(alpaca_pos.realized_pl))

            return position
        except APIError:
            return None

    def close_position(self, symbol: str) -> bool:
        """Close a position by symbol."""
        self._check_connection()

        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")

        try:
            self.client.close_position(symbol)
            return True
        except APIError:
            return False

    def close_all_positions(self) -> bool:
        """Close all positions."""
        self._check_connection()

        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")

        try:
            self.client.close_all_positions()
            return True
        except APIError:
            return False

    def get_positions(self) -> list[Position]:
        """Get positions from Alpaca. Maps Alpaca positions to domain positions."""
        self._check_connection()

        # Get positions from Alpaca
        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")
        alpaca_positions = self.client.get_all_positions()

        # Map to domain positions
        domain_positions = []
        for alpaca_pos in alpaca_positions:
            # Create domain position from Alpaca position
            if not hasattr(alpaca_pos, "symbol"):
                continue
            # Get current price safely
            current_price_value = getattr(alpaca_pos, "current_price", None)
            current_price = (
                Decimal(str(current_price_value)) if current_price_value is not None else None
            )

            position = Position(
                symbol=getattr(alpaca_pos, "symbol", ""),
                quantity=Decimal(str(getattr(alpaca_pos, "qty", 0))),
                average_entry_price=Decimal(str(getattr(alpaca_pos, "avg_entry_price", 0))),
                current_price=current_price,
            )

            # Set P&L if available
            if hasattr(alpaca_pos, "realized_pl") and alpaca_pos.realized_pl is not None:
                position.realized_pnl = Decimal(str(alpaca_pos.realized_pl))

            domain_positions.append(position)

        return domain_positions

    def get_account_info(self) -> AccountInfo:
        """Get account info from Alpaca."""
        self._check_connection()

        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")

        try:
            account = self.client.get_account()
        except APIError as e:
            raise BrokerConnectionError(f"Failed to get account info: {e!s}") from e

        return AccountInfo(
            account_id=getattr(account, "account_number", ""),
            account_type="paper" if self.paper else "live",
            equity=Decimal(str(getattr(account, "equity", 0))),
            cash=Decimal(str(getattr(account, "cash", 0))),
            buying_power=Decimal(str(getattr(account, "buying_power", 0))),
            positions_value=Decimal(str(getattr(account, "long_market_value", 0))),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            margin_used=Decimal("0"),
            margin_available=Decimal(str(getattr(account, "buying_power", 0))),
            pattern_day_trader=bool(getattr(account, "pattern_day_trader", False)),
            trades_today=0,
            trades_remaining=None,
            last_updated=datetime.now(UTC),
        )

    def is_market_open(self) -> bool:
        """Check if market is open."""
        self._check_connection()
        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")
        clock = self.client.get_clock()
        return bool(getattr(clock, "is_open", False))

    def get_market_hours(self, date: datetime | None = None) -> MarketHours:
        """Get market hours from Alpaca for a specific date or today."""
        self._check_connection()
        if self.client is None:
            raise RuntimeError("Alpaca client not initialized")

        try:
            if date is None:
                # Get current market hours
                clock = self.client.get_clock()
                return MarketHours(
                    is_open=bool(getattr(clock, "is_open", False)),
                    next_open=getattr(clock, "next_open", None),
                    next_close=getattr(clock, "next_close", None),
                )
            else:
                # Get calendar for specific date
                from alpaca.trading import GetCalendarRequest

                # Convert datetime to date for the calendar request
                date_only = date.date()
                request = GetCalendarRequest(start=date_only, end=date_only)
                calendar = self.client.get_calendar(filters=request)

                if calendar:
                    cal_day = calendar[0]
                    return MarketHours(
                        is_open=True,
                        next_open=getattr(cal_day, "open", None),
                        next_close=getattr(cal_day, "close", None),
                    )
                else:
                    # Market is closed on this date
                    return MarketHours(
                        is_open=False,
                        next_open=None,
                        next_close=None,
                    )
        except APIError as e:
            raise BrokerConnectionError(f"Failed to get market hours: {e!s}") from e

    def _map_status(self, alpaca_status: AlpacaOrderStatus) -> OrderStatus:
        """Map Alpaca status to domain status."""
        mapping = {
            AlpacaOrderStatus.NEW: OrderStatus.SUBMITTED,
            AlpacaOrderStatus.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
            AlpacaOrderStatus.FILLED: OrderStatus.FILLED,
            AlpacaOrderStatus.CANCELED: OrderStatus.CANCELLED,
            AlpacaOrderStatus.EXPIRED: OrderStatus.EXPIRED,
            AlpacaOrderStatus.REJECTED: OrderStatus.REJECTED,
        }
        return mapping.get(alpaca_status, OrderStatus.PENDING)

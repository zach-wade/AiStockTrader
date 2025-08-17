"""
Alpaca Broker Implementation - Integration with Alpaca Trading API
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import logging
import os
from typing import Any
from uuid import UUID

# Third-party imports
from alpaca.common.exceptions import APIError
from alpaca.trading import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
from alpaca.trading import OrderSide as AlpacaOrderSide
from alpaca.trading import OrderStatus as AlpacaOrderStatus
from alpaca.trading import OrderType as AlpacaOrderType
from alpaca.trading import PositionSide, StopLimitOrderRequest, StopOrderRequest
from alpaca.trading import TimeInForce as AlpacaTimeInForce
from alpaca.trading import TradingClient
from alpaca.trading.models import Order as AlpacaOrder

# Local imports
from src.application.interfaces.broker import (
    AccountInfo,
    BrokerConnectionError,
    InsufficientFundsError,
    InvalidCredentialsError,
    InvalidOrderError,
    MarketClosedError,
    MarketHours,
    OrderNotFoundError,
    RateLimitError,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from src.domain.entities.position import Position

from .constants import HTTP_NOT_FOUND, HTTP_UNAUTHORIZED

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Alpaca broker implementation for both paper and live trading.

    Uses the alpaca-py SDK for API integration.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        base_url: str | None = None,
    ):
        """
        Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key (defaults to ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (defaults to ALPACA_SECRET_KEY env var)
            paper: Whether to use paper trading (defaults to True)
            base_url: Optional base URL override
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise InvalidCredentialsError(
                "Alpaca API credentials not provided. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        self.paper = paper
        self.base_url = base_url
        self.client: TradingClient | None = None
        self._order_map: dict[UUID, str] = {}  # Maps internal UUID to Alpaca order ID
        self._connected = False

        # Rate limiting tracking
        self._last_request_time = datetime.now(UTC)
        self._request_count = 0
        self._rate_limit_window = timedelta(minutes=1)
        self._max_requests_per_minute = 200  # Alpaca's rate limit

        logger.info(f"Initialized Alpaca broker (paper={paper})")

    def connect(self) -> None:
        """Establish connection to Alpaca"""
        try:
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper,
                url_override=self.base_url,
            )

            # Test connection by getting account info
            account = self.client.get_account()
            self._connected = True

            logger.info(
                f"Connected to Alpaca ({'paper' if self.paper else 'live'} trading), "
                f"Account: {account.account_number}"
            )

        except APIError as e:
            if e.code == HTTP_UNAUTHORIZED:
                raise InvalidCredentialsError(f"Invalid Alpaca credentials: {e}") from e
            else:
                raise BrokerConnectionError(f"Failed to connect to Alpaca: {e}") from e
        except Exception as e:
            raise BrokerConnectionError(f"Failed to connect to Alpaca: {e}") from e

    def disconnect(self) -> None:
        """Close connection to Alpaca"""
        self.client = None
        self._connected = False
        logger.info("Disconnected from Alpaca")

    def is_connected(self) -> bool:
        """Check if connected to Alpaca"""
        return self._connected and self.client is not None

    def _check_connection(self) -> None:
        """Ensure broker is connected"""
        if not self.is_connected():
            raise BrokerConnectionError("Not connected to Alpaca. Call connect() first.")

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting"""
        now = datetime.now(UTC)

        # Reset counter if window has passed
        if now - self._last_request_time > self._rate_limit_window:
            self._request_count = 0
            self._last_request_time = now

        # Check if limit exceeded
        if self._request_count >= self._max_requests_per_minute:
            wait_time = self._rate_limit_window - (now - self._last_request_time)
            raise RateLimitError(
                f"Rate limit exceeded. Please wait {wait_time.total_seconds():.1f} seconds."
            )

        self._request_count += 1

    def _convert_order_side(self, side: OrderSide) -> AlpacaOrderSide:
        """Convert domain OrderSide to Alpaca OrderSide"""
        if side == OrderSide.BUY:
            return AlpacaOrderSide.BUY
        else:
            return AlpacaOrderSide.SELL

    def _convert_time_in_force(self, tif: TimeInForce) -> AlpacaTimeInForce:
        """Convert domain TimeInForce to Alpaca TimeInForce"""
        mapping = {
            TimeInForce.DAY: AlpacaTimeInForce.DAY,
            TimeInForce.GTC: AlpacaTimeInForce.GTC,
            TimeInForce.IOC: AlpacaTimeInForce.IOC,
            TimeInForce.FOK: AlpacaTimeInForce.FOK,
        }
        return mapping[tif]

    def _convert_order_status(self, status: AlpacaOrderStatus) -> OrderStatus:
        """Convert Alpaca OrderStatus to domain OrderStatus"""
        mapping = {
            AlpacaOrderStatus.PENDING_NEW: OrderStatus.PENDING,
            AlpacaOrderStatus.NEW: OrderStatus.SUBMITTED,
            AlpacaOrderStatus.ACCEPTED: OrderStatus.SUBMITTED,
            AlpacaOrderStatus.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
            AlpacaOrderStatus.FILLED: OrderStatus.FILLED,
            AlpacaOrderStatus.CANCELED: OrderStatus.CANCELLED,
            AlpacaOrderStatus.REJECTED: OrderStatus.REJECTED,
            AlpacaOrderStatus.EXPIRED: OrderStatus.EXPIRED,
            AlpacaOrderStatus.PENDING_CANCEL: OrderStatus.SUBMITTED,
            AlpacaOrderStatus.PENDING_REPLACE: OrderStatus.SUBMITTED,
        }
        return mapping.get(status, OrderStatus.PENDING)

    def _create_order_request(self, order: Order) -> Any:
        """Create Alpaca order request from domain order"""
        side = self._convert_order_side(order.side)
        tif = self._convert_time_in_force(order.time_in_force)

        # Convert quantity to float for Alpaca
        qty = float(order.quantity)

        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(
                symbol=order.symbol,
                qty=qty,
                side=side,
                time_in_force=tif,
            )

        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise InvalidOrderError("Limit order requires limit_price")

            return LimitOrderRequest(
                symbol=order.symbol,
                qty=qty,
                side=side,
                time_in_force=tif,
                limit_price=float(order.limit_price),
            )

        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                raise InvalidOrderError("Stop order requires stop_price")

            return StopOrderRequest(
                symbol=order.symbol,
                qty=qty,
                side=side,
                time_in_force=tif,
                stop_price=float(order.stop_price),
            )

        elif order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                raise InvalidOrderError("Stop-limit order requires both stop_price and limit_price")

            return StopLimitOrderRequest(
                symbol=order.symbol,
                qty=qty,
                side=side,
                time_in_force=tif,
                stop_price=float(order.stop_price),
                limit_price=float(order.limit_price),
            )

        else:
            raise InvalidOrderError(f"Unsupported order type: {order.order_type}")

    def _update_order_from_alpaca(self, order: Order, alpaca_order: AlpacaOrder) -> Order:
        """Update domain order with data from Alpaca order"""
        order.broker_order_id = alpaca_order.id
        order.status = self._convert_order_status(alpaca_order.status)

        if alpaca_order.filled_qty:
            order.filled_quantity = Decimal(str(alpaca_order.filled_qty))

        if alpaca_order.filled_avg_price:
            order.average_fill_price = Decimal(str(alpaca_order.filled_avg_price))

        if alpaca_order.submitted_at:
            order.submitted_at = alpaca_order.submitted_at

        if alpaca_order.filled_at:
            order.filled_at = alpaca_order.filled_at

        if alpaca_order.canceled_at:
            order.cancelled_at = alpaca_order.canceled_at

        return order

    def submit_order(self, order: Order) -> Order:
        """Submit an order to Alpaca"""
        self._check_connection()
        self._check_rate_limit()

        try:
            # Check if market is open for market orders
            if order.order_type == OrderType.MARKET and not self.is_market_open():
                raise MarketClosedError("Cannot submit market order while market is closed")

            # Create and submit order
            order_request = self._create_order_request(order)
            alpaca_order = self.client.submit_order(order_request)

            # Update order with broker information
            order.submit(alpaca_order.id)
            self._order_map[order.id] = alpaca_order.id

            # Update with any immediate fills
            order = self._update_order_from_alpaca(order, alpaca_order)

            logger.info(f"Submitted order {order.id} to Alpaca: {alpaca_order.id}")
            return order

        except APIError as e:
            if "insufficient" in str(e).lower():
                raise InsufficientFundsError(f"Insufficient funds for order: {e}") from e
            elif "invalid" in str(e).lower():
                raise InvalidOrderError(f"Invalid order: {e}") from e
            else:
                raise BrokerConnectionError(f"Failed to submit order: {e}") from e
        except Exception as e:
            raise BrokerConnectionError(f"Failed to submit order: {e}") from e

    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel an order at Alpaca"""
        self._check_connection()
        self._check_rate_limit()

        try:
            # Get Alpaca order ID
            alpaca_order_id = self._order_map.get(order_id)
            if not alpaca_order_id:
                # Try to find by searching recent orders
                orders = self.get_recent_orders(limit=100)
                for o in orders:
                    if o.id == order_id:
                        alpaca_order_id = o.broker_order_id
                        break

            if not alpaca_order_id:
                raise OrderNotFoundError(f"Order {order_id} not found")

            # Cancel the order
            self.client.cancel_order_by_id(alpaca_order_id)
            logger.info(f"Cancelled order {order_id} (Alpaca: {alpaca_order_id})")
            return True

        except APIError as e:
            if e.code == HTTP_NOT_FOUND:
                raise OrderNotFoundError(f"Order {order_id} not found at Alpaca") from e
            else:
                logger.error(f"Failed to cancel order {order_id}: {e}")
                return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: UUID) -> OrderStatus:
        """Get current status of an order"""
        self._check_connection()
        self._check_rate_limit()

        try:
            # Get Alpaca order ID
            alpaca_order_id = self._order_map.get(order_id)
            if not alpaca_order_id:
                raise OrderNotFoundError(f"Order {order_id} not found")

            # Get order from Alpaca
            alpaca_order = self.client.get_order_by_id(alpaca_order_id)
            return self._convert_order_status(alpaca_order.status)

        except APIError as e:
            if e.code == HTTP_NOT_FOUND:
                raise OrderNotFoundError(f"Order {order_id} not found at Alpaca") from e
            else:
                raise BrokerConnectionError(f"Failed to get order status: {e}") from e
        except Exception as e:
            raise BrokerConnectionError(f"Failed to get order status: {e}") from e

    def update_order(self, order: Order) -> Order:
        """Update order with latest status from Alpaca"""
        self._check_connection()
        self._check_rate_limit()

        try:
            alpaca_order_id = order.broker_order_id or self._order_map.get(order.id)
            if not alpaca_order_id:
                raise OrderNotFoundError(f"Order {order.id} not found")

            # Get latest order data from Alpaca
            alpaca_order = self.client.get_order_by_id(alpaca_order_id)

            # Update order
            return self._update_order_from_alpaca(order, alpaca_order)

        except APIError as e:
            if e.code == HTTP_NOT_FOUND:
                raise OrderNotFoundError(f"Order {order.id} not found at Alpaca") from e
            else:
                raise BrokerConnectionError(f"Failed to update order: {e}") from e
        except Exception as e:
            raise BrokerConnectionError(f"Failed to update order: {e}") from e

    def get_recent_orders(self, limit: int = 100) -> list[Order]:
        """Get recent orders from Alpaca"""
        self._check_connection()
        self._check_rate_limit()

        try:
            # Get orders from Alpaca
            request = GetOrdersRequest(
                status="all",
                limit=limit,
            )
            alpaca_orders = self.client.get_orders(filter=request)

            # Convert to domain orders
            orders = []
            for alpaca_order in alpaca_orders:
                # Create basic order
                order = Order(
                    symbol=alpaca_order.symbol,
                    quantity=Decimal(str(alpaca_order.qty)),
                    side=(
                        OrderSide.BUY
                        if alpaca_order.side == AlpacaOrderSide.BUY
                        else OrderSide.SELL
                    ),
                    order_type=self._get_order_type(alpaca_order),
                )

                # Update with Alpaca data
                order = self._update_order_from_alpaca(order, alpaca_order)
                orders.append(order)

            return orders

        except APIError as e:
            raise BrokerConnectionError(f"Failed to get recent orders: {e}") from e
        except Exception as e:
            raise BrokerConnectionError(f"Failed to get recent orders: {e}") from e

    def _get_order_type(self, alpaca_order: AlpacaOrder) -> OrderType:
        """Determine order type from Alpaca order"""
        if alpaca_order.order_type == AlpacaOrderType.MARKET:
            return OrderType.MARKET
        elif alpaca_order.order_type == AlpacaOrderType.LIMIT:
            return OrderType.LIMIT
        elif alpaca_order.order_type == AlpacaOrderType.STOP:
            return OrderType.STOP
        elif alpaca_order.order_type == AlpacaOrderType.STOP_LIMIT:
            return OrderType.STOP_LIMIT
        else:
            return OrderType.MARKET

    def get_positions(self) -> list[Position]:
        """Get all current positions from Alpaca"""
        self._check_connection()
        self._check_rate_limit()

        try:
            alpaca_positions = self.client.get_all_positions()

            positions = []
            for alpaca_pos in alpaca_positions:
                # Determine if long or short
                qty = Decimal(str(alpaca_pos.qty))
                if alpaca_pos.side == PositionSide.SHORT:
                    qty = -qty

                position = Position(
                    symbol=alpaca_pos.symbol,
                    quantity=qty,
                    average_entry_price=Decimal(str(alpaca_pos.avg_entry_price)),
                    current_price=(
                        Decimal(str(alpaca_pos.current_price)) if alpaca_pos.current_price else None
                    ),
                    realized_pnl=Decimal("0"),  # Alpaca doesn't track this per position
                )

                # Update market value if available
                if alpaca_pos.current_price:
                    position.update_market_price(Decimal(str(alpaca_pos.current_price)))

                positions.append(position)

            return positions

        except APIError as e:
            raise BrokerConnectionError(f"Failed to get positions: {e}") from e
        except Exception as e:
            raise BrokerConnectionError(f"Failed to get positions: {e}") from e

    def get_account_info(self) -> AccountInfo:
        """Get account information from Alpaca"""
        self._check_connection()
        self._check_rate_limit()

        try:
            account = self.client.get_account()

            return AccountInfo(
                account_id=account.account_number,
                account_type="paper" if self.paper else "live",
                equity=Decimal(str(account.equity)),
                cash=Decimal(str(account.cash)),
                buying_power=Decimal(str(account.buying_power)),
                positions_value=Decimal(str(account.long_market_value))
                + Decimal(str(account.short_market_value)),
                unrealized_pnl=Decimal("0"),  # Calculate from positions if needed
                realized_pnl=Decimal("0"),  # Not directly available from Alpaca
                margin_used=(
                    Decimal(str(account.initial_margin)) if account.initial_margin else None
                ),
                margin_available=(
                    Decimal(str(account.regt_buying_power)) if account.regt_buying_power else None
                ),
                pattern_day_trader=account.pattern_day_trader,
                trades_today=account.daytrade_count if account.daytrade_count else 0,
                trades_remaining=None,  # Calculate based on PDT rules if needed
                last_updated=datetime.now(UTC),
            )

        except APIError as e:
            raise BrokerConnectionError(f"Failed to get account info: {e}") from e
        except Exception as e:
            raise BrokerConnectionError(f"Failed to get account info: {e}") from e

    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        self._check_connection()
        self._check_rate_limit()

        try:
            clock = self.client.get_clock()
            return clock.is_open

        except APIError as e:
            raise BrokerConnectionError(f"Failed to check market status: {e}") from e
        except Exception as e:
            raise BrokerConnectionError(f"Failed to check market status: {e}") from e

    def get_market_hours(self) -> MarketHours:
        """Get detailed market hours information"""
        self._check_connection()
        self._check_rate_limit()

        try:
            clock = self.client.get_clock()

            return MarketHours(
                is_open=clock.is_open,
                next_open=clock.next_open if clock.next_open else None,
                next_close=clock.next_close if clock.next_close else None,
            )

        except APIError as e:
            raise BrokerConnectionError(f"Failed to get market hours: {e}") from e
        except Exception as e:
            raise BrokerConnectionError(f"Failed to get market hours: {e}") from e

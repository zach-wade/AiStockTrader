"""
Alpaca broker implementation.

This module provides integration with the Alpaca trading API, implementing
the BrokerInterface for live and paper trading.
"""

# Standard library imports
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
import logging
from typing import Any

# Third-party imports
from alpaca.data.live import StockDataStream
from alpaca.data.models import Quote
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import TimeInForce as AlpacaTIF
from alpaca.trading.models import Order as AlpacaOrder
from alpaca.trading.models import Position as AlpacaPosition
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
)
from alpaca.trading.stream import TradingStream
import httpx

# Local imports
from main.models.common import (
    AccountInfo,
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TimeInForce,
)
from main.trading_engine.brokers.broker_interface import (
    BrokerConnectionError,
    BrokerInterface,
    InsufficientFundsError,
    OrderSubmissionError,
)
from main.utils.core import async_retry
from main.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class AlpacaBroker(BrokerInterface):
    """
    Alpaca broker implementation.

    Provides integration with Alpaca's trading API for both live and
    paper trading accounts.
    """

    def __init__(self, config: dict[str, Any], metrics_collector: MetricsCollector | None = None):
        """
        Initialize Alpaca broker.

        Args:
            config: Configuration with keys:
                - api_key: Alpaca API key
                - api_secret: Alpaca API secret
                - base_url: API base URL (optional, defaults to paper trading)
                - data_feed: Data feed type ('iex' or 'sip', defaults to 'iex')
            metrics_collector: Optional metrics collector
        """
        super().__init__(config, metrics_collector)

        self.api_key = config["api_key"]
        self.api_secret = config["api_secret"]
        self.base_url = config.get("base_url", "https://paper-api.alpaca.markets")
        self.data_feed = config.get("data_feed", "iex")

        # Initialize clients
        self.trading_client = TradingClient(
            api_key=self.api_key, secret_key=self.api_secret, url_override=self.base_url
        )

        self.trading_stream: TradingStream | None = None
        self.data_stream: StockDataStream | None = None
        self._stream_tasks: list[asyncio.Task] = []

    async def connect(self) -> None:
        """Connect to Alpaca."""
        try:
            # Test connection by getting account info
            account = self.trading_client.get_account()
            logger.info(f"Connected to Alpaca account: {account.id}")

            # Initialize streaming connections
            self.trading_stream = TradingStream(
                api_key=self.api_key,
                secret_key=self.api_secret,
                url_override=self.base_url.replace("https://", "wss://").replace("api", "stream"),
            )

            self.data_stream = StockDataStream(
                api_key=self.api_key, secret_key=self.api_secret, feed=self.data_feed
            )

            self._connected = True
            self._track_event("connected", {"account_id": account.id})

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise BrokerConnectionError(f"Alpaca connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self._connected = False

        # Cancel streaming tasks
        for task in self._stream_tasks:
            task.cancel()

        # Close streams
        if self.trading_stream:
            await self.trading_stream.close()

        if self.data_stream:
            await self.data_stream.close()

        logger.info("Disconnected from Alpaca")
        self._track_event("disconnected", {})

    async def is_connected(self) -> bool:
        """Check if connected to Alpaca."""
        return self._connected

    @async_retry(max_attempts=3, delay=1.0)
    async def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca."""
        try:
            # Create Alpaca order request based on order type
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=self._convert_side(order.side),
                    time_in_force=self._convert_tif(order.time_in_force),
                )
            elif order.order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=self._convert_side(order.side),
                    time_in_force=self._convert_tif(order.time_in_force),
                    limit_price=float(order.limit_price),
                )
            elif order.order_type == OrderType.STOP:
                request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=self._convert_side(order.side),
                    time_in_force=self._convert_tif(order.time_in_force),
                    stop_price=float(order.stop_price),
                )
            elif order.order_type == OrderType.STOP_LIMIT:
                request = StopLimitOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=self._convert_side(order.side),
                    time_in_force=self._convert_tif(order.time_in_force),
                    stop_price=float(order.stop_price),
                    limit_price=float(order.limit_price),
                )
            else:
                raise OrderSubmissionError(f"Unsupported order type: {order.order_type}")

            # Submit order
            alpaca_order = self.trading_client.submit_order(request)

            logger.info(f"Submitted order {alpaca_order.id} for {order.symbol}")
            self._track_event(
                "order_submitted",
                {
                    "order_id": alpaca_order.id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": float(order.quantity),
                },
            )

            return alpaca_order.id

        except Exception as e:
            if "insufficient" in str(e).lower():
                raise InsufficientFundsError(f"Insufficient funds for order: {e}")
            else:
                raise OrderSubmissionError(f"Failed to submit order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order {order_id}")
            self._track_event("order_cancelled", {"order_id": order_id})
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str) -> Order | None:
        """Get order from Alpaca."""
        try:
            alpaca_order = self.trading_client.get_order_by_id(order_id)
            return self._convert_order(alpaca_order)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def get_orders(
        self,
        status: OrderStatus | None = None,
        symbol: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Order]:
        """Get orders from Alpaca."""
        try:
            request = GetOrdersRequest(
                status=self._convert_status_filter(status) if status else None,
                symbols=[symbol] if symbol else None,
                after=start_time,
                until=end_time,
                limit=limit,
            )

            alpaca_orders = self.trading_client.get_orders(request)
            return [self._convert_order(order) for order in alpaca_orders]

        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    async def get_positions(self) -> list[Position]:
        """Get positions from Alpaca."""
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            return [self._convert_position(pos) for pos in alpaca_positions]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for symbol from Alpaca."""
        try:
            alpaca_position = self.trading_client.get_open_position(symbol)
            return self._convert_position(alpaca_position)
        except Exception as e:
            # Alpaca throws exception if no position exists
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None

    async def get_account_info(self) -> AccountInfo:
        """Get account info from Alpaca."""
        try:
            account = self.trading_client.get_account()

            return AccountInfo(
                account_id=account.id,
                cash=Decimal(str(account.cash)),
                buying_power=Decimal(str(account.buying_power)),
                portfolio_value=Decimal(str(account.portfolio_value)),
                positions_value=Decimal(str(account.long_market_value))
                + Decimal(str(account.short_market_value)),
                pattern_day_trader=account.pattern_day_trader,
                trading_blocked=account.trading_blocked,
                transfers_blocked=account.transfers_blocked,
                account_blocked=account.account_blocked,
                trade_suspended_by_user=account.trade_suspended_by_user,
                created_at=account.created_at,
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get market data from Alpaca."""
        try:
            # Get latest quote
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url.replace('api', 'data')}/v2/stocks/{symbol}/quotes/latest",
                    headers={
                        "APCA-API-KEY-ID": self.api_key,
                        "APCA-API-SECRET-KEY": self.api_secret,
                    },
                )
                response.raise_for_status()
                data = response.json()

            quote = data["quote"]

            return MarketData(
                symbol=symbol,
                bid=Decimal(str(quote["bp"])),
                ask=Decimal(str(quote["ap"])),
                bid_size=quote["bs"],
                ask_size=quote["as"],
                last=Decimal(str(quote["bp"] + quote["ap"])) / 2,  # Mid price
                volume=0,  # Not available in quote
                timestamp=datetime.fromisoformat(quote["t"].replace("Z", "+00:00")),
            )

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            raise

    async def stream_market_data(self, symbols: list[str]) -> AsyncIterator[MarketData]:
        """Stream market data from Alpaca."""
        if not self.data_stream:
            raise BrokerConnectionError("Data stream not initialized")

        # Create queue for market data
        data_queue: asyncio.Queue[MarketData] = asyncio.Queue()

        async def handle_quote(data: Quote):
            """Handle quote update."""
            market_data = MarketData(
                symbol=data.symbol,
                bid=Decimal(str(data.bid_price)),
                ask=Decimal(str(data.ask_price)),
                bid_size=data.bid_size,
                ask_size=data.ask_size,
                last=Decimal(str(data.bid_price + data.ask_price)) / 2,
                volume=0,
                timestamp=data.timestamp,
            )
            await data_queue.put(market_data)

        # Subscribe to quotes
        for symbol in symbols:
            self.data_stream.subscribe_quotes(handle_quote, symbol)

        # Start streaming
        stream_task = create_task_safely(self.data_stream.run())
        self._stream_tasks.append(stream_task)

        try:
            while True:
                data = await data_queue.get()
                yield data
        finally:
            # Unsubscribe
            for symbol in symbols:
                self.data_stream.unsubscribe_quotes(symbol)

    async def stream_order_updates(self) -> AsyncIterator[Order]:
        """Stream order updates from Alpaca."""
        if not self.trading_stream:
            raise BrokerConnectionError("Trading stream not initialized")

        # Create queue for order updates
        order_queue: asyncio.Queue[Order] = asyncio.Queue()

        async def handle_trade_update(data: Any):
            """Handle trade update."""
            if hasattr(data, "order"):
                order = self._convert_order(data.order)
                await order_queue.put(order)

        # Subscribe to trade updates
        self.trading_stream.subscribe_trade_updates(handle_trade_update)

        # Start streaming
        stream_task = create_task_safely(self.trading_stream.run())
        self._stream_tasks.append(stream_task)

        try:
            while True:
                order = await order_queue.get()
                yield order
        finally:
            # Unsubscribe
            self.trading_stream.unsubscribe_trade_updates()

    def _convert_side(self, side: OrderSide) -> AlpacaSide:
        """Convert order side to Alpaca format."""
        return AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL

    def _convert_tif(self, tif: TimeInForce) -> AlpacaTIF:
        """Convert time in force to Alpaca format."""
        mapping = {
            TimeInForce.DAY: AlpacaTIF.DAY,
            TimeInForce.GTC: AlpacaTIF.GTC,
            TimeInForce.IOC: AlpacaTIF.IOC,
            TimeInForce.FOK: AlpacaTIF.FOK,
        }
        return mapping.get(tif, AlpacaTIF.DAY)

    def _convert_status_filter(self, status: OrderStatus) -> str:
        """Convert order status for filtering."""
        if status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            return "open"
        elif status == OrderStatus.FILLED:
            return "closed"
        elif status == OrderStatus.CANCELED:
            return "canceled"
        else:
            return "all"

    def _convert_order(self, alpaca_order: AlpacaOrder) -> Order:
        """Convert Alpaca order to internal format."""
        # Map Alpaca status to internal status
        status_map = {
            "new": OrderStatus.NEW,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.CANCELED,
            "canceled": OrderStatus.CANCELED,
            "expired": OrderStatus.EXPIRED,
            "replaced": OrderStatus.CANCELED,
            "pending_cancel": OrderStatus.PENDING_CANCEL,
            "pending_replace": OrderStatus.NEW,
            "accepted": OrderStatus.NEW,
            "pending_new": OrderStatus.PENDING_NEW,
            "accepted_for_bidding": OrderStatus.NEW,
            "stopped": OrderStatus.STOPPED,
            "rejected": OrderStatus.REJECTED,
            "suspended": OrderStatus.SUSPENDED,
            "calculated": OrderStatus.NEW,
        }

        # Map Alpaca order type to internal type
        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT,
        }

        return Order(
            order_id=alpaca_order.id,
            client_order_id=alpaca_order.client_order_id,
            symbol=alpaca_order.symbol,
            side=OrderSide.BUY if alpaca_order.side == AlpacaSide.BUY else OrderSide.SELL,
            order_type=type_map.get(alpaca_order.order_type, OrderType.MARKET),
            quantity=Decimal(str(alpaca_order.qty)),
            filled_quantity=Decimal(str(alpaca_order.filled_qty)),
            limit_price=(
                Decimal(str(alpaca_order.limit_price)) if alpaca_order.limit_price else None
            ),
            stop_price=Decimal(str(alpaca_order.stop_price)) if alpaca_order.stop_price else None,
            status=status_map.get(alpaca_order.status, OrderStatus.UNKNOWN),
            time_in_force=TimeInForce.DAY,  # Simplified mapping
            created_at=alpaca_order.created_at,
            updated_at=alpaca_order.updated_at,
            filled_at=alpaca_order.filled_at,
        )

    def _convert_position(self, alpaca_position: AlpacaPosition) -> Position:
        """Convert Alpaca position to internal format."""
        return Position(
            symbol=alpaca_position.symbol,
            side=PositionSide.LONG if float(alpaca_position.qty) > 0 else PositionSide.SHORT,
            quantity=Decimal(str(abs(float(alpaca_position.qty)))),
            avg_entry_price=Decimal(str(alpaca_position.avg_entry_price)),
            current_price=(
                Decimal(str(alpaca_position.current_price))
                if alpaca_position.current_price
                else None
            ),
            market_value=Decimal(str(alpaca_position.market_value)),
            cost_basis=Decimal(str(alpaca_position.cost_basis)),
            unrealized_pnl=Decimal(str(alpaca_position.unrealized_pl)),
            realized_pnl=Decimal("0"),  # Not provided by Alpaca
            opened_at=None,  # Not provided by Alpaca
        )

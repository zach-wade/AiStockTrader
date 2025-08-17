# File: backtesting/engine/market_simulator.py

"""
Market Simulator for Backtesting Engine.

Simulates realistic order execution including:
- Order queuing and priority
- Partial fills
- Slippage modeling
- Market impact
- Order rejection scenarios
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import heapq
from typing import Any

# Local imports
from main.events.types import FillEvent
from main.interfaces.events import OrderEvent
from main.models.common import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from main.utils.core import ErrorHandlingMixin, get_logger

from .cost_model import CostComponents, CostModel

logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Market simulation execution modes."""

    OPTIMISTIC = "optimistic"  # Fill at limit price or better
    REALISTIC = "realistic"  # Consider spread and market depth
    PESSIMISTIC = "pessimistic"  # Conservative fills with slippage


@dataclass
class OrderBookLevel:
    """Single level in the order book."""

    price: float
    size: int
    timestamp: datetime


@dataclass
class SimulatedOrderBook:
    """Simulated order book for a symbol."""

    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    last_size: int
    timestamp: datetime

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = (self.bid + self.ask) / 2
        return (self.spread / mid) * 100 if mid > 0 else 0

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2


@dataclass
class PendingOrder:
    """Order awaiting execution."""

    order: Order
    received_time: datetime
    priority: int  # For order queue priority

    def __lt__(self, other):
        """Compare orders for priority queue."""
        # Price-time priority
        if self.order.order_type == OrderType.LIMIT:
            if self.order.side == OrderSide.BUY:
                # Higher price = higher priority for buy orders
                if self.order.limit_price != other.order.limit_price:
                    return self.order.limit_price > other.order.limit_price
            elif self.order.limit_price != other.order.limit_price:
                return self.order.limit_price < other.order.limit_price

        # Same price or market orders: time priority
        return self.received_time < other.received_time


class MarketSimulator(ErrorHandlingMixin):
    """
    Simulates market execution for backtesting.

    Provides realistic order execution with configurable slippage,
    market impact, and execution delays.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        execution_mode: ExecutionMode = ExecutionMode.REALISTIC,
        fill_ratio: float = 1.0,  # Fraction of orders that fill
        execution_delay_ms: int = 100,
        use_order_book: bool = True,
        default_spread_bps: float = 10.0,
    ):  # 10 basis points
        """
        Initialize market simulator.

        Args:
            cost_model: Cost model for trading costs
            execution_mode: How optimistic/pessimistic execution should be
            fill_ratio: Probability of order filling (for partial fill simulation)
            execution_delay_ms: Simulated execution delay
            use_order_book: Whether to simulate order book dynamics
            default_spread_bps: Default spread in basis points
        """
        self.cost_model = cost_model
        self.execution_mode = execution_mode
        self.fill_ratio = fill_ratio
        self.execution_delay = timedelta(milliseconds=execution_delay_ms)
        self.use_order_book = use_order_book
        self.default_spread_bps = default_spread_bps

        # Order management
        self.pending_orders: dict[str, PendingOrder] = {}
        self.order_queues: dict[str, list[PendingOrder]] = {}  # By symbol

        # Simulated order books
        self.order_books: dict[str, SimulatedOrderBook] = {}

        # Execution statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.partial_fills = 0

        logger.info(f"MarketSimulator initialized in {execution_mode.value} mode")

    def update_market_data(
        self,
        symbol: str,
        bid: float,
        ask: float,
        bid_size: int,
        ask_size: int,
        last_price: float,
        timestamp: datetime,
    ):
        """Update order book for a symbol."""
        self.order_books[symbol] = SimulatedOrderBook(
            symbol=symbol,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=last_price,
            last_size=0,
            timestamp=timestamp,
        )

    def submit_order(self, order: Order, timestamp: datetime) -> OrderEvent:
        """
        Submit an order for execution.

        Args:
            order: Order to submit
            timestamp: Current simulation time

        Returns:
            OrderEvent confirming submission
        """
        self.total_orders += 1

        # Validate order
        if not self._validate_order(order):
            self.rejected_orders += 1
            return self._create_rejection_event(order, timestamp, "Order validation failed")

        # Create pending order
        pending = PendingOrder(order=order, received_time=timestamp, priority=self.total_orders)

        # Add to pending orders
        self.pending_orders[order.order_id] = pending

        # Add to symbol queue
        if order.symbol not in self.order_queues:
            self.order_queues[order.symbol] = []
        heapq.heappush(self.order_queues[order.symbol], pending)

        # Update order status
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = timestamp

        # Create submission event
        event = OrderEvent(timestamp=timestamp, order=order, event_type="submitted")

        logger.debug(
            f"Order submitted: {order.order_id} - {order.symbol} "
            f"{order.side.value} {order.quantity}"
        )

        return event

    def process_orders(self, timestamp: datetime) -> list[FillEvent]:
        """
        Process pending orders and generate fills.

        Args:
            timestamp: Current simulation time

        Returns:
            List of fill events
        """
        fills = []

        # Process each symbol's order queue
        for symbol in list(self.order_queues.keys()):
            if symbol not in self.order_books:
                continue

            order_book = self.order_books[symbol]
            symbol_fills = self._process_symbol_orders(symbol, order_book, timestamp)
            fills.extend(symbol_fills)

        return fills

    def _process_symbol_orders(
        self, symbol: str, order_book: SimulatedOrderBook, timestamp: datetime
    ) -> list[FillEvent]:
        """Process orders for a specific symbol."""
        fills = []
        queue = self.order_queues.get(symbol, [])
        processed_orders = []

        while queue:
            pending = heapq.heappop(queue)
            order = pending.order

            # Check if order expired
            if self._is_order_expired(order, timestamp):
                self._expire_order(order, timestamp)
                continue

            # Check execution delay
            if timestamp < pending.received_time + self.execution_delay:
                processed_orders.append(pending)
                continue

            # Try to fill order
            fill = self._try_fill_order(order, order_book, timestamp)
            if fill:
                fills.append(fill)
                self.filled_orders += 1

                # Remove from pending
                del self.pending_orders[order.order_id]
            else:
                # Keep in queue if not filled
                processed_orders.append(pending)

        # Re-add unfilled orders to queue
        for pending in processed_orders:
            heapq.heappush(queue, pending)

        return fills

    def _try_fill_order(
        self, order: Order, order_book: SimulatedOrderBook, timestamp: datetime
    ) -> FillEvent | None:
        """Try to fill an order against the order book."""
        # Get execution price
        execution_price = self._get_execution_price(order, order_book)

        if execution_price is None:
            return None

        # Check if order should fill
        if not self._should_fill(order, execution_price, order_book):
            return None

        # Determine fill quantity
        fill_quantity = self._get_fill_quantity(order, order_book)

        if fill_quantity == 0:
            return None

        # Calculate costs
        costs = self._calculate_costs(order, execution_price, fill_quantity, order_book)

        # Update order
        order.filled_quantity += fill_quantity
        order.avg_fill_price = (
            order.avg_fill_price * (order.filled_quantity - fill_quantity)
            + execution_price * fill_quantity
        ) / order.filled_quantity

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = timestamp
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            self.partial_fills += 1

        # Create fill event
        fill = FillEvent(
            timestamp=timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=execution_price,
            commission=costs.commission,
            slippage=costs.slippage,
        )

        logger.debug(f"Order filled: {order.order_id} - {fill_quantity} @ {execution_price}")

        return fill

    def _get_execution_price(self, order: Order, order_book: SimulatedOrderBook) -> float | None:
        """Determine execution price based on order type and market conditions."""
        if order.order_type == OrderType.MARKET:
            # Market orders execute at bid/ask
            if order.side == OrderSide.BUY:
                price = order_book.ask
            else:
                price = order_book.bid

            # Apply execution mode adjustments
            if self.execution_mode == ExecutionMode.PESSIMISTIC:
                # Add extra slippage
                slippage = order_book.spread * 0.5
                if order.side == OrderSide.BUY:
                    price += slippage
                else:
                    price -= slippage

            return price

        else:  # LIMIT order
            limit_price = order.limit_price

            if order.side == OrderSide.BUY:
                # Buy limit can execute at ask or better
                if limit_price >= order_book.ask:
                    if self.execution_mode == ExecutionMode.OPTIMISTIC:
                        return limit_price  # Fill at limit
                    else:
                        return order_book.ask  # Fill at market
                else:
                    return None  # Cannot fill
            elif limit_price <= order_book.bid:
                if self.execution_mode == ExecutionMode.OPTIMISTIC:
                    return limit_price  # Fill at limit
                else:
                    return order_book.bid  # Fill at market
            else:
                return None  # Cannot fill

    def _should_fill(
        self, order: Order, execution_price: float, order_book: SimulatedOrderBook
    ) -> bool:
        """Determine if order should fill based on market conditions."""
        if self.execution_mode == ExecutionMode.OPTIMISTIC:
            return True

        # Consider fill probability
        # Local imports
        from main.utils.core import secure_uniform

        if secure_uniform(0, 1) > self.fill_ratio:
            return False

        # Check market depth for large orders
        if self.execution_mode == ExecutionMode.REALISTIC:
            available_size = (
                order_book.ask_size if order.side == OrderSide.BUY else order_book.bid_size
            )

            # Large orders may not fill completely
            if order.quantity > available_size * 2:
                return secure_uniform(0, 1) < 0.5

        return True

    def _get_fill_quantity(self, order: Order, order_book: SimulatedOrderBook) -> int:
        """Determine fill quantity based on available liquidity."""
        remaining = order.quantity - order.filled_quantity

        if self.execution_mode == ExecutionMode.OPTIMISTIC:
            return remaining

        # Consider available size
        available = order_book.ask_size if order.side == OrderSide.BUY else order_book.bid_size

        if self.execution_mode == ExecutionMode.REALISTIC:
            # Can take up to 50% of displayed size
            max_fill = min(remaining, available // 2)
        else:  # PESSIMISTIC
            # Can only take 25% of displayed size
            max_fill = min(remaining, available // 4)

        # Partial fills for large orders
        if remaining > max_fill and remaining > 100:
            # Local imports
            from main.utils.core import secure_randint

            return secure_randint(max_fill // 2, max_fill)

        return min(remaining, max_fill)

    def _calculate_costs(
        self,
        order: Order,
        execution_price: float,
        fill_quantity: int,
        order_book: SimulatedOrderBook,
    ) -> CostComponents:
        """Calculate trading costs for the fill."""
        if not self.cost_model:
            return CostComponents()

        # Estimate volatility from spread
        volatility = order_book.spread_pct / 100 * 2  # Rough estimate

        # Calculate costs
        costs = self.cost_model.calculate_trade_cost(
            quantity=fill_quantity,
            price=execution_price,
            order_side=order.side,
            order_type=order.order_type,
            spread=order_book.spread_pct / 100,
            volatility=volatility,
            avg_daily_volume=1000000,  # Default ADV
        )

        return costs

    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        if order.quantity <= 0:
            logger.warning(f"Invalid order quantity: {order.quantity}")
            return False

        if order.order_type == OrderType.LIMIT and order.limit_price <= 0:
            logger.warning(f"Invalid limit price: {order.limit_price}")
            return False

        return True

    def _is_order_expired(self, order: Order, timestamp: datetime) -> bool:
        """Check if order has expired."""
        if order.time_in_force == TimeInForce.DAY:
            # Assume market closes at 4 PM
            market_close = timestamp.replace(hour=16, minute=0, second=0)
            return timestamp > market_close

        return False

    def _expire_order(self, order: Order, timestamp: datetime):
        """Expire an order."""
        order.status = OrderStatus.EXPIRED
        order.cancelled_at = timestamp
        del self.pending_orders[order.order_id]
        logger.debug(f"Order expired: {order.order_id}")

    def _create_rejection_event(self, order: Order, timestamp: datetime, reason: str) -> OrderEvent:
        """Create order rejection event."""
        order.status = OrderStatus.REJECTED
        order.cancelled_at = timestamp

        return OrderEvent(
            timestamp=timestamp, order=order, event_type="rejected", data={"reason": reason}
        )

    def cancel_order(self, order_id: str, timestamp: datetime) -> OrderEvent | None:
        """Cancel a pending order."""
        if order_id not in self.pending_orders:
            return None

        pending = self.pending_orders[order_id]
        order = pending.order

        # Update order status
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = timestamp

        # Remove from pending
        del self.pending_orders[order_id]

        # Remove from queue
        symbol = order.symbol
        if symbol in self.order_queues:
            self.order_queues[symbol] = [
                p for p in self.order_queues[symbol] if p.order.order_id != order_id
            ]
            heapq.heapify(self.order_queues[symbol])

        # Create cancellation event
        return OrderEvent(timestamp=timestamp, order=order, event_type="cancelled")

    def get_pending_orders(self, symbol: str | None = None) -> list[Order]:
        """Get list of pending orders."""
        if symbol:
            return [p.order for p in self.order_queues.get(symbol, [])]
        else:
            return [p.order for p in self.pending_orders.values()]

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics."""
        fill_rate = self.filled_orders / self.total_orders * 100 if self.total_orders > 0 else 0

        return {
            "total_orders": self.total_orders,
            "filled_orders": self.filled_orders,
            "rejected_orders": self.rejected_orders,
            "partial_fills": self.partial_fills,
            "pending_orders": len(self.pending_orders),
            "fill_rate": fill_rate,
            "execution_mode": self.execution_mode.value,
        }

    def reset(self):
        """Reset simulator state."""
        self.pending_orders.clear()
        self.order_queues.clear()
        self.order_books.clear()
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.partial_fills = 0
        logger.info("MarketSimulator reset")

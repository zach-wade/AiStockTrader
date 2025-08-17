"""
Paper Trading Broker - Simulated broker for testing without real money
"""

# Standard library imports
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
import logging
import secrets
from uuid import UUID

# Local imports
from src.application.interfaces.broker import (
    AccountInfo,
    BrokerConnectionError,
    InsufficientFundsError,
    InvalidOrderError,
    MarketClosedError,
    MarketHours,
    OrderNotFoundError,
)
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position

from .constants import (
    DAYS_TO_MONDAY_FROM_FRIDAY,
    DAYS_TO_MONDAY_FROM_SATURDAY,
    RANDOM_MIN_FACTOR,
    WEEKDAY_FRIDAY,
    WEEKDAY_SATURDAY,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulatedFill:
    """Represents a simulated order fill"""

    order_id: UUID
    fill_price: Decimal
    fill_quantity: Decimal
    timestamp: datetime
    slippage: Decimal = Decimal("0")


class PaperBroker:
    """
    Paper trading broker for simulated order execution.

    Simulates realistic order fills with configurable slippage and delays.
    All data is kept in memory - no real trades are executed.
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        slippage_pct: Decimal = Decimal("0.001"),  # 0.1% default slippage
        fill_delay_seconds: int = 1,
        commission_per_share: Decimal = Decimal("0.01"),
        min_commission: Decimal = Decimal("1.0"),
        simulate_partial_fills: bool = False,
    ):
        """
        Initialize paper broker.

        Args:
            initial_capital: Starting capital for the account
            slippage_pct: Percentage slippage to apply to market orders
            fill_delay_seconds: Simulated delay for order fills
            commission_per_share: Commission per share traded
            min_commission: Minimum commission per trade
            simulate_partial_fills: Whether to simulate partial fills
        """
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.fill_delay_seconds = fill_delay_seconds
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.simulate_partial_fills = simulate_partial_fills

        # Initialize portfolio
        self.portfolio = Portfolio(
            name="Paper Trading Portfolio",
            initial_capital=initial_capital,
            cash_balance=initial_capital,
        )

        # Order tracking
        self.orders: dict[UUID, Order] = {}
        self.pending_fills: list[SimulatedFill] = []

        # Market data (would normally come from a data provider)
        self.market_prices: dict[str, Decimal] = {}
        self.last_prices: dict[str, Decimal] = {}

        # Simulated market hours (NYSE regular hours)
        self.market_open_time = time(9, 30)  # 9:30 AM ET
        self.market_close_time = time(16, 0)  # 4:00 PM ET

        self._connected = False

        logger.info(
            f"Initialized paper broker with ${initial_capital} capital, "
            f"{slippage_pct}% slippage"
        )

    def connect(self) -> None:
        """Establish connection (simulated)"""
        self._connected = True
        logger.info("Connected to paper trading broker")

    def disconnect(self) -> None:
        """Close connection (simulated)"""
        self._connected = False
        logger.info("Disconnected from paper trading broker")

    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connected

    def _check_connection(self) -> None:
        """Ensure broker is connected"""
        if not self._connected:
            raise BrokerConnectionError("Not connected to paper broker. Call connect() first.")

    def set_market_price(self, symbol: str, price: Decimal) -> None:
        """
        Set the current market price for a symbol.

        This is used to simulate market data for paper trading.
        In production, this would come from a real data feed.
        """
        self.market_prices[symbol] = price
        self.last_prices[symbol] = price

        # Update portfolio positions with new price
        if symbol in self.portfolio.positions:
            self.portfolio.update_position_price(symbol, price)

        # Process any pending limit orders
        self._process_pending_orders(symbol, price)

    def _process_pending_orders(self, symbol: str, current_price: Decimal) -> None:
        """Process pending orders that might be triggered by price movement"""
        for order in self.orders.values():
            if order.symbol != symbol or not order.is_active():
                continue

            # Check limit orders
            if order.order_type == OrderType.LIMIT and order.limit_price:
                should_fill = False

                if (
                    order.side == OrderSide.BUY
                    and current_price <= order.limit_price
                    or order.side == OrderSide.SELL
                    and current_price >= order.limit_price
                ):
                    should_fill = True

                if should_fill:
                    self._simulate_fill(order, order.limit_price)

            # Check stop orders
            elif order.order_type == OrderType.STOP and order.stop_price:
                should_trigger = False

                if (
                    order.side == OrderSide.BUY
                    and current_price >= order.stop_price
                    or order.side == OrderSide.SELL
                    and current_price <= order.stop_price
                ):
                    should_trigger = True

                if should_trigger:
                    # Stop order becomes market order
                    fill_price = self._apply_slippage(current_price, order.side)
                    self._simulate_fill(order, fill_price)

    def _apply_slippage(self, price: Decimal, side: OrderSide) -> Decimal:
        """Apply slippage to a price based on order side"""
        slippage = price * self.slippage_pct

        # Add randomness to slippage (50% to 150% of configured slippage)
        # Use secrets for cryptographic randomness
        random_factor = RANDOM_MIN_FACTOR + (secrets.randbits(16) / 65536)
        slippage *= Decimal(str(random_factor))

        if side == OrderSide.BUY:
            # Buyers pay more (adverse selection)
            return price + slippage
        else:
            # Sellers receive less
            return price - slippage

    def _calculate_commission(self, quantity: Decimal) -> Decimal:
        """Calculate commission for a trade"""
        commission = quantity * self.commission_per_share
        return max(commission, self.min_commission)

    def _simulate_fill(
        self, order: Order, fill_price: Decimal, partial_quantity: Decimal | None = None
    ) -> None:
        """Simulate an order fill"""
        # Determine fill quantity
        if partial_quantity and self.simulate_partial_fills:
            fill_quantity = min(partial_quantity, order.get_remaining_quantity())
        else:
            fill_quantity = order.get_remaining_quantity()

        # Calculate commission
        commission = self._calculate_commission(fill_quantity)

        # Update order
        order.fill(fill_quantity, fill_price, datetime.now(UTC))

        # Update portfolio
        if order.side == OrderSide.BUY:
            # Opening or adding to long position
            if order.symbol in self.portfolio.positions:
                position = self.portfolio.positions[order.symbol]
                if not position.is_closed():
                    position.add_to_position(fill_quantity, fill_price, commission)
                else:
                    # Reopen position
                    self.portfolio.open_position(
                        symbol=order.symbol,
                        quantity=fill_quantity,
                        entry_price=fill_price,
                        commission=commission,
                    )
            else:
                self.portfolio.open_position(
                    symbol=order.symbol,
                    quantity=fill_quantity,
                    entry_price=fill_price,
                    commission=commission,
                )
        elif order.symbol in self.portfolio.positions:
            position = self.portfolio.positions[order.symbol]
            if position.is_long():
                # Closing or reducing long position
                position.reduce_position(fill_quantity, fill_price, commission)
            else:
                # Opening or adding to short position
                position.add_to_position(-fill_quantity, fill_price, commission)
        else:
            # Opening short position
            self.portfolio.open_position(
                symbol=order.symbol,
                quantity=-fill_quantity,
                entry_price=fill_price,
                commission=commission,
            )

        logger.info(
            f"Filled order {order.id}: {order.side.value} {fill_quantity} {order.symbol} "
            f"@ ${fill_price:.2f} (commission: ${commission:.2f})"
        )

    def submit_order(self, order: Order) -> Order:
        """Submit an order for paper execution"""
        self._check_connection()

        # Validate order
        if order.quantity <= 0:
            raise InvalidOrderError("Order quantity must be positive")

        # Check market hours for market orders
        if order.order_type == OrderType.MARKET and not self.is_market_open():
            raise MarketClosedError("Cannot submit market order while market is closed")

        # Get current price
        current_price = self.market_prices.get(order.symbol)
        if not current_price:
            # Use a default price for testing if not set
            current_price = Decimal("100.00")
            self.market_prices[order.symbol] = current_price
            logger.warning(f"No market price for {order.symbol}, using default ${current_price}")

        # Check buying power
        required_capital = order.quantity * current_price
        commission = self._calculate_commission(order.quantity)

        if (
            order.side == OrderSide.BUY
            and required_capital + commission > self.portfolio.cash_balance
        ):
            raise InsufficientFundsError(
                f"Insufficient funds: ${self.portfolio.cash_balance:.2f} available, "
                f"${required_capital + commission:.2f} required"
            )

        # Generate broker order ID
        broker_order_id = f"PAPER-{order.id}"
        order.submit(broker_order_id)

        # Store order
        self.orders[order.id] = order

        # Process based on order type
        if order.order_type == OrderType.MARKET:
            # Simulate immediate fill with slippage
            fill_price = self._apply_slippage(current_price, order.side)

            # Add delay if configured
            if self.fill_delay_seconds > 0:
                fill_time = datetime.now(UTC) + timedelta(seconds=self.fill_delay_seconds)
                self.pending_fills.append(
                    SimulatedFill(
                        order_id=order.id,
                        fill_price=fill_price,
                        fill_quantity=order.quantity,
                        timestamp=fill_time,
                        slippage=fill_price - current_price,
                    )
                )
            else:
                self._simulate_fill(order, fill_price)

        elif order.order_type == OrderType.LIMIT:
            # Check if limit order can be filled immediately
            if order.limit_price and (
                (order.side == OrderSide.BUY and current_price <= order.limit_price)
                or (order.side == OrderSide.SELL and current_price >= order.limit_price)
            ):
                self._simulate_fill(order, order.limit_price)

        logger.info(f"Submitted paper order: {order}")
        return order

    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel a paper order"""
        self._check_connection()

        if order_id not in self.orders:
            raise OrderNotFoundError(f"Order {order_id} not found")

        order = self.orders[order_id]

        if not order.is_active():
            return False

        order.cancel("User requested cancellation")

        # Remove from pending fills
        self.pending_fills = [f for f in self.pending_fills if f.order_id != order_id]

        logger.info(f"Cancelled paper order {order_id}")
        return True

    def get_order_status(self, order_id: UUID) -> OrderStatus:
        """Get status of a paper order"""
        self._check_connection()

        if order_id not in self.orders:
            raise OrderNotFoundError(f"Order {order_id} not found")

        # Process any pending fills
        self._process_pending_fills()

        return self.orders[order_id].status

    def update_order(self, order: Order) -> Order:
        """Update order with latest status"""
        self._check_connection()

        if order.id not in self.orders:
            raise OrderNotFoundError(f"Order {order.id} not found")

        # Process any pending fills
        self._process_pending_fills()

        # Return the stored order (which has the latest status)
        return self.orders[order.id]

    def _process_pending_fills(self) -> None:
        """Process any pending simulated fills"""
        now = datetime.now(UTC)
        fills_to_process = []

        for fill in self.pending_fills[:]:
            if fill.timestamp <= now:
                fills_to_process.append(fill)
                self.pending_fills.remove(fill)

        for fill in fills_to_process:
            if fill.order_id in self.orders:
                order = self.orders[fill.order_id]
                if order.is_active():
                    self._simulate_fill(order, fill.fill_price, fill.fill_quantity)

    def get_recent_orders(self, limit: int = 100) -> list[Order]:
        """Get recent paper orders"""
        self._check_connection()

        # Process pending fills first
        self._process_pending_fills()

        # Return most recent orders
        all_orders = list(self.orders.values())
        all_orders.sort(key=lambda o: o.created_at, reverse=True)

        return all_orders[:limit]

    def get_positions(self) -> list[Position]:
        """Get all current positions"""
        self._check_connection()

        return self.portfolio.get_open_positions()

    def get_account_info(self) -> AccountInfo:
        """Get paper account information"""
        self._check_connection()

        # Process pending fills first
        self._process_pending_fills()

        portfolio_value = self.portfolio.get_total_value()
        positions_value = self.portfolio.get_positions_value()

        return AccountInfo(
            account_id="PAPER-001",
            account_type="paper",
            equity=portfolio_value,
            cash=self.portfolio.cash_balance,
            buying_power=self.portfolio.cash_balance,  # Simplified - no margin
            positions_value=positions_value,
            unrealized_pnl=self.portfolio.get_unrealized_pnl(),
            realized_pnl=self.portfolio.total_realized_pnl,
            margin_used=Decimal("0"),  # No margin in paper trading
            margin_available=Decimal("0"),
            pattern_day_trader=False,
            trades_today=len(
                [o for o in self.orders.values() if o.created_at.date() == datetime.now(UTC).date()]
            ),
            trades_remaining=None,
            last_updated=datetime.now(UTC),
        )

    def is_market_open(self) -> bool:
        """Check if market is open (simulated)"""
        now = datetime.now(UTC)
        current_time = now.time()

        # Check if it's a weekday (Monday = 0, Sunday = 6)
        if now.weekday() > WEEKDAY_FRIDAY:  # Weekend
            return False

        # Check if within market hours
        # Note: This is simplified and doesn't account for holidays
        return self.market_open_time <= current_time <= self.market_close_time

    def get_market_hours(self) -> MarketHours:
        """Get market hours information (simulated)"""
        now = datetime.now(UTC)
        is_open = self.is_market_open()

        # Calculate next open/close
        if is_open:
            # Market is open, next event is close
            next_close = now.replace(
                hour=self.market_close_time.hour,
                minute=self.market_close_time.minute,
                second=0,
                microsecond=0,
            )
            next_open = None
        else:
            # Market is closed
            next_close = None

            # If it's before market open today
            if now.time() < self.market_open_time and now.weekday() <= WEEKDAY_FRIDAY:
                next_open = now.replace(
                    hour=self.market_open_time.hour,
                    minute=self.market_open_time.minute,
                    second=0,
                    microsecond=0,
                )
            else:
                # Next open is tomorrow or Monday
                days_ahead = 1
                if now.weekday() == WEEKDAY_FRIDAY:  # Friday
                    days_ahead = DAYS_TO_MONDAY_FROM_FRIDAY
                elif now.weekday() == WEEKDAY_SATURDAY:  # Saturday
                    days_ahead = DAYS_TO_MONDAY_FROM_SATURDAY

                next_open = (now + timedelta(days=days_ahead)).replace(
                    hour=self.market_open_time.hour,
                    minute=self.market_open_time.minute,
                    second=0,
                    microsecond=0,
                )

        return MarketHours(
            is_open=is_open,
            next_open=next_open,
            next_close=next_close,
        )

    def reset(self) -> None:
        """Reset the paper broker to initial state"""
        self.portfolio = Portfolio(
            name="Paper Trading Portfolio",
            initial_capital=self.initial_capital,
            cash_balance=self.initial_capital,
        )
        self.orders.clear()
        self.pending_fills.clear()
        self.market_prices.clear()
        self.last_prices.clear()

        logger.info("Reset paper broker to initial state")

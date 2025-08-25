"""
Order Processor - Domain service for order execution logic.

This module provides the OrderProcessor service which handles the core business logic
for processing order fills and managing the resulting position updates. It serves as
the bridge between order execution events and portfolio state changes.

The OrderProcessor encapsulates complex order processing logic including position
creation, scaling, reversal, and commission allocation. It ensures that all order
fills are properly reflected in portfolio positions while maintaining data consistency.

Key Responsibilities:
    - Processing order fills and updating portfolio positions
    - Handling position lifecycle (create, add, reduce, close, reverse)
    - Calculating execution prices considering order types
    - Commission allocation for partial fills
    - Determining when orders should be filled based on market conditions

Design Patterns:
    - Domain Service: Centralizes order processing logic outside of entities
    - Strategy Pattern: Different fill price calculation strategies per order type
    - Repository Pattern: Uses IPositionRepository protocol for data access

Architectural Decisions:
    - Separated from broker infrastructure to maintain domain purity
    - Uses protocol for repository to avoid infrastructure dependencies
    - All monetary calculations use Decimal for precision
    - Commission splitting for partial fills ensures accurate P&L

Example:
    >>> from decimal import Decimal
    >>> from datetime import datetime
    >>> from domain.services import OrderProcessor
    >>> from domain.entities import Order, Portfolio, OrderSide
    >>> from domain.value_objects import Price, Quantity, Money
    >>>
    >>> processor = OrderProcessor()
    >>> portfolio = Portfolio(cash_balance=Decimal("10000"))
    >>>
    >>> # Process a buy order fill
    >>> order = Order(symbol="AAPL", quantity=100, side=OrderSide.BUY)
    >>> fill_details = FillDetails(
    ...     order=order,
    ...     fill_price=Price(Decimal("150.00")),
    ...     fill_quantity=Quantity(100),
    ...     commission=Money(Decimal("1.00")),
    ...     timestamp=datetime.now()
    ... )
    >>> processor.process_fill(fill_details, portfolio)

Note:
    This service maintains no state and is designed to be concurrent-safe. All state
    changes are made to the provided portfolio and order entities.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from ..entities.order import Order, OrderSide
from ..entities.portfolio import Portfolio, PositionRequest
from ..entities.position import Position
from ..value_objects.money import Money
from ..value_objects.price import Price
from ..value_objects.quantity import Quantity

logger = logging.getLogger(__name__)


@dataclass
class FillDetails:
    """Details of an order fill.

    Encapsulates all information about a single order fill event. This data class
    serves as a parameter object for the order processing workflow, ensuring all
    required fill information is provided together.

    Attributes:
        order: The order being filled. Will be updated with fill information.
        fill_price: Actual execution price for this fill.
        fill_quantity: Number of shares filled (always positive).
        commission: Total commission charged for this fill.
        timestamp: Exact time when the fill occurred.

    Note:
        For partial fills, multiple FillDetails instances would be created,
        each representing a portion of the total order.
    """

    order: Order
    fill_price: Price
    fill_quantity: Quantity
    commission: Money
    timestamp: datetime


class IPositionRepository(Protocol):
    """Protocol for position repository operations needed by order processor.

    Defines the interface required for position persistence operations. This protocol
    allows the OrderProcessor to remain decoupled from infrastructure concerns while
    still being able to persist position changes.

    Implementations of this protocol would typically be found in the infrastructure
    layer, connecting to databases or other persistence mechanisms.

    Note:
        This protocol follows the Dependency Inversion Principle, allowing the
        domain service to depend on an abstraction rather than concrete implementations.
    """

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol.

        Args:
            symbol: Stock symbol to retrieve position for.

        Returns:
            Position if exists, None otherwise.
        """
        ...

    def persist_position(self, position: Position) -> Position:
        """Persist a position.

        Args:
            position: Position to persist.

        Returns:
            Persisted position (may include generated IDs).
        """
        ...


class OrderProcessor:
    """
    Domain service for processing order fills and managing positions.

    Encapsulates the business logic for order execution, position updates,
    and portfolio management. This removes business logic from infrastructure
    layer (brokers) and maintains it in the domain layer.

    The OrderProcessor is the central component for translating order execution
    events into portfolio state changes. It handles the complexity of position
    management including creating new positions, scaling existing positions,
    and handling position reversals.

    Attributes:
        None - This service is stateless

    Concurrency Safety:
        All methods are concurrent-safe as the service maintains no internal state.
        However, concurrent modifications to the same portfolio should be synchronized
        at a higher level.
    """

    async def process_fill(self, fill_details: FillDetails, portfolio: Portfolio) -> None:
        """
        Process an order fill and update portfolio positions.

        This is the main entry point for processing order fills. It orchestrates
        the entire fill processing workflow including order updates, position
        changes, and commission handling.

        This method encapsulates the business logic for:
        - Determining position changes based on order side
        - Opening new positions
        - Adding to existing positions
        - Reducing or closing positions
        - Handling both long and short positions

        Args:
            fill_details: Complete details of the order fill including price,
                quantity, commission, and timestamp.
            portfolio: The portfolio to update. Will be modified in place with
                new or updated positions.

        Side Effects:
            - Updates the order's fill status and average price
            - Modifies portfolio positions (create/update/close)
            - Logs the fill processing for audit trail

        Example:
            >>> order = Order(symbol="GOOGL", quantity=10, side=OrderSide.BUY)
            >>> fill = FillDetails(
            ...     order=order,
            ...     fill_price=Price(Decimal("2500")),
            ...     fill_quantity=Quantity(10),
            ...     commission=Money(Decimal("1.00")),
            ...     timestamp=datetime.now()
            ... )
            >>> processor.process_fill(fill, portfolio)
            >>> # Portfolio now has a new GOOGL position

        Note:
            This method assumes the fill has been previously validated and authorized.
            It does not perform risk checks or validate available funds.
        """
        order = fill_details.order

        # Update the order with fill information
        order.fill(
            fill_details.fill_quantity.value, fill_details.fill_price.value, fill_details.timestamp
        )

        # Process position updates based on order side
        is_buy = order.side == OrderSide.BUY
        await self._process_fill_with_direction(
            order,
            fill_details.fill_quantity,
            fill_details.fill_price,
            fill_details.commission,
            portfolio,
            is_buy,
        )

        logger.info(
            "Processed fill for order",
            extra={
                "order_id": str(order.id),
                "side": order.side.value,
                "quantity": float(fill_details.fill_quantity.value),
                "symbol": order.symbol,
                "price": float(fill_details.fill_price.value),
                "commission": float(fill_details.commission.amount),
            },
        )

    async def _process_fill_with_direction(
        self,
        order: Order,
        quantity: Quantity,
        price: Price,
        commission: Money,
        portfolio: Portfolio,
        is_buy: bool,
    ) -> None:
        """
        Process a fill based on direction (buy/sell).

        This unified method handles both buy and sell orders, reducing code duplication
        while maintaining clarity about the business logic. It determines the appropriate
        action based on the current position state and order direction.

        Args:
            order: The order being filled.
            quantity: Fill quantity (always positive regardless of side).
            price: Fill price for this execution.
            commission: Commission charged for this fill.
            portfolio: Portfolio to update with position changes.
            is_buy: True for buy orders, False for sell orders.

        Logic Flow:
            1. Check if position exists for the symbol
            2. If no position: Open new position
            3. If same direction: Add to existing position
            4. If opposite direction: Reduce or reverse position

        Note:
            The signed_quantity (positive for buys, negative for sells) is used
            internally to simplify position calculations while maintaining clarity
            about direction.
        """
        position = portfolio.positions.get(order.symbol)

        # Determine the signed quantity based on order side
        signed_quantity = quantity.value if is_buy else -quantity.value

        if position is None or position.is_closed():
            # Create new position with appropriate sign
            await self._create_new_position(
                portfolio, order.symbol, Quantity(signed_quantity), price, commission
            )
        elif self._is_same_direction(position, is_buy):
            # Add to existing position in same direction
            position.add_to_position(abs(signed_quantity), price.value, commission.amount)
        else:
            # Handle position reversal (long to short or short to long)
            await self._handle_position_reversal(
                position, quantity, price, commission, portfolio, order.symbol, is_buy
            )

    def _is_same_direction(self, position: Position, is_buy: bool) -> bool:
        """
        Check if the position and order are in the same direction.

        Determines whether an order would increase or decrease an existing position.
        This is used to decide whether to add to a position or reduce/reverse it.

        Args:
            position: Current position to check.
            is_buy: True if order is a buy, False if sell.

        Returns:
            bool: True if position and order are in same direction:
                - Long position + Buy order = True (adding to long)
                - Short position + Sell order = True (adding to short)
                - Long position + Sell order = False (reducing long)
                - Short position + Buy order = False (covering short)

        Example:
            >>> position = Position(symbol="TSLA", quantity=100)  # Long position
            >>> is_same = processor._is_same_direction(position, is_buy=True)
            >>> assert is_same == True  # Buy adds to long position
        """
        return (position.is_long() and is_buy) or (position.is_short() and not is_buy)

    async def _handle_position_reversal(
        self,
        position: Position,
        quantity: Quantity,
        price: Price,
        commission: Money,
        portfolio: Portfolio,
        symbol: str,
        is_buy: bool,
    ) -> None:
        """
        Handle position reversal (flipping from long to short or vice versa).

        Manages the complex case where an order is large enough to not only close
        an existing position but also create a new position in the opposite direction.
        Commission is proportionally allocated between the closing and creation portions.

        Args:
            position: Current position to potentially reverse.
            quantity: Order quantity (always positive).
            price: Fill price for both closing and creation.
            commission: Total commission to be split proportionally.
            portfolio: Portfolio to update with position changes.
            symbol: Symbol being traded.
            is_buy: True if buying (short to long), False if selling (long to short).

        Behavior:
            - If order quantity <= position size: Just reduces position
            - If order quantity > position size:
                1. Closes existing position completely
                2. Creates new position in opposite direction with remaining quantity
                3. Splits commission proportionally between close and create

        Example:
            >>> # Current position: Long 100 shares
            >>> # Sell order for 150 shares
            >>> # Result: Close long 100, create short 50
            >>> position = Position(symbol="AAPL", quantity=100)
            >>> processor._handle_position_reversal(
            ...     position, Quantity(150), Price(Decimal("150")),
            ...     Money(Decimal("2.00")), portfolio, "AAPL", is_buy=False
            ... )

        Note:
            Commission splitting ensures accurate P&L tracking for both the
            closing and creation portions of the reversal.
        """
        current_position_size = abs(position.quantity)

        if current_position_size >= quantity.value:
            # Just reducing the position, not reversing
            position.reduce_position(quantity.value, price.value, commission.amount)
        else:
            # Close current position and create opposite position
            remaining_quantity = quantity.value - current_position_size

            # Calculate proportional commission for each part
            close_commission = self._split_commission(
                commission, current_position_size, quantity.value
            )
            create_commission = self._split_commission(
                commission, remaining_quantity, quantity.value
            )

            # First close the current position
            position.reduce_position(current_position_size, price.value, close_commission.amount)

            # Then create new position in opposite direction
            signed_remaining = remaining_quantity if is_buy else -remaining_quantity
            await self._create_new_position(
                portfolio, symbol, Quantity(signed_remaining), price, create_commission
            )

    def _split_commission(
        self, total_commission: Money, partial_qty: Decimal, total_qty: Decimal
    ) -> Money:
        """
        Calculate proportional commission for partial fills.

        Allocates commission proportionally based on quantity when an order fill
        is split between multiple actions (e.g., closing and creating positions).

        Args:
            total_commission: Total commission for the entire fill.
            partial_qty: Quantity for this portion of the fill.
            total_qty: Total quantity being filled.

        Returns:
            Money: Proportional commission amount for the partial quantity.
                Returns Money(0) if total_qty is zero to avoid division errors.

        Formula:
            Partial Commission = Total Commission × (Partial Qty / Total Qty)

        Example:
            >>> total_comm = Money(Decimal("10.00"))
            >>> partial_comm = processor._split_commission(
            ...     total_comm, Decimal("30"), Decimal("100")
            ... )
            >>> assert partial_comm.amount == Decimal("3.00")  # 10 * (30/100)

        Note:
            This ensures commission is accurately allocated for P&L calculations,
            particularly important for position reversals where part of the fill
            closes an existing position and part creates a new one.
        """
        if total_qty == 0:
            return Money(Decimal("0"))
        ratio = partial_qty / total_qty
        return Money(total_commission.amount * ratio)

    async def _create_new_position(
        self, portfolio: Portfolio, symbol: str, quantity: Quantity, price: Price, commission: Money
    ) -> None:
        """
        Create a new position in the portfolio.

        Creates and adds a new position to the portfolio. This method handles
        both long positions (positive quantity) and short positions (negative quantity).

        Args:
            portfolio: Portfolio to update with the new position.
            symbol: Stock symbol for the position.
            quantity: Signed quantity where positive indicates long position
                and negative indicates short position.
            price: Entry price for the position.
            commission: Commission charged for creating the position.

        Side Effects:
            - Creates a new position in the portfolio
            - Updates portfolio's position tracking

        Example:
            >>> # Create a long position
            >>> processor._create_new_position(
            ...     portfolio, "MSFT", Quantity(100),
            ...     Price(Decimal("300")), Money(Decimal("1.00"))
            ... )
            >>> # Create a short position
            >>> processor._create_new_position(
            ...     portfolio, "MSFT", Quantity(-50),
            ...     Price(Decimal("300")), Money(Decimal("1.00"))
            ... )

        Note:
            This method uses the portfolio's open_position method which handles
            validation and state management.
        """
        request = PositionRequest(
            symbol=symbol,
            quantity=quantity.value,
            entry_price=price.value,
            commission=commission.amount,
        )
        portfolio.open_position(request)

    def calculate_fill_price(
        self, order: Order, market_price: Price, slippage_model: object | None = None
    ) -> Price:
        """
        Calculate the actual fill price for an order.

        Determines the execution price based on order type and market conditions.
        For limit orders, ensures execution at limit price or better. For stop
        orders, triggers at stop price and executes at market. Market orders
        execute at current market price.

        This method determines the execution price based on:
        - Order type (market, limit, stop)
        - Current market price
        - Slippage model (if provided)

        Args:
            order: The order to fill. Order type and prices determine fill logic.
            market_price: Current market price for the symbol.
            slippage_model: Optional slippage model for realistic fills. If provided,
                would apply market impact and bid-ask costs (not implemented here).

        Returns:
            Price: The calculated fill price based on order type and market conditions.
                - Limit orders: Best of limit price or market price
                - Stop orders: Market price when triggered
                - Market orders: Current market price

        Fill Logic:
            Limit Buy: Fill at min(market, limit) if market <= limit
            Limit Sell: Fill at max(market, limit) if market >= limit
            Stop Buy: Fill at market if market >= stop
            Stop Sell: Fill at market if market <= stop
            Market: Always fill at market price

        Example:
            >>> # Limit buy at $100, market at $98
            >>> order = Order(symbol="AAPL", side=OrderSide.BUY, limit_price=Decimal("100"))
            >>> fill_price = processor.calculate_fill_price(order, Price(Decimal("98")))
            >>> assert fill_price.value == Decimal("98")  # Better than limit

        Note:
            In production, the slippage_model would adjust the fill price based
            on order size, market depth, and volatility. This implementation
            provides the base price before slippage.
        """
        # Handle limit orders
        if order.limit_price:
            is_buy = order.side == OrderSide.BUY
            price_favorable = (
                market_price.value <= order.limit_price
                if is_buy
                else market_price.value >= order.limit_price
            )

            if price_favorable:
                # Execute at limit price or better
                optimal_price = (
                    min(market_price.value, order.limit_price)
                    if is_buy
                    else max(market_price.value, order.limit_price)
                )
                return Price(optimal_price)

        # Handle stop orders
        if order.stop_price:
            is_buy = order.side == OrderSide.BUY
            stop_triggered = (
                market_price.value >= order.stop_price
                if is_buy
                else market_price.value <= order.stop_price
            )

            if stop_triggered:
                # Stop triggered, execute at market
                return market_price

        # For market orders or triggered stops, use market price
        # (slippage would be applied by the slippage model if provided)
        return market_price

    def should_fill_order(self, order: Order, market_price: Price) -> bool:
        """
        Determine if an order should be filled at the current market price.

        Evaluates whether market conditions allow an order to be executed.
        This is used by trading systems to determine when pending orders
        should be converted to fills.

        Args:
            order: The order to check. Must be in an active status.
            market_price: Current market price for the order's symbol.

        Returns:
            bool: True if the order should be filled, False otherwise.
                - Market orders: Always true (during market hours)
                - Limit orders: True when price is favorable
                - Stop orders: True when stop is triggered

        Fill Conditions:
            - Order must be in active status (PENDING, PARTIALLY_FILLED)
            - Market Order: Always fills
            - Limit Buy: Fills when market <= limit price
            - Limit Sell: Fills when market >= limit price
            - Stop Buy: Fills when market >= stop price
            - Stop Sell: Fills when market <= stop price

        Example:
            >>> # Limit buy at $50, check at market $48
            >>> order = Order(
            ...     symbol="AMD", side=OrderSide.BUY,
            ...     limit_price=Decimal("50.00")
            ... )
            >>> should_fill = processor.should_fill_order(
            ...     order, Price(Decimal("48.00"))
            ... )
            >>> assert should_fill == True  # Can buy below limit

        Note:
            This method doesn't consider market hours, liquidity, or other
            market microstructure factors. Those would be handled by the
            execution venue or broker.
        """
        if not order.is_active():
            return False

        # Market orders always fill (during market hours)
        if order.limit_price is None and order.stop_price is None:
            return True

        is_buy = order.side == OrderSide.BUY

        # Check limit orders
        if order.limit_price:
            return (
                market_price.value <= order.limit_price
                if is_buy
                else market_price.value >= order.limit_price
            )

        # Check stop orders
        if order.stop_price:
            return (
                market_price.value >= order.stop_price
                if is_buy
                else market_price.value <= order.stop_price
            )

        return False

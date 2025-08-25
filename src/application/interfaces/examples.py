"""
Repository Interface Usage Examples

Demonstrates how to use the repository interfaces with domain entities.
These examples show the intended usage patterns for clean architecture.
"""

# Standard library imports
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast
from uuid import UUID, uuid4

# Local imports
from src.domain.entities.order import Order, OrderRequest, OrderSide, OrderStatus
from src.domain.entities.position import Position
from src.domain.value_objects.price import Price
from src.domain.value_objects.quantity import Quantity

from .exceptions import OrderNotFoundError, PositionNotFoundError
from .unit_of_work import ITransactionManager, IUnitOfWork


class TradingService:
    """
    Example service showing how to use repository interfaces.

    This demonstrates the application layer using repository contracts
    without depending on specific implementations.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        transaction_manager: ITransactionManager,
    ) -> None:
        self._uow = unit_of_work
        self._tx_manager = transaction_manager

    async def place_market_order(
        self, symbol: str, quantity: Decimal, side: OrderSide, reason: str | None = None
    ) -> Order:
        """
        Place a market order and save it to the repository.

        Example of using repository interface for simple operations.
        """
        # Create domain entity
        request = OrderRequest(symbol=symbol, quantity=Quantity(quantity), side=side, reason=reason)
        order = Order.create_market_order(request)

        # Save using repository interface
        async with self._uow:
            saved_order = await self._uow.orders.save_order(order)
            await self._uow.commit()

        return saved_order

    async def execute_trade(
        self, order_id: UUID, fill_price: Decimal, fill_quantity: Decimal
    ) -> tuple[Order, Position]:
        """
        Execute a trade by updating order and creating/updating position.

        Example of using Unit of Work for atomic operations across repositories.
        """

        async def trade_operation(uow: IUnitOfWork) -> tuple[Order, Position]:
            # Get order
            order = await uow.orders.get_order_by_id(order_id)
            if not order:
                raise OrderNotFoundError(order_id)

            # Fill the order
            order.fill(Quantity(fill_quantity), Price(fill_price))

            # Get or create position
            position = await uow.positions.get_position_by_symbol(order.symbol)
            if not position:
                # Create new position
                position = Position.open_position(
                    symbol=order.symbol,
                    quantity=Quantity(
                        fill_quantity if order.side == OrderSide.BUY else -fill_quantity
                    ),
                    entry_price=Price(fill_price),
                )
            elif order.side == OrderSide.BUY:
                position.add_to_position(Quantity(fill_quantity), Price(fill_price))
            else:
                position.reduce_position(Quantity(fill_quantity), Price(fill_price))

            # Save changes
            updated_order = await uow.orders.update_order(order)
            updated_position = await uow.positions.persist_position(position)

            return updated_order, updated_position

        result = await self._tx_manager.execute_in_transaction(trade_operation)
        return cast(tuple[Order, Position], result)

    async def close_position_completely(
        self, symbol: str, exit_price: Decimal
    ) -> tuple[Position, Order]:
        """
        Close a position completely by creating a closing order.

        Example of complex business logic using multiple repositories.
        """

        async def close_operation(uow: IUnitOfWork) -> tuple[Position, Order]:
            # Get current position
            position = await uow.positions.get_position_by_symbol(symbol)
            if not position or position.is_closed():
                raise PositionNotFoundError(position.id if position else uuid4())

            # Create closing order
            closing_side = OrderSide.SELL if position.is_long() else OrderSide.BUY
            close_request = OrderRequest(
                symbol=symbol,
                quantity=Quantity(abs(position.quantity)),
                side=closing_side,
                reason=f"Close position {position.id}",
            )
            closing_order = Order.create_market_order(close_request)

            # Mark order as filled immediately (assuming market execution)
            closing_order.submit(f"broker_{closing_order.id}")
            closing_order.fill(Quantity(abs(position.quantity.value)), Price(exit_price))

            # Close the position
            position.close_position(Price(exit_price))

            # Save changes
            saved_order = await uow.orders.save_order(closing_order)
            updated_position = await uow.positions.update_position(position)

            return updated_position, saved_order

        result = await self._tx_manager.execute_in_transaction(close_operation)
        return cast(tuple[Position, Order], result)

    async def get_portfolio_summary(self, portfolio_id: UUID) -> dict[str, Any]:
        """
        Get portfolio summary with positions and recent orders.

        Example of read operations across multiple repositories.
        """
        async with self._uow:
            # Get portfolio
            portfolio = await self._uow.portfolios.get_portfolio_by_id(portfolio_id)
            if not portfolio:
                raise PositionNotFoundError(portfolio_id)

            # Get active positions
            active_positions = await self._uow.positions.get_active_positions()

            # Get recent orders (last 24 hours)
            recent_orders = await self._uow.orders.get_orders_by_date_range(
                start_date=datetime.now(UTC).replace(hour=0, minute=0, second=0),
                end_date=datetime.now(UTC),
            )

            return {
                "portfolio": portfolio.to_dict(),
                "active_positions": [
                    {
                        "symbol": pos.symbol,
                        "quantity": float(pos.quantity),
                        "entry_price": float(pos.average_entry_price),
                        "current_value": float(pos.get_position_value() or 0),
                        "unrealized_pnl": float(pos.get_unrealized_pnl() or 0),
                    }
                    for pos in active_positions
                ],
                "recent_orders": [
                    {
                        "id": str(order.id),
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": float(order.quantity),
                        "status": order.status.value,
                        "created_at": order.created_at.isoformat(),
                    }
                    for order in recent_orders
                ],
                "summary": {
                    "total_value": float(portfolio.get_total_value()),
                    "cash_balance": float(portfolio.cash_balance),
                    "unrealized_pnl": float(portfolio.get_unrealized_pnl()),
                    "total_pnl": float(portfolio.get_total_pnl()),
                    "return_percentage": float(portfolio.get_return_percentage()),
                },
            }

    async def bulk_update_positions(self, price_updates: dict[str, Decimal]) -> list[Position]:
        """
        Update multiple position prices in a single transaction.

        Example of bulk operations with transaction management.
        """

        async def bulk_update_operation(uow: IUnitOfWork) -> list[Position]:
            updated_positions = []

            for symbol, price in price_updates.items():
                position = await uow.positions.get_position_by_symbol(symbol)
                if position and not position.is_closed():
                    position.update_market_price(Price(price))
                    updated_position = await uow.positions.update_position(position)
                    updated_positions.append(updated_position)

            return updated_positions

        result = await self._tx_manager.execute_in_transaction(bulk_update_operation)
        return cast(list[Position], result)

    async def get_trading_metrics(self, symbol: str) -> dict[str, Any]:
        """
        Get comprehensive trading metrics for a symbol.

        Example of analytical queries across repositories.
        """
        async with self._uow:
            # Get all orders for symbol
            all_orders = await self._uow.orders.get_orders_by_symbol(symbol)

            # Get all positions for symbol
            all_positions = await self._uow.positions.get_positions_by_symbol(symbol)

            # Calculate metrics
            total_orders = len(all_orders)
            filled_orders = [o for o in all_orders if o.status == OrderStatus.FILLED]
            cancelled_orders = [o for o in all_orders if o.status == OrderStatus.CANCELLED]

            closed_positions = [p for p in all_positions if p.is_closed()]
            winning_positions = [p for p in closed_positions if p.realized_pnl > 0]

            return {
                "symbol": symbol,
                "order_stats": {
                    "total_orders": total_orders,
                    "filled_orders": len(filled_orders),
                    "cancelled_orders": len(cancelled_orders),
                    "fill_rate": len(filled_orders) / total_orders if total_orders > 0 else 0,
                },
                "position_stats": {
                    "total_positions": len(all_positions),
                    "closed_positions": len(closed_positions),
                    "winning_positions": len(winning_positions),
                    "win_rate": (
                        len(winning_positions) / len(closed_positions) if closed_positions else 0
                    ),
                },
                "pnl_stats": {
                    "total_realized_pnl": float(sum(p.realized_pnl for p in closed_positions)),
                    "average_win": float(
                        sum(p.realized_pnl for p in winning_positions) / len(winning_positions)
                        if winning_positions
                        else 0
                    ),
                    "average_loss": float(
                        sum(p.realized_pnl for p in closed_positions if p.realized_pnl < 0)
                        / len([p for p in closed_positions if p.realized_pnl < 0])
                        if any(p.realized_pnl < 0 for p in closed_positions)
                        else 0
                    ),
                },
            }


# Example usage patterns for testing and documentation
async def example_usage_patterns() -> None:
    """
    Example patterns showing repository interface usage.

    Note: This is for documentation purposes only.
    Actual implementations would inject concrete repositories.
    """

    # This would be injected in real application
    # uow: IUnitOfWork = SomeConcreteUnitOfWork()
    # tx_manager: ITransactionManager = SomeConcreteTransactionManager()
    # service = TradingService(uow, tx_manager)

    # Example operations:
    # order = await service.place_market_order("AAPL", Decimal("100"), OrderSide.BUY)
    # order, position = await service.execute_trade(order.id, Decimal("150.50"), Decimal("100"))
    # summary = await service.get_portfolio_summary(portfolio_id)

    pass

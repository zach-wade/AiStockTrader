"""Factory functions for creating test entities."""

from decimal import Decimal
from typing import Any

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.value_objects import Money, Price, Quantity


class OrderFactory:
    """Factory for creating test Order instances."""

    @staticmethod
    def create(
        symbol: str = "AAPL",
        side: OrderSide = OrderSide.BUY,
        quantity: int | Decimal = 100,
        order_type: OrderType = OrderType.MARKET,
        status: OrderStatus = OrderStatus.PENDING,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        **kwargs: Any,
    ) -> Order:
        """Create a test Order with sensible defaults."""
        order_data = {
            "symbol": symbol,
            "side": side,
            "quantity": Quantity(Decimal(str(quantity))),
            "order_type": order_type,
            "status": status,
        }

        if limit_price is not None:
            order_data["limit_price"] = Price(Decimal(str(limit_price)))
        if stop_price is not None:
            order_data["stop_price"] = Price(Decimal(str(stop_price)))

        # Add any additional kwargs
        order_data.update(kwargs)

        return Order(**order_data)


class PositionFactory:
    """Factory for creating test Position instances."""

    @staticmethod
    def create(
        symbol: str = "AAPL",
        quantity: int | Decimal = 100,
        entry_price: Decimal = Decimal("150.00"),
        **kwargs: Any,
    ) -> Position:
        """Create a test Position with sensible defaults."""
        position_data = {
            "symbol": symbol,
            "quantity": Quantity(Decimal(str(quantity))),
            "average_entry_price": Price(Decimal(str(entry_price))),
        }

        # Add any additional kwargs
        position_data.update(kwargs)

        return Position(**position_data)


class PortfolioFactory:
    """Factory for creating test Portfolio instances."""

    @staticmethod
    def create(
        name: str = "Test Portfolio",
        cash_balance: Decimal = Decimal("100000.00"),
        initial_capital: Decimal = Decimal("100000.00"),
        **kwargs: Any,
    ) -> Portfolio:
        """Create a test Portfolio with sensible defaults."""
        portfolio_data = {
            "name": name,
            "cash_balance": Money(Decimal(str(cash_balance))),
            "initial_capital": Money(Decimal(str(initial_capital))),
        }

        # Add any additional kwargs
        portfolio_data.update(kwargs)

        return Portfolio(**portfolio_data)


# Convenience functions
def create_test_order(**kwargs: Any) -> Order:
    """Create a test order with defaults."""
    return OrderFactory.create(**kwargs)


def create_test_position(**kwargs: Any) -> Position:
    """Create a test position with defaults."""
    return PositionFactory.create(**kwargs)


def create_test_portfolio(**kwargs: Any) -> Portfolio:
    """Create a test portfolio with defaults."""
    return PortfolioFactory.create(**kwargs)

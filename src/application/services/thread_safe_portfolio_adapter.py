"""
ThreadSafePortfolioAdapter - Infrastructure layer thread safety adapter for Portfolio operations

This adapter provides thread-safe wrappers around Portfolio domain entity operations
using threading locks to coordinate concurrent access. It follows the Adapter pattern
to add infrastructure concerns (threading) without polluting the domain layer.
"""

import threading
from decimal import Decimal

from ...domain.entities.portfolio import Portfolio, PositionRequest
from ...domain.entities.position import Position
from ...domain.value_objects.money import Money
from ...domain.value_objects.price import Price
from .portfolio_service import PortfolioService


class ThreadSafePortfolioAdapter:
    """
    Infrastructure adapter providing thread-safe operations for Portfolio entities.

    This adapter wraps Portfolio domain entities and application services, adding
    thread-safety coordination through threading locks. The domain entities
    themselves remain pure and synchronous, maintaining clean architecture.

    This is an infrastructure concern that should not exist in domain or application layers.
    """

    def __init__(self, portfolio_service: PortfolioService | None = None) -> None:
        """Initialize thread-safety locks and wrapped service.

        Args:
            portfolio_service: The portfolio service to wrap.
                              If not provided, creates a new instance.
        """
        # Use RLock (reentrant lock) to allow same thread to acquire lock multiple times
        self._position_lock = threading.RLock()
        self._cash_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        self._portfolio_service = portfolio_service or PortfolioService()

    def open_position(
        self,
        portfolio: Portfolio,
        request: PositionRequest,
    ) -> Position:
        """Open a new position in the portfolio (thread-safe).

        Args:
            portfolio: Portfolio domain entity
            request: Position request with parameters

        Returns:
            Newly opened position

        Raises:
            ValueError: If position cannot be opened
        """
        # Validate ability to open (read-only check, no lock needed)
        can_open, reason = self._portfolio_service.can_open_position(
            portfolio, request.symbol, request.quantity, request.entry_price
        )
        if not can_open:
            raise ValueError(f"Cannot open position: {reason}")

        # Atomic update of portfolio state
        with self._position_lock:
            # Double-check after acquiring lock
            if (
                request.symbol in portfolio.positions
                and not portfolio.positions[request.symbol].is_closed()
            ):
                raise ValueError(f"Position already exists for {request.symbol}")

            with self._cash_lock:
                # Delegate to portfolio service for business logic
                position = self._portfolio_service.open_position(portfolio, request)

        return position

    def close_position(
        self,
        portfolio: Portfolio,
        symbol: str,
        exit_price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> Money:
        """
        Close a position and update portfolio (thread-safe).

        Args:
            portfolio: Portfolio domain entity
            symbol: Symbol of position to close
            exit_price: Exit price for the position
            commission: Commission paid for closing

        Returns:
            Realized P&L from closing the position
        """
        with self._position_lock:
            with self._cash_lock:
                with self._stats_lock:
                    # Delegate to portfolio service for business logic
                    return self._portfolio_service.close_position(
                        portfolio, symbol, Price(exit_price), Money(commission)
                    )

    def update_position_price(self, portfolio: Portfolio, symbol: str, price: Decimal) -> None:
        """Update market price for a position (thread-safe)

        Args:
            portfolio: Portfolio domain entity
            symbol: Symbol to update price for
            price: New market price
        """
        with self._position_lock:
            # Update position price directly
            if symbol in portfolio.positions:
                portfolio.positions[symbol].update_market_price(Price(price))

        with self._stats_lock:
            # Update portfolio timestamp
            from datetime import UTC, datetime

            portfolio.last_updated = datetime.now(UTC)

    def update_all_prices(self, portfolio: Portfolio, prices: dict[str, Decimal]) -> None:
        """Update market prices for multiple positions (thread-safe)

        Args:
            portfolio: Portfolio domain entity
            prices: Dictionary of symbol to price mappings
        """
        with self._position_lock:
            # Update all position prices
            for symbol, price in prices.items():
                if symbol in portfolio.positions:
                    portfolio.positions[symbol].update_market_price(Price(price))

        with self._stats_lock:
            # Update portfolio timestamp
            from datetime import UTC, datetime

            portfolio.last_updated = datetime.now(UTC)

    def get_total_value(self, portfolio: Portfolio) -> Money:
        """Calculate total portfolio value (cash + positions) - thread-safe

        Args:
            portfolio: Portfolio domain entity

        Returns:
            Total portfolio value
        """
        with self._cash_lock:
            with self._position_lock:
                # Delegate to operations service for calculation
                return self._portfolio_service.get_total_value(portfolio)

    def get_total_value_sync(self, portfolio: Portfolio) -> Money:
        """Synchronous wrapper for get_total_value

        Args:
            portfolio: Portfolio domain entity

        Returns:
            Total portfolio value
        """
        # Now that get_total_value is synchronous, just call it directly
        return self.get_total_value(portfolio)

    def open_position_sync(
        self,
        portfolio: Portfolio,
        request: PositionRequest,
    ) -> Position:
        """Synchronous wrapper for open_position

        Args:
            portfolio: Portfolio domain entity
            request: Position request with parameters

        Returns:
            Newly opened position
        """
        # Now that open_position is synchronous, just call it directly
        return self.open_position(portfolio, request)

    def close_position_sync(
        self,
        portfolio: Portfolio,
        symbol: str,
        exit_price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> Money:
        """Synchronous wrapper for close_position

        Args:
            portfolio: Portfolio domain entity
            symbol: Symbol of position to close
            exit_price: Exit price for the position
            commission: Commission paid for closing

        Returns:
            Realized P&L from closing the position
        """
        # Now that close_position is synchronous, just call it directly
        return self.close_position(portfolio, symbol, exit_price, commission)

    def update_position_price_sync(self, portfolio: Portfolio, symbol: str, price: Decimal) -> None:
        """Synchronous wrapper for update_position_price

        Args:
            portfolio: Portfolio domain entity
            symbol: Symbol to update price for
            price: New market price
        """
        # Now that update_position_price is synchronous, just call it directly
        self.update_position_price(portfolio, symbol, price)

    def update_all_prices_sync(self, portfolio: Portfolio, prices: dict[str, Decimal]) -> None:
        """Synchronous wrapper for update_all_prices

        Args:
            portfolio: Portfolio domain entity
            prices: Dictionary of symbol to price mappings
        """
        # Now that update_all_prices is synchronous, just call it directly
        self.update_all_prices(portfolio, prices)

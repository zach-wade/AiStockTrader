"""
ThreadSafePortfolioService - Application-layer thread safety for Portfolio operations

Provides thread-safe wrappers around Portfolio domain entity operations
using asyncio locks to coordinate concurrent access.
"""

import asyncio
from decimal import Decimal

from ...domain.entities.portfolio import Portfolio, PositionRequest
from ...domain.entities.position import Position


class ThreadSafePortfolioService:
    """
    Application service providing thread-safe operations for Portfolio entities.

    This service acts as a wrapper around Portfolio domain entities, adding
    thread-safety coordination through asyncio locks. The domain entities
    themselves remain pure and synchronous.
    """

    def __init__(self) -> None:
        """Initialize thread-safety locks"""
        self._position_lock = asyncio.Lock()
        self._cash_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()

    async def open_position(
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
        can_open, reason = portfolio.can_open_position(
            request.symbol, request.quantity, request.entry_price
        )
        if not can_open:
            raise ValueError(f"Cannot open position: {reason}")

        # Atomic update of portfolio state
        async with self._position_lock:
            # Double-check after acquiring lock
            if (
                request.symbol in portfolio.positions
                and not portfolio.positions[request.symbol].is_closed()
            ):
                raise ValueError(f"Position already exists for {request.symbol}")

            async with self._cash_lock:
                # Delegate to domain entity for business logic
                position = portfolio.open_position(request)

        return position

    async def close_position(
        self,
        portfolio: Portfolio,
        symbol: str,
        exit_price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> Decimal:
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
        async with self._position_lock:
            async with self._cash_lock:
                async with self._stats_lock:
                    # Delegate to domain entity for business logic
                    return portfolio.close_position(symbol, exit_price, commission)

    async def update_position_price(
        self, portfolio: Portfolio, symbol: str, price: Decimal
    ) -> None:
        """Update market price for a position (thread-safe)

        Args:
            portfolio: Portfolio domain entity
            symbol: Symbol to update price for
            price: New market price
        """
        async with self._position_lock:
            # Delegate to domain entity for business logic
            portfolio.update_position_price(symbol, price)

        async with self._stats_lock:
            # Update portfolio timestamp
            from datetime import UTC, datetime

            portfolio.last_updated = datetime.now(UTC)

    async def update_all_prices(self, portfolio: Portfolio, prices: dict[str, Decimal]) -> None:
        """Update market prices for multiple positions (thread-safe)

        Args:
            portfolio: Portfolio domain entity
            prices: Dictionary of symbol to price mappings
        """
        async with self._position_lock:
            # Delegate to domain entity for business logic
            portfolio.update_all_prices(prices)

        async with self._stats_lock:
            # Update portfolio timestamp
            from datetime import UTC, datetime

            portfolio.last_updated = datetime.now(UTC)

    async def get_total_value(self, portfolio: Portfolio) -> Decimal:
        """Calculate total portfolio value (cash + positions) - thread-safe

        Args:
            portfolio: Portfolio domain entity

        Returns:
            Total portfolio value
        """
        async with self._cash_lock:
            async with self._position_lock:
                # Delegate to domain entity for business logic
                return portfolio.get_total_value()

    def get_total_value_sync(self, portfolio: Portfolio) -> Decimal:
        """Synchronous wrapper for get_total_value

        Args:
            portfolio: Portfolio domain entity

        Returns:
            Total portfolio value
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(asyncio.run, self.get_total_value(portfolio)).result()
            else:
                return asyncio.run(self.get_total_value(portfolio))
        except RuntimeError:
            return asyncio.run(self.get_total_value(portfolio))

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
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.open_position(portfolio, request))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.open_position(portfolio, request))

    def close_position_sync(
        self,
        portfolio: Portfolio,
        symbol: str,
        exit_price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> Decimal:
        """Synchronous wrapper for close_position

        Args:
            portfolio: Portfolio domain entity
            symbol: Symbol of position to close
            exit_price: Exit price for the position
            commission: Commission paid for closing

        Returns:
            Realized P&L from closing the position
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self.close_position(portfolio, symbol, exit_price, commission)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self.close_position(portfolio, symbol, exit_price, commission))

    def update_position_price_sync(self, portfolio: Portfolio, symbol: str, price: Decimal) -> None:
        """Synchronous wrapper for update_position_price

        Args:
            portfolio: Portfolio domain entity
            symbol: Symbol to update price for
            price: New market price
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.submit(
                        asyncio.run, self.update_position_price(portfolio, symbol, price)
                    ).result()
            else:
                asyncio.run(self.update_position_price(portfolio, symbol, price))
        except RuntimeError:
            asyncio.run(self.update_position_price(portfolio, symbol, price))

    def update_all_prices_sync(self, portfolio: Portfolio, prices: dict[str, Decimal]) -> None:
        """Synchronous wrapper for update_all_prices

        Args:
            portfolio: Portfolio domain entity
            prices: Dictionary of symbol to price mappings
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.submit(asyncio.run, self.update_all_prices(portfolio, prices)).result()
            else:
                asyncio.run(self.update_all_prices(portfolio, prices))
        except RuntimeError:
            asyncio.run(self.update_all_prices(portfolio, prices))

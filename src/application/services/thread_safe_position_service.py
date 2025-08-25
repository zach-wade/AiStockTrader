"""
ThreadSafePositionService - Application-layer thread safety for Position operations

Provides thread-safe wrappers around Position domain entity operations
using asyncio locks to coordinate concurrent access.
"""

import asyncio
from decimal import Decimal

from ...domain.entities.position import Position


class ThreadSafePositionService:
    """
    Application service providing thread-safe operations for Position entities.

    This service acts as a wrapper around Position domain entities, adding
    thread-safety coordination through asyncio locks. The domain entities
    themselves remain pure and synchronous.
    """

    def __init__(self) -> None:
        """Initialize thread-safety locks"""
        self._lock = asyncio.Lock()

    async def add_to_position(
        self,
        position: Position,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> None:
        """Add to existing position (same direction) - thread-safe

        Args:
            position: Position domain entity
            quantity: Quantity to add to position
            price: Price for the additional quantity
            commission: Commission paid for the addition
        """
        async with self._lock:
            # Delegate to domain entity for business logic
            position.add_to_position(quantity, price, commission)

    async def reduce_position(
        self,
        position: Position,
        quantity: Decimal,
        exit_price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Reduce position size and calculate realized P&L (thread-safe).

        Args:
            position: Position domain entity
            quantity: Quantity to reduce from position
            exit_price: Exit price for the reduced quantity
            commission: Commission paid for the reduction

        Returns:
            Realized P&L from this reduction
        """
        async with self._lock:
            # Delegate to domain entity for business logic
            return position.reduce_position(quantity, exit_price, commission)

    async def update_market_price(self, position: Position, price: Decimal) -> None:
        """Update current market price (thread-safe)

        Args:
            position: Position domain entity
            price: New market price
        """
        async with self._lock:
            # Delegate to domain entity for business logic
            position.update_market_price(price)

    def add_to_position_sync(
        self,
        position: Position,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> None:
        """Synchronous wrapper for add_to_position

        Args:
            position: Position domain entity
            quantity: Quantity to add to position
            price: Price for the additional quantity
            commission: Commission paid for the addition
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.submit(
                        asyncio.run, self.add_to_position(position, quantity, price, commission)
                    ).result()
            else:
                asyncio.run(self.add_to_position(position, quantity, price, commission))
        except RuntimeError:
            asyncio.run(self.add_to_position(position, quantity, price, commission))

    def reduce_position_sync(
        self,
        position: Position,
        quantity: Decimal,
        exit_price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> Decimal:
        """Synchronous wrapper for reduce_position

        Args:
            position: Position domain entity
            quantity: Quantity to reduce from position
            exit_price: Exit price for the reduced quantity
            commission: Commission paid for the reduction

        Returns:
            Realized P&L from this reduction
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(
                        asyncio.run,
                        self.reduce_position(position, quantity, exit_price, commission),
                    ).result()
            else:
                return asyncio.run(self.reduce_position(position, quantity, exit_price, commission))
        except RuntimeError:
            return asyncio.run(self.reduce_position(position, quantity, exit_price, commission))

    def update_market_price_sync(self, position: Position, price: Decimal) -> None:
        """Synchronous wrapper for update_market_price

        Args:
            position: Position domain entity
            price: New market price
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.submit(asyncio.run, self.update_market_price(position, price)).result()
            else:
                asyncio.run(self.update_market_price(position, price))
        except RuntimeError:
            asyncio.run(self.update_market_price(position, price))

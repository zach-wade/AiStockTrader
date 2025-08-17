"""
Position Tracker

Focused position state management and tracking.
Handles position state updates, price updates, and position queries.
"""

# Standard library imports
import asyncio
from datetime import datetime
from decimal import Decimal
import logging

# Local imports
from main.models.common import Position
from main.utils.cache import CacheType
from main.utils.database import execute_query

logger = logging.getLogger(__name__)


class PositionTracker:
    """
    Focused position state management and tracking.

    Responsibilities:
    - Track current position states
    - Update position prices
    - Provide position queries
    - Manage position persistence
    """

    def __init__(self, use_cache: bool = True, use_db: bool = True):
        """
        Initialize position tracker.

        Args:
            use_cache: Whether to use caching for performance
            use_db: Whether to persist positions to database
        """
        # Position state storage
        self.positions: dict[str, Position] = {}

        # Position history tracking
        self.position_history: list[dict] = []

        # Configuration
        self.use_cache = use_cache
        self.use_db = use_db

        # Cache setup
        self.cache = get_global_cache(CacheType.MEMORY) if use_cache else None

        # Thread safety
        self._lock = asyncio.Lock()

        # Performance tracking
        self.last_update_time = datetime.now()
        self.update_count = 0

        logger.info("✅ PositionTracker initialized")

    async def update_position(self, position: Position) -> Position | None:
        """
        Update a position in the tracker.

        Args:
            position: Updated position object

        Returns:
            Previous position state (if any)
        """
        async with self._lock:
            try:
                old_position = self.positions.get(position.symbol)

                # Update position
                self.positions[position.symbol] = position

                # Update cache
                if self.cache:
                    await self._update_cache(position)

                # Persist to database
                if self.use_db:
                    await self._persist_position(position)

                # Track update
                self.update_count += 1
                self.last_update_time = datetime.now()

                # Add to history
                self._add_to_history(position, old_position)

                logger.debug(
                    f"Updated position {position.symbol}: {position.quantity} @ {position.current_price}"
                )

                return old_position

            except Exception as e:
                logger.error(f"Error updating position {position.symbol}: {e}")
                return None

    async def update_positions(self, positions: dict[str, Position]) -> dict[str, Position | None]:
        """
        Update multiple positions atomically.

        Args:
            positions: Dictionary of symbol -> Position

        Returns:
            Dictionary of symbol -> previous position state
        """
        old_positions = {}

        async with self._lock:
            for symbol, position in positions.items():
                old_positions[symbol] = await self.update_position(position)

        return old_positions

    async def update_prices(self, price_updates: dict[str, Decimal]) -> list[Position]:
        """
        Update current prices for existing positions.

        Args:
            price_updates: Dictionary of symbol -> new price

        Returns:
            List of updated positions
        """
        updated_positions = []

        async with self._lock:
            for symbol, new_price in price_updates.items():
                if symbol in self.positions:
                    old_position = self.positions[symbol]

                    # Create updated position with new price
                    updated_position = Position(
                        symbol=old_position.symbol,
                        quantity=old_position.quantity,
                        avg_entry_price=old_position.avg_entry_price,
                        current_price=float(new_price),
                        market_value=old_position.quantity * float(new_price),
                        cost_basis=old_position.cost_basis,
                        unrealized_pnl=(float(new_price) - old_position.avg_entry_price)
                        * old_position.quantity,
                        unrealized_pnl_pct=(
                            ((float(new_price) / old_position.avg_entry_price) - 1) * 100
                            if old_position.avg_entry_price > 0
                            else 0
                        ),
                        realized_pnl=old_position.realized_pnl,
                        side=old_position.side,
                        timestamp=datetime.now(),
                    )

                    # Update position
                    self.positions[symbol] = updated_position
                    updated_positions.append(updated_position)

                    # Update cache
                    if self.cache:
                        await self._update_cache(updated_position)

                    logger.debug(f"Updated price for {symbol}: {new_price}")

        return updated_positions

    async def remove_position(self, symbol: str) -> Position | None:
        """
        Remove a position from tracking.

        Args:
            symbol: Symbol to remove

        Returns:
            Removed position (if any)
        """
        async with self._lock:
            try:
                removed_position = self.positions.pop(symbol, None)

                if removed_position:
                    # Remove from cache
                    if self.cache:
                        await self._remove_from_cache(symbol)

                    # Mark as closed in database
                    if self.use_db:
                        await self._mark_position_closed(symbol)

                    logger.debug(f"Removed position {symbol}")

                return removed_position

            except Exception as e:
                logger.error(f"Error removing position {symbol}: {e}")
                return None

    def get_position(self, symbol: str) -> Position | None:
        """Get a specific position."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Get all tracked positions."""
        return self.positions.copy()

    def get_positions_by_side(self, side: str) -> dict[str, Position]:
        """Get positions by side (long/short)."""
        return {
            symbol: position for symbol, position in self.positions.items() if position.side == side
        }

    def get_symbols(self) -> set[str]:
        """Get all tracked symbols."""
        return set(self.positions.keys())

    def get_position_count(self) -> int:
        """Get total number of positions."""
        return len(self.positions)

    def get_total_market_value(self) -> Decimal:
        """Get total market value of all positions."""
        return Decimal(str(sum(position.market_value for position in self.positions.values())))

    def get_total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L."""
        return Decimal(str(sum(position.unrealized_pnl for position in self.positions.values())))

    def get_position_metrics(self) -> dict[str, float]:
        """Get position tracking metrics."""
        long_positions = self.get_positions_by_side("long")
        short_positions = self.get_positions_by_side("short")

        total_long_value = sum(pos.market_value for pos in long_positions.values())
        total_short_value = sum(abs(pos.market_value) for pos in short_positions.values())

        return {
            "total_positions": len(self.positions),
            "long_positions": len(long_positions),
            "short_positions": len(short_positions),
            "total_market_value": float(self.get_total_market_value()),
            "total_unrealized_pnl": float(self.get_total_unrealized_pnl()),
            "long_exposure": total_long_value,
            "short_exposure": total_short_value,
            "net_exposure": total_long_value - total_short_value,
            "gross_exposure": total_long_value + total_short_value,
            "update_count": self.update_count,
            "last_update": self.last_update_time.isoformat(),
        }

    async def _update_cache(self, position: Position):
        """Update position in cache."""
        if not self.cache:
            return

        try:
            cache_key = f"position:{position.symbol}"
            cache_data = {
                "symbol": position.symbol,
                "quantity": position.quantity,
                "avg_entry_price": position.avg_entry_price,
                "current_price": position.current_price,
                "market_value": position.market_value,
                "unrealized_pnl": position.unrealized_pnl,
                "timestamp": position.timestamp.isoformat(),
            }

            await self.cache.set(cache_key, cache_data, ttl=300)  # 5 minute TTL

        except Exception as e:
            logger.error(f"Error updating cache for {position.symbol}: {e}")

    async def _remove_from_cache(self, symbol: str):
        """Remove position from cache."""
        if not self.cache:
            return

        try:
            cache_key = f"position:{symbol}"
            await self.cache.delete(cache_key)

        except Exception as e:
            logger.error(f"Error removing {symbol} from cache: {e}")

    async def _persist_position(self, position: Position):
        """Persist position to database."""
        if not self.use_db:
            return

        try:
            query = """
                INSERT INTO positions (symbol, quantity, avg_entry_price, current_price,
                                     market_value, unrealized_pnl, side, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_entry_price = excluded.avg_entry_price,
                    current_price = excluded.current_price,
                    market_value = excluded.market_value,
                    unrealized_pnl = excluded.unrealized_pnl,
                    side = excluded.side,
                    timestamp = excluded.timestamp
            """

            params = (
                position.symbol,
                position.quantity,
                position.avg_entry_price,
                position.current_price,
                position.market_value,
                position.unrealized_pnl,
                position.side,
                position.timestamp,
            )

            await execute_query(query, params)

        except Exception as e:
            logger.error(f"Error persisting position {position.symbol}: {e}")

    async def _mark_position_closed(self, symbol: str):
        """Mark position as closed in database."""
        if not self.use_db:
            return

        try:
            query = "UPDATE positions SET status = 'closed', closed_at = ? WHERE symbol = ?"
            await execute_query(query, (datetime.now(), symbol))

        except Exception as e:
            logger.error(f"Error marking position {symbol} as closed: {e}")

    def _add_to_history(self, position: Position, old_position: Position | None):
        """Add position update to history."""
        try:
            history_entry = {
                "symbol": position.symbol,
                "timestamp": position.timestamp.isoformat(),
                "quantity": position.quantity,
                "current_price": position.current_price,
                "market_value": position.market_value,
                "unrealized_pnl": position.unrealized_pnl,
                "old_quantity": old_position.quantity if old_position else 0,
                "old_price": old_position.current_price if old_position else 0,
            }

            self.position_history.append(history_entry)

            # Keep only last 1000 entries
            if len(self.position_history) > 1000:
                self.position_history = self.position_history[-500:]

        except Exception as e:
            logger.error(f"Error adding to history: {e}")

    async def load_positions_from_db(self) -> dict[str, Position]:
        """Load positions from database on startup."""
        if not self.use_db:
            return {}

        try:
            query = """
                SELECT symbol, quantity, avg_entry_price, current_price, market_value,
                       cost_basis, unrealized_pnl, unrealized_pnl_pct, realized_pnl,
                       side, timestamp
                FROM positions
                WHERE status = 'open'
            """

            rows = await fetch_all(query)
            positions = {}

            for row in rows:
                position = Position(
                    symbol=row["symbol"],
                    quantity=row["quantity"],
                    avg_entry_price=row["avg_entry_price"],
                    current_price=row["current_price"],
                    market_value=row["market_value"],
                    cost_basis=row["cost_basis"],
                    unrealized_pnl=row["unrealized_pnl"],
                    unrealized_pnl_pct=row["unrealized_pnl_pct"],
                    realized_pnl=row["realized_pnl"],
                    side=row["side"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
                positions[position.symbol] = position

            self.positions = positions
            logger.info(f"Loaded {len(positions)} positions from database")
            return positions

        except Exception as e:
            logger.error(f"Error loading positions from database: {e}")
            return {}

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Final persistence
            if self.use_db:
                for position in self.positions.values():
                    await self._persist_position(position)

            # Clear cache
            if self.cache:
                for symbol in self.positions:
                    await self._remove_from_cache(symbol)

            logger.info("✅ PositionTracker cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

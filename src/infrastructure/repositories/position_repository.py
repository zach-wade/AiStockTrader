"""
PostgreSQL Position Repository Implementation

Concrete implementation of IPositionRepository using PostgreSQL database.
Handles position persistence, retrieval, and mapping between domain entities and database records.
"""

# Standard library imports
from datetime import datetime
import logging
from uuid import UUID

# Third-party imports
from psycopg.rows import Row

# Local imports
from src.application.interfaces.exceptions import PositionNotFoundError, RepositoryError
from src.application.interfaces.repositories import IPositionRepository
from src.domain.entities.position import Position
from src.infrastructure.database.adapter import PostgreSQLAdapter

logger = logging.getLogger(__name__)


class PostgreSQLPositionRepository(IPositionRepository):
    """
    PostgreSQL implementation of IPositionRepository.

    Provides position persistence and retrieval using PostgreSQL database.
    Maps between Position domain entities and database records.
    """

    def __init__(self, adapter: PostgreSQLAdapter) -> None:
        """
        Initialize repository with database adapter.

        Args:
            adapter: PostgreSQL database adapter
        """
        self.adapter = adapter

    async def save_position(self, position: Position) -> Position:
        """
        Save a new position or update an existing position.

        Args:
            position: The position entity to save

        Returns:
            The saved position entity

        Raises:
            RepositoryError: If save operation fails
        """
        try:
            # Check if position exists
            existing_position = await self.get_position_by_id(position.id)

            if existing_position:
                return await self.update_position(position)
            else:
                return await self._insert_position(position)

        except Exception as e:
            logger.error(f"Failed to save position {position.id}: {e}")
            raise RepositoryError(f"Failed to save position: {e}") from e

    async def _insert_position(self, position: Position) -> Position:
        """Insert a new position into the database."""
        insert_query = """
        INSERT INTO positions (
            id, symbol, quantity, average_entry_price, current_price,
            last_updated, realized_pnl, commission_paid, stop_loss_price,
            take_profit_price, max_position_value, opened_at, closed_at,
            strategy, tags
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        await self.adapter.execute_query(
            insert_query,
            position.id,
            position.symbol,
            position.quantity,
            position.average_entry_price,
            position.current_price,
            position.last_updated,
            position.realized_pnl,
            position.commission_paid,
            position.stop_loss_price,
            position.take_profit_price,
            position.max_position_value,
            position.opened_at,
            position.closed_at,
            position.strategy,
            position.tags,
        )

        logger.debug(f"Inserted position {position.id}")
        return position

    async def get_position_by_id(self, position_id: UUID) -> Position | None:
        """
        Retrieve a position by its ID.

        Args:
            position_id: The unique identifier of the position

        Returns:
            The position entity if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, symbol, quantity, average_entry_price, current_price,
                   last_updated, realized_pnl, commission_paid, stop_loss_price,
                   take_profit_price, max_position_value, opened_at, closed_at,
                   strategy, tags
            FROM positions
            WHERE id = %s
            """

            record = await self.adapter.fetch_one(query, position_id)

            if record is None:
                return None

            return self._map_record_to_position(record)

        except Exception as e:
            logger.error(f"Failed to get position {position_id}: {e}")
            raise RepositoryError(f"Failed to retrieve position: {e}") from e

    async def get_position_by_symbol(self, symbol: str) -> Position | None:
        """
        Retrieve the current position for a symbol.

        Args:
            symbol: The trading symbol (e.g., 'AAPL')

        Returns:
            The position entity if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, symbol, quantity, average_entry_price, current_price,
                   last_updated, realized_pnl, commission_paid, stop_loss_price,
                   take_profit_price, max_position_value, opened_at, closed_at,
                   strategy, tags
            FROM positions
            WHERE symbol = %s AND closed_at IS NULL
            ORDER BY opened_at DESC
            LIMIT 1
            """

            record = await self.adapter.fetch_one(query, symbol)

            if record is None:
                return None

            return self._map_record_to_position(record)

        except Exception as e:
            logger.error(f"Failed to get position for symbol {symbol}: {e}")
            raise RepositoryError(f"Failed to retrieve position for symbol: {e}") from e

    async def get_positions_by_symbol(self, symbol: str) -> list[Position]:
        """
        Retrieve all positions (including historical) for a symbol.

        Args:
            symbol: The trading symbol (e.g., 'AAPL')

        Returns:
            List of positions for the symbol, empty if none found

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, symbol, quantity, average_entry_price, current_price,
                   last_updated, realized_pnl, commission_paid, stop_loss_price,
                   take_profit_price, max_position_value, opened_at, closed_at,
                   strategy, tags
            FROM positions
            WHERE symbol = %s
            ORDER BY opened_at DESC
            """

            records = await self.adapter.fetch_all(query, symbol)
            return [self._map_record_to_position(record) for record in records]

        except Exception as e:
            logger.error(f"Failed to get positions for symbol {symbol}: {e}")
            raise RepositoryError(f"Failed to retrieve positions for symbol: {e}") from e

    async def get_active_positions(self) -> list[Position]:
        """
        Retrieve all currently open positions.

        Returns:
            List of open positions

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, symbol, quantity, average_entry_price, current_price,
                   last_updated, realized_pnl, commission_paid, stop_loss_price,
                   take_profit_price, max_position_value, opened_at, closed_at,
                   strategy, tags
            FROM positions
            WHERE closed_at IS NULL
            ORDER BY opened_at DESC
            """

            records = await self.adapter.fetch_all(query)
            return [self._map_record_to_position(record) for record in records]

        except Exception as e:
            logger.error(f"Failed to get active positions: {e}")
            raise RepositoryError(f"Failed to retrieve active positions: {e}") from e

    async def get_closed_positions(self) -> list[Position]:
        """
        Retrieve all closed positions.

        Returns:
            List of closed positions

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, symbol, quantity, average_entry_price, current_price,
                   last_updated, realized_pnl, commission_paid, stop_loss_price,
                   take_profit_price, max_position_value, opened_at, closed_at,
                   strategy, tags
            FROM positions
            WHERE closed_at IS NOT NULL
            ORDER BY closed_at DESC
            """

            records = await self.adapter.fetch_all(query)
            return [self._map_record_to_position(record) for record in records]

        except Exception as e:
            logger.error(f"Failed to get closed positions: {e}")
            raise RepositoryError(f"Failed to retrieve closed positions: {e}") from e

    async def get_positions_by_strategy(self, strategy: str) -> list[Position]:
        """
        Retrieve positions associated with a specific strategy.

        Args:
            strategy: The strategy name

        Returns:
            List of positions for the strategy

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, symbol, quantity, average_entry_price, current_price,
                   last_updated, realized_pnl, commission_paid, stop_loss_price,
                   take_profit_price, max_position_value, opened_at, closed_at,
                   strategy, tags
            FROM positions
            WHERE strategy = %s
            ORDER BY opened_at DESC
            """

            records = await self.adapter.fetch_all(query, strategy)
            return [self._map_record_to_position(record) for record in records]

        except Exception as e:
            logger.error(f"Failed to get positions for strategy {strategy}: {e}")
            raise RepositoryError(f"Failed to retrieve positions for strategy: {e}") from e

    async def get_positions_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[Position]:
        """
        Retrieve positions opened within a date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of positions in date range

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, symbol, quantity, average_entry_price, current_price,
                   last_updated, realized_pnl, commission_paid, stop_loss_price,
                   take_profit_price, max_position_value, opened_at, closed_at,
                   strategy, tags
            FROM positions
            WHERE opened_at >= %s AND opened_at <= %s
            ORDER BY opened_at DESC
            """

            records = await self.adapter.fetch_all(query, start_date, end_date)
            return [self._map_record_to_position(record) for record in records]

        except Exception as e:
            logger.error(f"Failed to get positions by date range: {e}")
            raise RepositoryError(f"Failed to retrieve positions by date range: {e}") from e

    async def update_position(self, position: Position) -> Position:
        """
        Update an existing position.

        Args:
            position: The position entity to update

        Returns:
            The updated position entity

        Raises:
            RepositoryError: If update operation fails
            PositionNotFoundError: If position doesn't exist
        """
        try:
            update_query = """
            UPDATE positions SET
                symbol = %s, quantity = %s, average_entry_price = %s,
                current_price = %s, last_updated = %s, realized_pnl = %s,
                commission_paid = %s, stop_loss_price = %s, take_profit_price = %s,
                max_position_value = %s, closed_at = %s, strategy = %s, tags = %s
            WHERE id = %s
            """

            result = await self.adapter.execute_query(
                update_query,
                position.symbol,
                position.quantity,
                position.average_entry_price,
                position.current_price,
                position.last_updated,
                position.realized_pnl,
                position.commission_paid,
                position.stop_loss_price,
                position.take_profit_price,
                position.max_position_value,
                position.closed_at,
                position.strategy,
                position.tags,
                position.id,  # id moved to end for WHERE clause
            )

            if "UPDATE 0" in result:
                raise PositionNotFoundError(position.id)

            logger.debug(f"Updated position {position.id}")
            return position

        except PositionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update position {position.id}: {e}")
            raise RepositoryError(f"Failed to update position: {e}") from e

    async def close_position(self, position_id: UUID) -> bool:
        """
        Mark a position as closed.

        Args:
            position_id: The unique identifier of the position

        Returns:
            True if position was closed, False if not found

        Raises:
            RepositoryError: If close operation fails
        """
        try:
            close_query = """
            UPDATE positions SET
                closed_at = NOW(),
                quantity = 0
            WHERE id = %s AND closed_at IS NULL
            """

            result = await self.adapter.execute_query(close_query, position_id)
            success = "UPDATE 1" in result

            if success:
                logger.debug(f"Closed position {position_id}")
            else:
                logger.debug(f"Position {position_id} not found or already closed")

            return success

        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}")
            raise RepositoryError(f"Failed to close position: {e}") from e

    async def delete_position(self, position_id: UUID) -> bool:
        """
        Delete a position by ID.

        Args:
            position_id: The unique identifier of the position

        Returns:
            True if position was deleted, False if not found

        Raises:
            RepositoryError: If delete operation fails
        """
        try:
            delete_query = "DELETE FROM positions WHERE id = %s"
            result = await self.adapter.execute_query(delete_query, position_id)

            success = "DELETE 1" in result

            if success:
                logger.debug(f"Deleted position {position_id}")
            else:
                logger.debug(f"Position {position_id} not found for deletion")

            return success

        except Exception as e:
            logger.error(f"Failed to delete position {position_id}: {e}")
            raise RepositoryError(f"Failed to delete position: {e}") from e

    def _map_record_to_position(self, record: Row) -> Position:
        """
        Map database record to Position entity.

        Args:
            record: Database record

        Returns:
            Position entity
        """
        return Position(
            id=record["id"],
            symbol=record["symbol"],
            quantity=record["quantity"],
            average_entry_price=record["average_entry_price"],
            current_price=record["current_price"],
            last_updated=record["last_updated"],
            realized_pnl=record["realized_pnl"],
            commission_paid=record["commission_paid"],
            stop_loss_price=record["stop_loss_price"],
            take_profit_price=record["take_profit_price"],
            max_position_value=record["max_position_value"],
            opened_at=record["opened_at"],
            closed_at=record["closed_at"],
            strategy=record["strategy"],
            tags=record["tags"] or {},
        )

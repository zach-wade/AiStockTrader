"""
PostgreSQL Portfolio Repository Implementation

Concrete implementation of IPortfolioRepository using PostgreSQL database.
Handles portfolio persistence, retrieval, and mapping between domain entities and database records.
Implements optimistic locking for concurrent operations safety.
"""

# Standard library imports
import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

# Local imports
from src.application.interfaces.exceptions import (
    ConcurrencyError,
    PortfolioNotFoundError,
    RepositoryError,
)
from src.application.interfaces.repositories import IPortfolioRepository
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.exceptions import DeadlockException, OptimisticLockException, StaleDataException
from src.infrastructure.database.adapter import PostgreSQLAdapter, Row
from src.infrastructure.resilience.concurrency_service import ConcurrencyService

logger = logging.getLogger(__name__)


# OptimisticLockException is now imported from domain.exceptions


class PostgreSQLPortfolioRepository(IPortfolioRepository):
    """
    PostgreSQL implementation of IPortfolioRepository.

    Provides portfolio persistence and retrieval using PostgreSQL database.
    Maps between Portfolio domain entities and database records.
    Implements optimistic locking with version numbers for concurrent safety.
    Note: Positions are stored separately and loaded as needed.
    """

    def __init__(
        self,
        adapter: PostgreSQLAdapter,
        max_retries: int = 3,
        concurrency_service: ConcurrencyService | None = None,
    ) -> None:
        """
        Initialize repository with database adapter.

        Args:
            adapter: PostgreSQL database adapter
            max_retries: Maximum retries for optimistic lock conflicts
            concurrency_service: Optional concurrency service for advanced locking
        """
        self.adapter = adapter
        self.max_retries = max_retries
        self._version_lock = asyncio.Lock()
        self.concurrency_service = concurrency_service or ConcurrencyService(
            max_retries=max_retries
        )

    async def save_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """
        Save a new portfolio or update an existing portfolio.

        Args:
            portfolio: The portfolio entity to save

        Returns:
            The saved portfolio entity

        Raises:
            RepositoryError: If save operation fails
        """
        try:
            # Check if portfolio exists
            existing_portfolio = await self.get_portfolio_by_id(portfolio.id)

            if existing_portfolio:
                return await self.update_portfolio(portfolio)
            else:
                return await self._insert_portfolio(portfolio)

        except Exception as e:
            logger.error(f"Failed to save portfolio {portfolio.id}: {e}")
            raise RepositoryError(f"Failed to save portfolio: {e}") from e

    async def _insert_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """Insert a new portfolio into the database with initial version."""
        # Set initial version
        if not hasattr(portfolio, "version"):
            portfolio.version = 1

        insert_query = """
        INSERT INTO portfolios (
            id, name, initial_capital, cash_balance, max_position_size,
            max_portfolio_risk, max_positions, max_leverage, total_realized_pnl,
            total_commission_paid, trades_count, winning_trades, losing_trades,
            created_at, last_updated, strategy, tags, version
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        await self.adapter.execute_query(
            insert_query,
            portfolio.id,
            portfolio.name,
            portfolio.initial_capital,
            portfolio.cash_balance,
            portfolio.max_position_size,
            portfolio.max_portfolio_risk,
            portfolio.max_positions,
            portfolio.max_leverage,
            portfolio.total_realized_pnl,
            portfolio.total_commission_paid,
            portfolio.trades_count,
            portfolio.winning_trades,
            portfolio.losing_trades,
            portfolio.created_at,
            portfolio.last_updated,
            portfolio.strategy,
            portfolio.tags,
            1,  # Initial version
        )

        logger.debug(f"Inserted portfolio {portfolio.id} with version 1")
        return portfolio

    async def get_portfolio_by_id(self, portfolio_id: UUID) -> Portfolio | None:
        """
        Retrieve a portfolio by its ID.

        Args:
            portfolio_id: The unique identifier of the portfolio

        Returns:
            The portfolio entity if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, name, initial_capital, cash_balance, max_position_size,
                   max_portfolio_risk, max_positions, max_leverage, total_realized_pnl,
                   total_commission_paid, trades_count, winning_trades, losing_trades,
                   created_at, last_updated, strategy, tags, COALESCE(version, 1) as version
            FROM portfolios
            WHERE id = %s
            """

            record = await self.adapter.fetch_one(query, portfolio_id)

            if record is None:
                return None

            portfolio = self._map_record_to_portfolio(record)

            # Load positions for this portfolio
            await self._load_positions(portfolio)

            return portfolio

        except Exception as e:
            logger.error(f"Failed to get portfolio {portfolio_id}: {e}")
            raise RepositoryError(f"Failed to retrieve portfolio: {e}") from e

    async def get_portfolio_by_name(self, name: str) -> Portfolio | None:
        """
        Retrieve a portfolio by its name.

        Args:
            name: The portfolio name

        Returns:
            The portfolio entity if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, name, initial_capital, cash_balance, max_position_size,
                   max_portfolio_risk, max_positions, max_leverage, total_realized_pnl,
                   total_commission_paid, trades_count, winning_trades, losing_trades,
                   created_at, last_updated, strategy, tags, COALESCE(version, 1) as version
            FROM portfolios
            WHERE name = %s
            """

            record = await self.adapter.fetch_one(query, name)

            if record is None:
                return None

            portfolio = self._map_record_to_portfolio(record)

            # Load positions for this portfolio
            await self._load_positions(portfolio)

            return portfolio

        except Exception as e:
            logger.error(f"Failed to get portfolio by name {name}: {e}")
            raise RepositoryError(f"Failed to retrieve portfolio by name: {e}") from e

    async def get_current_portfolio(self) -> Portfolio | None:
        """
        Retrieve the current active portfolio.

        Returns:
            The current portfolio entity if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            # For now, return the most recently updated portfolio
            # In a real system, you might have an "active" flag or similar
            query = """
            SELECT id, name, initial_capital, cash_balance, max_position_size,
                   max_portfolio_risk, max_positions, max_leverage, total_realized_pnl,
                   total_commission_paid, trades_count, winning_trades, losing_trades,
                   created_at, last_updated, strategy, tags, COALESCE(version, 1) as version
            FROM portfolios
            ORDER BY COALESCE(last_updated, created_at) DESC
            LIMIT 1
            """

            record = await self.adapter.fetch_one(query)

            if record is None:
                return None

            portfolio = self._map_record_to_portfolio(record)

            # Load positions for this portfolio
            await self._load_positions(portfolio)

            return portfolio

        except Exception as e:
            logger.error(f"Failed to get current portfolio: {e}")
            raise RepositoryError(f"Failed to retrieve current portfolio: {e}") from e

    async def get_all_portfolios(self) -> list[Portfolio]:
        """
        Retrieve all portfolios.

        Returns:
            List of all portfolios

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, name, initial_capital, cash_balance, max_position_size,
                   max_portfolio_risk, max_positions, max_leverage, total_realized_pnl,
                   total_commission_paid, trades_count, winning_trades, losing_trades,
                   created_at, last_updated, strategy, tags, COALESCE(version, 1) as version
            FROM portfolios
            ORDER BY created_at DESC
            """

            records = await self.adapter.fetch_all(query)
            portfolios = [self._map_record_to_portfolio(record) for record in records]

            # Load positions for all portfolios
            for portfolio in portfolios:
                await self._load_positions(portfolio)

            return portfolios

        except Exception as e:
            logger.error(f"Failed to get all portfolios: {e}")
            raise RepositoryError(f"Failed to retrieve all portfolios: {e}") from e

    async def get_portfolios_by_strategy(self, strategy: str) -> list[Portfolio]:
        """
        Retrieve portfolios associated with a specific strategy.

        Args:
            strategy: The strategy name

        Returns:
            List of portfolios for the strategy

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, name, initial_capital, cash_balance, max_position_size,
                   max_portfolio_risk, max_positions, max_leverage, total_realized_pnl,
                   total_commission_paid, trades_count, winning_trades, losing_trades,
                   created_at, last_updated, strategy, tags, COALESCE(version, 1) as version
            FROM portfolios
            WHERE strategy = %s
            ORDER BY created_at DESC
            """

            records = await self.adapter.fetch_all(query, strategy)
            portfolios = [self._map_record_to_portfolio(record) for record in records]

            # Load positions for all portfolios
            for portfolio in portfolios:
                await self._load_positions(portfolio)

            return portfolios

        except Exception as e:
            logger.error(f"Failed to get portfolios for strategy {strategy}: {e}")
            raise RepositoryError(f"Failed to retrieve portfolios for strategy: {e}") from e

    async def get_portfolio_history(
        self, portfolio_id: UUID, start_date: datetime, end_date: datetime
    ) -> list[Portfolio]:
        """
        Retrieve historical snapshots of a portfolio.

        Args:
            portfolio_id: The portfolio identifier
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of portfolio snapshots in date range

        Note:
            This is a simplified implementation. In a real system, you might
            have a separate table for portfolio snapshots or audit history.

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            # For now, just return the current portfolio if it was last updated in the range
            query = """
            SELECT id, name, initial_capital, cash_balance, max_position_size,
                   max_portfolio_risk, max_positions, max_leverage, total_realized_pnl,
                   total_commission_paid, trades_count, winning_trades, losing_trades,
                   created_at, last_updated, strategy, tags, COALESCE(version, 1) as version
            FROM portfolios
            WHERE id = %s
              AND (last_updated BETWEEN %s AND %s OR created_at BETWEEN %s AND %s)
            ORDER BY COALESCE(last_updated, created_at) DESC
            """

            records = await self.adapter.fetch_all(
                query, portfolio_id, start_date, end_date, start_date, end_date
            )
            portfolios = [self._map_record_to_portfolio(record) for record in records]

            # Load positions for all portfolios
            for portfolio in portfolios:
                await self._load_positions(portfolio)

            return portfolios

        except Exception as e:
            logger.error(f"Failed to get portfolio history: {e}")
            raise RepositoryError(f"Failed to retrieve portfolio history: {e}") from e

    async def update_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """
        Update an existing portfolio with optimistic locking.

        Args:
            portfolio: The portfolio entity to update

        Returns:
            The updated portfolio entity

        Raises:
            RepositoryError: If update operation fails
            PortfolioNotFoundError: If portfolio doesn't exist
            OptimisticLockException: If version conflict occurs
        """

        async def _do_update() -> Portfolio:
            async with self._version_lock:
                # Get current version from portfolio
                current_version = getattr(portfolio, "version", 1)

                # Note: The portfolio entity should have already incremented its version
                # via _increment_version() when its state was modified
                expected_new_version = current_version

                # For backward compatibility, if version wasn't incremented, do it now
                if portfolio.last_updated and portfolio.last_updated > (
                    portfolio.created_at
                    if hasattr(portfolio, "created_at")
                    else datetime.min.replace(tzinfo=UTC)
                ):
                    # Portfolio was modified but version might not have been incremented
                    if not hasattr(portfolio, "_version_incremented"):
                        expected_new_version = current_version + 1
                else:
                    expected_new_version = current_version

                # Update with version check using SELECT FOR UPDATE for pessimistic locking
                # This prevents other transactions from modifying the row
                lock_query = """
                SELECT version FROM portfolios
                WHERE id = %s
                FOR UPDATE NOWAIT
                """

                try:
                    lock_result = await self.adapter.fetch_one(lock_query, portfolio.id)
                except Exception as e:
                    if "could not obtain lock" in str(e).lower():
                        # Another transaction has the lock
                        raise StaleDataException(
                            entity_type="Portfolio",
                            entity_id=portfolio.id,
                            expected_version=current_version,
                            actual_version=None,
                        )
                    raise

                if lock_result is None:
                    raise PortfolioNotFoundError(portfolio.id)

                db_version = lock_result["version"]

                # Check if the database version matches what we expect
                if db_version != current_version:
                    raise StaleDataException(
                        entity_type="Portfolio",
                        entity_id=portfolio.id,
                        expected_version=current_version,
                        actual_version=db_version,
                    )

                # Now perform the update
                update_query = """
                UPDATE portfolios SET
                    name = %s, initial_capital = %s, cash_balance = %s,
                    max_position_size = %s, max_portfolio_risk = %s, max_positions = %s,
                    max_leverage = %s, total_realized_pnl = %s, total_commission_paid = %s,
                    trades_count = %s, winning_trades = %s, losing_trades = %s,
                    last_updated = %s, strategy = %s, tags = %s, version = %s
                WHERE id = %s AND version = %s
                RETURNING version
                """

                result = await self.adapter.execute_query(
                    update_query,
                    portfolio.name,
                    portfolio.initial_capital,
                    portfolio.cash_balance,
                    portfolio.max_position_size,
                    portfolio.max_portfolio_risk,
                    portfolio.max_positions,
                    portfolio.max_leverage,
                    portfolio.total_realized_pnl,
                    portfolio.total_commission_paid,
                    portfolio.trades_count,
                    portfolio.winning_trades,
                    portfolio.losing_trades,
                    portfolio.last_updated or datetime.now(UTC),
                    portfolio.strategy,
                    portfolio.tags,
                    expected_new_version,  # New version
                    portfolio.id,
                    current_version,  # Check current version
                )

                if "UPDATE 0" in result:
                    # This shouldn't happen due to our lock, but handle it anyway
                    raise StaleDataException(
                        entity_type="Portfolio",
                        entity_id=portfolio.id,
                        expected_version=current_version,
                        actual_version=None,
                    )

                # Update successful
                portfolio.version = expected_new_version
                logger.debug(f"Updated portfolio {portfolio.id} to version {expected_new_version}")
                return portfolio

        # Use the concurrency service for retry logic
        try:
            return await self.concurrency_service.async_retry_on_version_conflict(_do_update)
        except OptimisticLockException:
            raise
        except (PortfolioNotFoundError, StaleDataException):
            raise
        except Exception as e:
            # Check if it's a deadlock
            if self.concurrency_service.detect_deadlock(e):
                raise DeadlockException(
                    message=f"Deadlock detected updating portfolio {portfolio.id}",
                    entities=[("Portfolio", portfolio.id)],
                )
            logger.error(f"Failed to update portfolio {portfolio.id}: {e}")
            raise RepositoryError(f"Failed to update portfolio: {e}") from e

    async def delete_portfolio(self, portfolio_id: UUID) -> bool:
        """
        Delete a portfolio by ID.

        Args:
            portfolio_id: The unique identifier of the portfolio

        Returns:
            True if portfolio was deleted, False if not found

        Raises:
            RepositoryError: If delete operation fails
        """
        try:
            delete_query = "DELETE FROM portfolios WHERE id = %s"
            result = await self.adapter.execute_query(delete_query, portfolio_id)

            success = "DELETE 1" in result

            if success:
                logger.debug(f"Deleted portfolio {portfolio_id}")
            else:
                logger.debug(f"Portfolio {portfolio_id} not found for deletion")

            return success

        except Exception as e:
            logger.error(f"Failed to delete portfolio {portfolio_id}: {e}")
            raise RepositoryError(f"Failed to delete portfolio: {e}") from e

    async def get_portfolio_for_update(self, portfolio_id: UUID) -> Portfolio | None:
        """
        Retrieve a portfolio with pessimistic locking (SELECT FOR UPDATE).

        This method acquires a row-level lock on the portfolio, preventing
        other transactions from modifying it until the current transaction
        completes.

        Args:
            portfolio_id: The unique identifier of the portfolio

        Returns:
            The locked portfolio entity if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
            ConcurrencyError: If unable to acquire lock
        """
        try:
            # Use SELECT FOR UPDATE to lock the row
            query = """
            SELECT id, name, initial_capital, cash_balance, max_position_size,
                   max_portfolio_risk, max_positions, max_leverage, total_realized_pnl,
                   total_commission_paid, trades_count, winning_trades, losing_trades,
                   created_at, last_updated, strategy, tags, COALESCE(version, 1) as version
            FROM portfolios
            WHERE id = %s
            FOR UPDATE NOWAIT
            """

            try:
                record = await self.adapter.fetch_one(query, portfolio_id)
            except Exception as e:
                if "could not obtain lock" in str(e).lower():
                    raise ConcurrencyError(
                        entity_type="Portfolio",
                        identifier=portfolio_id,
                    )
                raise

            if record is None:
                return None

            portfolio = self._map_record_to_portfolio(record)

            # Load positions for this portfolio
            await self._load_positions(portfolio)

            return portfolio

        except ConcurrencyError:
            raise
        except Exception as e:
            logger.error(f"Failed to get portfolio {portfolio_id} for update: {e}")
            raise RepositoryError(f"Failed to retrieve portfolio for update: {e}") from e

    async def update_portfolio_atomic(
        self,
        portfolio_id: UUID,
        updates: dict[str, Any],
        expected_version: int | None = None,
    ) -> Portfolio:
        """
        Atomically update specific fields of a portfolio.

        This method allows updating only specific fields without loading
        the entire portfolio, reducing the chance of conflicts.

        Args:
            portfolio_id: The portfolio to update
            updates: Dictionary of field names to new values
            expected_version: If provided, ensures version matches before update

        Returns:
            The updated portfolio

        Raises:
            PortfolioNotFoundError: If portfolio doesn't exist
            StaleDataException: If version mismatch
            RepositoryError: If update fails
        """
        try:
            async with self._version_lock:
                # Build the UPDATE query dynamically
                update_fields = []
                params = []

                # Add each field to update
                for field, value in updates.items():
                    if field not in ["id", "version", "created_at"]:
                        update_fields.append(f"{field} = %s")
                        params.append(value)

                if not update_fields:
                    raise ValueError("No fields to update")

                # Always update version and last_updated
                update_fields.append("version = version + 1")
                update_fields.append("last_updated = %s")
                params.append(datetime.now(UTC))

                # Build WHERE clause
                where_clause = "WHERE id = %s"
                params.append(portfolio_id)

                if expected_version is not None:
                    where_clause += " AND version = %s"
                    params.append(expected_version)

                # Execute the update
                query = f"""
                UPDATE portfolios
                SET {', '.join(update_fields)}
                {where_clause}
                RETURNING *
                """

                record = await self.adapter.fetch_one(query, *params)

                if record is None:
                    # Check if portfolio exists
                    check_query = "SELECT id, version FROM portfolios WHERE id = %s"
                    check_result = await self.adapter.fetch_one(check_query, portfolio_id)

                    if check_result is None:
                        raise PortfolioNotFoundError(portfolio_id)
                    else:
                        # Version mismatch
                        raise StaleDataException(
                            entity_type="Portfolio",
                            entity_id=portfolio_id,
                            expected_version=expected_version or 0,
                            actual_version=check_result["version"],
                        )

                portfolio = self._map_record_to_portfolio(record)
                await self._load_positions(portfolio)

                logger.debug(
                    f"Atomically updated portfolio {portfolio_id} "
                    f"to version {portfolio.version}"
                )

                return portfolio

        except (PortfolioNotFoundError, StaleDataException):
            raise
        except Exception as e:
            if self.concurrency_service.detect_deadlock(e):
                raise DeadlockException(
                    message=f"Deadlock detected updating portfolio {portfolio_id}",
                    entities=[("Portfolio", portfolio_id)],
                )
            logger.error(f"Failed to atomically update portfolio {portfolio_id}: {e}")
            raise RepositoryError(f"Failed to atomically update portfolio: {e}") from e

    async def create_portfolio_snapshot(self, portfolio: Portfolio) -> Portfolio:
        """
        Create a point-in-time snapshot of a portfolio.

        Args:
            portfolio: The portfolio to snapshot

        Returns:
            The created snapshot

        Note:
            This is a simplified implementation that just saves the current state.
            In a real system, you might create a separate snapshot table.

        Raises:
            RepositoryError: If snapshot creation fails
        """
        try:
            # For now, just update the last_updated timestamp
            portfolio.last_updated = datetime.now(UTC)
            return await self.update_portfolio(portfolio)

        except Exception as e:
            logger.error(f"Failed to create portfolio snapshot: {e}")
            raise RepositoryError(f"Failed to create portfolio snapshot: {e}") from e

    async def _load_positions(self, portfolio: Portfolio) -> None:
        """
        Load positions for a portfolio from the database.

        Args:
            portfolio: Portfolio to load positions for
        """
        try:
            # Load active positions for this portfolio
            # Note: In a real system, you might have a portfolio_id foreign key
            # For now, we'll load all active positions (this is simplified)
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

            # Clear existing positions and load from database
            portfolio.positions.clear()

            for record in records:
                position = self._map_record_to_position(record)
                portfolio.positions[position.symbol] = position

        except Exception as e:
            logger.warning(f"Failed to load positions for portfolio {portfolio.id}: {e}")
            # Don't raise error here as portfolio can exist without positions

    def _map_record_to_portfolio(self, record: Row) -> Portfolio:
        """
        Map database record to Portfolio entity.

        Args:
            record: Database record

        Returns:
            Portfolio entity
        """
        portfolio = Portfolio(
            id=record["id"],
            name=record["name"],
            initial_capital=record["initial_capital"],
            cash_balance=record["cash_balance"],
            positions={},  # Will be loaded separately
            max_position_size=record["max_position_size"],
            max_portfolio_risk=record["max_portfolio_risk"],
            max_positions=record["max_positions"],
            max_leverage=record["max_leverage"],
            total_realized_pnl=record["total_realized_pnl"],
            total_commission_paid=record["total_commission_paid"],
            trades_count=record["trades_count"],
            winning_trades=record["winning_trades"],
            losing_trades=record["losing_trades"],
            created_at=record["created_at"],
            last_updated=record["last_updated"],
            strategy=record["strategy"],
            tags=record["tags"] or {},
        )
        # Add version if present
        if "version" in record:
            portfolio.version = record["version"]
        return portfolio

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

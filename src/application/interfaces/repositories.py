"""
Repository Interface Definitions

Defines the contracts that infrastructure repositories must implement.
Following the Repository pattern and clean architecture principles.
"""

# Standard library imports
from abc import abstractmethod
from datetime import datetime
from typing import Protocol
from uuid import UUID

# Local imports
from src.application.interfaces.market_data import Bar
from src.domain.entities.order import Order, OrderStatus
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position


class IOrderRepository(Protocol):
    """
    Order repository interface.

    Defines operations for persisting and retrieving Order entities.
    The infrastructure layer must implement this interface.
    """

    @abstractmethod
    async def save_order(self, order: Order) -> Order:
        """
        Save a new order or update an existing order.

        Args:
            order: The order entity to save

        Returns:
            The saved order entity

        Raises:
            RepositoryError: If save operation fails
        """
        ...

    @abstractmethod
    async def get_order_by_id(self, order_id: UUID) -> Order | None:
        """
        Retrieve an order by its ID.

        Args:
            order_id: The unique identifier of the order

        Returns:
            The order entity if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def get_orders_by_symbol(self, symbol: str) -> list[Order]:
        """
        Retrieve all orders for a specific symbol.

        Args:
            symbol: The trading symbol (e.g., 'AAPL')

        Returns:
            List of orders for the symbol, empty if none found

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def get_orders_by_status(self, status: OrderStatus) -> list[Order]:
        """
        Retrieve all orders with a specific status.

        Args:
            status: The order status to filter by

        Returns:
            List of orders with the specified status

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def get_active_orders(self) -> list[Order]:
        """
        Retrieve all active orders (pending, submitted, partially filled).

        Returns:
            List of active orders

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def get_orders_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[Order]:
        """
        Retrieve orders created within a date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of orders in date range

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def update_order(self, order: Order) -> Order:
        """
        Update an existing order.

        Args:
            order: The order entity to update

        Returns:
            The updated order entity

        Raises:
            RepositoryError: If update operation fails
            OrderNotFoundError: If order doesn't exist
        """
        ...

    @abstractmethod
    async def delete_order(self, order_id: UUID) -> bool:
        """
        Delete an order by ID.

        Args:
            order_id: The unique identifier of the order

        Returns:
            True if order was deleted, False if not found

        Raises:
            RepositoryError: If delete operation fails
        """
        ...

    @abstractmethod
    async def get_orders_by_broker_id(self, broker_order_id: str) -> list[Order]:
        """
        Retrieve orders by broker order ID.

        Args:
            broker_order_id: The broker's order identifier

        Returns:
            List of orders with matching broker ID

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...


class IPositionRepository(Protocol):
    """
    Position repository interface.

    Defines operations for persisting and retrieving Position entities.
    The infrastructure layer must implement this interface.
    """

    @abstractmethod
    async def persist_position(self, position: Position) -> Position:
        """
        Save a new position or update an existing position.

        Args:
            position: The position entity to save

        Returns:
            The saved position entity

        Raises:
            RepositoryError: If save operation fails
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def get_active_positions(self) -> list[Position]:
        """
        Retrieve all currently open positions.

        Returns:
            List of open positions

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def get_closed_positions(self) -> list[Position]:
        """
        Retrieve all closed positions.

        Returns:
            List of closed positions

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...


class IMarketDataRepository(Protocol):
    """
    Market data repository interface.

    Defines operations for persisting and retrieving market data (bars, quotes, etc.).
    The infrastructure layer must implement this interface for caching market data.
    """

    @abstractmethod
    async def save_bar(self, bar: Bar) -> None:
        """
        Save a market data bar.

        Args:
            bar: The bar data to save

        Raises:
            RepositoryError: If save operation fails
        """
        ...

    @abstractmethod
    async def save_bars(self, bars: list[Bar]) -> None:
        """
        Save multiple market data bars in batch.

        Args:
            bars: List of bars to save

        Raises:
            RepositoryError: If save operation fails
        """
        ...

    @abstractmethod
    async def get_latest_bar(self, symbol: str) -> Bar | None:
        """
        Get the most recent bar for a symbol.

        Args:
            symbol: The trading symbol

        Returns:
            The latest bar if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def get_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1min"
    ) -> list[Bar]:
        """
        Get bars for a symbol within a date range.

        Args:
            symbol: The trading symbol
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            timeframe: Bar timeframe (e.g., "1min", "5min", "1hour", "1day")

        Returns:
            List of bars ordered by timestamp (ascending)

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def get_bars_by_count(
        self, symbol: str, count: int, end: datetime | None = None, timeframe: str = "1min"
    ) -> list[Bar]:
        """
        Get a specific number of most recent bars.

        Args:
            symbol: The trading symbol
            count: Number of bars to retrieve
            end: End datetime (defaults to now)
            timeframe: Bar timeframe

        Returns:
            List of bars ordered by timestamp (ascending)

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def delete_bars_before(self, timestamp: datetime, symbol: str | None = None) -> int:
        """
        Delete bars older than a specified timestamp.

        Args:
            timestamp: Delete bars before this time
            symbol: Optional - only delete bars for this symbol (safety feature)

        Returns:
            Number of bars deleted

        Raises:
            RepositoryError: If delete operation fails
        """
        ...

    @abstractmethod
    async def get_symbols_with_data(self) -> list[str]:
        """
        Get list of symbols that have stored data.

        Returns:
            List of symbol strings

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def get_data_range(self, symbol: str) -> tuple[datetime, datetime] | None:
        """
        Get the date range of available data for a symbol.

        Args:
            symbol: The trading symbol

        Returns:
            Tuple of (earliest_timestamp, latest_timestamp) or None if no data

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...


class IPortfolioRepository(Protocol):
    """
    Portfolio repository interface.

    Defines operations for persisting and retrieving Portfolio entities.
    The infrastructure layer must implement this interface.
    """

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def get_current_portfolio(self) -> Portfolio | None:
        """
        Retrieve the current active portfolio.

        Returns:
            The current portfolio entity if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def get_all_portfolios(self) -> list[Portfolio]:
        """
        Retrieve all portfolios.

        Returns:
            List of all portfolios

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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

        Raises:
            RepositoryError: If retrieval operation fails
        """
        ...

    @abstractmethod
    async def update_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """
        Update an existing portfolio.

        Args:
            portfolio: The portfolio entity to update

        Returns:
            The updated portfolio entity

        Raises:
            RepositoryError: If update operation fails
            PortfolioNotFoundError: If portfolio doesn't exist
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def create_portfolio_snapshot(self, portfolio: Portfolio) -> Portfolio:
        """
        Create a point-in-time snapshot of a portfolio.

        Args:
            portfolio: The portfolio to snapshot

        Returns:
            The created snapshot

        Raises:
            RepositoryError: If snapshot creation fails
        """
        ...

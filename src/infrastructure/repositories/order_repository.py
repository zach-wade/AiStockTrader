"""
PostgreSQL Order Repository Implementation

Concrete implementation of IOrderRepository using PostgreSQL database.
Handles order persistence, retrieval, and mapping between domain entities and database records.
"""

# Standard library imports
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

# Local imports
from src.application.interfaces.exceptions import OrderNotFoundError, RepositoryError
from src.application.interfaces.repositories import IOrderRepository
from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from src.infrastructure.database.adapter import PostgreSQLAdapter

logger = logging.getLogger(__name__)


class PostgreSQLOrderRepository(IOrderRepository):
    """
    PostgreSQL implementation of IOrderRepository.

    Provides order persistence and retrieval using PostgreSQL database.
    Maps between Order domain entities and database records.
    """

    def __init__(self, adapter: PostgreSQLAdapter) -> None:
        """
        Initialize repository with database adapter.

        Args:
            adapter: PostgreSQL database adapter
        """
        self.adapter = adapter

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
        try:
            # Check if order exists
            existing_order = await self.get_order_by_id(order.id)

            if existing_order:
                return await self.update_order(order)
            else:
                return await self._insert_order(order)

        except Exception as e:
            logger.error(f"Failed to save order {order.id}: {e}")
            raise RepositoryError(f"Failed to save order: {e}") from e

    async def _insert_order(self, order: Order) -> Order:
        """Insert a new order into the database."""
        insert_query = """
        INSERT INTO orders (
            id, symbol, side, order_type, status, quantity,
            limit_price, stop_price, time_in_force, broker_order_id,
            filled_quantity, average_fill_price, created_at,
            submitted_at, filled_at, cancelled_at, reason, tags
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        await self.adapter.execute_query(
            insert_query,
            order.id,
            order.symbol,
            order.side.value,
            order.order_type.value,
            order.status.value,
            order.quantity,
            order.limit_price,
            order.stop_price,
            order.time_in_force.value,
            order.broker_order_id,
            order.filled_quantity,
            order.average_fill_price,
            order.created_at,
            order.submitted_at,
            order.filled_at,
            order.cancelled_at,
            order.reason,
            order.tags,
        )

        logger.debug(f"Inserted order {order.id}")
        return order

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
        try:
            query = """
            SELECT id, symbol, side, order_type, status, quantity,
                   limit_price, stop_price, time_in_force, broker_order_id,
                   filled_quantity, average_fill_price, created_at,
                   submitted_at, filled_at, cancelled_at, reason, tags
            FROM orders
            WHERE id = %s
            """

            record = await self.adapter.fetch_one(query, order_id)

            if record is None:
                return None

            return self._map_record_to_order(record)

        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise RepositoryError(f"Failed to retrieve order: {e}") from e

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
        try:
            query = """
            SELECT id, symbol, side, order_type, status, quantity,
                   limit_price, stop_price, time_in_force, broker_order_id,
                   filled_quantity, average_fill_price, created_at,
                   submitted_at, filled_at, cancelled_at, reason, tags
            FROM orders
            WHERE symbol = %s
            ORDER BY created_at DESC
            """

            records = await self.adapter.fetch_all(query, symbol)
            return [self._map_record_to_order(record) for record in records]

        except Exception as e:
            logger.error(f"Failed to get orders for symbol {symbol}: {e}")
            raise RepositoryError(f"Failed to retrieve orders for symbol: {e}") from e

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
        try:
            query = """
            SELECT id, symbol, side, order_type, status, quantity,
                   limit_price, stop_price, time_in_force, broker_order_id,
                   filled_quantity, average_fill_price, created_at,
                   submitted_at, filled_at, cancelled_at, reason, tags
            FROM orders
            WHERE status = %s
            ORDER BY created_at DESC
            """

            records = await self.adapter.fetch_all(query, status.value)
            return [self._map_record_to_order(record) for record in records]

        except Exception as e:
            logger.error(f"Failed to get orders with status {status}: {e}")
            raise RepositoryError(f"Failed to retrieve orders by status: {e}") from e

    async def get_active_orders(self) -> list[Order]:
        """
        Retrieve all active orders (pending, submitted, partially filled).

        Returns:
            List of active orders

        Raises:
            RepositoryError: If retrieval operation fails
        """
        try:
            query = """
            SELECT id, symbol, side, order_type, status, quantity,
                   limit_price, stop_price, time_in_force, broker_order_id,
                   filled_quantity, average_fill_price, created_at,
                   submitted_at, filled_at, cancelled_at, reason, tags
            FROM orders
            WHERE status IN ('pending', 'submitted', 'partially_filled')
            ORDER BY created_at DESC
            """

            records = await self.adapter.fetch_all(query)
            return [self._map_record_to_order(record) for record in records]

        except Exception as e:
            logger.error(f"Failed to get active orders: {e}")
            raise RepositoryError(f"Failed to retrieve active orders: {e}") from e

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
        try:
            query = """
            SELECT id, symbol, side, order_type, status, quantity,
                   limit_price, stop_price, time_in_force, broker_order_id,
                   filled_quantity, average_fill_price, created_at,
                   submitted_at, filled_at, cancelled_at, reason, tags
            FROM orders
            WHERE created_at >= %s AND created_at <= %s
            ORDER BY created_at DESC
            """

            records = await self.adapter.fetch_all(query, start_date, end_date)
            return [self._map_record_to_order(record) for record in records]

        except Exception as e:
            logger.error(f"Failed to get orders by date range: {e}")
            raise RepositoryError(f"Failed to retrieve orders by date range: {e}") from e

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
        try:
            update_query = """
            UPDATE orders SET
                symbol = %s, side = %s, order_type = %s, status = %s,
                quantity = %s, limit_price = %s, stop_price = %s,
                time_in_force = %s, broker_order_id = %s,
                filled_quantity = %s, average_fill_price = %s,
                submitted_at = %s, filled_at = %s, cancelled_at = %s,
                reason = %s, tags = %s
            WHERE id = %s
            """

            result = await self.adapter.execute_query(
                update_query,
                order.symbol,
                order.side.value,
                order.order_type.value,
                order.status.value,
                order.quantity,
                order.limit_price,
                order.stop_price,
                order.time_in_force.value,
                order.broker_order_id,
                order.filled_quantity,
                order.average_fill_price,
                order.submitted_at,
                order.filled_at,
                order.cancelled_at,
                order.reason,
                order.tags,
                order.id,  # id moved to end for WHERE clause
            )

            if "UPDATE 0" in result:
                raise OrderNotFoundError(order.id)

            logger.debug(f"Updated order {order.id}")
            return order

        except OrderNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update order {order.id}: {e}")
            raise RepositoryError(f"Failed to update order: {e}") from e

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
        try:
            delete_query = "DELETE FROM orders WHERE id = %s"
            result = await self.adapter.execute_query(delete_query, order_id)

            success = "DELETE 1" in result

            if success:
                logger.debug(f"Deleted order {order_id}")
            else:
                logger.debug(f"Order {order_id} not found for deletion")

            return success

        except Exception as e:
            logger.error(f"Failed to delete order {order_id}: {e}")
            raise RepositoryError(f"Failed to delete order: {e}") from e

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
        try:
            query = """
            SELECT id, symbol, side, order_type, status, quantity,
                   limit_price, stop_price, time_in_force, broker_order_id,
                   filled_quantity, average_fill_price, created_at,
                   submitted_at, filled_at, cancelled_at, reason, tags
            FROM orders
            WHERE broker_order_id = %s
            ORDER BY created_at DESC
            """

            records = await self.adapter.fetch_all(query, broker_order_id)
            return [self._map_record_to_order(record) for record in records]

        except Exception as e:
            logger.error(f"Failed to get orders by broker ID {broker_order_id}: {e}")
            raise RepositoryError(f"Failed to retrieve orders by broker ID: {e}") from e

    def _map_record_to_order(self, record: dict[str, Any]) -> Order:
        """
        Map database record to Order entity.

        Args:
            record: Database record

        Returns:
            Order entity
        """
        return Order(
            id=record["id"],
            symbol=record["symbol"],
            side=OrderSide(record["side"]),
            order_type=OrderType(record["order_type"]),
            status=OrderStatus(record["status"]),
            quantity=record["quantity"],
            limit_price=record["limit_price"],
            stop_price=record["stop_price"],
            time_in_force=TimeInForce(record["time_in_force"]),
            broker_order_id=record["broker_order_id"],
            filled_quantity=record["filled_quantity"],
            average_fill_price=record["average_fill_price"],
            created_at=record["created_at"],
            submitted_at=record["submitted_at"],
            filled_at=record["filled_at"],
            cancelled_at=record["cancelled_at"],
            reason=record["reason"],
            tags=record["tags"] or {},
        )

"""
Order lifecycle management.

This module manages the complete lifecycle of orders from creation through
execution, including tracking, status updates, and fill processing.
"""

# Standard library imports
import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from typing import Any
import uuid

# Local imports
from main.models.common import Order, OrderSide, OrderStatus, OrderType, Signal, TimeInForce
from main.trading_engine.brokers.broker_interface import BrokerInterface
from main.utils.core import AITraderException, async_retry, create_event_tracker, create_task_safely
from main.utils.database import DatabasePool
from main.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class OrderManagerException(AITraderException):
    """Base exception for order manager errors."""

    pass


class OrderNotFoundError(OrderManagerException):
    """Raised when order is not found."""

    pass


@dataclass
class OrderEvent:
    """Event in order lifecycle."""

    order_id: str
    event_type: str  # created, submitted, filled, canceled, rejected, etc.
    timestamp: datetime
    old_status: OrderStatus | None
    new_status: OrderStatus
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderStatistics:
    """Statistics for order execution."""

    total_orders: int = 0
    filled_orders: int = 0
    canceled_orders: int = 0
    rejected_orders: int = 0
    partial_fills: int = 0
    avg_fill_time: timedelta | None = None
    fill_rate: Decimal = Decimal("0")


class OrderManager:
    """
    Manages order lifecycle and execution.

    Tracks orders from creation through execution, maintains order state,
    processes fills, and provides order analytics.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        database: DatabasePool,
        metrics_collector: MetricsCollector | None = None,
    ):
        """
        Initialize order manager.

        Args:
            broker: Broker interface for order execution
            database: Database for order persistence
            metrics_collector: Optional metrics collector
        """
        self.broker = broker
        self.db = database
        self.metrics = metrics_collector
        self.event_tracker = create_event_tracker("order_manager")

        # Order tracking
        self._orders: dict[str, Order] = {}
        self._order_events: dict[str, list[OrderEvent]] = defaultdict(list)
        self._signal_orders: dict[str, set[str]] = defaultdict(set)  # signal_id -> order_ids
        self._symbol_orders: dict[str, set[str]] = defaultdict(set)  # symbol -> order_ids

        # Order callbacks
        self._status_callbacks: dict[OrderStatus, list[Callable]] = defaultdict(list)

        # Background tasks
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Start order manager."""
        logger.info("Starting order manager")
        self._running = True

        # Start order update stream
        update_task = create_task_safely(self._process_order_updates())
        self._tasks.append(update_task)

        # Start order timeout monitor
        timeout_task = create_task_safely(self._monitor_order_timeouts())
        self._tasks.append(timeout_task)

        # Load active orders from database
        await self._load_active_orders()

    async def stop(self) -> None:
        """Stop order manager."""
        logger.info("Stopping order manager")
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def create_order(
        self,
        signal: Signal,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Create a new order.

        Args:
            signal: Signal that generated the order
            symbol: Symbol to trade
            side: Order side (BUY/SELL)
            order_type: Order type
            quantity: Order quantity
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force
            client_order_id: Optional client order ID

        Returns:
            Created order
        """
        # Generate client order ID if not provided
        if not client_order_id:
            client_order_id = f"ai_trader_{uuid.uuid4().hex[:8]}"

        # Create order object
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            status=OrderStatus.PENDING_NEW,
            created_at=datetime.utcnow(),
        )

        # Store order
        self._orders[client_order_id] = order
        self._signal_orders[signal.signal_id].add(client_order_id)
        self._symbol_orders[symbol].add(client_order_id)

        # Record event
        self._record_event(
            order,
            "created",
            None,
            OrderStatus.PENDING_NEW,
            {"signal_id": signal.signal_id, "signal_strength": float(signal.strength)},
        )

        # Persist to database
        await self._save_order(order)

        logger.info(f"Created order {client_order_id} for {symbol} {side.value}")
        return order

    @async_retry(max_attempts=3, delay=1.0)
    async def submit_order(self, order: Order) -> str:
        """
        Submit order to broker.

        Args:
            order: Order to submit

        Returns:
            Broker order ID

        Raises:
            OrderManagerException: If submission fails
        """
        try:
            # Update status
            old_status = order.status
            order.status = OrderStatus.PENDING_SUBMIT
            self._record_event(order, "submitting", old_status, order.status)

            # Submit to broker
            broker_order_id = await self.broker.submit_order(order)

            # Update order with broker ID
            order.order_id = broker_order_id
            order.status = OrderStatus.NEW
            order.submitted_at = datetime.utcnow()

            # Record event
            self._record_event(
                order,
                "submitted",
                OrderStatus.PENDING_SUBMIT,
                order.status,
                {"broker_order_id": broker_order_id},
            )

            # Update database
            await self._save_order(order)

            # Track metrics
            if self.metrics:
                self.metrics.increment(
                    "order_manager.orders_submitted",
                    tags={
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "type": order.order_type.value,
                    },
                )

            logger.info(f"Submitted order {order.client_order_id} to broker as {broker_order_id}")
            return broker_order_id

        except Exception as e:
            # Update status to rejected
            order.status = OrderStatus.REJECTED
            order.rejection_reason = str(e)

            self._record_event(
                order, "rejected", OrderStatus.PENDING_SUBMIT, order.status, {"reason": str(e)}
            )

            await self._save_order(order)

            logger.error(f"Failed to submit order {order.client_order_id}: {e}")
            raise OrderManagerException(f"Order submission failed: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Client order ID

        Returns:
            True if cancellation successful
        """
        order = self._orders.get(order_id)
        if not order:
            raise OrderNotFoundError(f"Order {order_id} not found")

        # Check if order can be canceled
        if order.status not in [
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_NEW,
        ]:
            logger.warning(f"Cannot cancel order {order_id} in status {order.status}")
            return False

        try:
            # Update status
            old_status = order.status
            order.status = OrderStatus.PENDING_CANCEL
            self._record_event(order, "cancel_requested", old_status, order.status)

            # Cancel with broker
            if order.order_id:  # Has broker ID
                success = await self.broker.cancel_order(order.order_id)

                if success:
                    order.status = OrderStatus.CANCELED
                    order.canceled_at = datetime.utcnow()
                    self._record_event(order, "canceled", OrderStatus.PENDING_CANCEL, order.status)
                else:
                    # Revert status if cancellation failed
                    order.status = old_status
                    self._record_event(
                        order, "cancel_failed", OrderStatus.PENDING_CANCEL, old_status
                    )

            else:
                # Order not yet submitted to broker
                order.status = OrderStatus.CANCELED
                order.canceled_at = datetime.utcnow()
                self._record_event(order, "canceled", OrderStatus.PENDING_CANCEL, order.status)
                success = True

            # Update database
            await self._save_order(order)

            # Track metrics
            if self.metrics and success:
                self.metrics.increment(
                    "order_manager.orders_canceled", tags={"symbol": order.symbol}
                )

            return success

        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            # Revert status
            order.status = old_status
            await self._save_order(order)
            return False

    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def get_orders_by_signal(self, signal_id: str) -> list[Order]:
        """Get all orders for a signal."""
        order_ids = self._signal_orders.get(signal_id, set())
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    async def get_orders_by_symbol(
        self, symbol: str, status: OrderStatus | None = None
    ) -> list[Order]:
        """Get orders for a symbol."""
        order_ids = self._symbol_orders.get(symbol, set())
        orders = [self._orders[oid] for oid in order_ids if oid in self._orders]

        if status:
            orders = [o for o in orders if o.status == status]

        return orders

    async def get_active_orders(self) -> list[Order]:
        """Get all active orders."""
        active_statuses = {
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_NEW,
            OrderStatus.PENDING_SUBMIT,
        }

        return [order for order in self._orders.values() if order.status in active_statuses]

    def register_status_callback(
        self, status: OrderStatus, callback: Callable[[Order], None]
    ) -> None:
        """Register callback for order status changes."""
        self._status_callbacks[status].append(callback)

    async def get_order_statistics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        symbol: str | None = None,
    ) -> OrderStatistics:
        """Get order execution statistics."""
        # Filter orders
        orders = list(self._orders.values())

        if start_time:
            orders = [o for o in orders if o.created_at >= start_time]
        if end_time:
            orders = [o for o in orders if o.created_at <= end_time]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        # Calculate statistics
        stats = OrderStatistics(total_orders=len(orders))

        fill_times = []
        for order in orders:
            if order.status == OrderStatus.FILLED:
                stats.filled_orders += 1
                if order.filled_at and order.submitted_at:
                    fill_times.append(order.filled_at - order.submitted_at)
            elif order.status == OrderStatus.CANCELED:
                stats.canceled_orders += 1
            elif order.status == OrderStatus.REJECTED:
                stats.rejected_orders += 1
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                stats.partial_fills += 1

        # Calculate fill rate
        if stats.total_orders > 0:
            stats.fill_rate = Decimal(stats.filled_orders) / Decimal(stats.total_orders)

        # Calculate average fill time
        if fill_times:
            avg_seconds = sum(ft.total_seconds() for ft in fill_times) / len(fill_times)
            stats.avg_fill_time = timedelta(seconds=avg_seconds)

        return stats

    async def _process_order_updates(self) -> None:
        """Process order updates from broker."""
        while self._running:
            try:
                # Stream order updates from broker
                async for broker_order in self.broker.stream_order_updates():
                    await self._handle_broker_update(broker_order)

            except Exception as e:
                logger.error(f"Error processing order updates: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _handle_broker_update(self, broker_order: Order) -> None:
        """Handle order update from broker."""
        # Find our order by broker ID
        our_order = None
        for order in self._orders.values():
            if order.order_id == broker_order.order_id:
                our_order = order
                break

        if not our_order:
            logger.warning(f"Received update for unknown order {broker_order.order_id}")
            return

        # Update order status
        old_status = our_order.status
        our_order.status = broker_order.status
        our_order.filled_quantity = broker_order.filled_quantity
        our_order.avg_fill_price = broker_order.avg_fill_price
        our_order.updated_at = datetime.utcnow()

        if broker_order.filled_at:
            our_order.filled_at = broker_order.filled_at

        # Record event
        self._record_event(
            our_order,
            "status_update",
            old_status,
            our_order.status,
            {
                "filled_qty": (
                    float(broker_order.filled_quantity) if broker_order.filled_quantity else 0
                ),
                "avg_price": (
                    float(broker_order.avg_fill_price) if broker_order.avg_fill_price else None
                ),
            },
        )

        # Process fills if any
        if broker_order.filled_quantity > our_order.filled_quantity:
            await self._process_fill(our_order, broker_order)

        # Save to database
        await self._save_order(our_order)

        # Trigger status callbacks
        await self._trigger_callbacks(our_order)

        # Track metrics
        if self.metrics:
            self.metrics.increment(
                "order_manager.status_updates",
                tags={"old_status": old_status.value, "new_status": our_order.status.value},
            )

    async def _process_fill(self, our_order: Order, broker_order: Order) -> None:
        """Process order fill."""
        # Calculate fill details
        fill_qty = broker_order.filled_quantity - our_order.filled_quantity

        if fill_qty <= 0:
            return

        fill = Fill(
            order_id=our_order.order_id,
            client_order_id=our_order.client_order_id,
            symbol=our_order.symbol,
            side=our_order.side,
            quantity=fill_qty,
            price=broker_order.avg_fill_price,
            timestamp=broker_order.filled_at or datetime.utcnow(),
        )

        # Save fill to database
        await self.db.save_fill(fill)

        logger.info(
            f"Processed fill for order {our_order.client_order_id}: "
            f"{fill_qty} @ {broker_order.avg_fill_price}"
        )

    async def _monitor_order_timeouts(self) -> None:
        """Monitor orders for timeouts."""
        while self._running:
            try:
                now = datetime.utcnow()

                for order in self._orders.values():
                    # Check Day orders for expiration
                    if order.time_in_force == TimeInForce.DAY and order.status == OrderStatus.NEW:
                        if order.created_at.date() < now.date():
                            # Day order expired
                            old_status = order.status
                            order.status = OrderStatus.EXPIRED
                            order.expired_at = now

                            self._record_event(order, "expired", old_status, order.status)
                            await self._save_order(order)
                            await self._trigger_callbacks(order)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error monitoring order timeouts: {e}")
                await asyncio.sleep(60)

    async def _trigger_callbacks(self, order: Order) -> None:
        """Trigger callbacks for order status."""
        callbacks = self._status_callbacks.get(order.status, [])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")

    def _record_event(
        self,
        order: Order,
        event_type: str,
        old_status: OrderStatus | None,
        new_status: OrderStatus,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record order event."""
        event = OrderEvent(
            order_id=order.client_order_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            old_status=old_status,
            new_status=new_status,
            details=details or {},
        )

        self._order_events[order.client_order_id].append(event)

        # Track event
        self.event_tracker.track(
            "order_event",
            {
                "order_id": order.client_order_id,
                "event_type": event_type,
                "symbol": order.symbol,
                "side": order.side.value,
                "old_status": old_status.value if old_status else None,
                "new_status": new_status.value,
            },
        )

    async def _save_order(self, order: Order) -> None:
        """Save order to database."""
        try:
            await self.db.save_order(order)
        except Exception as e:
            logger.error(f"Error saving order {order.client_order_id}: {e}")

    async def _load_active_orders(self) -> None:
        """Load active orders from database on startup."""
        try:
            active_orders = await self.db.get_active_orders()

            for order in active_orders:
                self._orders[order.client_order_id] = order
                self._symbol_orders[order.symbol].add(order.client_order_id)

            logger.info(f"Loaded {len(active_orders)} active orders from database")

        except Exception as e:
            logger.error(f"Error loading active orders: {e}")


class OrderValidator:
    """Validates order parameters before submission."""

    def __init__(self):
        self.valid_symbols = set()  # Can be populated from config
        self.min_quantity = 1
        self.max_quantity = 10000

    def validate_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType,
        limit_price: float | None = None,
        stop_price: float | None = None,
        **kwargs,
    ) -> list[str]:
        """Validate order parameters.

        Returns:
            List of validation error messages, empty if valid
        """
        errors = []

        # Basic validations
        if not symbol or len(symbol) == 0:
            errors.append("Symbol cannot be empty")
        elif len(symbol) > 10:
            errors.append("Symbol too long")

        if quantity <= 0:
            errors.append("Quantity must be positive")
        elif quantity > self.max_quantity:
            errors.append(f"Quantity exceeds maximum of {self.max_quantity}")

        # Order type specific validations
        if order_type == OrderType.LIMIT and limit_price is None:
            errors.append("Limit order requires limit_price")
        elif order_type == OrderType.STOP and stop_price is None:
            errors.append("Stop order requires stop_price")
        elif order_type == OrderType.STOP_LIMIT and (limit_price is None or stop_price is None):
            errors.append("Stop limit order requires both limit_price and stop_price")

        # Price validations
        if limit_price is not None and limit_price <= 0:
            errors.append("Limit price must be positive")
        if stop_price is not None and stop_price <= 0:
            errors.append("Stop price must be positive")

        return errors

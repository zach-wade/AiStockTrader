# File: trading_engine/brokers/paper_broker.py
# Standard library imports
import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
import logging
from typing import Any
import uuid

# Local imports
from main.models.common import (
    AccountInfo,
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from main.trading_engine.brokers.broker_interface import BrokerInterface

logger = logging.getLogger(__name__)


class PaperBroker(BrokerInterface):
    """Paper trading broker for simulated trading."""

    def __init__(self, config):
        # Handle both Config object and dict for backward compatibility
        if hasattr(config, "_raw_config"):
            # It's a Config object
            super().__init__(config)
            config_dict = config._raw_config
        else:
            # It's a raw dict - create minimal Config-like object
            # Standard library imports
            from types import SimpleNamespace

            config_obj = SimpleNamespace()
            config_obj._raw_config = config
            super().__init__(config_obj)
            config_dict = config

        self.config_dict = config_dict
        self.connected = False

        # Account state
        self.account_balance = config_dict.get("trading", {}).get("starting_cash", 100000)
        self.buying_power = self.account_balance
        self.positions = {}
        self.orders = {}
        self.order_history = []

        # Market data (simplified)
        self.last_prices = {}

        logger.info(f"Paper broker initialized with balance: ${self.account_balance}")

    async def connect(self) -> bool:
        """Connect to paper broker (simulated)."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True
        self._connected = True
        logger.info("Connected to paper broker")
        return True

    async def disconnect(self) -> None:
        """Disconnect from paper broker."""
        self.connected = False
        self._connected = False
        logger.info("Disconnected from paper broker")

    async def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected

    async def get_account_status(self) -> dict[str, Any]:
        """Get account status."""
        return {
            "account_value": self.account_balance,
            "buying_power": self.buying_power,
            "positions_value": sum(pos["market_value"] for pos in self.positions.values()),
            "cash": self.buying_power,
            "pattern_day_trader": False,
            "trading_blocked": False,
            "account_blocked": False,
        }

    async def get_account_info(self) -> AccountInfo:
        """Get account information as AccountInfo model."""
        positions_value = sum(pos["market_value"] for pos in self.positions.values())
        total_equity = self.buying_power + positions_value

        return AccountInfo(
            account_id="paper_trading_account",
            buying_power=self.buying_power,
            cash=self.buying_power,
            portfolio_value=total_equity,
            equity=total_equity,
            last_equity=total_equity,
            long_market_value=positions_value,
            short_market_value=0.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            sma=self.buying_power,
            daytrade_count=0,
            balance_asof=datetime.now(UTC),
            pattern_day_trader=False,
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            trade_suspended_by_user=False,
            currency="USD",
        )

    async def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Submit an order."""
        if not self.connected:
            raise Exception("Broker not connected")

        # Generate order ID
        order_id = str(uuid.uuid4())

        # Get current price (in real implementation, would fetch from data source)
        price = await self._get_current_price(order["symbol"])

        # Calculate order value
        order_value = price * order["quantity"]

        # Check buying power for buy orders
        if order["side"].value == "buy" and order_value > self.buying_power:
            return {"id": order_id, "status": "rejected", "reason": "Insufficient buying power"}

        # Simulate order execution (instant fill for market orders)
        if order["type"].value == "market":
            # Execute order
            fill_price = price * (
                1.0001 if order["side"].value == "buy" else 0.9999
            )  # Simulate slippage

            # Update positions
            if order["side"].value == "buy":
                if order["symbol"] in self.positions:
                    # Add to existing position
                    pos = self.positions[order["symbol"]]
                    total_cost = (pos["quantity"] * pos["entry_price"]) + (
                        order["quantity"] * fill_price
                    )
                    pos["quantity"] += order["quantity"]
                    pos["entry_price"] = total_cost / pos["quantity"]
                    pos["market_value"] = pos["quantity"] * price
                else:
                    # Create new position
                    self.positions[order["symbol"]] = {
                        "symbol": order["symbol"],
                        "quantity": order["quantity"],
                        "entry_price": fill_price,
                        "market_value": order["quantity"] * price,
                        "cost_basis": order["quantity"] * fill_price,
                    }

                # Update buying power
                self.buying_power -= order_value

            elif order["symbol"] in self.positions:
                pos = self.positions[order["symbol"]]

                if order["quantity"] <= pos["quantity"]:
                    # Partial or full sell
                    pos["quantity"] -= order["quantity"]

                    if pos["quantity"] == 0:
                        # Close position
                        del self.positions[order["symbol"]]
                    else:
                        # Update position
                        pos["market_value"] = pos["quantity"] * price

                    # Update buying power
                    self.buying_power += order_value
                else:
                    return {"id": order_id, "status": "rejected", "reason": "Insufficient shares"}
            else:
                return {"id": order_id, "status": "rejected", "reason": "No position to sell"}

            # Record order
            order_record = {
                "id": order_id,
                "symbol": order["symbol"],
                "side": order["side"].value,
                "quantity": order["quantity"],
                "type": order["type"].value,
                "status": "filled",
                "fill_price": fill_price,
                "filled_at": datetime.now(UTC),
                "commission": 0,  # No commission in paper trading
            }

            self.orders[order_id] = order_record
            self.order_history.append(order_record)

            logger.info(
                f"Order filled: {order['side'].value} {order['quantity']} {order['symbol']} @ ${fill_price:.2f}"
            )

            return order_record

        else:
            # For limit/stop orders, create pending order with full details
            order_record = {
                "id": order_id,
                "symbol": order["symbol"],
                "side": order["side"].value,
                "quantity": order["quantity"],
                "type": order["type"].value,
                "status": "pending",
                "limit_price": order.get("limit_price"),
                "stop_price": order.get("stop_price"),
                "created_at": datetime.now(UTC),
                "commission": 0,  # No commission in paper trading
            }

            self.orders[order_id] = order_record
            self.order_history.append(order_record)

            logger.info(
                f"Order created: {order['side'].value} {order['quantity']} {order['symbol']} (pending)"
            )

            return order_record

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id in self.orders and self.orders[order_id]["status"] == "pending":
            self.orders[order_id]["status"] = "cancelled"
            logger.info(f"Order {order_id} cancelled")
            return True
        return False

    async def get_positions(self) -> dict[str, Position]:
        """Get all positions."""
        result = {}
        for symbol in self.positions:
            pos = await self.get_position(symbol)
            if pos:
                result[symbol] = pos
        return result

    async def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        """Get orders, optionally filtered by status."""
        orders = []
        for order_id in self.orders:
            order = await self.get_order(order_id)
            if order and (status is None or order.status == status):
                orders.append(order)
        return orders

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol (simulated)."""
        # In real implementation, would fetch from data source
        # For now, return a simulated price
        # Local imports
        from main.utils.core import secure_uniform

        if symbol not in self.last_prices:
            # Initialize with a random price
            self.last_prices[symbol] = secure_uniform(50, 500)
        else:
            # Simulate price movement
            change = secure_uniform(-0.02, 0.02)  # ±2% change
            self.last_prices[symbol] *= 1 + change

        return round(self.last_prices[symbol], 2)

    async def get_market_hours(self) -> dict[str, Any]:
        """Get market hours."""
        # Simplified market hours
        return {
            "is_open": True,  # Always open for paper trading
            "next_open": None,
            "next_close": None,
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        total_value = self.buying_power + sum(
            pos["market_value"] for pos in self.positions.values()
        )
        initial_balance = self.config_dict.get("trading", {}).get("starting_cash", 100000)

        return {
            "total_value": total_value,
            "cash": self.buying_power,
            "positions_value": sum(pos["market_value"] for pos in self.positions.values()),
            "total_pnl": total_value - initial_balance,
            "total_pnl_percent": ((total_value - initial_balance) / initial_balance) * 100,
            "total_trades": len(self.order_history),
            "open_positions": len(self.positions),
        }

    # BrokerInterface abstract methods that need to be implemented
    async def modify_order(
        self,
        order_id: str,
        limit_price: float | None = None,
        stop_price: float | None = None,
        quantity: float | None = None,
    ) -> Order:
        """
        Modify existing order.

        Args:
            order_id: Order ID to modify
            limit_price: New limit price (if applicable)
            stop_price: New stop price (if applicable)
            quantity: New quantity

        Returns:
            Modified Order object

        Raises:
            ValueError: If order not found or cannot be modified
        """
        if not self.connected:
            raise Exception("Broker not connected")

        # Check if order exists
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")

        order = self.orders[order_id]

        # Check if order can be modified (only pending orders can be modified)
        if order["status"] not in ["pending", "submitted"]:
            raise ValueError(
                f"Cannot modify order with status '{order['status']}'. Only pending orders can be modified."
            )

        # Track what was modified for logging
        modifications = []

        # Update fields as provided
        if limit_price is not None:
            old_limit_price = order.get("limit_price")
            order["limit_price"] = limit_price
            modifications.append(f"limit_price={old_limit_price}->{limit_price}")

        if stop_price is not None:
            old_stop_price = order.get("stop_price")
            order["stop_price"] = stop_price
            modifications.append(f"stop_price={old_stop_price}->{stop_price}")

        if quantity is not None:
            old_quantity = order.get("quantity", 0)
            order["quantity"] = quantity
            modifications.append(f"quantity={old_quantity}->{quantity}")

        # Add modification timestamp
        order["modified_at"] = datetime.now(UTC)

        # Log the modification
        if modifications:
            logger.info(f"Order {order_id} modified: {', '.join(modifications)}")

        # Return the modified order as Order object
        return await self.get_order(order_id)

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for specific symbol."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            return Position(
                symbol=symbol,
                quantity=pos["quantity"],
                avg_entry_price=pos["entry_price"],
                current_price=await self._get_current_price(symbol),
                market_value=pos["market_value"],
                cost_basis=pos["cost_basis"],
                unrealized_pnl=pos["market_value"] - pos["cost_basis"],
                unrealized_pnl_pct=(
                    ((pos["market_value"] / pos["cost_basis"]) - 1) * 100
                    if pos["cost_basis"] > 0
                    else 0
                ),
                realized_pnl=0.0,
                side="long" if pos["quantity"] > 0 else "short",
                timestamp=datetime.now(UTC),
            )
        return None

    async def get_order(self, order_id: str) -> Order | None:
        """Get specific order by ID."""
        if order_id in self.orders:
            order = self.orders[order_id]
            return Order(
                order_id=order_id,
                symbol=order["symbol"],
                side=OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL,
                quantity=order["quantity"],
                order_type=OrderType.MARKET if order["type"] == "market" else OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                status=OrderStatus.FILLED if order["status"] == "filled" else OrderStatus.PENDING,
                created_at=order.get("filled_at", datetime.now(UTC)),
            )
        return None

    async def get_recent_orders(self) -> list[Order]:
        """Get a list of recently executed or updated orders."""
        orders = []
        for order_id, order in self.orders.items():
            orders.append(await self.get_order(order_id))
        return [o for o in orders if o is not None]

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol."""
        price = await self._get_current_price(symbol)
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            last=price,
            bid=price * 0.999,
            ask=price * 1.001,
            volume=1000000,
        )

    async def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Fetches the latest quote for a symbol."""
        price = await self._get_current_price(symbol)
        return {
            "symbol": symbol,
            "bid_price": price * 0.999,
            "ask_price": price * 1.001,
            "bid_size": 100,
            "ask_size": 100,
            "timestamp": datetime.now(UTC),
        }

    async def subscribe_market_data(self, symbols: list[str]) -> None:
        """Subscribe to real-time market data."""
        logger.info(f"Paper broker: Simulating subscription to {symbols}")

    async def unsubscribe_market_data(self, symbols: list[str]) -> None:
        """Unsubscribe from market data."""
        logger.info(f"Paper broker: Simulating unsubscription from {symbols}")

    async def get_historical_data(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> list[dict[str, Any]]:
        """Get historical market data."""
        # Return empty for paper broker
        return []

    async def get_tradable_symbols(self) -> list[str]:
        """Get list of tradable symbols."""
        # Return common symbols for paper broker
        return ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]

    async def is_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable."""
        tradable_symbols = await self.get_tradable_symbols()
        return symbol in tradable_symbols

    async def stream_market_data(self, symbols: list[str]) -> AsyncIterator[MarketData]:
        """
        Stream real-time market data for symbols.

        Args:
            symbols: List of symbols to stream

        Yields:
            Market data updates
        """
        logger.info(f"Paper broker: Starting market data stream for {symbols}")

        # Simulate streaming market data
        while self.connected:
            for symbol in symbols:
                # Simulate price updates every 1-5 seconds
                await asyncio.sleep(secure_uniform(1, 5))

                # Generate market data update
                price = await self._get_current_price(symbol)
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(UTC),
                    last=price,
                    bid=price * 0.999,
                    ask=price * 1.001,
                    volume=secure_randint(100000, 5000000),
                    open=price * secure_uniform(0.98, 1.02),
                    high=price * secure_uniform(1.0, 1.03),
                    low=price * secure_uniform(0.97, 1.0),
                    close=price,
                )

                yield market_data

    async def stream_order_updates(self) -> AsyncIterator[Order]:
        """
        Stream real-time order updates.

        Yields:
            Order updates
        """
        logger.info("Paper broker: Starting order updates stream")

        # Track processed orders to avoid duplicates
        processed_orders = set()

        while self.connected:
            # Check for pending orders that might be filled
            for order_id, order_data in list(self.orders.items()):
                if order_id in processed_orders:
                    continue

                if order_data["status"] == "pending":
                    # Simulate order fills for limit orders
                    current_price = await self._get_current_price(order_data["symbol"])

                    should_fill = False
                    if order_data["type"] == "limit":
                        if (
                            order_data["side"] == "buy"
                            and current_price <= order_data.get("limit_price", float("inf"))
                            or order_data["side"] == "sell"
                            and current_price >= order_data.get("limit_price", 0)
                        ):
                            should_fill = True

                    if should_fill:
                        # Update order status
                        order_data["status"] = "filled"
                        order_data["fill_price"] = current_price
                        order_data["filled_at"] = datetime.now(UTC)

                        # Update positions and buying power
                        if order_data["side"] == "buy":
                            # Update or create position
                            symbol = order_data["symbol"]
                            quantity = order_data["quantity"]

                            if symbol in self.positions:
                                pos = self.positions[symbol]
                                total_cost = (pos["quantity"] * pos["entry_price"]) + (
                                    quantity * current_price
                                )
                                pos["quantity"] += quantity
                                pos["entry_price"] = total_cost / pos["quantity"]
                                pos["market_value"] = pos["quantity"] * current_price
                            else:
                                self.positions[symbol] = {
                                    "symbol": symbol,
                                    "quantity": quantity,
                                    "entry_price": current_price,
                                    "market_value": quantity * current_price,
                                    "cost_basis": quantity * current_price,
                                }

                            self.buying_power -= quantity * current_price
                        else:  # sell
                            # Update position
                            symbol = order_data["symbol"]
                            if symbol in self.positions:
                                pos = self.positions[symbol]
                                pos["quantity"] -= order_data["quantity"]

                                if pos["quantity"] <= 0:
                                    del self.positions[symbol]
                                else:
                                    pos["market_value"] = pos["quantity"] * current_price

                                self.buying_power += order_data["quantity"] * current_price

                        # Yield order update
                        order = await self.get_order(order_id)
                        if order:
                            yield order
                            processed_orders.add(order_id)
                            logger.info(f"Order {order_id} filled at ${current_price:.2f}")

            # Wait before checking again
            await asyncio.sleep(2)

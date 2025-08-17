"""
Backtest broker implementation for historical strategy testing.
Provides realistic order execution simulation with configurable slippage and commission models.
"""

# Standard library imports
import asyncio
from collections import defaultdict, deque
from datetime import UTC, datetime
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.config.config_manager import get_config
from main.models.common import (
    AccountInfo,
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from main.utils.core import ErrorHandlingMixin

# Import base interface and common models
from .broker_interface import BrokerInterface

logger = logging.getLogger(__name__)


class BacktestBroker(BrokerInterface, ErrorHandlingMixin):
    """
    Backtest broker for historical strategy testing.

    Features:
    - Realistic order execution with slippage models
    - Transaction cost modeling
    - Market impact simulation
    - Order book depth simulation
    - Historical data playback
    - Position tracking and P&L calculation
    """

    def __init__(self, config: Any = None):
        """
        Initialize backtest broker.

        Args:
            config: Configuration object
        """
        if config is None:
            config = get_config()
        super().__init__(config)
        ErrorHandlingMixin.__init__(self)

        # Backtest-specific configuration
        self.initial_capital = config.get("backtesting.initial_capital", 100000)
        self.commission_rate = config.get("backtesting.commission_rate", 0.001)
        self.slippage_model = config.get("backtesting.slippage_model", "linear")
        self.slippage_bps = config.get("backtesting.slippage_bps", 5)  # basis points
        self.market_impact_model = config.get("backtesting.market_impact_model", "square_root")
        self.fill_ratio = config.get("backtesting.fill_ratio", 1.0)  # Partial fill simulation

        # Account state
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.buying_power = self.initial_capital
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

        # Historical data storage
        self.historical_data: dict[str, pd.DataFrame] = {}
        self.current_timestamp: datetime | None = None
        self.data_index: dict[str, int] = defaultdict(int)

        # Order management
        self._next_order_id = 1
        self._pending_orders: deque = deque()
        self._order_history: list[Order] = []

        # Position tracking with detailed cost basis
        self._position_details: dict[str, list[dict]] = defaultdict(list)  # FIFO lots

        # Market state cache for current bar
        self._current_market_state: dict[str, dict] = {}

        # Performance tracking
        self.equity_curve: list[dict] = []
        self.trade_history: list[dict] = []

        logger.info(f"BacktestBroker initialized with ${self.initial_capital:,.2f} capital")

    def load_historical_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Load historical data for a symbol.

        Args:
            symbol: Trading symbol
            data: DataFrame with columns: open, high, low, close, volume
                  Index should be datetime
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Sort by time and store
        self.historical_data[symbol] = data.sort_index()
        self.data_index[symbol] = 0

        logger.info(
            f"Loaded {len(data)} bars for {symbol} from {data.index[0]} to {data.index[-1]}"
        )

    def set_current_time(self, timestamp: datetime) -> None:
        """
        Set the current simulation time.

        Args:
            timestamp: Current backtest timestamp
        """
        self.current_timestamp = timestamp

        # Update market state for all symbols
        self._update_market_state()

        # Process any pending orders
        asyncio.create_task(self._process_pending_orders())

        # Update account metrics
        self._update_account_metrics()

    def _update_market_state(self) -> None:
        """Update current market state from historical data."""
        for symbol, data in self.historical_data.items():
            # Find the appropriate bar for current timestamp
            mask = data.index <= self.current_timestamp
            if mask.any():
                idx = mask.sum() - 1
                self.data_index[symbol] = idx

                if idx < len(data):
                    bar = data.iloc[idx]
                    self._current_market_state[symbol] = {
                        "timestamp": data.index[idx],
                        "open": bar["open"],
                        "high": bar["high"],
                        "low": bar["low"],
                        "close": bar["close"],
                        "volume": bar["volume"],
                        "bid": bar["close"] * 0.9999,  # Simulated bid
                        "ask": bar["close"] * 1.0001,  # Simulated ask
                        "mid": bar["close"],
                    }

    def _calculate_slippage(self, order: Order, base_price: float) -> float:
        """
        Calculate slippage for order execution.

        Args:
            order: Order being executed
            base_price: Base execution price

        Returns:
            Adjusted price after slippage
        """
        if self.slippage_model == "zero":
            return base_price

        # Base slippage in basis points
        slippage_pct = self.slippage_bps / 10000.0

        if self.slippage_model == "linear":
            # Linear slippage based on order size
            size_factor = min(order.quantity / 1000, 2.0)  # Cap at 2x
            slippage_pct *= size_factor

        elif self.slippage_model == "square_root":
            # Square root market impact
            size_factor = min(np.sqrt(order.quantity / 1000), 2.0)
            slippage_pct *= size_factor

        # Apply slippage (adverse to order direction)
        if order.side == OrderSide.BUY:
            return base_price * (1 + slippage_pct)
        else:
            return base_price * (1 - slippage_pct)

    def _calculate_commission(self, order: Order, fill_price: float) -> float:
        """
        Calculate commission for order execution.

        Args:
            order: Executed order
            fill_price: Execution price

        Returns:
            Commission amount
        """
        # Simple percentage-based commission
        notional = order.quantity * fill_price
        return notional * self.commission_rate

    async def connect(self) -> bool:
        """Connect to backtest broker (always succeeds)."""
        self._connected = True
        logger.info("Connected to backtest broker")
        return True

    async def disconnect(self) -> None:
        """Disconnect from backtest broker."""
        self._connected = False
        logger.info("Disconnected from backtest broker")

    async def submit_order(self, order: Order) -> Order:
        """
        Submit order for execution.

        Args:
            order: Order to submit

        Returns:
            Updated order with broker order ID
        """
        try:
            # Validate order
            self.validate_order(order)

            # Generate broker order ID
            order.broker_order_id = f"BT-{self._next_order_id:06d}"
            self._next_order_id += 1

            # Set submission time
            order.submitted_at = self.current_timestamp or datetime.now(UTC)
            order.status = OrderStatus.SUBMITTED

            # Store order
            self._orders[order.broker_order_id] = order

            # Add to pending queue for processing
            self._pending_orders.append(order)

            # Log order submission
            logger.info(
                f"Order submitted: {order.broker_order_id} - {order.side.value} "
                f"{order.quantity} {order.symbol} @ {order.order_type.value}"
            )

            # Trigger callbacks
            self._trigger_order_callbacks(order)

            # Process immediately if market order
            if order.order_type == OrderType.MARKET:
                await self._process_pending_orders()

            return order

        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            order.status_message = str(e)
            raise

    async def _process_pending_orders(self) -> None:
        """Process pending orders against current market state."""
        if not self._pending_orders:
            return

        processed_orders = []

        for order in list(self._pending_orders):
            if order.symbol not in self._current_market_state:
                continue

            market_state = self._current_market_state[order.symbol]

            # Check if order should be filled
            should_fill, fill_price = self._check_order_fill(order, market_state)

            if should_fill:
                # Execute the order
                await self._execute_order(order, fill_price)
                processed_orders.append(order)

        # Remove processed orders from pending queue
        for order in processed_orders:
            self._pending_orders.remove(order)

    def _check_order_fill(self, order: Order, market_state: dict) -> tuple[bool, float | None]:
        """
        Check if order should be filled at current market state.

        Args:
            order: Order to check
            market_state: Current market data

        Returns:
            Tuple of (should_fill, fill_price)
        """
        high = market_state["high"]
        low = market_state["low"]
        close = market_state["close"]

        if order.order_type == OrderType.MARKET:
            # Market orders always fill at current close
            return True, close

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                # Buy limit fills if price drops to or below limit
                if low <= order.limit_price:
                    return True, min(order.limit_price, close)
            elif high >= order.limit_price:
                return True, max(order.limit_price, close)

        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                # Buy stop fills if price rises to or above stop
                if high >= order.stop_price:
                    return True, max(order.stop_price, close)
            elif low <= order.stop_price:
                return True, min(order.stop_price, close)

        elif order.order_type == OrderType.STOP_LIMIT:
            # First check if stop is triggered
            stop_triggered = False
            if order.side == OrderSide.BUY:
                stop_triggered = high >= order.stop_price
            else:
                stop_triggered = low <= order.stop_price

            if stop_triggered:
                # Then check limit price
                if order.side == OrderSide.BUY:
                    if low <= order.limit_price:
                        return True, min(order.limit_price, close)
                elif high >= order.limit_price:
                    return True, max(order.limit_price, close)

        return False, None

    async def _execute_order(self, order: Order, base_fill_price: float) -> None:
        """
        Execute order and update positions.

        Args:
            order: Order to execute
            base_fill_price: Base fill price before adjustments
        """
        # Calculate actual fill price with slippage
        fill_price = self._calculate_slippage(order, base_fill_price)

        # Calculate commission
        commission = self._calculate_commission(order, fill_price)

        # Calculate fill quantity (may be partial)
        fill_quantity = order.quantity * self.fill_ratio

        # Update order
        order.filled_quantity = fill_quantity
        order.average_fill_price = fill_price
        order.commission = commission
        order.filled_at = self.current_timestamp
        order.status = (
            OrderStatus.FILLED if fill_quantity == order.quantity else OrderStatus.PARTIALLY_FILLED
        )

        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= fill_quantity * fill_price + commission
        else:
            self.cash += fill_quantity * fill_price - commission

        # Update positions
        self._update_position(
            order.symbol,
            fill_quantity if order.side == OrderSide.BUY else -fill_quantity,
            fill_price,
        )

        # Record trade
        trade = {
            "timestamp": self.current_timestamp,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": fill_quantity,
            "price": fill_price,
            "commission": commission,
            "order_id": order.broker_order_id,
        }
        self.trade_history.append(trade)

        # Store in order history
        self._order_history.append(order)

        # Trigger callbacks
        self._trigger_order_callbacks(order)

        logger.info(
            f"Order filled: {order.broker_order_id} - {fill_quantity} @ ${fill_price:.2f} "
            f"(commission: ${commission:.2f})"
        )

    def _update_position(self, symbol: str, quantity_change: float, price: float) -> None:
        """
        Update position with FIFO cost basis tracking.

        Args:
            symbol: Symbol to update
            quantity_change: Change in quantity (positive for buy, negative for sell)
            price: Execution price
        """
        if quantity_change > 0:
            # Adding to position
            self._position_details[symbol].append(
                {"quantity": quantity_change, "price": price, "timestamp": self.current_timestamp}
            )
        else:
            # Reducing position (FIFO)
            remaining_to_sell = abs(quantity_change)
            realized_pnl = 0.0

            while remaining_to_sell > 0 and self._position_details[symbol]:
                lot = self._position_details[symbol][0]

                if lot["quantity"] <= remaining_to_sell:
                    # Sell entire lot
                    realized_pnl += lot["quantity"] * (price - lot["price"])
                    remaining_to_sell -= lot["quantity"]
                    self._position_details[symbol].pop(0)
                else:
                    # Partial sale of lot
                    realized_pnl += remaining_to_sell * (price - lot["price"])
                    lot["quantity"] -= remaining_to_sell
                    remaining_to_sell = 0

            self.realized_pnl += realized_pnl

        # Update position object
        total_quantity = sum(lot["quantity"] for lot in self._position_details[symbol])

        if total_quantity == 0:
            # Position closed
            if symbol in self._positions:
                del self._positions[symbol]
        else:
            # Calculate weighted average price
            total_value = sum(
                lot["quantity"] * lot["price"] for lot in self._position_details[symbol]
            )
            avg_price = total_value / total_quantity

            # Update or create position
            current_price = self._current_market_state.get(symbol, {}).get("close", price)

            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                average_price=avg_price,
                current_price=current_price,
                market_value=total_quantity * current_price,
                unrealized_pnl=(current_price - avg_price) * total_quantity,
                realized_pnl=self.realized_pnl,  # Total realized P&L
            )

    def _update_account_metrics(self) -> None:
        """Update account metrics based on current positions."""
        # Calculate total position value
        position_value = sum(pos.market_value for pos in self._positions.values())

        # Update equity
        self.equity = self.cash + position_value

        # Calculate unrealized P&L
        self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self._positions.values())

        # Update buying power (simplified - could add margin calculations)
        self.buying_power = self.cash

        # Record equity curve point
        self.equity_curve.append(
            {
                "timestamp": self.current_timestamp,
                "equity": self.equity,
                "cash": self.cash,
                "positions_value": position_value,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
            }
        )

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.

        Args:
            order_id: Broker order ID to cancel

        Returns:
            True if cancellation successful
        """
        if order_id not in self._orders:
            logger.warning(f"Order {order_id} not found for cancellation")
            return False

        order = self._orders[order_id]

        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
            return False

        # Remove from pending orders
        if order in self._pending_orders:
            self._pending_orders.remove(order)

        # Update order status
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = self.current_timestamp

        # Trigger callbacks
        self._trigger_order_callbacks(order)

        logger.info(f"Order cancelled: {order_id}")
        return True

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
            order_id: Broker order ID to modify
            limit_price: New limit price
            stop_price: New stop price
            quantity: New quantity

        Returns:
            Modified order
        """
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")

        order = self._orders[order_id]

        if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PENDING]:
            raise ValueError(f"Cannot modify order with status {order.status}")

        # Apply modifications
        if limit_price is not None:
            order.limit_price = limit_price
        if stop_price is not None:
            order.stop_price = stop_price
        if quantity is not None:
            order.quantity = quantity

        order.modified_at = self.current_timestamp

        # Trigger callbacks
        self._trigger_order_callbacks(order)

        logger.info(f"Order modified: {order_id}")
        return order

    async def get_positions(self) -> dict[str, Position]:
        """Get all positions."""
        # Update position prices with current market data
        for symbol, position in self._positions.items():
            if symbol in self._current_market_state:
                current_price = self._current_market_state[symbol]["close"]
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (
                    current_price - position.average_price
                ) * position.quantity

        return self._positions.copy()

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for specific symbol."""
        positions = await self.get_positions()
        return positions.get(symbol)

    async def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        """Get orders, optionally filtered by status."""
        orders = list(self._orders.values())

        if status:
            orders = [o for o in orders if o.status == status]

        return orders

    async def get_order(self, order_id: str) -> Order | None:
        """Get specific order by ID."""
        return self._orders.get(order_id)

    async def get_recent_orders(self) -> list[Order]:
        """Get recently executed orders."""
        # Return last 100 orders from history
        return self._order_history[-100:]

    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        return AccountInfo(
            account_id="BACKTEST-001",
            buying_power=self.buying_power,
            cash=self.cash,
            portfolio_value=self.equity,
            pattern_day_trader=False,
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            trade_suspended_by_user=False,
            multiplier=1.0,
            shorting_enabled=True,
            equity=self.equity,
            last_equity=self.equity,
            initial_margin=0.0,
            maintenance_margin=0.0,
            daytrade_count=0,
            balance_asof=self.current_timestamp or datetime.now(UTC),
        )

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol."""
        if symbol not in self._current_market_state:
            raise ValueError(f"No market data available for {symbol}")

        state = self._current_market_state[symbol]

        return MarketData(
            symbol=symbol,
            timestamp=state["timestamp"],
            bid=state["bid"],
            ask=state["ask"],
            last=state["close"],
            volume=state["volume"],
            open=state["open"],
            high=state["high"],
            low=state["low"],
            close=state["close"],
        )

    async def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Get latest quote for symbol."""
        if symbol not in self._current_market_state:
            return None

        state = self._current_market_state[symbol]

        return {
            "symbol": symbol,
            "timestamp": state["timestamp"].isoformat(),
            "bid": state["bid"],
            "ask": state["ask"],
            "bid_size": 100,  # Simulated
            "ask_size": 100,  # Simulated
            "last": state["close"],
            "volume": state["volume"],
        }

    async def subscribe_market_data(self, symbols: list[str]) -> None:
        """Subscribe to market data (no-op for backtest)."""
        logger.info(f"Subscribed to market data for: {symbols}")

    async def unsubscribe_market_data(self, symbols: list[str]) -> None:
        """Unsubscribe from market data (no-op for backtest)."""
        logger.info(f"Unsubscribed from market data for: {symbols}")

    async def get_historical_data(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> list[dict[str, Any]]:
        """Get historical market data."""
        if symbol not in self.historical_data:
            return []

        df = self.historical_data[symbol]

        # Filter by date range
        mask = (df.index >= start) & (df.index <= end)
        filtered_df = df[mask]

        # Convert to list of dicts using vectorized operations
        bars = [
            {
                "timestamp": timestamp,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            for timestamp, row in filtered_df.iterrows()
        ]

        return bars

    async def get_tradable_symbols(self) -> list[str]:
        """Get list of tradable symbols."""
        return list(self.historical_data.keys())

    async def is_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable."""
        return symbol in self.historical_data

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Calculate and return performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.equity_curve:
            return {}

        equity_series = pd.Series([e["equity"] for e in self.equity_curve])
        returns = equity_series.pct_change().dropna()

        # Calculate metrics
        total_return = (self.equity - self.initial_capital) / self.initial_capital

        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            max_drawdown = (equity_series / equity_series.cummax() - 1).min()
            win_rate = (
                len([t for t in self.trade_history if t.get("pnl", 0) > 0])
                / len(self.trade_history)
                if self.trade_history
                else 0
            )
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": len(self.trade_history),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "final_equity": self.equity,
            "initial_capital": self.initial_capital,
        }

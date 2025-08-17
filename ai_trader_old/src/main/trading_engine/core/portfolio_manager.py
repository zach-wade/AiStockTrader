# File: ai_trader/trading_engine/core/portfolio_manager.py

"""Portfolio and position management with real-time P&L tracking."""
# Standard library imports
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any

# Local imports
# NEW: Import common models (Position, AccountInfo) and enums
from main.models.common import OrderSide
from main.models.common import Position as CommonPosition

# NEW: Import BrokerInterface and Config
from main.trading_engine.brokers.broker_interface import BrokerInterface

# Import unified cache system
from main.utils.cache import CacheType, get_global_cache

logger = logging.getLogger(__name__)


# REVISED: Portfolio Dataclass (This remains local as it aggregates common.Position)
@dataclass(frozen=True)  # Portfolio state itself should be immutable where possible
class Portfolio:
    """Portfolio state and metrics."""

    cash: float
    # Use CommonPosition for the positions dictionary
    positions: dict[str, CommonPosition] = field(default_factory=dict)

    @property
    def total_value(self) -> float:
        """Calculates total portfolio value (cash + market value of positions)."""
        # Ensure market_value is accessed safely, as it's optional on the CommonPosition
        return self.cash + sum(
            pos.market_value for pos in self.positions.values() if pos.market_value is not None
        )

    @property
    def invested_value(self) -> float:
        """Calculates the total market value of all open positions."""
        return sum(
            pos.market_value for pos in self.positions.values() if pos.market_value is not None
        )

    @property
    def total_pnl(self) -> float:
        """Calculates the sum of unrealized P&L from all positions."""
        return sum(
            pos.unrealized_pnl for pos in self.positions.values() if pos.unrealized_pnl is not None
        )

    @property
    def position_count(self) -> int:
        """Returns the number of open positions."""
        return len(self.positions)


class PortfolioManager:
    """Deadlock-free portfolio state and positions manager with broker synchronization."""

    def __init__(self, broker: BrokerInterface, config: Any):
        self.broker = broker
        self.config_obj = config

        # Access config values from the structured Config object's raw dict
        self.initial_capital = config.get("trading.starting_cash", 100000.0)
        self.max_positions_limit = config.get("risk.max_positions", 10)

        # Initialize portfolio state (will be overwritten by broker synchronization)
        self.portfolio = Portfolio(cash=self.initial_capital, positions={})

        self.trade_history: list[dict[str, Any]] = []
        self.daily_pnl: list[float] = []

        # DEADLOCK FIX: Separate locks for different operations
        self._update_lock = asyncio.Lock()  # For portfolio updates from broker
        self._position_lock = asyncio.Lock()  # For position operations
        self._calculation_lock = asyncio.Lock()  # For calculations

        # Unified caching to reduce broker calls
        self.cache = get_global_cache()
        self.cache_ttl_seconds = 5  # 5-second cache TTL

        # Timeout configurations to prevent deadlocks
        self.broker_timeout = 10.0  # 10 seconds for broker calls
        self.lock_timeout = 30.0  # 30 seconds for lock acquisition

        logger.info("Deadlock-free Portfolio Manager initialized.")

    @asynccontextmanager
    async def _timed_lock(self, lock: asyncio.Lock, timeout: float = None):
        """Context manager for locks with timeout to prevent deadlocks."""
        timeout = timeout or self.lock_timeout
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            try:
                yield
            finally:
                lock.release()
        except TimeoutError:
            logger.error(f"Lock acquisition timeout after {timeout}s - potential deadlock detected")
            raise RuntimeError(f"Lock timeout after {timeout}s - deadlock prevented")

    async def _get_cached_portfolio(self) -> Portfolio | None:
        """Get portfolio from cache if valid, otherwise return None."""
        try:
            cached_portfolio = await self.cache.get(CacheType.PORTFOLIO, "current_portfolio")
            if cached_portfolio is not None:
                logger.debug("Using cached portfolio data")
                return cached_portfolio
        except Exception as e:
            logger.debug(f"Cache retrieval failed: {e}")
        return None

    async def _update_cache(self, portfolio: Portfolio):
        """Update portfolio cache with TTL."""
        try:
            await self.cache.set(
                CacheType.PORTFOLIO, "current_portfolio", portfolio, self.cache_ttl_seconds
            )
            logger.debug("Portfolio cache updated")
        except Exception as e:
            logger.debug(f"Cache update failed: {e}")

    async def initialize_portfolio_from_broker(self):
        """Fetches initial account info and positions from the broker to set up the portfolio."""
        async with self._timed_lock(self._update_lock, timeout=15.0):
            try:
                # Fetch current account info with timeout
                account_info = await asyncio.wait_for(
                    self.broker.get_account_info(), timeout=self.broker_timeout
                )

                # Fetch all open positions from the broker with timeout
                broker_positions_dict = await asyncio.wait_for(
                    self.broker.get_positions(), timeout=self.broker_timeout
                )

                # Create a new Portfolio object (immutable update)
                new_portfolio_positions: dict[str, CommonPosition] = {}
                for symbol, broker_pos in broker_positions_dict.items():
                    new_portfolio_positions[symbol] = broker_pos

                self.portfolio = Portfolio(
                    cash=account_info.cash, positions=new_portfolio_positions
                )

                # Update cache
                await self._update_cache(self.portfolio)

                logger.info(
                    f"Portfolio initialized/synchronized with broker. Cash: ${self.portfolio.cash:,.2f}, Positions: {self.portfolio.position_count}."
                )
            except TimeoutError:
                logger.error("Broker API timeout during portfolio initialization")
                self.portfolio = Portfolio(cash=self.initial_capital, positions={})
            except Exception as e:
                logger.error(f"Failed to initialize portfolio from broker: {e}", exc_info=True)
                self.portfolio = Portfolio(cash=self.initial_capital, positions={})

    async def _update_portfolio_internal(self) -> bool:
        """Internal method to update portfolio without acquiring lock. Returns success status."""
        try:
            # Fetch current account info with timeout
            account_info = await asyncio.wait_for(
                self.broker.get_account_info(), timeout=self.broker_timeout
            )

            # Fetch all open positions from the broker with timeout
            broker_positions_dict = await asyncio.wait_for(
                self.broker.get_positions(), timeout=self.broker_timeout
            )

            # Create a new dictionary of positions to replace the old one (for immutability)
            new_portfolio_positions: dict[str, CommonPosition] = {}
            for symbol, broker_pos in broker_positions_dict.items():
                new_portfolio_positions[symbol] = broker_pos

            # Create a new immutable Portfolio object with the updated state
            new_portfolio = Portfolio(cash=account_info.cash, positions=new_portfolio_positions)

            # Update both portfolio and cache atomically
            self.portfolio = new_portfolio
            await self._update_cache(new_portfolio)

            logger.debug(
                f"Portfolio state synchronized with broker successfully. Cash: {self.portfolio.cash}, Equity: {self.portfolio.total_value}"
            )
            return True

        except TimeoutError:
            logger.warning("Broker API timeout during portfolio update")
            return False
        except Exception as e:
            logger.error(f"Failed to update portfolio from broker: {e}", exc_info=True)
            return False

    async def update_portfolio(self) -> bool:
        """Fetches latest account info and positions from the broker and updates local state."""
        async with self._timed_lock(self._update_lock, timeout=15.0):
            return await self._update_portfolio_internal()

    async def can_open_position(self) -> bool:
        """Check if we can open a new position based on max_positions limit."""
        # Try cache first to avoid unnecessary broker calls
        portfolio = await self._get_cached_portfolio()

        if portfolio is None:
            # Update portfolio outside any position locks to prevent deadlock
            if not await self._update_portfolio_internal():
                logger.error("Failed to get current portfolio data")
                return False
            portfolio = self.portfolio

        return portfolio.position_count < self.max_positions_limit

    async def get_position_size(self, symbol: str, price: float, risk_pct: float = 0.02) -> float:
        """Calculate position size - deadlock free with cache-first approach."""
        # Use separate calculation lock with short timeout
        async with self._timed_lock(self._calculation_lock, timeout=5.0):
            # Get fresh portfolio data (cache-first approach)
            portfolio = await self._get_cached_portfolio()

        # Update outside calculation lock if needed to prevent deadlock
        if portfolio is None:
            if not await self._update_portfolio_internal():
                logger.warning("Portfolio data unavailable for position sizing")
                return 0.0
            portfolio = self.portfolio

        if portfolio.total_value <= 0:
            logger.warning(
                "Portfolio total value is zero or negative. Cannot calculate position size."
            )
            return 0.0

        # Calculate position size (no locks needed for math)
        risk_amount = portfolio.total_value * risk_pct
        shares = risk_amount / price if price > 0 else 0.0

        # Ensure we have enough cash
        if shares * price > portfolio.cash:
            shares = portfolio.cash / price if price > 0 else 0.0

        return shares

    async def open_position(
        self, symbol: str, quantity: float, price: float, side: OrderSide, strategy: str = "manual"
    ) -> bool:
        """
        Updates local portfolio state after a BUY order is filled.
        This method assumes the trade has already occurred via the broker.
        """
        async with self._timed_lock(self._position_lock, timeout=10.0):
            # Create a new positions dictionary for immutable update
            new_positions = dict(self.portfolio.positions)  # Copy current positions

            if symbol in new_positions:
                # Position already exists, update existing immutable position
                existing_pos = new_positions[symbol]

                new_total_quantity = existing_pos.quantity + quantity
                new_total_cost = (existing_pos.quantity * existing_pos.avg_entry_price) + (
                    quantity * price
                )

                new_positions[symbol] = CommonPosition(
                    symbol=existing_pos.symbol,
                    quantity=new_total_quantity,
                    avg_entry_price=new_total_cost / new_total_quantity,
                    current_price=price,  # Update current price too
                    side=existing_pos.side,  # Side doesn't change on average down/up
                    timestamp=datetime.now(),
                    market_value=new_total_quantity * price,
                    cost_basis=new_total_quantity
                    * (new_total_cost / new_total_quantity),  # Update cost basis
                    unrealized_pnl=(price - (new_total_cost / new_total_quantity))
                    * new_total_quantity,
                    unrealized_pnl_pct=(
                        (price / (new_total_cost / new_total_quantity) - 1) * 100
                        if (new_total_cost / new_total_quantity)
                        else 0.0
                    ),
                    realized_pnl=existing_pos.realized_pnl,  # Realized PnL carries over
                )
                logger.info(
                    f"Updated existing position for {symbol}: new quantity {new_total_quantity}, avg price {new_positions[symbol].avg_entry_price:.2f}"
                )
            else:
                # Open a new position
                new_positions[symbol] = CommonPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=price,
                    current_price=price,
                    side=side.value,  # Store enum value as string 'long' or 'short'
                    timestamp=datetime.now(),
                    market_value=quantity * price,
                    cost_basis=quantity * price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    realized_pnl=0.0,
                )
                logger.info(f"Opened new position: {symbol} {quantity} @ ${price:.2f}")

            # Create a new immutable Portfolio object with updated positions
            # Cash is updated by the broker, so we don't adjust it here.
            new_portfolio = Portfolio(cash=self.portfolio.cash, positions=new_positions)
            self.portfolio = new_portfolio

            # Update cache with new portfolio state
            await self._update_cache(new_portfolio)

            self.trade_history.append(
                {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "action": side.value.upper(),  # 'BUY' or 'SELL'
                    "quantity": quantity,
                    "price": price,
                    "strategy": strategy,
                }
            )

            return True

    async def close_position(self, symbol: str, quantity: float, price: float) -> float | None:
        """
        Updates local portfolio state after a SELL order is filled.
        Returns P&L from the closed portion.
        """
        async with self._timed_lock(self._position_lock, timeout=10.0):
            if symbol not in self.portfolio.positions:
                logger.warning(f"No position found for {symbol} to close.")
                return None

            position_to_close = self.portfolio.positions[symbol]

            if quantity > position_to_close.quantity:
                logger.warning(
                    f"Attempted to close {quantity} of {symbol}, but only {position_to_close.quantity} held. Closing all available."
                )
                quantity_to_close_actual = position_to_close.quantity
            else:
                quantity_to_close_actual = quantity

            # Calculate P&L for the portion being closed
            pnl_realized = (price - position_to_close.avg_entry_price) * quantity_to_close_actual

            # Create a new positions dictionary for immutable update
            new_positions = dict(self.portfolio.positions)

            if quantity_to_close_actual == position_to_close.quantity:
                # Position fully closed
                del new_positions[symbol]
                logger.info(
                    f"Fully closed position: {symbol} at ${price:.2f}, Realized P&L: ${pnl_realized:.2f}"
                )
            else:
                # Partial close: create a new immutable Position object for the updated state
                new_quantity = position_to_close.quantity - quantity_to_close_actual
                new_positions[symbol] = CommonPosition(
                    symbol=position_to_close.symbol,
                    quantity=new_quantity,
                    avg_entry_price=position_to_close.avg_entry_price,  # Avg entry price doesn't change on partial close
                    current_price=price,
                    side=position_to_close.side,
                    timestamp=datetime.now(),
                    market_value=new_quantity * price,
                    cost_basis=new_quantity * position_to_close.avg_entry_price,
                    unrealized_pnl=(price - position_to_close.avg_entry_price) * new_quantity,
                    unrealized_pnl_pct=(
                        (price / position_to_close.avg_entry_price - 1) * 100
                        if position_to_close.avg_entry_price
                        else 0.0
                    ),
                    realized_pnl=position_to_close.realized_pnl
                    + pnl_realized,  # Accumulate realized PnL
                )
                logger.info(
                    f"Partially closed position: {symbol} {quantity_to_close_actual} shares at ${price:.2f}. Remaining: {new_positions[symbol].quantity}."
                )

            # Create a new immutable Portfolio object with updated positions
            new_portfolio = Portfolio(cash=self.portfolio.cash, positions=new_positions)
            self.portfolio = new_portfolio

            # Update cache with new portfolio state
            await self._update_cache(new_portfolio)

            self.trade_history.append(
                {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "action": "SELL",
                    "quantity": quantity_to_close_actual,
                    "price": price,
                    "pnl_realized": pnl_realized,
                    "strategy": (
                        position_to_close.strategy
                        if hasattr(position_to_close, "strategy")
                        else "manual_close"
                    ),
                }
            )

            return pnl_realized

    async def update_prices(self, prices: dict[str, float]):
        """Update current prices for all tracked positions."""
        async with self._timed_lock(self._position_lock, timeout=5.0):
            # Create a new positions dictionary to update
            new_positions = dict(self.portfolio.positions)
            updated_count = 0
            for symbol, price in prices.items():
                if symbol in new_positions:
                    existing_pos = new_positions[symbol]
                    # Create a new immutable Position object with updated price
                    new_positions[symbol] = CommonPosition(
                        symbol=existing_pos.symbol,
                        quantity=existing_pos.quantity,
                        avg_entry_price=existing_pos.avg_entry_price,
                        current_price=price,  # Update current price
                        side=existing_pos.side,
                        timestamp=datetime.now(),  # Update timestamp for price
                        market_value=existing_pos.quantity * price,  # Recalculate market value
                        cost_basis=existing_pos.cost_basis,
                        realized_pnl=existing_pos.realized_pnl,  # Carries over
                        unrealized_pnl=(price - existing_pos.avg_entry_price)
                        * existing_pos.quantity,  # Recalculate P&L
                        unrealized_pnl_pct=(
                            (price / existing_pos.avg_entry_price - 1) * 100
                            if existing_pos.avg_entry_price
                            else 0.0
                        ),
                    )
                    updated_count += 1

            if updated_count > 0:
                # Replace the portfolio with a new immutable version
                new_portfolio = Portfolio(cash=self.portfolio.cash, positions=new_positions)
                self.portfolio = new_portfolio

                # Update cache with new portfolio state
                await self._update_cache(new_portfolio)
                logger.debug(f"Updated prices for {updated_count} positions.")

    async def get_positions(self) -> list[CommonPosition]:
        """Returns current open positions - deadlock free."""
        portfolio = await self._get_cached_portfolio()

        if portfolio is None:
            if not await self._update_portfolio_internal():
                logger.error("Failed to get current positions")
                return []
            portfolio = self.portfolio

        return list(portfolio.positions.values())

    async def get_position_by_symbol(self, symbol: str) -> CommonPosition | None:
        """Returns a specific position by symbol - deadlock free."""
        portfolio = await self._get_cached_portfolio()

        if portfolio is None:
            if not await self._update_portfolio_internal():
                logger.error("Failed to get position data")
                return None
            portfolio = self.portfolio

        return portfolio.positions.get(symbol)

    async def get_cash_balance(self) -> float:
        """Returns the current cash balance - deadlock free."""
        portfolio = await self._get_cached_portfolio()

        if portfolio is None:
            if not await self._update_portfolio_internal():
                logger.warning("Failed to get current cash balance")
                return 0.0
            portfolio = self.portfolio

        return portfolio.cash

    async def get_equity(self) -> float:
        """Returns the current total equity - deadlock free."""
        portfolio = await self._get_cached_portfolio()

        if portfolio is None:
            if not await self._update_portfolio_internal():
                logger.warning("Failed to get current equity")
                return 0.0
            portfolio = self.portfolio

        return portfolio.total_value

    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get current portfolio summary - deadlock free."""
        portfolio = await self._get_cached_portfolio()

        if portfolio is None:
            if not await self._update_portfolio_internal():
                logger.warning("Failed to get portfolio summary")
                return {
                    "total_value": 0.0,
                    "cash": 0.0,
                    "invested": 0.0,
                    "total_pnl_unrealized": 0.0,
                    "position_count": 0,
                    "positions": {},
                }
            portfolio = self.portfolio

        return {
            "total_value": portfolio.total_value,
            "cash": portfolio.cash,
            "invested": portfolio.invested_value,
            "total_pnl_unrealized": portfolio.total_pnl,
            "position_count": portfolio.position_count,
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.pnl if hasattr(pos, "pnl") else pos.unrealized_pnl,
                    "unrealized_pnl_pct": (
                        pos.pnl_pct if hasattr(pos, "pnl_pct") else pos.unrealized_pnl_pct
                    ),
                    "market_value": pos.market_value,
                    "cost_basis": pos.cost_basis,
                    "realized_pnl_cumulative": pos.realized_pnl,
                    "side": pos.side,
                    "timestamp": pos.timestamp.isoformat(),
                }
                for symbol, pos in portfolio.positions.items()
            },
        }

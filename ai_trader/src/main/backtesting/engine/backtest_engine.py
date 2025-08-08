# File: backtesting/engine/backtest_engine.py

"""
Event-Driven Backtesting Engine.

Core backtesting engine that orchestrates:
- Event processing and distribution
- Market data replay
- Strategy execution
- Order management and execution
- Performance tracking
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Type
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import heapq
import pandas as pd

from main.utils.core import (
    get_logger, ErrorHandlingMixin, ensure_utc, 
    AsyncCircuitBreaker, RateLimiter, process_in_batches
)
from main.interfaces.events import Event, MarketEvent, OrderEvent, EventType
from main.events.types import FillEvent
from main.events.core import EventBusFactory
from main.models.common import Order, OrderType, OrderSide, Strategy
from .portfolio import Portfolio
from .market_simulator import MarketSimulator, ExecutionMode
from .cost_model import CostModel, create_default_cost_model
from .bar_aggregator import BarAggregator

logger = get_logger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes."""
    SINGLE_SYMBOL = "single_symbol"
    MULTI_SYMBOL = "multi_symbol"
    PORTFOLIO = "portfolio"
    WALK_FORWARD = "walk_forward"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_cash: float = 100000.0
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1day"
    mode: BacktestMode = BacktestMode.PORTFOLIO
    use_adjusted_prices: bool = True
    include_dividends: bool = True
    include_splits: bool = True
    execution_mode: ExecutionMode = ExecutionMode.REALISTIC
    commission_model: Optional[str] = "percentage"
    slippage_model: Optional[str] = "spread"
    enable_shorting: bool = True
    margin_ratio: float = 2.0  # 2:1 margin
    
    def validate(self) -> bool:
        """Validate configuration."""
        if self.start_date >= self.end_date:
            logger.error("Start date must be before end date")
            return False
        
        if self.initial_cash <= 0:
            logger.error("Initial cash must be positive")
            return False
        
        if not self.symbols:
            logger.error("At least one symbol required")
            return False
        
        return True


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    config: BacktestConfig
    portfolio_history: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    positions_history: List[Dict[str, Any]]
    events_processed: int
    execution_time: float
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_return': self.metrics.get('total_return', 0),
            'sharpe_ratio': self.metrics.get('sharpe_ratio', 0),
            'max_drawdown': self.metrics.get('max_drawdown', 0),
            'win_rate': self.metrics.get('win_rate', 0),
            'total_trades': len(self.trades),
            'profit_factor': self.metrics.get('profit_factor', 0),
            'execution_time': self.execution_time
        }


class BacktestEngine(ErrorHandlingMixin):
    """
    Event-driven backtesting engine.
    
    Coordinates market data replay, strategy execution, and performance tracking.
    """
    
    def __init__(self,
                 config: BacktestConfig,
                 strategy: Strategy,
                 data_source: Optional[Any] = None,
                 cost_model: Optional[CostModel] = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
            strategy: Trading strategy to test
            data_source: Data source for market data
            cost_model: Cost model for execution
        """
        if not config.validate():
            raise ValueError("Invalid backtest configuration")
        
        self.config = config
        self.strategy = strategy
        self.data_source = data_source
        self.cost_model = cost_model or create_default_cost_model()
        
        # Core components
        self.portfolio = Portfolio(
            initial_cash=config.initial_cash,
            margin_ratio=config.margin_ratio
        )
        
        self.market_simulator = MarketSimulator(
            cost_model=self.cost_model,
            execution_mode=config.execution_mode
        )
        
        self.bar_aggregator = BarAggregator()
        
        # Event management
        self.event_bus = None  # Will be initialized in setup()
        self.event_queue: List[Event] = []
        self.current_time = config.start_date
        
        # Performance tracking
        self.events_processed = 0
        self.market_events_count = 0
        self.order_events_count = 0
        self.fill_events_count = 0
        
        # Circuit breaker for resilience
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=10,
            recovery_timeout=60
        )
        
        logger.info(f"BacktestEngine initialized for {len(config.symbols)} symbols")
    
    async def setup(self):
        """Setup backtest engine and components."""
        # Initialize event bus
        self.event_bus = EventBusFactory.create()
        
        # Start the event bus
        await self.event_bus.start()
        
        # Subscribe to events
        await self._setup_event_handlers()
        
        # Initialize strategy
        if hasattr(self.strategy, 'initialize'):
            await self.strategy.initialize()
        
        logger.info("BacktestEngine setup complete")
    
    async def _setup_event_handlers(self):
        """Setup event handlers."""
        # Market events trigger strategy
        self.event_bus.subscribe(
            EventType.MARKET_DATA,
            self._handle_market_event
        )
        
        # Order events go to simulator
        self.event_bus.subscribe(
            EventType.ORDER_PLACED,
            self._handle_order_event
        )
        
        # Fill events update portfolio
        self.event_bus.subscribe(
            EventType.ORDER_FILLED,
            self._handle_fill_event
        )
    
    async def run(self) -> BacktestResult:
        """
        Run the backtest.
        
        Returns:
            BacktestResult with performance metrics
        """
        start_time = datetime.now()
        
        try:
            # Setup
            await self.setup()
            
            # Load and process market data
            await self._run_backtest_loop()
            
            # Calculate final metrics
            result = self._generate_results()
            
            # Cleanup
            await self.cleanup()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            logger.info(f"Backtest completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def _run_backtest_loop(self):
        """Main backtest event processing loop."""
        # Load market data
        market_data = await self._load_market_data()
        
        # Create initial market events
        await self._create_market_events(market_data)
        
        # Process events
        while self.event_queue:
            # Get next event
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp
            self.events_processed += 1
            
            # Process event
            await self.event_bus.publish(event)
            
            # Process any pending orders
            if self.events_processed % 10 == 0:  # Check every 10 events
                fills = self.market_simulator.process_orders(self.current_time)
                for fill in fills:
                    await self.event_bus.publish(fill)
            
            # Take portfolio snapshot periodically
            if self.events_processed % 100 == 0:
                self.portfolio.take_snapshot()
    
    async def _load_market_data(self) -> pd.DataFrame:
        """Load market data for backtest period."""
        if not self.data_source:
            raise ValueError("No data source configured")
        
        # Load data for all symbols
        all_data = []
        
        for symbol in self.config.symbols:
            data = await self.data_source.get_historical_data(
                symbol=symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                timeframe=self.config.timeframe,
                adjusted=self.config.use_adjusted_prices
            )
            
            data['symbol'] = symbol
            all_data.append(data)
        
        # Combine all data
        market_data = pd.concat(all_data, ignore_index=True)
        market_data.sort_values(['timestamp', 'symbol'], inplace=True)
        
        logger.info(f"Loaded {len(market_data)} bars for {len(self.config.symbols)} symbols")
        return market_data
    
    async def _create_market_events(self, market_data: pd.DataFrame):
        """Create market events from historical data."""
        for _, row in market_data.iterrows():
            # Create market event
            event = MarketEvent(
                timestamp=ensure_utc(row['timestamp']),
                symbol=row['symbol'],
                data={
                    'symbol': row['symbol'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'timeframe': self.config.timeframe
                }
            )
            
            # Add to event queue
            heapq.heappush(self.event_queue, event)
            
            # Handle bar aggregation for multi-timeframe
            if self.config.timeframe == '1minute':
                # Aggregate to higher timeframes
                higher_events = self.bar_aggregator.process_minute_bar(
                    symbol=row['symbol'],
                    timestamp=row['timestamp'],
                    ohlcv={
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                )
                
                for higher_event in higher_events:
                    heapq.heappush(self.event_queue, higher_event)
    
    async def _handle_market_event(self, event: MarketEvent):
        """Handle market data event."""
        self.market_events_count += 1
        
        # Update portfolio prices
        symbol = event.data['symbol']
        price = event.data['close']
        self.portfolio.update_position_price(symbol, price, event.timestamp)
        
        # Update market simulator
        spread_bps = self.market_simulator.default_spread_bps
        spread = price * (spread_bps / 10000)
        
        self.market_simulator.update_market_data(
            symbol=symbol,
            bid=price - spread/2,
            ask=price + spread/2,
            bid_size=10000,
            ask_size=10000,
            last_price=price,
            timestamp=event.timestamp
        )
        
        # Pass to strategy
        if hasattr(self.strategy, 'on_market_data'):
            orders = await self.circuit_breaker.call(
                self.strategy.on_market_data,
                event
            )
            
            # Submit any orders generated
            if orders:
                for order in orders:
                    await self._submit_order(order)
    
    async def _handle_order_event(self, event: OrderEvent):
        """Handle order event."""
        self.order_events_count += 1
        
        # Order events are handled by market simulator
        # which generates fill events
        pass
    
    async def _handle_fill_event(self, event: FillEvent):
        """Handle fill event."""
        self.fill_events_count += 1
        
        # Calculate costs
        costs = self.cost_model.calculate_trade_cost(
            quantity=event.quantity,
            price=event.price,
            order_side=event.side,
            order_type=OrderType.MARKET  # Assume market for now
        )
        
        # Update portfolio
        success = self.portfolio.process_fill(event, costs)
        
        if success:
            logger.debug(f"Fill processed: {event.symbol} {event.quantity} @ {event.price}")
        else:
            logger.warning(f"Fill rejected: {event.symbol} - insufficient buying power")
        
        # Notify strategy
        if hasattr(self.strategy, 'on_fill'):
            await self.strategy.on_fill(event)
    
    async def _submit_order(self, order: Order):
        """Submit an order to the market simulator."""
        # Validate order
        if not self._validate_order(order):
            logger.warning(f"Invalid order rejected: {order}")
            return
        
        # Submit to simulator
        order_event = self.market_simulator.submit_order(order, self.current_time)
        
        # Add to event queue
        heapq.heappush(self.event_queue, order_event)
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order before submission."""
        # Check if symbol is in universe
        if order.symbol not in self.config.symbols:
            return False
        
        # Check shorting permission
        if order.side == OrderSide.SELL:
            position = self.portfolio.get_position(order.symbol)
            if not position and not self.config.enable_shorting:
                return False
        
        return True
    
    def _generate_results(self) -> BacktestResult:
        """Generate backtest results."""
        # Get portfolio history
        portfolio_history = self.portfolio.get_history_dataframe()
        
        # Get trades
        trades_df = pd.DataFrame(self.portfolio.trades)
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_history, trades_df)
        
        # Create equity curve
        equity_curve = portfolio_history['total_equity'] if not portfolio_history.empty else pd.Series()
        
        # Calculate drawdown
        drawdown_curve = self._calculate_drawdown(equity_curve) if not equity_curve.empty else pd.Series()
        
        # Get position history
        positions_history = []
        for snapshot in self.portfolio.history:
            positions = {
                'timestamp': snapshot.timestamp,
                'positions': len(self.portfolio.positions),
                'long_exposure': snapshot.long_exposure,
                'short_exposure': snapshot.short_exposure,
                'net_exposure': snapshot.net_exposure
            }
            positions_history.append(positions)
        
        return BacktestResult(
            config=self.config,
            portfolio_history=portfolio_history,
            trades=trades_df,
            metrics=metrics,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            positions_history=positions_history,
            events_processed=self.events_processed,
            execution_time=0  # Set later
        )
    
    def _calculate_metrics(self, portfolio_history: pd.DataFrame,
                          trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        if portfolio_history.empty:
            return metrics
        
        # Returns
        initial_equity = self.config.initial_cash
        final_equity = portfolio_history['total_equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        metrics['total_return'] = total_return * 100
        metrics['final_equity'] = final_equity
        
        # Daily returns
        equity = portfolio_history['total_equity']
        returns = equity.pct_change().dropna()
        
        if len(returns) > 0:
            # Sharpe ratio (assuming 252 trading days)
            metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
            
            # Max drawdown
            drawdown = self._calculate_drawdown(equity)
            metrics['max_drawdown'] = drawdown.min() * 100
            
            # Volatility
            metrics['volatility'] = returns.std() * (252 ** 0.5) * 100
        
        # Trade metrics
        if not trades.empty:
            # Calculate P&L for each trade
            # This is simplified - in reality would need to match buys/sells
            metrics['total_trades'] = len(trades)
            
            # Win rate (simplified)
            profitable_trades = 0  # Would need proper P&L calculation
            metrics['win_rate'] = 0
            
            # Average trade
            metrics['avg_trade_size'] = trades['value'].mean()
            
            # Commissions
            metrics['total_commission'] = trades['commission'].sum()
        
        return metrics
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.event_bus:
            await self.event_bus.stop()
        
        logger.info("BacktestEngine cleanup complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'events_processed': self.events_processed,
            'market_events': self.market_events_count,
            'order_events': self.order_events_count,
            'fill_events': self.fill_events_count,
            'portfolio_metrics': self.portfolio.get_portfolio_metrics(),
            'simulator_stats': self.market_simulator.get_statistics(),
            'cost_model_stats': self.cost_model.get_statistics()
        }


def create_backtest_engine(config: BacktestConfig,
                          strategy: Strategy,
                          data_source: Any) -> BacktestEngine:
    """Create a configured backtest engine."""
    return BacktestEngine(
        config=config,
        strategy=strategy,
        data_source=data_source,
        cost_model=create_default_cost_model()
    )
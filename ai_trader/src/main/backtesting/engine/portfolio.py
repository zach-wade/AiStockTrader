# File: backtesting/engine/portfolio.py

"""
Portfolio Management for Backtesting Engine.

Handles:
- Position tracking and management
- Cash balance management
- P&L calculation
- Portfolio metrics computation
- Risk metrics tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import pandas as pd

from main.utils.core import get_logger, ErrorHandlingMixin, ensure_utc
from main.models.common import OrderSide, Position
from main.events.types import FillEvent
from .cost_model import CostComponents

logger = get_logger(__name__)


@dataclass
class PortfolioPosition:
    """Extended position information for portfolio tracking."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    side: OrderSide
    entry_time: datetime
    last_update_time: datetime
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return abs(self.quantity) * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return abs(self.quantity) * self.avg_cost
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L for the position."""
        if self.quantity > 0:  # Long position
            return self.quantity * (self.current_price - self.avg_cost)
        else:  # Short position
            return abs(self.quantity) * (self.avg_cost - self.current_price)
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_pct(self) -> float:
        """Return percentage on position."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'current_price': self.current_price,
            'side': self.side.value,
            'entry_time': self.entry_time.isoformat(),
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'return_pct': self.return_pct,
            'commission_paid': self.commission_paid
        }


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""
    timestamp: datetime
    cash: float
    positions_value: float
    total_equity: float
    margin_used: float
    buying_power: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    position_count: int
    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cash': self.cash,
            'positions_value': self.positions_value,
            'total_equity': self.total_equity,
            'margin_used': self.margin_used,
            'buying_power': self.buying_power,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'position_count': self.position_count,
            'long_exposure': self.long_exposure,
            'short_exposure': self.short_exposure,
            'net_exposure': self.net_exposure,
            'gross_exposure': self.gross_exposure
        }


class Portfolio(ErrorHandlingMixin):
    """
    Portfolio manager for backtesting.
    
    Tracks positions, cash, and performance metrics throughout the backtest.
    """
    
    def __init__(self,
                 initial_cash: float = 100000.0,
                 margin_ratio: float = 1.0,  # 1.0 = cash account, 2.0 = 50% margin
                 min_position_size: float = 0.01,
                 track_history: bool = True):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash balance
            margin_ratio: Margin multiplier (1.0 for cash, 2.0 for 2:1 margin)
            min_position_size: Minimum position size to track
            track_history: Whether to track portfolio history
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.margin_ratio = margin_ratio
        self.min_position_size = min_position_size
        self.track_history = track_history
        
        # Position tracking
        self.positions: Dict[str, PortfolioPosition] = {}
        
        # Performance tracking
        self.realized_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        # Trade history
        self.trades: List[Dict[str, Any]] = []
        
        # Portfolio history
        self.history: List[PortfolioSnapshot] = []
        
        # Current timestamp
        self.current_time = ensure_utc(datetime.now())
        
        logger.info(f"Portfolio initialized with ${initial_cash:,.2f}")
    
    def update_position_price(self, symbol: str, price: float, timestamp: datetime):
        """Update current price for a position."""
        if symbol in self.positions:
            self.positions[symbol].current_price = price
            self.positions[symbol].last_update_time = timestamp
    
    def update_all_prices(self, prices: Dict[str, float], timestamp: datetime):
        """Update prices for all positions."""
        self.current_time = timestamp
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.update_position_price(symbol, price, timestamp)
    
    def process_fill(self, fill_event: FillEvent, costs: CostComponents) -> bool:
        """
        Process a fill event and update portfolio.
        
        Args:
            fill_event: Fill event from execution
            costs: Trading costs for this fill
            
        Returns:
            True if fill was processed successfully
        """
        try:
            symbol = fill_event.symbol
            quantity = fill_event.quantity
            price = fill_event.price
            side = fill_event.side
            timestamp = fill_event.timestamp
            
            # Calculate cash impact
            trade_value = quantity * price
            total_cost = costs.total_cost
            
            if side == OrderSide.BUY:
                cash_impact = -(trade_value + total_cost)
            else:  # SELL
                cash_impact = trade_value - total_cost
            
            # Check if we have sufficient buying power
            if not self._check_buying_power(cash_impact):
                logger.warning(f"Insufficient buying power for {symbol} trade")
                return False
            
            # Update cash
            self.cash += cash_impact
            
            # Update position
            self._update_position(symbol, quantity, price, side, timestamp, costs)
            
            # Update statistics
            self.total_commission += costs.commission
            self.total_slippage += costs.slippage
            
            # Record trade
            if self.trades is not None:
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': side.value,
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value,
                    'commission': costs.commission,
                    'total_cost': total_cost,
                    'cash_impact': cash_impact
                })
            
            logger.debug(f"Processed fill: {symbol} {side.value} {quantity} @ {price}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing fill: {e}")
            return False
    
    def _check_buying_power(self, cash_required: float) -> bool:
        """Check if sufficient buying power exists."""
        if cash_required >= 0:  # Cash inflow
            return True
        
        buying_power = self.get_buying_power()
        return buying_power >= abs(cash_required)
    
    def _update_position(self, symbol: str, quantity: int, price: float,
                        side: OrderSide, timestamp: datetime, costs: CostComponents):
        """Update or create position."""
        if symbol not in self.positions:
            # New position
            if side == OrderSide.BUY:
                position_quantity = quantity
            else:
                position_quantity = -quantity
            
            self.positions[symbol] = PortfolioPosition(
                symbol=symbol,
                quantity=position_quantity,
                avg_cost=price,
                current_price=price,
                side=side,
                entry_time=timestamp,
                last_update_time=timestamp,
                commission_paid=costs.commission
            )
        else:
            # Update existing position
            position = self.positions[symbol]
            old_quantity = position.quantity
            
            if side == OrderSide.BUY:
                new_quantity = old_quantity + quantity
            else:
                new_quantity = old_quantity - quantity
            
            # Handle position changes
            if abs(new_quantity) < self.min_position_size:
                # Position closed
                self._close_position(symbol, price, costs)
            elif old_quantity * new_quantity < 0:
                # Position flipped (long to short or vice versa)
                self._flip_position(symbol, new_quantity, price, timestamp, costs)
            else:
                # Position increased or decreased
                if abs(new_quantity) > abs(old_quantity):
                    # Position increased - update average cost
                    total_cost = abs(old_quantity) * position.avg_cost + quantity * price
                    position.avg_cost = total_cost / abs(new_quantity)
                else:
                    # Position decreased - calculate realized P&L
                    if old_quantity > 0:  # Long position reduced
                        realized = quantity * (price - position.avg_cost)
                    else:  # Short position reduced  
                        realized = quantity * (position.avg_cost - price)
                    position.realized_pnl += realized
                    self.realized_pnl += realized
                
                position.quantity = new_quantity
                position.current_price = price
                position.last_update_time = timestamp
                position.commission_paid += costs.commission
    
    def _close_position(self, symbol: str, price: float, costs: CostComponents):
        """Close a position completely."""
        position = self.positions[symbol]
        
        # Calculate final realized P&L
        if position.quantity > 0:  # Closing long
            realized = position.quantity * (price - position.avg_cost)
        else:  # Closing short
            realized = abs(position.quantity) * (position.avg_cost - price)
        
        position.realized_pnl += realized
        self.realized_pnl += realized
        
        # Remove position
        del self.positions[symbol]
        logger.info(f"Closed position {symbol}: P&L = ${realized:,.2f}")
    
    def _flip_position(self, symbol: str, new_quantity: int, price: float,
                      timestamp: datetime, costs: CostComponents):
        """Handle position flip from long to short or vice versa."""
        position = self.positions[symbol]
        
        # First close the old position
        if position.quantity > 0:  # Was long
            realized = position.quantity * (price - position.avg_cost)
        else:  # Was short
            realized = abs(position.quantity) * (position.avg_cost - price)
        
        position.realized_pnl += realized
        self.realized_pnl += realized
        
        # Create new position in opposite direction
        position.quantity = new_quantity
        position.avg_cost = price
        position.current_price = price
        position.side = OrderSide.BUY if new_quantity > 0 else OrderSide.SELL
        position.entry_time = timestamp
        position.last_update_time = timestamp
        position.commission_paid += costs.commission
        
        logger.info(f"Flipped position {symbol}: realized P&L = ${realized:,.2f}")
    
    def get_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PortfolioPosition]:
        """Get all open positions."""
        return self.positions.copy()
    
    def get_total_equity(self) -> float:
        """Calculate total portfolio equity."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_buying_power(self) -> float:
        """Calculate available buying power."""
        total_equity = self.get_total_equity()
        margin_used = self.get_margin_used()
        max_margin = total_equity * self.margin_ratio
        return max_margin - margin_used
    
    def get_margin_used(self) -> float:
        """Calculate margin currently in use."""
        long_value = sum(pos.market_value for pos in self.positions.values() 
                        if pos.quantity > 0)
        short_value = sum(pos.market_value for pos in self.positions.values()
                         if pos.quantity < 0)
        return long_value + short_value * 2  # Shorts require 2x margin
    
    def get_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.get_unrealized_pnl()
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio metrics."""
        total_equity = self.get_total_equity()
        positions_value = sum(pos.market_value for pos in self.positions.values())
        
        long_exposure = sum(pos.market_value for pos in self.positions.values()
                           if pos.quantity > 0)
        short_exposure = sum(pos.market_value for pos in self.positions.values()
                            if pos.quantity < 0)
        
        return {
            'cash': self.cash,
            'positions_value': positions_value,
            'total_equity': total_equity,
            'initial_cash': self.initial_cash,
            'margin_used': self.get_margin_used(),
            'buying_power': self.get_buying_power(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.get_total_pnl(),
            'return_pct': ((total_equity - self.initial_cash) / self.initial_cash) * 100,
            'position_count': len(self.positions),
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'gross_exposure': long_exposure + short_exposure,
            'cash_percentage': (self.cash / total_equity * 100) if total_equity > 0 else 100,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage
        }
    
    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state."""
        metrics = self.get_portfolio_metrics()
        
        snapshot = PortfolioSnapshot(
            timestamp=self.current_time,
            cash=metrics['cash'],
            positions_value=metrics['positions_value'],
            total_equity=metrics['total_equity'],
            margin_used=metrics['margin_used'],
            buying_power=metrics['buying_power'],
            unrealized_pnl=metrics['unrealized_pnl'],
            realized_pnl=metrics['realized_pnl'],
            total_pnl=metrics['total_pnl'],
            position_count=metrics['position_count'],
            long_exposure=metrics['long_exposure'],
            short_exposure=metrics['short_exposure'],
            net_exposure=metrics['net_exposure'],
            gross_exposure=metrics['gross_exposure']
        )
        
        if self.track_history:
            self.history.append(snapshot)
        
        return snapshot
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Get portfolio history as pandas DataFrame."""
        if not self.history:
            return pd.DataFrame()
        
        data = [snapshot.to_dict() for snapshot in self.history]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self.realized_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.trades.clear()
        self.history.clear()
        logger.info("Portfolio reset to initial state")
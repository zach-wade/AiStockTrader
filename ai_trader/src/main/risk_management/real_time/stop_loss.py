"""Dynamic stop loss management system."""
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StopLossType(Enum):
    """Types of stop loss orders."""
    FIXED = "fixed"
    TRAILING = "trailing"
    ATR_BASED = "atr_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    TIME_BASED = "time_based"
    CHANDELIER = "chandelier"

@dataclass
class StopLoss:
    """Stop loss configuration."""
    position_id: str
    symbol: str
    stop_type: StopLossType
    initial_stop: float
    current_stop: float
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    time_limit: Optional[datetime] = None
    high_water_mark: Optional[float] = None

class DynamicStopLossManager:
    """Manage dynamic stop losses for all positions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stop_losses: Dict[str, StopLoss] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.execution_callbacks = []
        self._lock = asyncio.Lock()
        
        # Configuration
        self.default_stop_pct = config.get('default_stop_pct', 0.02)
        self.use_atr = config.get('use_atr_stops', True)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.trail_activation_pct = config.get('trail_activation_pct', 0.01)
        self.time_stop_hours = config.get('time_stop_hours', 48)
    
    async def create_stop_loss(self, 
                              position_id: str,
                              symbol: str,
                              entry_price: float,
                              quantity: float,
                              position_side: str = 'LONG') -> StopLoss:
        """Create initial stop loss for a new position."""
        async with self._lock:
            # Determine stop type based on market conditions
            stop_type = await self._determine_stop_type(symbol)
            
            # Calculate initial stop price
            initial_stop = await self._calculate_initial_stop(
                symbol, entry_price, stop_type, position_side
            )
            
            # Create stop loss object
            stop_loss = StopLoss(
                position_id=position_id,
                symbol=symbol,
                stop_type=stop_type,
                initial_stop=initial_stop,
                current_stop=initial_stop,
                high_water_mark=entry_price if position_side == 'LONG' else None
            )
            
            # Set trailing parameters if applicable
            if stop_type == StopLossType.TRAILING:
                stop_loss.trail_percent = 0.01  # 1% trailing
            
            # Set time limit if applicable
            if stop_type == StopLossType.TIME_BASED:
                stop_loss.time_limit = datetime.now() + timedelta(hours=self.time_stop_hours)
            
            self.stop_losses[position_id] = stop_loss
            
            logger.info(f"Created {stop_type.value} stop loss for {symbol} at ${initial_stop:.2f}")
            
            return stop_loss
    
    async def update_stop_losses(self, market_data: Dict[str, Dict[str, float]]):
        """Update all stop losses with new market data."""
        async with self._lock:
            positions_to_close = []
            
            for position_id, stop_loss in self.stop_losses.items():
                symbol = stop_loss.symbol
                
                if symbol not in market_data:
                    continue
                
                current_price = market_data[symbol]['price']
                
                # Check if stop is hit
                if self._is_stop_triggered(stop_loss, current_price):
                    positions_to_close.append(position_id)
                    continue
                
                # Update stop based on type
                if stop_loss.stop_type == StopLossType.TRAILING:
                    await self._update_trailing_stop(stop_loss, current_price)
                
                elif stop_loss.stop_type == StopLossType.ATR_BASED:
                    await self._update_atr_stop(stop_loss, symbol)
                
                elif stop_loss.stop_type == StopLossType.VOLATILITY_ADJUSTED:
                    await self._update_volatility_stop(stop_loss, symbol)
                
                elif stop_loss.stop_type == StopLossType.TIME_BASED:
                    await self._update_time_stop(stop_loss, current_price)
                
                elif stop_loss.stop_type == StopLossType.CHANDELIER:
                    await self._update_chandelier_stop(stop_loss, symbol)
            
            # Execute stops
            for position_id in positions_to_close:
                await self._execute_stop_loss(position_id)
    
    async def _determine_stop_type(self, symbol: str) -> StopLossType:
        """Determine best stop type based on market conditions."""
        if symbol not in self.market_data or len(self.market_data[symbol]) < 20:
            return StopLossType.FIXED
        
        data = self.market_data[symbol]
        
        # Calculate recent volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # High volatility: Use ATR-based stops
        if volatility > 0.03:
            return StopLossType.ATR_BASED
        
        # Trending market: Use trailing stops
        sma_20 = data['close'].rolling(20).mean()
        if data['close'].iloc[-1] > sma_20.iloc[-1] * 1.02:
            return StopLossType.TRAILING
        
        # Default to volatility-adjusted
        return StopLossType.VOLATILITY_ADJUSTED
    
    async def _calculate_initial_stop(self, 
                                    symbol: str,
                                    entry_price: float,
                                    stop_type: StopLossType,
                                    position_side: str) -> float:
        """Calculate initial stop loss price."""
        if stop_type == StopLossType.FIXED:
            # Simple percentage-based stop
            if position_side == 'LONG':
                return entry_price * (1 - self.default_stop_pct)
            else:
                return entry_price * (1 + self.default_stop_pct)
        
        elif stop_type == StopLossType.ATR_BASED and symbol in self.market_data:
            # ATR-based stop
            data = self.market_data[symbol]
            atr = self._calculate_atr(data, period=14)
            
            if position_side == 'LONG':
                return entry_price - (atr * self.atr_multiplier)
            else:
                return entry_price + (atr * self.atr_multiplier)
        
        elif stop_type == StopLossType.VOLATILITY_ADJUSTED and symbol in self.market_data:
            # Volatility-adjusted stop
            data = self.market_data[symbol]
            volatility = data['close'].pct_change().std()
            stop_distance = entry_price * volatility * 2
            
            if position_side == 'LONG':
                return entry_price - stop_distance
            else:
                return entry_price + stop_distance
        
        # Default to fixed stop
        return entry_price * (1 - self.default_stop_pct)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr
    
    def _is_stop_triggered(self, stop_loss: StopLoss, current_price: float) -> bool:
        """Check if stop loss is triggered."""
        # For long positions
        if stop_loss.high_water_mark is not None:
            return current_price <= stop_loss.current_stop
        
        # For short positions (not implemented in this example)
        return False
    
    async def _update_trailing_stop(self, stop_loss: StopLoss, current_price: float):
        """Update trailing stop loss."""
        if stop_loss.high_water_mark is None:
            return
        
        # Update high water mark
        if current_price > stop_loss.high_water_mark:
            stop_loss.high_water_mark = current_price
            
            # Update stop loss
            if stop_loss.trail_percent:
                new_stop = current_price * (1 - stop_loss.trail_percent)
            elif stop_loss.trail_amount:
                new_stop = current_price - stop_loss.trail_amount
            else:
                new_stop = current_price * (1 - self.default_stop_pct)
            
            # Only move stop up, never down
            if new_stop > stop_loss.current_stop:
                stop_loss.current_stop = new_stop
                logger.info(f"Updated trailing stop for {stop_loss.symbol} to ${new_stop:.2f}")
    
    async def _update_atr_stop(self, stop_loss: StopLoss, symbol: str):
        """Update ATR-based stop loss."""
        if symbol not in self.market_data:
            return
        
        data = self.market_data[symbol]
        current_price = data['close'].iloc[-1]
        atr = self._calculate_atr(data)
        
        # Calculate new stop
        new_stop = current_price - (atr * self.atr_multiplier)
        
        # Only move stop up for long positions
        if new_stop > stop_loss.current_stop:
            stop_loss.current_stop = new_stop
            logger.info(f"Updated ATR stop for {symbol} to ${new_stop:.2f}")
    
    async def _update_volatility_stop(self, stop_loss: StopLoss, symbol: str):
        """Update volatility-adjusted stop loss."""
        if symbol not in self.market_data:
            return
        
        data = self.market_data[symbol]
        current_price = data['close'].iloc[-1]
        
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        
        # Adjust stop distance based on volatility
        stop_distance = current_price * volatility * 2
        new_stop = current_price - stop_distance
        
        # Only move stop up
        if new_stop > stop_loss.current_stop:
            stop_loss.current_stop = new_stop
    
    async def _update_time_stop(self, stop_loss: StopLoss, current_price: float):
        """Update time-based stop loss."""
        if stop_loss.time_limit is None:
            return
        
        time_remaining = (stop_loss.time_limit - datetime.now()).total_seconds() / 3600
        
        if time_remaining <= 0:
            # Time limit reached, tighten stop
            stop_loss.current_stop = current_price * 0.999  # Very tight stop
        else:
            # Gradually tighten stop as time passes
            progress = 1 - (time_remaining / self.time_stop_hours)
            
            # Linear tightening from initial stop to current price
            stop_range = current_price - stop_loss.initial_stop
            stop_loss.current_stop = stop_loss.initial_stop + (stop_range * progress * 0.8)
    
    async def _update_chandelier_stop(self, stop_loss: StopLoss, symbol: str):
        """Update Chandelier stop loss."""
        if symbol not in self.market_data:
            return
        
        data = self.market_data[symbol]
        
        # Calculate highest high over lookback period
        lookback = 22
        highest_high = data['high'].rolling(lookback).max().iloc[-1]
        
        # Calculate ATR
        atr = self._calculate_atr(data)
        
        # Chandelier stop = Highest High - ATR * Multiplier
        new_stop = highest_high - (atr * self.atr_multiplier)
        
        # Only move stop up
        if new_stop > stop_loss.current_stop:
            stop_loss.current_stop = new_stop
    
    async def _execute_stop_loss(self, position_id: str):
        """Execute stop loss order."""
        stop_loss = self.stop_losses[position_id]
        
        logger.warning(f"Stop loss triggered for {stop_loss.symbol} at ${stop_loss.current_stop:.2f}")
        
        # Call execution callbacks
        for callback in self.execution_callbacks:
            await callback(position_id, stop_loss)
        
        # Remove stop loss
        del self.stop_losses[position_id]
    
    def add_execution_callback(self, callback):
        """Add callback for stop loss execution."""
        self.execution_callbacks.append(callback)
    
    async def update_market_data(self, symbol: str, data: pd.DataFrame):
        """Update market data for stop loss calculations."""
        self.market_data[symbol] = data
    
    async def remove_stop_loss(self, position_id: str):
        """Remove stop loss (when position is closed)."""
        async with self._lock:
            if position_id in self.stop_losses:
                del self.stop_losses[position_id]
    
    async def get_stop_loss_summary(self) -> Dict[str, Any]:
        """Get summary of all active stop losses."""
        async with self._lock:
            return {
                'active_stops': len(self.stop_losses),
                'stops_by_type': self._count_by_type(),
                'average_stop_distance': self._calculate_avg_stop_distance(),
                'stops': {
                    pid: {
                        'symbol': sl.symbol,
                        'type': sl.stop_type.value,
                        'current_stop': sl.current_stop,
                        'initial_stop': sl.initial_stop
                    }
                    for pid, sl in self.stop_losses.items()
                }
            }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count stop losses by type."""
        counts = {}
        for sl in self.stop_losses.values():
            stop_type = sl.stop_type.value
            counts[stop_type] = counts.get(stop_type, 0) + 1
        return counts
    
    def _calculate_avg_stop_distance(self) -> float:
        """Calculate average stop distance percentage."""
        distances = []
        
        for sl in self.stop_losses.values():
            if sl.symbol in self.market_data:
                current_price = self.market_data[sl.symbol]['close'].iloc[-1]
                distance = abs(current_price - sl.current_stop) / current_price
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
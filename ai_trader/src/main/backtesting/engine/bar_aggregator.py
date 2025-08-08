"""
Bar Aggregator for Real-time Multi-timeframe Construction
Perfect time machine implementation - builds higher timeframes as time progresses.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

from .backtest_engine import MarketEvent

logger = logging.getLogger(__name__)


class BarAggregator:
    """
    Real-time bar aggregator that builds higher timeframes from 1-minute bars.
    Ensures no look-ahead bias by only completing bars when time boundaries are crossed.
    """
    
    def __init__(self):
        # Store incomplete bars for each symbol/timeframe combination
        self.incomplete_bars: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
        
        # Track bar boundaries
        self.timeframe_minutes = {
            '5minute': 5,
            '15minute': 15,
            '1hour': 60,
            '1day': 1440  # 24 * 60
        }
        
        # Completed events to inject back into queue
        self.completed_events: List[MarketEvent] = []
    
    def process_minute_bar(self, symbol: str, timestamp: datetime, ohlcv: Dict[str, float]) -> List[MarketEvent]:
        """
        Process a 1-minute bar and return any completed higher timeframe bars.
        
        Args:
            symbol: Stock symbol
            timestamp: Timestamp of the 1-minute bar
            ohlcv: Dict with open, high, low, close, volume
            
        Returns:
            List of MarketEvent objects for completed higher timeframe bars
        """
        completed_events = []
        
        # Process each timeframe
        for timeframe, minutes in self.timeframe_minutes.items():
            completed_bar = self._aggregate_bar(symbol, timeframe, timestamp, ohlcv, minutes)
            
            if completed_bar:
                # Create market event for completed bar
                event = MarketEvent(
                    timestamp=completed_bar['timestamp'],
                    data={
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'ohlcv': {
                            'open': completed_bar['open'],
                            'high': completed_bar['high'],
                            'low': completed_bar['low'],
                            'close': completed_bar['close'],
                            'volume': completed_bar['volume']
                        }
                    }
                )
                completed_events.append(event)
                logger.debug(f"Completed {timeframe} bar for {symbol} at {completed_bar['timestamp']}")
        
        return completed_events
    
    def _aggregate_bar(self, symbol: str, timeframe: str, timestamp: datetime, ohlcv: Dict[str, float], minutes: int) -> Optional[Dict]:
        """
        Aggregate a 1-minute bar into a higher timeframe bar.
        
        Returns:
            Dict with completed bar data if bar boundary was crossed, None otherwise
        """
        # Calculate bar start time based on timeframe
        bar_start = self._get_bar_start_time(timestamp, minutes)
        bar_key = f"{symbol}_{timeframe}_{bar_start.isoformat()}"
        
        # Get or create incomplete bar
        if bar_key not in self.incomplete_bars[symbol][timeframe]:
            # Starting new bar
            self.incomplete_bars[symbol][timeframe][bar_key] = {
                'start_time': bar_start,
                'end_time': bar_start + timedelta(minutes=minutes),
                'open': ohlcv['open'],
                'high': ohlcv['high'],
                'low': ohlcv['low'],
                'close': ohlcv['close'],
                'volume': ohlcv['volume'],
                'bar_count': 1
            }
            return None  # Bar not complete yet
        
        # Update existing incomplete bar
        bar = self.incomplete_bars[symbol][timeframe][bar_key]
        bar['high'] = max(bar['high'], ohlcv['high'])
        bar['low'] = min(bar['low'], ohlcv['low'])
        bar['close'] = ohlcv['close']  # Always use latest close
        bar['volume'] += ohlcv['volume']
        bar['bar_count'] += 1
        
        # Check if we've moved to next bar period
        current_bar_start = self._get_bar_start_time(timestamp, minutes)
        if current_bar_start > bar['start_time']:
            # Previous bar is complete, return it
            completed_bar = {
                'timestamp': bar['end_time'],
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume']
            }
            
            # Remove completed bar and start new one
            del self.incomplete_bars[symbol][timeframe][bar_key]
            
            # Start new bar with current minute
            new_bar_key = f"{symbol}_{timeframe}_{current_bar_start.isoformat()}"
            self.incomplete_bars[symbol][timeframe][new_bar_key] = {
                'start_time': current_bar_start,
                'end_time': current_bar_start + timedelta(minutes=minutes),
                'open': ohlcv['open'],
                'high': ohlcv['high'],
                'low': ohlcv['low'],
                'close': ohlcv['close'],
                'volume': ohlcv['volume'],
                'bar_count': 1
            }
            
            return completed_bar
        
        return None  # Bar still incomplete
    
    def _get_bar_start_time(self, timestamp: datetime, minutes: int) -> datetime:
        """
        Calculate the start time of the bar that contains the given timestamp.
        Ensures proper alignment for different timeframes.
        """
        if minutes == 1440:  # Daily bars
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif minutes == 60:  # Hourly bars
            return timestamp.replace(minute=0, second=0, microsecond=0)
        else:  # 5min, 15min bars
            # Round down to nearest timeframe boundary
            minutes_since_hour = timestamp.minute
            aligned_minute = (minutes_since_hour // minutes) * minutes
            return timestamp.replace(minute=aligned_minute, second=0, microsecond=0)
    
    def get_current_bars(self, symbol: str) -> Dict[str, Dict]:
        """
        Get current incomplete bars for a symbol (for debugging/monitoring).
        """
        return dict(self.incomplete_bars[symbol])
    
    def finalize_remaining_bars(self, end_time: datetime) -> List[MarketEvent]:
        """
        Finalize any remaining incomplete bars at the end of backtesting.
        """
        completed_events = []
        
        for symbol in self.incomplete_bars:
            for timeframe in self.incomplete_bars[symbol]:
                for bar_key, bar in list(self.incomplete_bars[symbol][timeframe].items()):
                    # Complete the bar with end_time as close time
                    completed_bar = {
                        'timestamp': min(bar['end_time'], end_time),
                        'open': bar['open'],
                        'high': bar['high'],
                        'low': bar['low'],
                        'close': bar['close'],
                        'volume': bar['volume']
                    }
                    
                    event = MarketEvent(
                        timestamp=completed_bar['timestamp'],
                        data={
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'ohlcv': completed_bar
                        }
                    )
                    completed_events.append(event)
        
        # Clear all incomplete bars
        self.incomplete_bars.clear()
        
        return completed_events
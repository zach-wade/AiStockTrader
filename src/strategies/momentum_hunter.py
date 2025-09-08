"""
Momentum Hunter Strategy
Quick momentum detection for immediate trading opportunities
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paper_trading_alpaca import AlpacaMarketData


class MomentumHunter:
    """
    Hunts for momentum opportunities using technical indicators.
    Designed for quick deployment with minimal dependencies.
    """
    
    def __init__(self, 
                 symbols: Optional[List[str]] = None,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 volume_surge_multiplier: float = 2.0,
                 min_price: float = 10.0,
                 max_price: float = 500.0):
        """
        Initialize the momentum hunter.
        
        Args:
            symbols: List of symbols to monitor (default: top liquid stocks)
            rsi_oversold: RSI threshold for oversold condition
            rsi_overbought: RSI threshold for overbought condition
            volume_surge_multiplier: Multiplier for volume surge detection
            min_price: Minimum stock price to consider
            max_price: Maximum stock price to consider
        """
        self.symbols = symbols or [
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 
            'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD'
        ]
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_surge_multiplier = volume_surge_multiplier
        self.min_price = min_price
        self.max_price = max_price
        self.market_data = AlpacaMarketData()
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices: List of prices (oldest to newest)
            period: RSI period (default 14)
            
        Returns:
            RSI value between 0 and 100
        """
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: List[float]) -> Optional[Dict[str, float]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: List of prices (oldest to newest)
            
        Returns:
            Dict with MACD line, signal line, and histogram
        """
        if len(prices) < 26:
            return None
        
        # Calculate EMAs
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        if ema12 is None or ema26 is None:
            return None
        
        # MACD line
        macd_line = ema12 - ema26
        
        # Signal line (9-period EMA of MACD)
        macd_values = []
        for i in range(26, len(prices)):
            ema12_i = self._calculate_ema(prices[:i+1], 12)
            ema26_i = self._calculate_ema(prices[:i+1], 26)
            if ema12_i and ema26_i:
                macd_values.append(ema12_i - ema26_i)
        
        if len(macd_values) < 9:
            return None
        
        signal_line = self._calculate_ema(macd_values, 9)
        
        if signal_line is None:
            return None
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_ema(self, values: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(values) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = values[0]
        
        for value in values[1:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def detect_volume_surge(self, volumes: List[float]) -> bool:
        """
        Detect if there's a volume surge.
        
        Args:
            volumes: List of volumes (oldest to newest)
            
        Returns:
            True if volume surge detected
        """
        if len(volumes) < 20:
            return False
        
        avg_volume = np.mean(volumes[:-1])
        current_volume = volumes[-1]
        
        return current_volume > avg_volume * self.volume_surge_multiplier
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single symbol for momentum opportunities.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Signal dict if opportunity found, None otherwise
        """
        try:
            # Get current price
            current_price = float(self.market_data.get_latest_price(symbol))
            
            # Check price range
            if current_price < self.min_price or current_price > self.max_price:
                return None
            
            # Get historical data (would need to implement)
            # For now, using mock data
            prices = self._get_mock_prices(symbol, current_price)
            volumes = self._get_mock_volumes(symbol)
            
            # Calculate indicators
            rsi = self.calculate_rsi(prices)
            macd_data = self.calculate_macd(prices)
            volume_surge = self.detect_volume_surge(volumes)
            
            # Generate signals
            signal = None
            confidence = 0.5
            reasons = []
            
            if rsi:
                if rsi < self.rsi_oversold:
                    signal = 'BUY'
                    confidence += 0.2
                    reasons.append(f'RSI oversold ({rsi:.1f})')
                elif rsi > self.rsi_overbought:
                    signal = 'BUY'  # Changed to BUY for testing paper trading
                    confidence += 0.2
                    reasons.append(f'RSI momentum ({rsi:.1f})')
            
            if macd_data and macd_data['histogram'] > 0:
                if signal != 'SELL':
                    signal = 'BUY'
                    confidence += 0.15
                    reasons.append('MACD bullish crossover')
            
            if volume_surge:
                confidence += 0.15
                reasons.append('Volume surge detected')
            
            # Only return strong signals
            if signal and confidence >= 0.65:
                return {
                    'symbol': symbol,
                    'action': signal,
                    'price': current_price,
                    'confidence': min(confidence, 0.95),
                    'reasons': reasons,
                    'timestamp': datetime.now(timezone.utc),
                    'indicators': {
                        'rsi': rsi,
                        'macd': macd_data['macd'] if macd_data else None,
                        'volume_surge': volume_surge
                    }
                }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
        
        return None
    
    def _get_mock_prices(self, symbol: str, current_price: float, days: int = 30) -> List[float]:
        """Generate mock historical prices for testing."""
        prices = []
        price = current_price
        
        for i in range(days):
            # Random walk
            change = np.random.normal(0, current_price * 0.02)
            price = max(1, price - change)  # Going backwards in time
            prices.append(price)
        
        prices.reverse()  # Oldest to newest
        prices.append(current_price)
        
        return prices
    
    def _get_mock_volumes(self, symbol: str, days: int = 30) -> List[float]:
        """Generate mock volume data for testing."""
        base_volume = 10_000_000
        volumes = []
        
        for _ in range(days):
            volume = base_volume * np.random.uniform(0.5, 1.5)
            volumes.append(volume)
        
        # Add potential surge
        if np.random.random() > 0.7:
            volumes[-1] *= 2.5
        
        return volumes
    
    def scan(self) -> List[Dict[str, Any]]:
        """
        Scan all symbols for momentum opportunities.
        
        Returns:
            List of trading signals sorted by confidence
        """
        signals = []
        
        print(f"🔍 Scanning {len(self.symbols)} symbols for momentum...")
        
        for symbol in self.symbols:
            signal = self.analyze_symbol(symbol)
            if signal:
                signals.append(signal)
                print(f"  ✅ {symbol}: {signal['action']} signal (confidence: {signal['confidence']:.2f})")
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals
    
    def get_position_size(self, 
                         account_value: float,
                         signal_confidence: float,
                         max_position_pct: float = 0.10) -> float:
        """
        Calculate position size using Kelly Criterion variant.
        
        Args:
            account_value: Total account value
            signal_confidence: Confidence in the signal (0-1)
            max_position_pct: Maximum position size as % of account
            
        Returns:
            Position size in dollars
        """
        # Kelly fraction modified by confidence
        # More aggressive for paper trading testing
        kelly_fraction = signal_confidence * 0.5  # Medium Kelly
        position_pct = min(kelly_fraction, max_position_pct)
        
        # Minimum position size of 5% for testing
        position_pct = max(position_pct, 0.05)
        
        return account_value * position_pct
    
    def calculate_stops(self, 
                       entry_price: float,
                       signal_type: str,
                       atr_multiplier: float = 2.0) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            signal_type: 'BUY' or 'SELL'
            atr_multiplier: ATR multiplier for stops
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        # Simple percentage-based stops for now
        # Would use ATR in production
        stop_pct = 0.02  # 2% stop loss
        target_pct = 0.04  # 4% take profit (2:1 R/R)
        
        if signal_type == 'BUY':
            stop_loss = entry_price * (1 - stop_pct)
            take_profit = entry_price * (1 + target_pct)
        else:  # SELL
            stop_loss = entry_price * (1 + stop_pct)
            take_profit = entry_price * (1 - target_pct)
        
        return stop_loss, take_profit


def test_momentum_hunter():
    """Test the momentum hunter strategy."""
    print("Testing Momentum Hunter Strategy...")
    print("=" * 50)
    
    # Initialize hunter
    hunter = MomentumHunter(
        symbols=['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ'],
        rsi_oversold=35,
        rsi_overbought=65
    )
    
    # Scan for opportunities
    signals = hunter.scan()
    
    if signals:
        print(f"\n📊 Found {len(signals)} trading opportunities:")
        print("-" * 50)
        
        for signal in signals[:3]:  # Top 3 signals
            print(f"\n{signal['symbol']}:")
            print(f"  Action: {signal['action']}")
            print(f"  Price: ${signal['price']:.2f}")
            print(f"  Confidence: {signal['confidence']:.2%}")
            print(f"  Reasons: {', '.join(signal['reasons'])}")
            
            # Calculate position size
            account_value = 10000  # $10K account
            position_size = hunter.get_position_size(account_value, signal['confidence'])
            shares = int(position_size / signal['price'])
            
            print(f"  Position: {shares} shares (${position_size:.2f})")
            
            # Calculate stops
            stop_loss, take_profit = hunter.calculate_stops(signal['price'], signal['action'])
            print(f"  Stop Loss: ${stop_loss:.2f}")
            print(f"  Take Profit: ${take_profit:.2f}")
    else:
        print("\n❌ No trading opportunities found")
    
    print("\n" + "=" * 50)
    print("✅ Momentum Hunter test complete")


if __name__ == "__main__":
    test_momentum_hunter()
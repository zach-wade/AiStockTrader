"""
Mean Reversion Strategy Example

This example demonstrates a complete mean reversion trading strategy
that identifies oversold conditions and trades the bounce.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from main.trading.strategies.base_strategy import BaseStrategy
from main.config.config_manager import get_config
from main.data_providers.alpaca.market_data import AlpacaMarketClient
from main.utils.resilience.recovery_manager import get_global_recovery_manager
import structlog

logger = structlog.get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy that:
    - Identifies oversold conditions using multiple indicators
    - Enters positions when price is below moving average
    - Uses dynamic position sizing based on volatility
    - Implements stop loss and take profit
    """
    
    def __init__(
        self,
        config,
        lookback_period: int = 20,
        entry_threshold: float = -2.0,  # Z-score threshold
        exit_threshold: float = 0.5,
        max_position_size: float = 0.1,  # Max 10% per position
        stop_loss_pct: float = 0.03,  # 3% stop loss
        take_profit_pct: float = 0.05  # 5% take profit
    ):
        super().__init__(config)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Initialize market data client
        recovery_manager = get_global_recovery_manager()
        self.market_client = AlpacaMarketClient(config, recovery_manager)
        
        # Track positions
        self.positions = {}
    
    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze symbol for mean reversion opportunity."""
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period * 2)
        
        bars = await self.market_client.get_bars(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe='1Day'
        )
        
        if not bars or len(bars) < self.lookback_period:
            return {'signal': 0, 'reason': 'insufficient_data'}
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars
        ])
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators
        indicators = self._calculate_indicators(df)
        
        # Generate signal
        signal = self._generate_signal(indicators)
        
        return {
            'signal': signal['action'],
            'reason': signal['reason'],
            'indicators': indicators,
            'current_price': df['close'].iloc[-1],
            'position_size': signal.get('position_size', 0),
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit')
        }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate mean reversion indicators."""
        # Simple moving average
        sma = df['close'].rolling(window=self.lookback_period).mean()
        
        # Standard deviation
        std = df['close'].rolling(window=self.lookback_period).std()
        
        # Z-score
        z_score = (df['close'] - sma) / std
        
        # RSI
        rsi = self._calculate_rsi(df['close'], period=14)
        
        # Bollinger Bands
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_sma = sma.iloc[-1]
        current_z_score = z_score.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Price position relative to bands
        band_position = (current_price - lower_band.iloc[-1]) / (
            upper_band.iloc[-1] - lower_band.iloc[-1]
        )
        
        # Volume analysis
        avg_volume = df['volume'].rolling(window=self.lookback_period).mean()
        volume_ratio = df['volume'].iloc[-1] / avg_volume.iloc[-1]
        
        # Volatility
        volatility = std.iloc[-1] / current_sma
        
        return {
            'price': current_price,
            'sma': current_sma,
            'z_score': current_z_score,
            'rsi': current_rsi,
            'band_position': band_position,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'upper_band': upper_band.iloc[-1],
            'lower_band': lower_band.iloc[-1]
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _generate_signal(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Generate trading signal based on indicators."""
        z_score = indicators['z_score']
        rsi = indicators['rsi']
        band_position = indicators['band_position']
        volatility = indicators['volatility']
        volume_ratio = indicators['volume_ratio']
        
        # Entry conditions
        if (z_score < self.entry_threshold and 
            rsi < 35 and 
            band_position < 0.2 and
            volume_ratio > 0.8):  # Ensure decent volume
            
            # Calculate position size based on volatility
            position_size = self._calculate_position_size(volatility)
            
            # Calculate stop loss and take profit
            stop_loss = indicators['price'] * (1 - self.stop_loss_pct)
            take_profit = indicators['price'] * (1 + self.take_profit_pct)
            
            return {
                'action': 1,  # Buy signal
                'reason': 'oversold_conditions',
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': min(abs(z_score) / 3, 1.0)  # 0-1 confidence
            }
        
        # Exit conditions
        elif (z_score > self.exit_threshold or
              rsi > 65 or
              band_position > 0.8):
            
            return {
                'action': -1,  # Sell signal
                'reason': 'mean_reversion_complete',
                'confidence': min(z_score / 2, 1.0)
            }
        
        # Hold
        else:
            return {
                'action': 0,  # No signal
                'reason': 'no_clear_signal'
            }
    
    def _calculate_position_size(self, volatility: float) -> float:
        """Calculate position size based on volatility."""
        # Lower position size for higher volatility
        volatility_factor = max(0.5, 1 - (volatility * 2))
        position_size = self.max_position_size * volatility_factor
        
        # Ensure minimum position size
        return max(0.02, min(position_size, self.max_position_size))
    
    def get_parameters(self) -> Dict:
        """Return strategy parameters."""
        return {
            'lookback_period': self.lookback_period,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }


async def main():
    """Example usage of mean reversion strategy."""
    
    # Load configuration
    config = get_config(config_name='prod', environment='dev')
    
    # Create strategy
    strategy = MeanReversionStrategy(
        config,
        lookback_period=20,
        entry_threshold=-2.0,
        exit_threshold=0.5
    )
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    print("=== Mean Reversion Strategy Example ===\n")
    print("Strategy Parameters:")
    params = strategy.get_parameters()
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\n=== Analyzing Symbols ===\n")
    
    signals = []
    
    for symbol in symbols:
        try:
            print(f"\nAnalyzing {symbol}...")
            analysis = await strategy.analyze(symbol)
            
            # Display results
            action_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
            action = action_map[analysis['signal']]
            
            print(f"  Signal: {action}")
            print(f"  Reason: {analysis['reason']}")
            
            if 'indicators' in analysis:
                ind = analysis['indicators']
                print(f"  Price: ${ind['price']:.2f}")
                print(f"  Z-Score: {ind['z_score']:.2f}")
                print(f"  RSI: {ind['rsi']:.2f}")
                print(f"  Band Position: {ind['band_position']:.2%}")
                print(f"  Volatility: {ind['volatility']:.2%}")
            
            if analysis['signal'] == 1:  # Buy signal
                print(f"  Position Size: {analysis['position_size']:.2%} of portfolio")
                print(f"  Stop Loss: ${analysis['stop_loss']:.2f}")
                print(f"  Take Profit: ${analysis['take_profit']:.2f}")
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': ind['price'],
                    'position_size': analysis['position_size']
                })
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    print("\n\n=== Signal Summary ===")
    if signals:
        print(f"Total buy signals: {len(signals)}")
        for signal in signals:
            print(f"  {signal['symbol']}: BUY at ${signal['price']:.2f} "
                  f"(Size: {signal['position_size']:.2%})")
    else:
        print("No buy signals generated")


if __name__ == "__main__":
    asyncio.run(main())
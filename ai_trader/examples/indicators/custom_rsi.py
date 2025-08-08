"""
Custom RSI Indicator Example

This example demonstrates how to create a custom RSI indicator
that integrates with the AI Trader indicator framework.
"""

import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from main.data_pipeline.indicators.base_indicator import BaseIndicator
from main.config.config_manager import get_config
import structlog

logger = structlog.get_logger(__name__)


class CustomRSI(BaseIndicator):
    """
    Custom RSI implementation with additional features:
    - Divergence detection
    - Overbought/oversold zones
    - Signal smoothing
    """
    
    def __init__(self, config, period: int = 14, smooth_period: int = 3):
        super().__init__(config)
        self.period = period
        self.smooth_period = smooth_period
        self.overbought = 70
        self.oversold = 30
    
    async def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom RSI with additional features."""
        if len(data) < self.period + 1:
            return pd.DataFrame()
        
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=self.period).mean()
        avg_loss = losses.rolling(window=self.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Smooth RSI
        rsi_smooth = rsi.rolling(window=self.smooth_period).mean()
        
        # Create result dataframe
        result = pd.DataFrame(index=data.index)
        result['rsi'] = rsi
        result['rsi_smooth'] = rsi_smooth
        
        # Add overbought/oversold signals
        result['overbought'] = rsi > self.overbought
        result['oversold'] = rsi < self.oversold
        
        # Detect divergences
        result['bullish_divergence'] = self._detect_divergence(
            data['close'], rsi, 'bullish'
        )
        result['bearish_divergence'] = self._detect_divergence(
            data['close'], rsi, 'bearish'
        )
        
        # Generate trading signals
        result['signal'] = self._generate_signals(result)
        
        return result
    
    def _detect_divergence(
        self, 
        price: pd.Series, 
        rsi: pd.Series, 
        divergence_type: str,
        lookback: int = 20
    ) -> pd.Series:
        """Detect RSI divergences."""
        divergence = pd.Series(False, index=price.index)
        
        if len(price) < lookback:
            return divergence
        
        for i in range(lookback, len(price)):
            window_price = price.iloc[i-lookback:i]
            window_rsi = rsi.iloc[i-lookback:i]
            
            if divergence_type == 'bullish':
                # Price making lower lows, RSI making higher lows
                price_lows = window_price[window_price == window_price.min()]
                if len(price_lows) >= 2:
                    if price.iloc[i] < price_lows.iloc[0]:
                        rsi_at_price_lows = rsi.iloc[price_lows.index]
                        if rsi.iloc[i] > rsi_at_price_lows.iloc[0]:
                            divergence.iloc[i] = True
            
            elif divergence_type == 'bearish':
                # Price making higher highs, RSI making lower highs
                price_highs = window_price[window_price == window_price.max()]
                if len(price_highs) >= 2:
                    if price.iloc[i] > price_highs.iloc[0]:
                        rsi_at_price_highs = rsi.iloc[price_highs.index]
                        if rsi.iloc[i] < rsi_at_price_highs.iloc[0]:
                            divergence.iloc[i] = True
        
        return divergence
    
    def _generate_signals(self, rsi_data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on RSI conditions."""
        signals = pd.Series(0, index=rsi_data.index)
        
        # Buy signals
        buy_condition = (
            (rsi_data['oversold']) |
            (rsi_data['bullish_divergence']) |
            ((rsi_data['rsi'] > 30) & (rsi_data['rsi'].shift(1) <= 30))
        )
        signals[buy_condition] = 1
        
        # Sell signals
        sell_condition = (
            (rsi_data['overbought']) |
            (rsi_data['bearish_divergence']) |
            ((rsi_data['rsi'] < 70) & (rsi_data['rsi'].shift(1) >= 70))
        )
        signals[sell_condition] = -1
        
        return signals
    
    def get_parameters(self) -> Dict:
        """Return indicator parameters."""
        return {
            'period': self.period,
            'smooth_period': self.smooth_period,
            'overbought': self.overbought,
            'oversold': self.oversold
        }


async def main():
    """Example usage of custom RSI indicator."""
    
    # Load configuration
    config = get_config(config_name='prod', environment='dev')
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    data = pd.DataFrame({
        'close': prices,
        'volume': np.secure_randint(1000000, 5000000, 100)
    }, index=dates)
    
    print("=== Custom RSI Indicator Example ===\n")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Create and calculate custom RSI
    rsi_indicator = CustomRSI(config, period=14, smooth_period=3)
    rsi_result = await rsi_indicator.calculate(data)
    
    # Display results
    print("\n=== RSI Calculation Results ===")
    print(rsi_result.tail(10).round(2))
    
    # Show signal summary
    signals = rsi_result['signal']
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    
    print(f"\n=== Signal Summary ===")
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    # Show divergence detection
    bullish_div = rsi_result['bullish_divergence'].sum()
    bearish_div = rsi_result['bearish_divergence'].sum()
    
    print(f"\n=== Divergence Detection ===")
    print(f"Bullish divergences: {bullish_div}")
    print(f"Bearish divergences: {bearish_div}")
    
    # Show current state
    latest = rsi_result.iloc[-1]
    print(f"\n=== Current State ===")
    print(f"RSI: {latest['rsi']:.2f}")
    print(f"Smoothed RSI: {latest['rsi_smooth']:.2f}")
    print(f"Overbought: {latest['overbought']}")
    print(f"Oversold: {latest['oversold']}")
    print(f"Signal: {latest['signal']}")


if __name__ == "__main__":
    asyncio.run(main())
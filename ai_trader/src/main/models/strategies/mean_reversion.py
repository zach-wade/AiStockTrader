"""
Mean Reversion Strategy - refactored to be a stateless component that adheres
to the BaseStrategy contract and uses the UnifiedFeatureEngine.
"""
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

# Import required components from our established structure
from .base_strategy import BaseStrategy, Signal
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    A statistical arbitrage strategy that generates signals when a security's price
    deviates significantly from its recent mean.
    """
    
    def __init__(self, config: Dict[str, Any], feature_engine: UnifiedFeatureEngine):
        super().__init__(config, feature_engine)
        self.name = 'mean_reversion'  # REFACTOR: Added strategy name
        
        # Load strategy-specific parameters from the config
        strategy_conf = self.config.get('strategies', {}).get(self.name, {})
        self.zscore_threshold = strategy_conf.get('zscore_threshold', 2.0)
        self.lookback_period = strategy_conf.get('lookback_period', 20)
        
    def get_required_feature_sets(self) -> List[str]:
        """
        This strategy needs technical features, specifically a moving average
        which is typically included in the 'technical' feature set.
        """
        return ['technical']
        
    async def generate_signals(self, symbol: str, features: pd.DataFrame, current_position: Optional[Dict]) -> List[Signal]:
        """
        REFACTOR: The method is now async and accepts `current_position` to
        adhere to the stateless BaseStrategy contract.
        """
        if 'close' not in features.columns or len(features) < self.lookback_period:
            return []
            
        try:
            price = features['close']
            
            # Calculate rolling z-score
            # Use pre-calculated SMA from feature engine if available
            sma_col = f'sma_{self.lookback_period}'
            mean = features[sma_col] if sma_col in features.columns else price.rolling(self.lookback_period).mean()
            std = price.rolling(self.lookback_period).std()
            
            # Avoid division by zero if standard deviation is zero (flat price)
            if std.iloc[-1] < 1e-8:
                return []
                
            zscore = (price - mean) / std
            
            latest_zscore = zscore.iloc[-1]
            
            # --- Buy Signal Logic (Price too low, expect reversion up) ---
            if latest_zscore < -self.zscore_threshold:
                # Only generate a buy signal if we are not already long
                if not current_position or current_position.get('direction') != 'long':
                    confidence = min(1.0, abs(latest_zscore) / 3.0)
                    signal = Signal(
                        symbol=symbol,
                        direction='buy',
                        confidence=confidence,
                        metadata={
                            'strategy_name': self.name,
                            'zscore': latest_zscore,
                            'mean': mean.iloc[-1],
                        }
                    )
                    return [signal]
            
            # --- Sell Signal Logic (Price too high, expect reversion down) ---
            elif latest_zscore > self.zscore_threshold:
                # Only generate a sell signal if we are not already short
                if not current_position or current_position.get('direction') != 'short':
                    confidence = min(1.0, abs(latest_zscore) / 3.0)
                    signal = Signal(
                        symbol=symbol,
                        direction='sell',
                        confidence=confidence,
                        metadata={
                            'strategy_name': self.name,
                            'zscore': latest_zscore,
                            'mean': mean.iloc[-1],
                        }
                    )
                    return [signal]

            # --- Exit Signal Logic (Z-score returned to the mean) ---
            # Generate a 'hold' signal (effectively close) if we have a position and z-score is near zero
            if current_position and abs(latest_zscore) < 0.5:
                 return [Signal(symbol=symbol, direction='hold', confidence=1.0)]

        except Exception as e:
            logger.error(f"Error in {self.name} for {symbol}: {e}", exc_info=True)

        return []

    # REFACTOR: Removed the unused `calculate_half_life` method and the buggy
    # call to a non-existent `filter_signals` method. The class is now clean
    # and focused only on generating signals based on its core logic.
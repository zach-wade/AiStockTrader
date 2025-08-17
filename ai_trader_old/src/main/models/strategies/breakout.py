"""
Breakout Strategy - identifies price breakouts from consolidation patterns.
This version is refactored to comply with the stateless BaseStrategy contract.
"""

# Standard library imports
import logging
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
# We now need the UnifiedFeatureEngine for the constructor type hint
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    Identifies breakouts from low-volatility consolidation patterns,
    adhering to the stateless BaseStrategy interface.
    """

    def __init__(self, config: Dict[str, Any], feature_engine: UnifiedFeatureEngine):
        # Pass both config and the required feature_engine to the parent
        super().__init__(config, feature_engine)
        self.name = "breakout"

        # Strategy parameters are loaded from the main config object
        strategy_conf = self.config.get("strategies", {}).get(self.name, {})
        self.lookback_period = strategy_conf.get("lookback_period", 20)
        self.breakout_pct = strategy_conf.get("breakout_pct", 0.02)  # 2% move
        self.volume_multiplier = strategy_conf.get("volume_multiplier", 1.5)
        self.consolidation_range_max_pct = strategy_conf.get(
            "consolidation_range_max_pct", 0.05
        )  # Refactored
        self.stop_loss_pct = strategy_conf.get("stop_loss_pct", 0.05)

    def get_required_feature_sets(self) -> List[str]:
        """This strategy requires technical indicators like price and volume."""
        return ["technical"]  # Simplified from microstructure, assuming OHLCV is in 'technical'

    async def generate_signals(
        self, symbol: str, features: pd.DataFrame, current_position: Optional[Dict]
    ) -> List[Signal]:
        """
        Core alpha logic for generating breakout signals.
        This method is now async and accepts current_position.
        """
        if len(features) < self.lookback_period:
            return []

        try:
            lookback_features = features.tail(self.lookback_period)

            # Calculate key levels from the lookback window
            recent_high = lookback_features["high"].max()
            recent_low = lookback_features["low"].min()
            avg_volume = lookback_features["volume"].mean()

            current_price = features["close"].iloc[-1]
            current_volume = features["volume"].iloc[-1]

            # 1. Check for a consolidation pattern (low volatility)
            consolidation_range = (recent_high - recent_low) / np.mean([recent_high, recent_low])
            is_consolidating = consolidation_range < self.consolidation_range_max_pct

            if not is_consolidating:
                return []  # Not in a consolidation, so no breakout is possible

            # 2. Check for breakout confirmation
            is_volume_breakout = current_volume > (avg_volume * self.volume_multiplier)
            is_price_breakout_up = current_price > recent_high
            is_price_breakout_down = current_price < recent_low

            # --- Upward Breakout Logic ---
            # Generate a buy signal only if we are not already long
            if (
                is_price_breakout_up
                and is_volume_breakout
                and (not current_position or current_position.get("direction") != "long")
            ):
                confidence = min(1.0, (current_volume / avg_volume - 1.0) * 0.5 + 0.5)

                signal = Signal(
                    symbol=symbol,
                    direction="buy",
                    confidence=confidence,
                    metadata={
                        "strategy_name": self.name,
                        "breakout_type": "upward",
                        "resistance_level": recent_high,
                        "volume_ratio": current_volume / avg_volume,
                        "stop_loss_level": current_price * (1 - self.stop_loss_pct),
                    },
                )
                logger.info(f"Generated upward breakout signal for {symbol} at {current_price:.2f}")
                return [signal]

            # --- Downward Breakout Logic ---
            # Generate a sell signal only if we are not already short
            if (
                is_price_breakout_down
                and is_volume_breakout
                and (not current_position or current_position.get("direction") != "short")
            ):
                confidence = min(1.0, (current_volume / avg_volume - 1.0) * 0.5 + 0.5)

                signal = Signal(
                    symbol=symbol,
                    direction="sell",
                    confidence=confidence,
                    metadata={
                        "strategy_name": self.name,
                        "breakout_type": "downward",
                        "support_level": recent_low,
                        "volume_ratio": current_volume / avg_volume,
                        "stop_loss_level": current_price * (1 + self.stop_loss_pct),
                    },
                )
                logger.info(
                    f"Generated downward breakout signal for {symbol} at {current_price:.2f}"
                )
                return [signal]

        except Exception as e:
            logger.error(f"Error in {self.name} for {symbol}: {e}", exc_info=True)

        return []

    # REFACTOR: Removed `calculate_position_size` and `should_exit_position`.
    # Sizing is handled by the base class's _get_position_size, and exits are
    # managed by a higher-level system using the metadata we provide.

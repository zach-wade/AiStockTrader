# Standard library imports
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

from .base_strategy import Signal

# REFACTOR: Inherit from the universe-aware base class
from .base_universe_strategy import BaseUniverseStrategy

logger = logging.getLogger(__name__)

DYNAMIC_PAIRS_PATH = Path("data/analysis_results/tradable_pairs.json")


class PairsTradingStrategy(BaseUniverseStrategy):
    """
    Implements a stateless pairs trading strategy that consumes pre-computed
    hedge ratios and operates on a universe-wide view of market data.
    """

    def __init__(self, config: Dict[str, Any], feature_engine: UnifiedFeatureEngine):
        super().__init__(config, feature_engine)
        self.name = "pairs_trading"

        strategy_conf = self.config.get("strategies", {}).get(self.name, {})

        self.tradable_pairs = self._load_pairs_with_ratios(strategy_conf)
        self.lookback_period = strategy_conf.get("lookback_days", 60)
        self.zscore_entry_threshold = strategy_conf.get("entry_zscore", 2.0)
        self.zscore_exit_threshold = strategy_conf.get("exit_zscore", 0.5)

        logger.info(
            f"PairsTradingStrategy initialized with {len(self.tradable_pairs)} pre-calculated pairs."
        )

    def _load_pairs_with_ratios(self, config: dict) -> List[Dict]:
        """Loads pairs and their pre-computed hedge ratios from the analysis file."""
        if DYNAMIC_PAIRS_PATH.exists():
            logger.info(f"Loading dynamically discovered pairs from {DYNAMIC_PAIRS_PATH}")
            with open(DYNAMIC_PAIRS_PATH, "r") as f:
                data = json.load(f)
                return data.get("tradable_pairs", [])
        logger.warning("Dynamic pairs file not found. Pairs trading strategy will be inactive.")
        return []

    def get_required_feature_sets(self) -> List[str]:
        """This strategy only needs the 'close' price from the technical feature set."""
        return ["technical"]

    async def generate_universe_signals(
        self, market_features: Dict[str, pd.DataFrame], portfolio_state: Dict
    ) -> List[Signal]:
        """
        Analyzes all pre-defined pairs using the universe-wide market data.
        """
        signals = []
        if not self.tradable_pairs:
            return signals

        for pair_info in self.tradable_pairs:
            symbol1, symbol2 = pair_info["pair"]
            hedge_ratio = pair_info["hedge_ratio"]

            # Ensure we have the data for both legs of the pair for the current tick
            if symbol1 not in market_features or symbol2 not in market_features:
                continue

            price_series1 = market_features[symbol1]["close"]
            price_series2 = market_features[symbol2]["close"]

            if (
                len(price_series1) < self.lookback_period
                or len(price_series2) < self.lookback_period
            ):
                continue  # Not enough historical data in the provided features

            # Calculate the spread and its Z-score
            spread = price_series1 - (hedge_ratio * price_series2)
            rolling_mean = spread.rolling(window=self.lookback_period).mean()
            rolling_std = spread.rolling(window=self.lookback_period).std()

            if rolling_std.empty or rolling_std.iloc[-1] == 0:
                continue

            z_score = (spread.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

            # Check current position status from the portfolio state passed in
            pair_key = f"{symbol1}_{symbol2}"
            is_open = portfolio_state.get("pairs", {}).get(pair_key, False)

            # --- Trading Logic (now stateless) ---
            confidence = min(1.0, abs(z_score) / 3.0)

            # Entry condition
            if not is_open:
                if z_score > self.zscore_entry_threshold:  # Sell the spread
                    signals.append(Signal(symbol=symbol1, direction="sell", confidence=confidence))
                    signals.append(Signal(symbol=symbol2, direction="buy", confidence=confidence))
                elif z_score < -self.zscore_entry_threshold:  # Buy the spread
                    signals.append(Signal(symbol=symbol1, direction="buy", confidence=confidence))
                    signals.append(Signal(symbol=symbol2, direction="sell", confidence=confidence))

            # Exit condition
            elif is_open and abs(z_score) < self.zscore_exit_threshold:
                signals.append(Signal(symbol=symbol1, direction="close", confidence=1.0))
                signals.append(Signal(symbol=symbol2, direction="close", confidence=1.0))

        return signals

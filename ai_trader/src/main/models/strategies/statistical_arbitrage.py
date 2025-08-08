import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

# REFACTOR: Inherits from the correct base class for multi-asset strategies
from .base_universe_strategy import BaseUniverseStrategy 
from .base_strategy import Signal
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

logger = logging.getLogger(__name__)

# The strategy now consumes the output of our offline analysis script
ANALYSIS_RESULTS_PATH = Path("data/analysis_results/stat_arb_pairs.json")

class StatisticalArbitrageStrategy(BaseUniverseStrategy):
    """
    Implements a stateless pairs trading strategy that consumes pre-computed
    pair statistics and operates on a universe-wide view of market data.
    """
    def __init__(self, config: Dict[str, Any], feature_engine: UnifiedFeatureEngine):
        super().__init__(config, feature_engine)
        self.name = 'statistical_arbitrage'
        
        strategy_conf = self.config.get('strategies', {}).get(self.name, {})
        
        # Load the pre-computed pairs and their parameters
        self.tradable_pairs = self._load_tradable_pairs()
        
        # Load trading thresholds from config
        self.zscore_entry_threshold = strategy_conf.get('entry_zscore', 2.0)
        self.zscore_exit_threshold = strategy_conf.get('exit_zscore', 0.5)
        self.stop_loss_zscore = strategy_conf.get('stop_loss_zscore', 3.5)
        
        logger.info(f"StatisticalArbitrageStrategy initialized with {len(self.tradable_pairs)} pre-analyzed pairs.")

    def _load_tradable_pairs(self) -> List[Dict]:
        """Loads pairs and their parameters from the analysis results file."""
        if not ANALYSIS_RESULTS_PATH.exists():
            logger.warning(f"Analysis results not found at {ANALYSIS_RESULTS_PATH}. Strategy will be inactive.")
            return []
        try:
            with open(ANALYSIS_RESULTS_PATH, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded pairs data analyzed on: {data.get('analysis_timestamp')}")
            return data.get('tradable_pairs', [])
        except Exception as e:
            logger.error(f"Failed to load or parse pair analysis file: {e}")
            return []

    def get_required_feature_sets(self) -> List[str]:
        """This strategy only needs the 'close' price from the technical feature set."""
        return ['technical']

    async def generate_universe_signals(self, market_features: Dict[str, pd.DataFrame], portfolio_state: Dict) -> List[Signal]:
        """
        Analyzes all pre-defined pairs using the universe-wide market data provided by the engine.
        """
        signals = []
        if not self.tradable_pairs:
            return signals

        for pair_info in self.tradable_pairs:
            symbol1, symbol2 = pair_info['pair']
            hedge_ratio = pair_info['hedge_ratio']

            if symbol1 not in market_features or symbol2 not in market_features:
                continue

            price1 = market_features[symbol1]['close'].iloc[-1]
            price2 = market_features[symbol2]['close'].iloc[-1]

            # Calculate the current spread and z-score using pre-computed stats
            current_spread = price1 - (hedge_ratio * price2)
            z_score = (current_spread - pair_info['spread_mean']) / pair_info['spread_std'] if pair_info['spread_std'] > 0 else 0
            
            # Get current position status from the portfolio state passed in by the engine
            pair_key = f"{symbol1}_{symbol2}"
            position_info = portfolio_state.get('active_pairs', {}).get(pair_key)

            confidence = min(1.0, abs(z_score) / 3.0)

            if not position_info: # No open position, look for entry
                if z_score > self.zscore_entry_threshold: # Sell the spread
                    signals.append(Signal(symbol=symbol1, direction='sell', confidence=confidence, metadata={'pair_key': pair_key}))
                    signals.append(Signal(symbol=symbol2, direction='buy', confidence=confidence, metadata={'pair_key': pair_key}))
                elif z_score < -self.zscore_entry_threshold: # Buy the spread
                    signals.append(Signal(symbol=symbol1, direction='buy', confidence=confidence, metadata={'pair_key': pair_key}))
                    signals.append(Signal(symbol=symbol2, direction='sell', confidence=confidence, metadata={'pair_key': pair_key}))
            
            else: # Position is open, look for exit
                # Exit condition 1: Mean reversion
                if (position_info['side'] == 'long_spread' and z_score > -self.zscore_exit_threshold) or \
                   (position_info['side'] == 'short_spread' and z_score < self.zscore_exit_threshold):
                    signals.append(Signal(symbol=symbol1, direction='close', confidence=1.0, metadata={'pair_key': pair_key, 'reason': 'mean_reversion'}))
                    signals.append(Signal(symbol=symbol2, direction='close', confidence=1.0, metadata={'pair_key': pair_key, 'reason': 'mean_reversion'}))
                
                # Exit condition 2: Stop loss
                elif abs(z_score) > self.stop_loss_zscore:
                    signals.append(Signal(symbol=symbol1, direction='close', confidence=1.0, metadata={'pair_key': pair_key, 'reason': 'stop_loss'}))
                    signals.append(Signal(symbol=symbol2, direction='close', confidence=1.0, metadata={'pair_key': pair_key, 'reason': 'stop_loss'}))
        
        return signals
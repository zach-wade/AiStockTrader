"""
Correlation-based trading strategy, refactored to be a stateless, universe-aware
component of the advanced ensemble.
"""
import logging
from typing import Dict, List, Optional

import pandas as pd

# Import the new base class and other components
from .base_universe_strategy import BaseUniverseStrategy # Assuming this file is created
from .base_strategy import Signal
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine
from main.backtesting.analysis.correlation_matrix import CorrelationMatrix

logger = logging.getLogger(__name__)


class CorrelationStrategy(BaseUniverseStrategy):
    """
    Analyzes cross-asset correlations for the entire market universe to find
    trading signals like breakdowns, regime shifts, and rotations.
    """
    
    def __init__(self, config: Dict, feature_engine: UnifiedFeatureEngine):
        super().__init__(config, feature_engine)
        self.name = "correlation"
        
        strategy_conf = self.config.get('strategies', {}).get(self.name, {})
        self.signal_threshold = strategy_conf.get('signal_threshold', 0.7)
        
        # The CorrelationMatrix tool is instantiated here.
        # It should be a stateless calculator.
        self.correlation_matrix = CorrelationMatrix(strategy_conf.get('correlation_matrix_config', {}))
        
        # This will hold asset classes defined in config, not hardcoded.
        self.asset_classes = strategy_conf.get('asset_classes', {})

    def get_required_feature_sets(self) -> List[str]:
        # This strategy needs basic price data for all symbols.
        return ['technical']

    async def generate_universe_signals(self, market_features: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        The new primary method. It receives features for all symbols at once.
        """
        all_signals = []
        if len(market_features) < 10:
            logger.warning("CorrelationStrategy requires a market view of at least 10 symbols.")
            return []

        try:
            # 1. Use the stateless correlation analysis tool.
            # This method should take the latest data and return a list of correlation events.
            correlation_events = self.correlation_matrix.analyze_latest_correlations(market_features)

            # 2. Convert these events into actionable trading signals.
            for event in correlation_events:
                if event.strength >= self.signal_threshold:
                    generated_signals = self._convert_event_to_signals(event)
                    all_signals.extend(generated_signals)
            
            # Note: Signal smoothing should be handled by a higher-level system if desired.
            
        except Exception as e:
            logger.error(f"Error during correlation universe analysis: {e}", exc_info=True)

        return all_signals

    def _convert_event_to_signals(self, event) -> List[Signal]:
        """
        Converts a single correlation event into one or more trading signals.
        This logic is now stateless and driven by configuration.
        """
        signals = []
        
        # Example for a 'rotation' event
        if event.signal_type == 'rotation':
            # Config defines which asset is the leader to buy
            leader_symbol = event.primary_asset
            # Config defines which assets are laggards to sell
            laggard_symbols = event.related_assets
            
            signals.append(Signal(symbol=leader_symbol, direction='buy', confidence=event.strength))
            for laggard in laggard_symbols:
                signals.append(Signal(symbol=laggard, direction='sell', confidence=event.strength))

        # Example for a 'risk_off' regime shift event
        elif event.signal_type == 'regime_shift' and event.direction == 'risk_off':
            risky_assets = self.asset_classes.get('RISKY', [])
            safe_assets = self.asset_classes.get('SAFE_HAVEN', [])
            
            for asset in event.affected_assets:
                if asset in risky_assets:
                    signals.append(Signal(symbol=asset, direction='sell', confidence=event.strength))
                elif asset in safe_assets:
                    signals.append(Signal(symbol=asset, direction='buy', confidence=event.strength))
        
        # Add logic for other event types...

        return signals
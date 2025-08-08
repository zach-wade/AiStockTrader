"""
Aggregates signals from multiple strategies into a single ensemble signal.
"""
import logging
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict

from ..base_strategy import Signal
from main.feature_pipeline.calculators.market_regime import MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class EnsembleSignal:
    """Enhanced signal with ensemble metadata."""
    symbol: str
    direction: str
    confidence: float
    size: float
    strategy_weights: Dict[str, float]
    contributing_strategies: List[str]
    consensus_strength: float
    risk_score: float
    time_horizon: int
    expected_return: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SignalAggregator:
    """
    Combines signals from various strategies using a selected method.
    Its single responsibility is signal fusion.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.aggregation_method = self.config.get('aggregation_method', 'weighted_average')
        logger.info(f"SignalAggregator initialized with method: {self.aggregation_method}")

    def aggregate(self, strategy_signals: Dict[str, List[Signal]], 
                  strategy_weights: Dict[str, float], 
                  symbol: str, 
                  regime: MarketRegime) -> List[EnsembleSignal]:
        """
        Primary aggregation method. Delegates to a specific implementation.
        """
        signals_by_symbol = self._group_signals_by_symbol(strategy_signals)
        
        if symbol not in signals_by_symbol:
            return []

        if self.aggregation_method == 'weighted_average':
            return self._weighted_average_aggregation(
                signals_by_symbol[symbol], strategy_weights, symbol, regime
            )
        # Add other aggregation methods like consensus voting here
        else:
            logger.warning(f"Unknown aggregation method: {self.aggregation_method}. Defaulting to weighted average.")
            return self._weighted_average_aggregation(
                signals_by_symbol[symbol], strategy_weights, symbol, regime
            )

    def _group_signals_by_symbol(self, strategy_signals: Dict[str, List[Signal]]) -> Dict:
        """Helper to group all signals by their symbol."""
        grouped = defaultdict(list)
        for strategy_name, signals in strategy_signals.items():
            for signal in signals:
                grouped[signal.symbol].append((strategy_name, signal))
        return grouped

    def _weighted_average_aggregation(self, signal_pairs: List, 
                                      strategy_weights: Dict[str, float], 
                                      symbol: str, 
                                      regime: MarketRegime) -> List[EnsembleSignal]:
        """Aggregates signals using a weighted average of their direction and confidence."""
        if not signal_pairs:
            return []

        total_weight, weighted_confidence, weighted_direction = 0.0, 0.0, 0.0
        contributing_strategies, strategy_weights_used = [], {}

        for name, signal in signal_pairs:
            weight = strategy_weights.get(name, 0)
            if weight > 0:
                total_weight += weight
                weighted_confidence += signal.confidence * weight
                dir_val = 1 if signal.direction == 'buy' else -1 if signal.direction == 'sell' else 0
                weighted_direction += dir_val * weight * signal.confidence
                contributing_strategies.append(name)
                strategy_weights_used[name] = weight

        if total_weight > 0:
            final_confidence = weighted_confidence / total_weight
            final_dir_val = weighted_direction / total_weight
            final_dir = 'buy' if final_dir_val > 0.1 else 'sell' if final_dir_val < -0.1 else 'hold'

            risk_score = self._calculate_risk_score(len(contributing_strategies), regime)

            return [EnsembleSignal(
                symbol=symbol,
                direction=final_dir,
                confidence=final_confidence,
                size=self._calculate_ensemble_size(final_confidence, risk_score),
                strategy_weights=strategy_weights_used,
                contributing_strategies=contributing_strategies,
                consensus_strength=self._calculate_consensus(signal_pairs),
                risk_score=risk_score,
                time_horizon=int(np.mean([s.metadata.get('time_horizon', 60) for _, s in signal_pairs])),
                expected_return=0.01, # Placeholder
                metadata={'aggregation_method': 'weighted_average', 'regime': regime.value}
            )]
        return []

    def _calculate_risk_score(self, num_strategies: int, regime: MarketRegime) -> float:
        """Calculates a risk score for the ensemble signal."""
        regime_risk = {'HIGH_VOLATILITY': 0.8, 'LOW_VOLATILITY': 0.2}.get(regime, 0.5)
        diversity_factor = 1 - (min(num_strategies, 5) / 10) # More strategies = less risk
        return regime_risk * diversity_factor

    def _calculate_consensus(self, signal_pairs: List) -> float:
        """Calculates consensus strength."""
        if len(signal_pairs) < 2: return 1.0
        dirs = [s.direction for _, s in signal_pairs]
        return max(dirs.count(d) for d in set(dirs)) / len(dirs)

    def _calculate_ensemble_size(self, confidence: float, risk_score: float) -> float:
        """Calculates position size."""
        base_size = self.config.get('base_position_size', 0.02)
        size_multiplier = confidence * (1 - risk_score * 0.5)
        return min(base_size * size_multiplier, self.config.get('max_position_per_symbol', 0.05))
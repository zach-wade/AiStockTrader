"""
The main AdvancedStrategyEnsemble class, refactored to be a high-level
orchestrator of specialized components for performance, allocation, and aggregation.
"""

# Standard library imports
import asyncio
import logging
from typing import Any, Dict, List

# Third-party imports
import pandas as pd

# Local imports
from main.feature_pipeline.calculators.market_regime import MarketRegimeDetector

from .aggregation import SignalAggregator
from .allocation import WeightAllocator

# Import sibling modules from the ensemble package
from .performance import PerformanceTracker

# Import other required components from the project
from ..base_strategy import BaseStrategy, Signal
from ..ml_momentum import MLMomentumStrategy

# Import all strategy classes that can be used by the ensemble
from ..regime_adaptive import RegimeAdaptiveStrategy
from ...hft.microstructure_alpha import MicrostructureAlphaStrategy

# ... import other strategy classes as needed

logger = logging.getLogger(__name__)

# Mapping to instantiate strategies from config
STRATEGY_CLASS_MAP = {
    "regime_adaptive": RegimeAdaptiveStrategy,
    "microstructure_alpha": MicrostructureAlphaStrategy,
    "ml_momentum": MLMomentumStrategy,
    # ... add other strategies here
}


class AdvancedStrategyEnsemble(BaseStrategy):
    """
    Orchestrates a portfolio of strategies, delegating responsibilities
    for performance tracking, allocation, and signal aggregation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "advanced_ensemble"
        self.ensemble_config = config.get("strategies", {}).get("advanced_ensemble", {})

        # 1. Initialize component strategies
        self.strategies = self._initialize_strategies()
        self.strategy_weights = {name: 1.0 / len(self.strategies) for name in self.strategies}

        # 2. Instantiate our specialized components (Composition over Inheritance)
        self.tracker = PerformanceTracker(list(self.strategies.keys()))
        self.allocator = WeightAllocator(self.ensemble_config)
        self.aggregator = SignalAggregator(self.ensemble_config)

        self.regime_detector = MarketRegimeDetector(config)

        logger.info(
            f"Advanced Strategy Ensemble initialized with {len(self.strategies)} strategies."
        )

    def _initialize_strategies(self) -> Dict[str, BaseStrategy]:
        """Initializes all enabled component strategies from the config."""
        strategies = {}
        enabled_strategies = self.ensemble_config.get("enabled_strategies", [])

        for name in enabled_strategies:
            if name in STRATEGY_CLASS_MAP:
                strategies[name] = STRATEGY_CLASS_MAP[name](self.config)
            else:
                logger.warning(f"Strategy '{name}' requested but not found in STRATEGY_CLASS_MAP.")
        return strategies

    async def generate_signals(self, symbol: str, features: pd.DataFrame) -> List[Signal]:
        """
        The main workflow for generating an ensemble signal. Now a clean, high-level process.
        """
        try:
            # 1. Rebalance weights if necessary, based on performance data
            if self.allocator.should_rebalance():
                perf_data = self.tracker.get_performance_data()
                self.strategy_weights = self.allocator.calculate_new_weights(perf_data)

            # 2. Detect current market regime
            regime, _ = self.regime_detector.detect_regime(features)

            # 3. Collect signals from all sub-strategies in parallel
            tasks = [s.generate_signals(symbol, features) for s in self.strategies.values()]
            signal_results = await asyncio.gather(*tasks, return_exceptions=True)

            raw_signals = {
                name: res
                for name, res in zip(self.strategies.keys(), signal_results)
                if isinstance(res, list)
            }

            # 4. Delegate aggregation to the SignalAggregator
            ensemble_signals = self.aggregator.aggregate(
                raw_signals, self.strategy_weights, symbol, regime
            )

            # 5. Convert to final Signal format (Portfolio optimization would go here)
            return [
                Signal(
                    symbol=es.symbol,
                    direction=es.direction,
                    confidence=es.confidence,
                    size=es.size,
                    metadata={"ensemble_data": es.metadata},
                )
                for es in ensemble_signals
            ]

        except Exception as e:
            logger.error(f"Error in ensemble signal generation for {symbol}: {e}", exc_info=True)
            return []

    async def update_performance(self, trade_result: Dict[str, Any]):
        """Delegates performance updates to the tracker."""
        self.tracker.update(trade_result)

    async def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Gathers and returns metrics from all components."""
        perf_data = self.tracker.get_performance_data()
        return {
            "strategy_weights": self.strategy_weights,
            "last_rebalance": self.allocator.last_rebalance.isoformat(),
            "aggregation_method": self.aggregator.aggregation_method,
            "performance_metrics": {
                name: {"win_rate": p.win_rate, "sharpe": p.sharpe_ratio}
                for name, p in perf_data.items()
            },
        }

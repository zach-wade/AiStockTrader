# File: models/strategies/base_strategy.py
"""
Base strategy class, using dependency injection and a unified feature engine.
This refactored version enforces a stricter, more robust contract for all strategies.
"""
# Standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Third-party imports
import pandas as pd

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    # Local imports
    from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A standardized trading signal object. This remains an excellent design."""

    symbol: str
    direction: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    size: Optional[float] = None  # Position size as a fraction of portfolio
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """
    Base class for all trading strategies, defining a strict execution contract.
    """

    def __init__(self, config: Dict[str, Any], feature_engine: "UnifiedFeatureEngine"):
        """
        Initializes the strategy.

        Args:
            config: A dictionary containing strategy-specific configurations.
            feature_engine: An instance of the UnifiedFeatureEngine.
                            This is a required dependency.
        """
        self.config = config
        self.name = "base"  # Should be overridden by child classes

        # REFACTOR: Strict Dependency Injection.
        # We no longer create a default engine. A valid feature_engine MUST be provided.
        # This makes the system's dependencies explicit and prevents it from running
        # in a poorly configured state.
        # Check that feature_engine has the required methods instead of isinstance
        if not hasattr(feature_engine, "calculate_features"):
            raise TypeError("A valid UnifiedFeatureEngine instance must be provided.")
        self.feature_engine = feature_engine

        # REFACTOR: Removed `self.positions`. Strategies should be stateless.
        # Position state is managed by a higher-level portfolio or execution manager.

    # REFACTOR: Added a formal, public `execute` method (Template Method Pattern)
    async def execute(
        self, symbol: str, data: pd.DataFrame, current_position: Optional[Dict] = None
    ) -> List[Signal]:
        """
        Public entry point to execute the strategy for a given symbol.
        This method orchestrates the workflow to ensure consistency.

        Args:
            symbol: The symbol to generate signals for.
            data: The market data (OHLCV) for the symbol.
            current_position: The current position held for this symbol, if any.

        Returns:
            A list of sized signals.
        """
        try:
            # Step 1: Prepare all necessary features for this strategy.
            features = self._prepare_features(symbol, data)

            # Step 2: Delegate to the core alpha logic to generate raw signals.
            # The statelessness principle is enacted here by passing the current position in.
            raw_signals = await self.generate_signals(symbol, features, current_position)

            # Step 3: Apply sizing logic to the generated signals.
            sized_signals = []
            for signal in raw_signals:
                if signal.direction not in ["buy", "sell"]:
                    signal.size = 0.0  # No size for 'hold' signals
                    sized_signals.append(signal)
                    continue

                if signal.size is None:
                    signal.size = self._get_position_size(symbol, signal, features)
                sized_signals.append(signal)

            return sized_signals
        except Exception as e:
            logger.error(
                f"FATAL: Error executing strategy {self.name} for {symbol}: {e}", exc_info=True
            )
            return []

    def _prepare_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Protected helper to calculate all necessary features using the feature engine.
        """
        required_calculators = self.get_required_feature_sets()
        return self.feature_engine.calculate_features(
            data=data, symbol=symbol, calculators=required_calculators, use_cache=True
        )

    def get_required_feature_sets(self) -> List[str]:
        """
        Child strategies must override this to declare their feature dependencies.
        Example: return ['technical', 'news_sentiment']
        """
        return ["technical"]  # A safe, minimal default

    @abstractmethod
    async def generate_signals(
        self, symbol: str, features: pd.DataFrame, current_position: Optional[Dict]
    ) -> List[Signal]:
        """
        The core alpha logic. Child strategies MUST implement this method.
        This is now conceptually a "protected" method, called only by `execute`.

        Args:
            symbol: The symbol being analyzed.
            features: A DataFrame containing all necessary pre-calculated features.
            current_position: A dictionary describing the current holding, if any.

        Returns:
            A list of raw (potentially unsized) Signal objects.
        """
        pass

    def _get_position_size(self, symbol: str, signal: Signal, features: pd.DataFrame) -> float:
        """
        Calculates position size. This default implementation can be overridden.
        """
        # A more robust way would be to pass a validated config object.
        strategy_conf = self.config.get("strategies", {}).get(self.name, {})
        base_size = strategy_conf.get("base_position_size", 0.01)

        return abs(base_size * signal.confidence)

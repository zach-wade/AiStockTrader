# File: models/strategies/__init__.py
"""
Trading Strategies Module

This package initializes all available trading strategies and provides a
central registry for the main application to discover and instantiate them.
"""
import logging

logger = logging.getLogger(__name__)

# Import the base classes first to make them available to other modules
from .base_strategy import BaseStrategy, Signal
from .base_universe_strategy import BaseUniverseStrategy

# Import all concrete, final strategy implementations
# Ensure the file names and class names here match your final, consolidated files.
from .sentiment import FinalSentimentStrategy
from .mean_reversion import MeanReversionStrategy
from .ml_momentum import MLMomentumStrategy
from .breakout import BreakoutStrategy
try:
    from .pairs_trading import PairsTradingStrategy
    pairs_trading_available = True
except ImportError as e:
    logger.warning(f"PairsTradingStrategy not available due to: {e}")
    pairs_trading_available = False
    PairsTradingStrategy = None

# Strategy imports with graceful error handling
try:
    from .regime_adaptive import RegimeAdaptiveStrategy
    regime_adaptive_available = True
except ImportError as e:
    logger.warning(f"RegimeAdaptiveStrategy not available due to: {e}")
    regime_adaptive_available = False
    RegimeAdaptiveStrategy = None

try:
    from .ensemble import AdvancedStrategyEnsemble
    # Alias the existing AdvancedStrategyEnsemble as EnsembleMetaLearningStrategy
    EnsembleMetaLearningStrategy = AdvancedStrategyEnsemble
    EnsembleStrategy = AdvancedStrategyEnsemble  # Add alias for EnsembleStrategy too
    ensemble_available = True
except ImportError as e:
    logger.warning(f"EnsembleMetaLearningStrategy not available due to: {e}")
    ensemble_available = False
    EnsembleMetaLearningStrategy = None
    EnsembleStrategy = None

try:
    from ..event_driven.news_analytics import NewsAnalyticsStrategy
    news_analytics_available = True
except ImportError as e:
    logger.warning(f"NewsAnalyticsStrategy not available due to: {e}")
    news_analytics_available = False
    NewsAnalyticsStrategy = None

try:
    from ..hft.microstructure_alpha import MicrostructureAlphaStrategy
    microstructure_alpha_available = True
except ImportError as e:
    logger.warning(f"MicrostructureAlphaStrategy not available due to: {e}")
    microstructure_alpha_available = False
    MicrostructureAlphaStrategy = None

try:
    from .statistical_arbitrage import StatisticalArbitrageStrategy
    stat_arb_available = True
except ImportError as e:
    logger.warning(f"StatisticalArbitrageStrategy not available due to: {e}")
    stat_arb_available = False
    StatisticalArbitrageStrategy = None

from .correlation_strategy import CorrelationStrategy

# Define the public API of this package for 'from strategies import *'
# This is a best practice that lists all classes intended for external use.
__all__ = [
    'BaseStrategy',
    'BaseUniverseStrategy',
    'Signal',
    'FinalSentimentStrategy',
    'MeanReversionStrategy',
    'MLMomentumStrategy',
    'CorrelationStrategy',
    'BreakoutStrategy',
]
# Add optional strategies if available
if pairs_trading_available:
    __all__.append('PairsTradingStrategy')
if stat_arb_available:
    __all__.append('StatisticalArbitrageStrategy')
if regime_adaptive_available:
    __all__.append('RegimeAdaptiveStrategy')
if ensemble_available:
    __all__.append('EnsembleMetaLearningStrategy')
    __all__.append('EnsembleStrategy')
if news_analytics_available:
    __all__.append('NewsAnalyticsStrategy')
if microstructure_alpha_available:
    __all__.append('MicrostructureAlphaStrategy')

# Create the central strategy registry.
# The keys (e.g., 'sentiment', 'ml_momentum') are what you should use in your
# config files to enable or disable a specific strategy.
STRATEGIES = {
    'sentiment': FinalSentimentStrategy,
    'mean_reversion': MeanReversionStrategy,
    'ml_momentum': MLMomentumStrategy,
    'correlation': CorrelationStrategy,
    'breakout': BreakoutStrategy,
}

# Add optional strategies if available
if pairs_trading_available:
    STRATEGIES['pairs_trading'] = PairsTradingStrategy
if stat_arb_available:
    STRATEGIES['stat_arb'] = StatisticalArbitrageStrategy
if regime_adaptive_available:
    STRATEGIES['regime_adaptive'] = RegimeAdaptiveStrategy
if ensemble_available:
    STRATEGIES['ensemble'] = EnsembleMetaLearningStrategy
if news_analytics_available:
    STRATEGIES['news_analytics'] = NewsAnalyticsStrategy
if microstructure_alpha_available:
    STRATEGIES['microstructure_alpha'] = MicrostructureAlphaStrategy

logger.info(f"✅ Strategy registry loaded with {len(STRATEGIES)} strategies.")
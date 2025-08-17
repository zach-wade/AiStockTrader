"""
Technical Analysis Module

Provides specialized technical indicator calculators organized by category.
Each calculator focuses on a specific type of technical analysis.
"""

# Base class
from .adaptive_indicators import AdaptiveIndicatorsCalculator
from .base_technical import BaseTechnicalCalculator
from .momentum_indicators import MomentumIndicatorsCalculator

# Specialized calculators
from .trend_indicators import TrendIndicatorsCalculator

# Unified facade for backward compatibility
from .unified_facade import UnifiedTechnicalIndicatorsFacade
from .volatility_indicators import VolatilityIndicatorsCalculator
from .volume_indicators import VolumeIndicatorsCalculator

__all__ = [
    # Base class
    "BaseTechnicalCalculator",
    # Specialized calculators
    "TrendIndicatorsCalculator",
    "MomentumIndicatorsCalculator",
    "VolatilityIndicatorsCalculator",
    "VolumeIndicatorsCalculator",
    "AdaptiveIndicatorsCalculator",
    # Facade
    "UnifiedTechnicalIndicatorsFacade",
]

# Registry for technical calculators
TECHNICAL_CALCULATOR_REGISTRY = {
    "trend": TrendIndicatorsCalculator,
    "momentum": MomentumIndicatorsCalculator,
    "volatility": VolatilityIndicatorsCalculator,
    "volume": VolumeIndicatorsCalculator,
    "adaptive": AdaptiveIndicatorsCalculator,
    "unified": UnifiedTechnicalIndicatorsFacade,
}

# Feature counts for each calculator
FEATURE_COUNTS = {
    "trend": 45,  # Moving averages, ADX, Aroon, channels
    "momentum": 52,  # RSI, MACD, Stochastic, momentum variants
    "volatility": 38,  # Bollinger Bands, ATR, volatility measures
    "volume": 35,  # Volume indicators, money flow, accumulation
    "adaptive": 40,  # Adaptive moving averages, dynamic indicators
}

# Total features across all calculators
TOTAL_TECHNICAL_FEATURES = sum(FEATURE_COUNTS.values())  # 210 total features


def get_technical_calculator(calculator_name: str, config: dict = None):
    """
    Get technical calculator instance by name.

    Args:
        calculator_name: Name of calculator from registry
        config: Optional configuration dictionary

    Returns:
        Calculator instance

    Raises:
        ValueError: If calculator name not found
    """
    if calculator_name not in TECHNICAL_CALCULATOR_REGISTRY:
        available = list(TECHNICAL_CALCULATOR_REGISTRY.keys())
        raise ValueError(f"Calculator '{calculator_name}' not found. Available: {available}")

    calc_class = TECHNICAL_CALCULATOR_REGISTRY[calculator_name]
    return calc_class(config)


def get_all_technical_calculators(config: dict = None):
    """
    Get instances of all technical calculators.

    Args:
        config: Optional configuration dictionary

    Returns:
        Dictionary mapping calculator names to instances
    """
    calculators = {}

    for name, calc_class in TECHNICAL_CALCULATOR_REGISTRY.items():
        if name != "unified":  # Skip facade to avoid duplication
            try:
                calculators[name] = calc_class(config)
            except Exception as e:
                print(f"Warning: Failed to initialize {name}: {e}")
                continue

    return calculators


def get_technical_feature_summary():
    """
    Get summary of features provided by each calculator.

    Returns:
        Dictionary with feature counts and descriptions
    """
    summary = {
        "total_features": TOTAL_TECHNICAL_FEATURES,
        "calculator_counts": FEATURE_COUNTS.copy(),
        "descriptions": {
            "trend": "Trend following indicators and directional analysis (45 features)",
            "momentum": "Momentum and oscillator indicators (52 features)",
            "volatility": "Volatility and range-based indicators (38 features)",
            "volume": "Volume analysis and money flow indicators (35 features)",
            "adaptive": "Adaptive and dynamic technical indicators (40 features)",
        },
        "categories": {
            "price_based": ["trend", "momentum", "volatility"],
            "volume_based": ["volume"],
            "advanced": ["adaptive"],
        },
    }

    return summary


# Common technical analysis configurations
def create_default_technical_config():
    """Create default technical analysis configuration."""
    return {
        # Common periods
        "lookback_periods": [5, 10, 20, 50, 200],
        # Trend indicators
        "ma_types": ["sma", "ema", "wma"],
        "adx_period": 14,
        "aroon_period": 25,
        # Momentum indicators
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "stoch_period": 14,
        # Volatility indicators
        "bb_period": 20,
        "bb_std": 2.0,
        "atr_period": 14,
        # Volume indicators
        "volume_ma_period": 20,
        "mfi_period": 14,
        # Adaptive settings
        "adaptive_enabled": True,
        "efficiency_ratio_period": 10,
    }


def create_fast_technical_config():
    """Create configuration for fast technical analysis."""
    return {
        "lookback_periods": [10, 20, 50],
        "ma_types": ["sma", "ema"],
        "rsi_period": 14,
        "bb_period": 20,
        "adaptive_enabled": False,
    }


def create_comprehensive_technical_config():
    """Create configuration for comprehensive technical analysis."""
    return {
        "lookback_periods": [5, 10, 20, 30, 50, 100, 200],
        "ma_types": ["sma", "ema", "wma", "dema", "tema"],
        "rsi_period": 14,
        "rsi_variations": [9, 14, 21],
        "macd_variations": [(12, 26, 9), (5, 35, 5)],
        "bb_variations": [(20, 2), (20, 2.5), (50, 2)],
        "adaptive_enabled": True,
        "include_advanced": True,
    }


# Version information
__version__ = "2.0.0"
__author__ = "AI Trader Technical Analysis Team"

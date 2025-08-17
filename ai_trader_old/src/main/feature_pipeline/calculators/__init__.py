"""
Feature Calculators Module

This module contains all feature calculation implementations for the AI Trader system.
Each calculator inherits from BaseFeatureCalculator and provides specialized features
for different data types and analysis approaches.

Calculator Categories:
- Technical Analysis: Technical indicators, price patterns, volume analysis
- Statistical Analysis: Advanced statistical features, entropy, fractals
- Cross-Asset Analysis: Correlations, sector rotation, market regime detection
- Alternative Data: News sentiment, social media, insider trading, options flow
- Market Structure: Microstructure features, liquidity, order flow analysis
"""

from .base_calculator import BaseFeatureCalculator

# Enhanced Correlation (use facade from correlation module)
from .correlation import EnhancedCorrelationCalculator

# Cross-Asset Analysis Calculators
from .cross_asset import CrossAssetCalculator
from .cross_sectional import CrossSectionalCalculator
from .enhanced_cross_sectional import EnhancedCrossSectionalCalculator

# Base calculator
# Options Analytics (new modular system)
# Risk Analytics (comprehensive risk management)
from . import options, risk
from .insider_analytics import InsiderAnalyticsCalculator

# Market Analysis Calculators
from .market_regime import MarketRegimeCalculator
from .microstructure import MicrostructureCalculator

# News Features (use facade from news module)
from .news import NewsFeatureCalculator

# Options Analytics (use facade from options module)
from .options import OptionsAnalyticsFacade as OptionsAnalyticsCalculator
from .sector_analytics import SectorAnalyticsCalculator

# Alternative Data Calculators
from .sentiment_features import SentimentFeaturesCalculator

# Statistical Analysis Calculators (refactored facade)
from .statistical import AdvancedStatisticalCalculator

# Refactored Technical Calculators
from .technical import (
    AdaptiveIndicatorsCalculator,
    MomentumIndicatorsCalculator,
    TrendIndicatorsCalculator,
    UnifiedTechnicalIndicatorsFacade,
    VolatilityIndicatorsCalculator,
    VolumeIndicatorsCalculator,
)

# Technical Analysis Calculators
from .technical_indicators import TechnicalIndicatorsCalculator

__all__ = [
    # Base class
    "BaseFeatureCalculator",
    # Technical Analysis
    "TechnicalIndicatorsCalculator",
    # Refactored Technical Calculators
    "TrendIndicatorsCalculator",
    "MomentumIndicatorsCalculator",
    "VolatilityIndicatorsCalculator",
    "VolumeIndicatorsCalculator",
    "AdaptiveIndicatorsCalculator",
    "UnifiedTechnicalIndicatorsFacade",
    # Statistical Analysis
    "AdvancedStatisticalCalculator",
    # Cross-Asset Analysis
    "CrossAssetCalculator",
    "CrossSectionalCalculator",
    "EnhancedCorrelationCalculator",
    "EnhancedCrossSectionalCalculator",
    # Market Analysis
    "MarketRegimeCalculator",
    "MicrostructureCalculator",
    # Alternative Data
    "NewsFeatureCalculator",
    "SentimentFeaturesCalculator",
    "InsiderAnalyticsCalculator",
    "SectorAnalyticsCalculator",
    # Options Analytics
    "OptionsAnalyticsCalculator",  # Legacy facade
    "options",  # New modular system
    # Risk Analytics
    "risk",  # Comprehensive risk management system
]

# Calculator registry for dynamic instantiation
CALCULATOR_REGISTRY = {
    "technical_indicators": TechnicalIndicatorsCalculator,
    # Refactored technical calculators
    "trend_indicators": TrendIndicatorsCalculator,
    "momentum_indicators": MomentumIndicatorsCalculator,
    "volatility_indicators": VolatilityIndicatorsCalculator,
    "volume_indicators": VolumeIndicatorsCalculator,
    "adaptive_indicators": AdaptiveIndicatorsCalculator,
    "unified_technical_facade": UnifiedTechnicalIndicatorsFacade,
    "advanced_statistical": AdvancedStatisticalCalculator,
    "cross_asset": CrossAssetCalculator,
    "cross_sectional": CrossSectionalCalculator,
    "enhanced_correlation": EnhancedCorrelationCalculator,
    "enhanced_cross_sectional": EnhancedCrossSectionalCalculator,
    "market_regime": MarketRegimeCalculator,
    "microstructure": MicrostructureCalculator,
    "news_features": NewsFeatureCalculator,
    "sentiment_features": SentimentFeaturesCalculator,
    "insider_analytics": InsiderAnalyticsCalculator,
    "sector_analytics": SectorAnalyticsCalculator,
    "options_analytics": OptionsAnalyticsCalculator,
    # Risk Analytics
    "risk_metrics": risk.RiskMetricsFacade,
    "var_calculator": risk.VaRCalculator,
    "volatility_calculator": risk.VolatilityCalculator,
    "drawdown_calculator": risk.DrawdownCalculator,
    "performance_calculator": risk.PerformanceCalculator,
    "stress_test_calculator": risk.StressTestCalculator,
    "tail_risk_calculator": risk.TailRiskCalculator,
}


def get_calculator(calculator_name: str, config: dict = None):
    """
    Get calculator instance by name.

    Args:
        calculator_name: Name of the calculator
        config: Configuration dictionary for the calculator

    Returns:
        Calculator instance

    Raises:
        ValueError: If calculator name is not found
    """
    if calculator_name not in CALCULATOR_REGISTRY:
        available = list(CALCULATOR_REGISTRY.keys())
        raise ValueError(f"Calculator '{calculator_name}' not found. Available: {available}")

    calculator_class = CALCULATOR_REGISTRY[calculator_name]
    return calculator_class(config=config)


def get_available_calculators():
    """Get list of available calculator names."""
    return list(CALCULATOR_REGISTRY.keys())

"""
Correlation Analysis Module

Specialized correlation calculators organized by functional area.
Each calculator focuses on a specific domain of correlation analysis.
"""

# Base utilities and configuration
from .base_correlation import BaseCorrelationCalculator
from .correlation_config import CorrelationConfig

# Specialized calculators
from .rolling_calculator import RollingCorrelationCalculator
from .beta_calculator import BetaAnalysisCalculator
from .stability_calculator import StabilityAnalysisCalculator
from .leadlag_calculator import LeadLagCalculator
from .pca_calculator import PCACorrelationCalculator
from .regime_calculator import RegimeCorrelationCalculator

# Unified facade for backward compatibility
from .enhanced_correlation_facade import EnhancedCorrelationCalculator

__all__ = [
    "BaseCorrelationCalculator",
    "CorrelationConfig",
    "RollingCorrelationCalculator",
    "BetaAnalysisCalculator", 
    "StabilityAnalysisCalculator",
    "LeadLagCalculator",
    "PCACorrelationCalculator",
    "RegimeCorrelationCalculator",
    "EnhancedCorrelationCalculator"
]

# Registry for correlation calculators
CORRELATION_CALCULATOR_REGISTRY = {
    "rolling": RollingCorrelationCalculator,
    "beta": BetaAnalysisCalculator,
    "stability": StabilityAnalysisCalculator,
    "leadlag": LeadLagCalculator,
    "pca": PCACorrelationCalculator,
    "regime": RegimeCorrelationCalculator,
    "facade": EnhancedCorrelationCalculator
}

def get_correlation_calculator(calculator_name: str, config: dict = None):
    """
    Get correlation calculator instance by name.
    
    Args:
        calculator_name: Name of the correlation calculator
        config: Configuration dictionary for the calculator
        
    Returns:
        Calculator instance
        
    Raises:
        ValueError: If calculator name is not found
    """
    if calculator_name not in CORRELATION_CALCULATOR_REGISTRY:
        available = list(CORRELATION_CALCULATOR_REGISTRY.keys())
        raise ValueError(f"Correlation calculator '{calculator_name}' not found. Available: {available}")
    
    calculator_class = CORRELATION_CALCULATOR_REGISTRY[calculator_name]
    return calculator_class(config=config)

def get_available_correlation_calculators():
    """Get list of available correlation calculator names."""
    return list(CORRELATION_CALCULATOR_REGISTRY.keys())
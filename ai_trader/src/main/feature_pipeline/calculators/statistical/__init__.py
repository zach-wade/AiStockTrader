"""
Statistical Analysis Module

Provides advanced statistical feature calculators for time series analysis.
Each calculator focuses on specific statistical properties and measures.
"""

# Base utilities and configuration
from .base_statistical import BaseStatisticalCalculator
from .statistical_config import (
    StatisticalConfig,
    EntropyConfig,
    MomentConfig,
    DistributionType,
    OutlierMethod,
    EntropyMethod,
    create_default_config,
    create_fast_config,
    create_comprehensive_config
)

# Specialized calculators
from .entropy_calculator import EntropyCalculator
from .moments_calculator import MomentsCalculator
from .fractal_calculator import FractalCalculator
from .multivariate_calculator import MultivariateCalculator
from .nonlinear_calculator import NonlinearCalculator
from .timeseries_calculator import TimeseriesCalculator
# Create alias for backward compatibility
TimeSeriesCalculator = TimeseriesCalculator

# Unified facade for backward compatibility
from .advanced_statistical_facade import AdvancedStatisticalCalculator

__all__ = [
    # Base classes and configuration
    "BaseStatisticalCalculator",
    "StatisticalConfig",
    "EntropyConfig",
    "MomentConfig",
    "DistributionType",
    "OutlierMethod",
    "EntropyMethod",
    
    # Configuration factories
    "create_default_config",
    "create_fast_config",
    "create_comprehensive_config",
    
    # Specialized calculators
    "EntropyCalculator",
    "MomentsCalculator",
    "FractalCalculator",
    "MultivariateCalculator",
    "NonlinearCalculator",
    "TimeSeriesCalculator",
    
    # Facade
    "AdvancedStatisticalCalculator"
]

# Registry for statistical calculators
STATISTICAL_CALCULATOR_REGISTRY = {
    "entropy": EntropyCalculator,
    "moments": MomentsCalculator,
    "fractal": FractalCalculator,
    "multivariate": MultivariateCalculator,
    "nonlinear": NonlinearCalculator,
    "timeseries": TimeSeriesCalculator,
    "facade": AdvancedStatisticalCalculator
}

# Feature counts for each calculator
FEATURE_COUNTS = {
    "entropy": 45,  # Shannon, Renyi, Tsallis, Approximate, Sample, Permutation, etc.
    "moments": 38,  # Raw moments, central moments, standardized moments, L-moments
    "fractal": 25,  # Hurst exponent, fractal dimension, DFA, multifractal spectrum
    "multivariate": 42,  # Multivariate distributions, copulas, dependencies
    "nonlinear": 35,  # Lyapunov exponent, recurrence quantification, chaos measures
    "timeseries": 40   # ARIMA features, stationarity tests, decomposition
}

# Total features across all calculators
TOTAL_STATISTICAL_FEATURES = sum(FEATURE_COUNTS.values())  # 225 total features


def get_statistical_calculator(calculator_name: str, config: dict = None):
    """
    Get statistical calculator instance by name.
    
    Args:
        calculator_name: Name of calculator from registry
        config: Optional configuration dictionary
        
    Returns:
        Calculator instance
        
    Raises:
        ValueError: If calculator name not found
    """
    if calculator_name not in STATISTICAL_CALCULATOR_REGISTRY:
        available = list(STATISTICAL_CALCULATOR_REGISTRY.keys())
        raise ValueError(f"Calculator '{calculator_name}' not found. Available: {available}")
    
    calc_class = STATISTICAL_CALCULATOR_REGISTRY[calculator_name]
    return calc_class(config)


def get_all_statistical_calculators(config: dict = None):
    """
    Get instances of all statistical calculators.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary mapping calculator names to instances
    """
    calculators = {}
    
    for name, calc_class in STATISTICAL_CALCULATOR_REGISTRY.items():
        if name != 'facade':  # Skip facade to avoid duplication
            try:
                calculators[name] = calc_class(config)
            except Exception as e:
                print(f"Warning: Failed to initialize {name}: {e}")
                continue
    
    return calculators


def get_statistical_feature_summary():
    """
    Get summary of features provided by each calculator.
    
    Returns:
        Dictionary with feature counts and descriptions
    """
    summary = {
        'total_features': TOTAL_STATISTICAL_FEATURES,
        'calculator_counts': FEATURE_COUNTS.copy(),
        'descriptions': {
            'entropy': 'Information theory and complexity measures (45 features)',
            'moments': 'Statistical moments and distribution properties (38 features)',
            'fractal': 'Fractal analysis and self-similarity (25 features)',
            'multivariate': 'Multivariate statistical analysis (42 features)',
            'nonlinear': 'Nonlinear dynamics and chaos theory (35 features)',
            'timeseries': 'Time series specific analysis (40 features)'
        },
        'categories': {
            'complexity': ['entropy', 'fractal', 'nonlinear'],
            'distribution': ['moments', 'multivariate'],
            'temporal': ['timeseries', 'nonlinear']
        }
    }
    
    return summary


# Version information
__version__ = "2.0.0"
__author__ = "AI Trader Statistical Analysis Team"
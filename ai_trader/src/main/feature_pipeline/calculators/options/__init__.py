"""
Options Analytics Module

Modular options analytics system providing specialized calculators for:
- Volume and flow analysis
- Put/Call ratio analysis  
- Implied volatility analysis
- Options Greeks computation
- Strike and moneyness analysis
- Unusual activity detection
- Market sentiment indicators
- Black-Scholes pricing utilities

Each calculator follows SOLID principles with single responsibility focus.
The facade maintains backward compatibility with existing code.
"""

# Configuration
from .options_config import OptionsConfig

# Base calculator
from .base_options import BaseOptionsCalculator

# Specialized calculators
from .volume_flow_calculator import VolumeFlowCalculator
from .putcall_calculator import PutCallAnalysisCalculator
from .iv_calculator import ImpliedVolatilityCalculator
from .greeks_calculator import GreeksCalculator
from .moneyness_calculator import MoneynessCalculator
from .unusual_activity_calculator import UnusualActivityCalculator
from .sentiment_calculator import OptionsSentimentCalculator as SentimentCalculator
from .blackscholes_calculator import BlackScholesCalculator

# Facade for backward compatibility
from .options_analytics_facade import OptionsAnalyticsFacade, OptionsAnalyticsCalculator

# Registry of all available calculators
CALCULATOR_REGISTRY = {
    'volume_flow': VolumeFlowCalculator,
    'putcall_analysis': PutCallAnalysisCalculator,
    'implied_volatility': ImpliedVolatilityCalculator,
    'greeks': GreeksCalculator,
    'moneyness': MoneynessCalculator,
    'unusual_activity': UnusualActivityCalculator,
    'sentiment': SentimentCalculator,
    'blackscholes': BlackScholesCalculator,
    'facade': OptionsAnalyticsFacade
}

# Feature counts for each calculator
FEATURE_COUNTS = {
    'volume_flow': 28,
    'putcall_analysis': 24,
    'implied_volatility': 26,
    'greeks': 32,
    'moneyness': 29,
    'unusual_activity': 27,
    'sentiment': 30,
    'blackscholes': 25
}

# Total features across all calculators
TOTAL_FEATURES = sum(FEATURE_COUNTS.values())  # 221 total features

def get_calculator(calculator_name: str, config=None):
    """
    Get a calculator instance by name.
    
    Args:
        calculator_name: Name of calculator from CALCULATOR_REGISTRY
        config: Optional configuration dictionary
        
    Returns:
        Calculator instance
        
    Raises:
        ValueError: If calculator name not found
    """
    if calculator_name not in CALCULATOR_REGISTRY:
        available = list(CALCULATOR_REGISTRY.keys())
        raise ValueError(f"Calculator '{calculator_name}' not found. Available: {available}")
    
    calc_class = CALCULATOR_REGISTRY[calculator_name]
    return calc_class(config)

def get_all_calculators(config=None):
    """
    Get instances of all specialized calculators.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary mapping calculator names to instances
    """
    calculators = {}
    
    for name, calc_class in CALCULATOR_REGISTRY.items():
        if name != 'facade':  # Skip facade to avoid duplication
            try:
                calculators[name] = calc_class(config)
            except Exception as e:
                print(f"Warning: Failed to initialize {name}: {e}")
                continue
    
    return calculators

def get_feature_summary():
    """
    Get summary of features provided by each calculator.
    
    Returns:
        Dictionary with feature counts and descriptions
    """
    summary = {
        'total_features': TOTAL_FEATURES,
        'calculator_counts': FEATURE_COUNTS.copy(),
        'descriptions': {
            'volume_flow': 'Options volume and flow analysis (28 features)',
            'putcall_analysis': 'Put/Call ratio and sentiment analysis (24 features)',
            'implied_volatility': 'IV term structure and volatility analysis (26 features)',
            'greeks': 'Options Greeks and risk exposure (32 features)',
            'moneyness': 'Strike distribution and moneyness analysis (29 features)',
            'unusual_activity': 'Unusual options flow detection (27 features)',
            'sentiment': 'Options-based market sentiment (30 features)',
            'blackscholes': 'Mathematical pricing and modeling (25 features)'
        }
    }
    
    return summary

# Version information
__version__ = "1.0.0"
__author__ = "AI Trader Options Analytics Team"
__description__ = "Modular options analytics system with 221 specialized features"

# Public API
__all__ = [
    # Configuration
    'OptionsConfig',
    
    # Base
    'BaseOptionsCalculator',
    
    # Specialized calculators
    'VolumeFlowCalculator',
    'PutCallAnalysisCalculator', 
    'ImpliedVolatilityCalculator',
    'GreeksCalculator',
    'MoneynessCalculator',
    'UnusualActivityCalculator',
    'SentimentCalculator',
    'BlackScholesCalculator',
    
    # Facade
    'OptionsAnalyticsFacade',
    'OptionsAnalyticsCalculator',  # Backward compatibility alias
    
    # Registry and utilities
    'CALCULATOR_REGISTRY',
    'FEATURE_COUNTS',
    'TOTAL_FEATURES',
    'get_calculator',
    'get_all_calculators',
    'get_feature_summary',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
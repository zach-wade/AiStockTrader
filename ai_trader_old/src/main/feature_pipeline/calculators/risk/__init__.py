"""
Risk Management Module

Provides comprehensive risk metric calculators for portfolio and position analysis.
Each calculator focuses on specific risk dimensions and measurement approaches.
"""

# Standard library imports
from typing import Optional

# Third-party imports
import pandas as pd

# Base class and configuration
from .base_risk import BaseRiskCalculator
from .drawdown_calculator import DrawdownCalculator
from .performance_calculator import PerformanceCalculator
from .risk_config import (
    RiskConfig,
    VaRMethod,
    VolatilityMethod,
    create_aggressive_risk_config,
    create_conservative_risk_config,
    create_default_risk_config,
)

# Unified risk metrics facade
from .risk_metrics_facade import RiskMetricsFacade
from .stress_test_calculator import StressTestCalculator
from .tail_risk_calculator import TailRiskCalculator

# Specialized risk calculators
from .var_calculator import VaRCalculator
from .volatility_calculator import VolatilityCalculator

__all__ = [
    # Base classes and configuration
    "BaseRiskCalculator",
    "RiskConfig",
    "VaRMethod",
    "VolatilityMethod",
    # Configuration factories
    "create_default_risk_config",
    "create_conservative_risk_config",
    "create_aggressive_risk_config",
    # Specialized calculators
    "VaRCalculator",
    "VolatilityCalculator",
    "DrawdownCalculator",
    "PerformanceCalculator",
    "StressTestCalculator",
    "TailRiskCalculator",
    # Facade
    "RiskMetricsFacade",
]

# Registry for risk calculators
RISK_CALCULATOR_REGISTRY = {
    "var": VaRCalculator,
    "volatility": VolatilityCalculator,
    "drawdown": DrawdownCalculator,
    "performance": PerformanceCalculator,
    "stress_test": StressTestCalculator,
    "tail_risk": TailRiskCalculator,
    "facade": RiskMetricsFacade,
}

# Feature counts for each calculator
FEATURE_COUNTS = {
    "var": 35,  # Value at Risk metrics across different methods
    "volatility": 42,  # Various volatility measures and forecasts
    "drawdown": 28,  # Drawdown analysis and recovery metrics
    "performance": 45,  # Risk-adjusted performance measures
    "stress_test": 30,  # Stress testing and scenario analysis
    "tail_risk": 25,  # Tail risk and extreme event analysis
}

# Total features across all calculators
TOTAL_RISK_FEATURES = sum(FEATURE_COUNTS.values())  # 205 total features

# Risk metric categories
RISK_CATEGORIES = {
    "downside_risk": ["var", "drawdown", "tail_risk"],
    "volatility_risk": ["volatility"],
    "performance_risk": ["performance"],
    "scenario_risk": ["stress_test"],
}


def get_risk_calculator(calculator_name: str, config: dict = None):
    """
    Get risk calculator instance by name.

    Args:
        calculator_name: Name of calculator from registry
        config: Optional configuration dictionary

    Returns:
        Calculator instance

    Raises:
        ValueError: If calculator name not found
    """
    if calculator_name not in RISK_CALCULATOR_REGISTRY:
        available = list(RISK_CALCULATOR_REGISTRY.keys())
        raise ValueError(f"Calculator '{calculator_name}' not found. Available: {available}")

    calc_class = RISK_CALCULATOR_REGISTRY[calculator_name]
    return calc_class(config)


def get_all_risk_calculators(config: dict = None):
    """
    Get instances of all risk calculators.

    Args:
        config: Optional configuration dictionary

    Returns:
        Dictionary mapping calculator names to instances
    """
    calculators = {}

    for name, calc_class in RISK_CALCULATOR_REGISTRY.items():
        if name != "facade":  # Skip facade to avoid duplication
            try:
                calculators[name] = calc_class(config)
            except Exception as e:
                print(f"Warning: Failed to initialize {name}: {e}")
                continue

    return calculators


def get_risk_feature_summary():
    """
    Get summary of features provided by each calculator.

    Returns:
        Dictionary with feature counts and descriptions
    """
    summary = {
        "total_features": TOTAL_RISK_FEATURES,
        "calculator_counts": FEATURE_COUNTS.copy(),
        "descriptions": {
            "var": "Value at Risk calculations across multiple methodologies (35 features)",
            "volatility": "Comprehensive volatility analysis and forecasting (42 features)",
            "drawdown": "Drawdown metrics and recovery analysis (28 features)",
            "performance": "Risk-adjusted performance metrics (45 features)",
            "stress_test": "Stress testing and scenario analysis (30 features)",
            "tail_risk": "Tail risk and extreme event metrics (25 features)",
        },
        "categories": RISK_CATEGORIES,
    }

    return summary


def calculate_portfolio_risk_metrics(
    returns: pd.DataFrame, weights: pd.Series | None = None, config: dict | None = None
):
    """
    Calculate comprehensive risk metrics for a portfolio.

    Args:
        returns: DataFrame of asset returns
        weights: Optional portfolio weights
        config: Risk calculation configuration

    Returns:
        Dictionary of risk metrics
    """
    # Use facade for comprehensive calculation
    facade = RiskMetricsFacade(config)

    # Set portfolio data if weights provided
    if weights is not None:
        portfolio_returns = (returns * weights).sum(axis=1)
    else:
        portfolio_returns = returns.mean(axis=1)

    # Calculate all risk metrics
    risk_metrics = facade.calculate(pd.DataFrame({"returns": portfolio_returns}))

    return risk_metrics.to_dict()


# Risk thresholds and limits
RISK_THRESHOLDS = {
    "max_var_95": 0.02,  # 2% VaR at 95% confidence
    "max_var_99": 0.05,  # 5% VaR at 99% confidence
    "max_volatility": 0.20,  # 20% annualized volatility
    "max_drawdown": 0.15,  # 15% maximum drawdown
    "min_sharpe": 0.5,  # Minimum Sharpe ratio
    "min_sortino": 0.7,  # Minimum Sortino ratio
    "max_tail_risk": 0.10,  # 10% tail risk threshold
}


def check_risk_limits(risk_metrics: dict) -> dict:
    """
    Check if risk metrics exceed predefined thresholds.

    Args:
        risk_metrics: Dictionary of calculated risk metrics

    Returns:
        Dictionary of limit breaches
    """
    breaches = {}

    # Check each threshold
    for metric, threshold in RISK_THRESHOLDS.items():
        if metric.startswith("max_"):
            actual_metric = metric.replace("max_", "")
            if actual_metric in risk_metrics and risk_metrics[actual_metric] > threshold:
                breaches[metric] = {
                    "actual": risk_metrics[actual_metric],
                    "limit": threshold,
                    "breach_pct": (risk_metrics[actual_metric] - threshold) / threshold,
                }
        elif metric.startswith("min_"):
            actual_metric = metric.replace("min_", "")
            if actual_metric in risk_metrics and risk_metrics[actual_metric] < threshold:
                breaches[metric] = {
                    "actual": risk_metrics[actual_metric],
                    "limit": threshold,
                    "breach_pct": (threshold - risk_metrics[actual_metric]) / threshold,
                }

    return breaches


# Version information
__version__ = "2.0.0"
__author__ = "AI Trader Risk Management Team"

# integration/__init__.py
"""
Risk Management Integration Modules

This module provides integration components for risk management including:
- TradingEngineIntegration: Integration with trading engine for risk management
"""

from .trading_engine_integration import TradingEngineIntegration

__all__ = [
    'TradingEngineIntegration',
]
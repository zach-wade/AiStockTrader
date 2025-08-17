"""
Action Processors Module

Provides strategy pattern implementations for processing different
types of corporate actions (dividends, splits, etc.).
"""

from .base import ActionProcessorConfig, BaseActionProcessor
from .dividend import DividendProcessor
from .split import SplitProcessor

__all__ = ["BaseActionProcessor", "ActionProcessorConfig", "DividendProcessor", "SplitProcessor"]

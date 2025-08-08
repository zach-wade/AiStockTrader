"""
Action Processors Module

Provides strategy pattern implementations for processing different
types of corporate actions (dividends, splits, etc.).
"""

from .base import BaseActionProcessor, ActionProcessorConfig
from .dividend import DividendProcessor
from .split import SplitProcessor

__all__ = [
    'BaseActionProcessor',
    'ActionProcessorConfig',
    'DividendProcessor',
    'SplitProcessor'
]
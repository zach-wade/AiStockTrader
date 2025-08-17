"""
Validation Rules Module

Modularized validation rules engine with separate components for
definitions, parsing, execution, and registry management.
"""

from .rule_definitions import (
    DEFAULT_FUNDAMENTALS_RULES,
    DEFAULT_MARKET_DATA_RULES,
    DEFAULT_NEWS_RULES,
    FailureAction,
    RuleExecutionResult,
    RuleProfile,
    ValidationRule,
)
from .rule_executor import RuleExecutor
from .rule_parser import RuleParser
from .rule_registry import RuleRegistry
from .rules_engine import ValidationRulesEngine

__all__ = [
    # Definitions
    "ValidationRule",
    "RuleProfile",
    "FailureAction",
    "RuleExecutionResult",
    "DEFAULT_MARKET_DATA_RULES",
    "DEFAULT_NEWS_RULES",
    "DEFAULT_FUNDAMENTALS_RULES",
    # Components
    "RuleParser",
    "RuleExecutor",
    "RuleRegistry",
    # Main engine
    "ValidationRulesEngine",
]

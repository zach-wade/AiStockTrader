"""
Unified Limit Checker - Backward Compatibility Facade

This module provides 100% backward compatibility with the original monolithic
unified_limit_checker.py file while leveraging the new modular architecture.

All original imports and functionality are preserved.
"""

import warnings

# Issue a deprecation warning for direct imports
warnings.warn(
    "Direct import from unified_limit_checker.py is deprecated. "
    "Please import from the unified_limit_checker package instead: "
    "from main.risk_management.pre_trade.unified_limit_checker import UnifiedLimitChecker",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new modular package
from .unified_limit_checker import *

# Re-export for backward compatibility
__all__ = [
    # Core types
    'LimitType',
    'LimitScope', 
    'ViolationSeverity',
    'LimitAction',
    'ComparisonOperator',
    
    # Data models
    'LimitDefinition',
    'LimitViolation',
    'LimitCheckResult',
    
    # Main classes
    'UnifiedLimitChecker',
    'LimitChecker',
    'SimpleThresholdChecker',
    'PositionSizeChecker',
    'DrawdownChecker',
    
    # Templates and utilities
    'LimitTemplates',
    'create_limit_checker',
    'create_basic_portfolio_limits',
]
"""
Unified Limit Checker Package

A comprehensive limit checking system for threshold validation across the codebase.
Provides modular, extensible, and configurable limit checking with event-driven architecture.

Features:
- Multiple limit types and specialized checkers
- Configurable violation actions and severity levels
- Historical violation tracking and analytics
- Event-driven architecture with customizable handlers
- Real-time monitoring and alerting
- Comprehensive reporting and audit trails
- 100% backward compatibility through facade pattern

Usage:
    from main.risk_management.pre_trade.unified_limit_checker import UnifiedLimitChecker

    # Create checker with defaults
    checker = UnifiedLimitChecker()

    # Add limits
    from main.risk_management.pre_trade.unified_limit_checker import LimitTemplates
    limit = LimitTemplates.create_position_size_limit("pos_limit", 10.0)
    checker.add_limit(limit)

    # Check limit
    result = checker.check_limit("pos_limit", 15.0, {"portfolio_value": 100.0})

    # Or use convenience functions
    from main.risk_management.pre_trade.unified_limit_checker import create_limit_checker
    checker = create_limit_checker()
"""

# Core types
# Specialized checkers
from .checkers import DrawdownChecker, PositionSizeChecker, SimpleThresholdChecker

# Configuration
from .config import LimitConfig, get_default_config

# Event system
from .events import CheckEvent, EventManager, ResolutionEvent, ViolationEvent

# Data models
from .models import LimitCheckResult, LimitDefinition, LimitViolation

# Registry and base classes
from .registry import CheckerRegistry, LimitChecker

# Templates and utilities
from .templates import LimitTemplates, create_basic_portfolio_limits
from .types import ComparisonOperator, LimitAction, LimitScope, LimitType, ViolationSeverity

# Main orchestrator
from .unified_limit_checker import UnifiedLimitChecker
from .utils import (
    create_limit_checker,
    create_limit_checker_with_defaults,
    format_limit_summary,
    setup_basic_portfolio_limits,
    setup_comprehensive_portfolio_limits,
    validate_limit_definition,
)

# Backward compatibility aliases
# These ensure that existing code continues to work
UnifiedLimitChecker = UnifiedLimitChecker
LimitChecker = LimitChecker
LimitDefinition = LimitDefinition
LimitViolation = LimitViolation
LimitCheckResult = LimitCheckResult
LimitType = LimitType
LimitScope = LimitScope
ViolationSeverity = ViolationSeverity
LimitAction = LimitAction
ComparisonOperator = ComparisonOperator
LimitTemplates = LimitTemplates

# Convenience functions for backward compatibility
create_limit_checker = create_limit_checker
create_basic_portfolio_limits = create_basic_portfolio_limits

__all__ = [
    # Core types
    "LimitType",
    "LimitScope",
    "ViolationSeverity",
    "LimitAction",
    "ComparisonOperator",
    # Data models
    "LimitDefinition",
    "LimitViolation",
    "LimitCheckResult",
    # Main classes
    "UnifiedLimitChecker",
    "LimitChecker",
    "CheckerRegistry",
    # Configuration
    "LimitConfig",
    "get_default_config",
    # Event system
    "EventManager",
    "ViolationEvent",
    "ResolutionEvent",
    "CheckEvent",
    # Specialized checkers
    "SimpleThresholdChecker",
    "PositionSizeChecker",
    "DrawdownChecker",
    # Templates and utilities
    "LimitTemplates",
    "create_basic_portfolio_limits",
    "create_limit_checker",
    "create_limit_checker_with_defaults",
    "setup_basic_portfolio_limits",
    "setup_comprehensive_portfolio_limits",
    "validate_limit_definition",
    "format_limit_summary",
]

# Package metadata
__version__ = "2.0.0"
__author__ = "AI Trading System Development Team"
__description__ = "Unified Limit Checker - Modular threshold validation system"

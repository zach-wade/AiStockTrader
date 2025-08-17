"""
Validation Rules Engine - Compatibility Layer

This file provides backward compatibility by importing from the new modular rules structure.
The actual implementation has been split into smaller, more manageable modules in the rules/ directory.
"""

# Import everything from the new modular structure
from ..rules import (
    FailureAction,
    RuleExecutionResult,
    RuleExecutor,
    RuleParser,
    RuleProfile,
    RuleRegistry,
    ValidationRule,
    ValidationRulesEngine,
)

# Re-export for backward compatibility
__all__ = [
    "ValidationRule",
    "RuleProfile",
    "FailureAction",
    "RuleExecutionResult",
    "RuleParser",
    "RuleExecutor",
    "RuleRegistry",
    "ValidationRulesEngine",
]

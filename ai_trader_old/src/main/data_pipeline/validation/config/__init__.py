"""
Validation Framework - Configuration Management

Configuration and rule management components.

Components:
- validation_profile_manager: Profile and field mapping management
- validation_rules_engine: Rules loading and evaluation engine

Note: ValidationConfig has been moved to main.config.validation_models.services.DataPipelineConfig.ValidationConfig
"""

from .validation_profile_manager import ValidationProfileManager, create_validation_profile_manager
from .validation_rules_engine import (
    FailureAction,
    RuleProfile,
    ValidationRule,
    ValidationRulesEngine,
)

__all__ = [
    # Profile management
    "ValidationProfileManager",
    # Rules engine
    "ValidationRulesEngine",
    "ValidationRule",
    "RuleProfile",
    "FailureAction",
    # Factory functions
    "create_validation_profile_manager",
]

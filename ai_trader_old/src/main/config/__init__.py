"""
Configuration management system for the AI Trader.
"""

# Core configuration manager
from .config_manager import (
    ConfigManager,
    get_config,
    get_config_manager,
    get_production_config_manager,
)

# Environment utilities
from .env_loader import (
    ensure_environment_loaded,
    get_env_var,
    get_environment_info,
    is_development,
    is_production,
    set_env_var,
    validate_required_env_vars,
)

# Field mappings
from .field_mappings import FieldMappingConfig, get_field_mapping_config

# Main configuration model
from .validation_models.main import AITraderConfig

# Validation utilities
from .validation_utils import (
    ConfigValidationError,
    ConfigValidator,
    validate_config_and_warn,
    validate_startup_config,
)

__all__ = [
    # Core configuration manager
    "ConfigManager",
    "get_config",
    "get_config_manager",
    "get_production_config_manager",
    "AITraderConfig",
    # Field mappings
    "FieldMappingConfig",
    "get_field_mapping_config",
    # Environment utilities
    "ensure_environment_loaded",
    "get_environment_info",
    "get_env_var",
    "set_env_var",
    "is_development",
    "is_production",
    "validate_required_env_vars",
    # Validation utilities
    "ConfigValidationError",
    "ConfigValidator",
    "validate_startup_config",
    "validate_config_and_warn",
]

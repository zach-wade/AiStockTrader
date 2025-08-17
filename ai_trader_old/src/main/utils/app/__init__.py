"""
AI Trader Application Utilities Package

Application-level utilities for the AI Trader, providing standardized patterns
for common application operations like context management, CLI creation, and workflows.
"""

# App context management
# CLI utilities
from .cli import (
    CLIAppConfig,
    StandardCLIHandler,
    async_command,
    create_cli_app,
    create_data_pipeline_app,
    create_training_app,
    create_validation_app,
    error_message,
    info_message,
    success_message,
    warning_message,
)
from .context import AppContextError, StandardAppContext, create_app_context, managed_app_context

# Workflow management imports removed - using HistoricalManager directly
# Validation utilities
from .validation import (
    AppConfigValidator,
    ConfigValidationError,
    ValidationResult,
    ensure_critical_config,
    validate_app_startup_config,
    validate_data_pipeline_config,
    validate_trading_config,
)

__all__ = [
    # Context management
    "StandardAppContext",
    "AppContextError",
    "create_app_context",
    "managed_app_context",
    # CLI utilities
    "create_cli_app",
    "CLIAppConfig",
    "StandardCLIHandler",
    "create_data_pipeline_app",
    "create_training_app",
    "create_validation_app",
    "async_command",
    "success_message",
    "error_message",
    "info_message",
    "warning_message",
    # Workflow management removed - using HistoricalManager directly
    # Validation utilities
    "AppConfigValidator",
    "ValidationResult",
    "ConfigValidationError",
    "validate_trading_config",
    "validate_data_pipeline_config",
    "ensure_critical_config",
    "validate_app_startup_config",
]

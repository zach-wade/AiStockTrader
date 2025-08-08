"""
Global Configuration Management

Global configuration instance management and convenience functions.
"""

import logging
from typing import Optional, Union, Dict, Any

from .wrapper import ConfigurationWrapper
from .schema import ConfigSchema
from .sources import ConfigFormat

logger = logging.getLogger(__name__)


# Global configuration instance
_global_config: Optional[ConfigurationWrapper] = None


def get_global_config() -> Optional[ConfigurationWrapper]:
    """Get the global configuration instance."""
    return _global_config


def set_global_config(config: ConfigurationWrapper):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config
    logger.info("Global configuration set")


def init_global_config(source: Union[str, Dict[str, Any]], 
                      format: Optional[ConfigFormat] = None,
                      schema: Optional[ConfigSchema] = None,
                      auto_reload: bool = False):
    """Initialize global configuration."""
    global _global_config
    _global_config = load_config(source, format, schema, auto_reload)
    logger.info("Global configuration initialized")


def load_config(source: Union[str, Dict[str, Any]], 
               format: Optional[ConfigFormat] = None,
               schema: Optional[ConfigSchema] = None,
               auto_reload: bool = False) -> ConfigurationWrapper:
    """
    Convenience function to load configuration.
    
    Args:
        source: Configuration source
        format: Configuration format
        schema: Configuration schema
        auto_reload: Enable automatic reloading
        
    Returns:
        ConfigurationWrapper instance
    """
    return ConfigurationWrapper(
        config_source=source,
        config_format=format,
        schema=schema,
        auto_reload=auto_reload
    )


def ensure_global_config() -> ConfigurationWrapper:
    """Ensure global configuration exists and return it."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigurationWrapper()
        logger.info("Created default global configuration")
    return _global_config


def reset_global_config():
    """Reset the global configuration instance."""
    global _global_config
    _global_config = None
    logger.info("Global configuration reset")


def is_global_config_initialized() -> bool:
    """Check if global configuration is initialized."""
    return _global_config is not None


def get_config_value(key: str, default: Any = None) -> Any:
    """Get value from global configuration."""
    config = get_global_config()
    if config is None:
        return default
    return config.get(key, default)


def set_config_value(key: str, value: Any):
    """Set value in global configuration."""
    config = ensure_global_config()
    config.set(key, value)


def has_config_key(key: str) -> bool:
    """Check if key exists in global configuration."""
    config = get_global_config()
    if config is None:
        return False
    return config.has(key)
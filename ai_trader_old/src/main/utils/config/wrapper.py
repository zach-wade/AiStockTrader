"""
Configuration Wrapper

Main configuration wrapper class for unified configuration access.
"""

# Standard library imports
from collections.abc import Callable
from contextlib import contextmanager
import copy
import logging
import threading
from typing import Any

from .loaders import load_from_dict, load_from_env, load_from_file
from .schema import ConfigSchema
from .sources import (
    ConfigFormat,
    ConfigSource,
    create_dict_source,
    create_env_source,
    create_file_source,
)

logger = logging.getLogger(__name__)


class ConfigurationWrapper:
    """
    Unified configuration wrapper supporting multiple formats and sources.

    Provides a consistent interface for accessing configuration data regardless
    of the underlying format or source.
    """

    def __init__(
        self,
        config_source: str | dict[str, Any] | None = None,
        config_format: ConfigFormat | None = None,
        schema: ConfigSchema | None = None,
        auto_reload: bool = False,
        reload_interval: float = 60.0,
    ):
        """
        Initialize configuration wrapper.

        Args:
            config_source: Configuration source (file path, dict, etc.)
            config_format: Configuration format
            schema: Configuration schema for validation
            auto_reload: Enable automatic reloading
            reload_interval: Reload check interval in seconds
        """
        self.config_source = config_source
        self.config_format = config_format
        self.schema = schema
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval

        # Configuration data
        self._config: dict[str, Any] = {}
        self._original_config: dict[str, Any] = {}
        self._lock = threading.RLock()

        # Source tracking
        self._sources: list[ConfigSource] = []
        self._watchers: list[Callable[[dict[str, Any]], None]] = []

        # Load initial configuration
        if config_source:
            self.load_configuration(config_source, config_format)

        logger.info("Configuration wrapper initialized")

    def load_configuration(
        self, source: str | dict[str, Any], format: ConfigFormat | None = None
    ) -> dict[str, Any]:
        """
        Load configuration from source.

        Args:
            source: Configuration source
            format: Configuration format (auto-detected if None)

        Returns:
            Loaded configuration
        """
        with self._lock:
            if isinstance(source, dict):
                # Direct dictionary source
                config = load_from_dict(source)
                self._add_source(create_dict_source())

            elif isinstance(source, str):
                # Standard library imports
                import os

                if os.path.isfile(source):
                    # File source
                    config = load_from_file(source, format)
                    self._add_source(create_file_source(source, format))
                else:
                    # Environment variable prefix
                    config = load_from_env(source)
                    self._add_source(create_env_source(source))
            else:
                raise ValueError(f"Unsupported configuration source: {type(source)}")

            # Apply schema defaults
            if self.schema:
                config = self.schema.apply_defaults(config)
                self.schema.validate(config)

            # Merge with existing config
            self._config.update(config)
            self._original_config = copy.deepcopy(self._config)

            # Notify watchers
            self._notify_watchers()

            logger.info(f"Configuration loaded from {source}")
            return config

    def _add_source(self, source: ConfigSource):
        """Add configuration source to tracking."""
        self._sources.append(source)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        with self._lock:
            return self._get_nested_value(self._config, key, default)

    def _get_nested_value(self, config: dict[str, Any], key: str, default: Any) -> Any:
        """Get nested configuration value using dot notation."""
        keys = key.split(".")
        current = config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def set(self, key: str, value: Any, validate: bool = True):
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            validate: Whether to validate the configuration
        """
        with self._lock:
            self._set_nested_value(self._config, key, value)

            if validate and self.schema:
                self.schema.validate(self._config)

            self._notify_watchers()

    def _set_nested_value(self, config: dict[str, Any], key: str, value: Any):
        """Set nested configuration value using dot notation."""
        keys = key.split(".")
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def update(self, updates: dict[str, Any], validate: bool = True):
        """
        Update multiple configuration values.

        Args:
            updates: Dictionary of updates
            validate: Whether to validate the configuration
        """
        with self._lock:
            self._config.update(updates)

            if validate and self.schema:
                self.schema.validate(self._config)

            self._notify_watchers()

    def has(self, key: str) -> bool:
        """Check if configuration key exists."""
        with self._lock:
            return self._get_nested_value(self._config, key, None) is not None

    def keys(self) -> list[str]:
        """Get all configuration keys."""
        with self._lock:
            return list(self._config.keys())

    def to_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary."""
        with self._lock:
            return copy.deepcopy(self._config)

    def add_watcher(self, callback: Callable[[dict[str, Any]], None]):
        """Add configuration change watcher."""
        self._watchers.append(callback)
        logger.debug("Added configuration watcher")

    def remove_watcher(self, callback: Callable[[dict[str, Any]], None]):
        """Remove configuration change watcher."""
        if callback in self._watchers:
            self._watchers.remove(callback)
            logger.debug("Removed configuration watcher")

    def _notify_watchers(self):
        """Notify all watchers of configuration changes."""
        config_copy = copy.deepcopy(self._config)

        for watcher in self._watchers:
            try:
                watcher(config_copy)
            except Exception as e:
                logger.error(f"Error in configuration watcher: {e}")

    def merge_configuration(self, other: "ConfigurationWrapper", deep: bool = True):
        """
        Merge configuration from another wrapper.

        Args:
            other: Other configuration wrapper
            deep: Whether to perform deep merge
        """
        with self._lock:
            other_config = other.to_dict()

            if deep:
                self._deep_merge(self._config, other_config)
            else:
                self._config.update(other_config)

            if self.schema:
                self.schema.validate(self._config)

            self._notify_watchers()

    def _deep_merge(self, target: dict[str, Any], source: dict[str, Any]):
        """Perform deep merge of dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def reset_to_original(self):
        """Reset configuration to original values."""
        with self._lock:
            self._config = copy.deepcopy(self._original_config)
            self._notify_watchers()
            logger.info("Configuration reset to original values")

    def get_changes(self) -> dict[str, Any]:
        """Get configuration changes from original."""
        with self._lock:
            changes = {}
            self._find_changes(self._original_config, self._config, "", changes)
            return changes

    def _find_changes(
        self,
        original: dict[str, Any],
        current: dict[str, Any],
        prefix: str,
        changes: dict[str, Any],
    ):
        """Find changes between original and current configuration."""
        for key in set(original.keys()) | set(current.keys()):
            full_key = f"{prefix}.{key}" if prefix else key

            if key not in original:
                changes[full_key] = {"action": "added", "value": current[key]}
            elif key not in current:
                changes[full_key] = {"action": "removed", "value": original[key]}
            elif original[key] != current[key]:
                if isinstance(original[key], dict) and isinstance(current[key], dict):
                    self._find_changes(original[key], current[key], full_key, changes)
                else:
                    changes[full_key] = {
                        "action": "changed",
                        "old_value": original[key],
                        "new_value": current[key],
                    }

    @contextmanager
    def temporary_config(self, updates: dict[str, Any]):
        """Context manager for temporary configuration changes."""
        original_values = {}

        try:
            # Save original values
            for key in updates:
                original_values[key] = self.get(key)

            # Apply updates
            for key, value in updates.items():
                self.set(key, value, validate=False)

            yield

        finally:
            # Restore original values
            for key, value in original_values.items():
                if value is not None:
                    self.set(key, value, validate=False)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Dictionary-style assignment."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Dictionary-style membership test."""
        return self.has(key)

    def __str__(self) -> str:
        """String representation."""
        return f"ConfigurationWrapper({len(self._config)} keys)"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"ConfigurationWrapper(keys={list(self._config.keys())})"

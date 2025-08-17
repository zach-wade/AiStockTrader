"""
Configuration Loaders

Configuration loading and parsing functionality.
"""

# Standard library imports
import json
import logging
import os
from typing import Any

# Third-party imports
import yaml

from .sources import ConfigFormat, detect_config_format

logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """Configuration loading error."""

    pass


def load_from_file(file_path: str, format: ConfigFormat | None = None) -> dict[str, Any]:
    """Load configuration from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    if format is None:
        format = detect_config_format(file_path)

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        if format == ConfigFormat.JSON:
            return json.loads(content)
        elif format == ConfigFormat.YAML:
            return yaml.safe_load(content) or {}
        else:
            raise ConfigLoadError(f"Unsupported file format: {format}")

    except json.JSONDecodeError as e:
        raise ConfigLoadError(f"Invalid JSON in {file_path}: {e}")
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Invalid YAML in {file_path}: {e}")
    except Exception as e:
        raise ConfigLoadError(f"Error loading config from {file_path}: {e}")


def load_from_env(prefix: str) -> dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}
    prefix = prefix.upper()

    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to lowercase
            config_key = key[len(prefix) :].lstrip("_").lower()

            # Try to parse as JSON first, then as string
            try:
                config[config_key] = json.loads(value)
            except json.JSONDecodeError:
                # Handle boolean strings
                if value.lower() in ("true", "false"):
                    config[config_key] = value.lower() == "true"
                # Handle numeric strings
                elif value.isdigit():
                    config[config_key] = int(value)
                elif value.replace(".", "", 1).isdigit():
                    config[config_key] = float(value)
                else:
                    config[config_key] = value

    return config


def load_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Load configuration from dictionary."""
    return data.copy()


def parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate type."""
    # Try JSON parsing first
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    # Handle boolean strings
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    elif value.lower() in ("false", "no", "0", "off"):
        return False

    # Handle numeric strings
    if value.isdigit():
        return int(value)

    # Handle float strings
    try:
        return float(value)
    except ValueError:
        pass

    # Handle lists (comma-separated)
    if "," in value:
        return [item.strip() for item in value.split(",")]

    # Return as string
    return value


def load_config_safely(file_path: str, default: dict[str, Any] = None) -> dict[str, Any]:
    """Load configuration safely with fallback to default."""
    if default is None:
        default = {}

    try:
        return load_from_file(file_path)
    except Exception as e:
        logger.warning(f"Failed to load config from {file_path}: {e}")
        return default


def validate_config_file(file_path: str) -> bool:
    """Validate that a configuration file can be loaded."""
    try:
        load_from_file(file_path)
        return True
    except Exception:
        return False


def get_config_file_info(file_path: str) -> dict[str, Any]:
    """Get information about a configuration file."""
    if not os.path.exists(file_path):
        return {"exists": False}

    try:
        format = detect_config_format(file_path)
        stat = os.stat(file_path)

        return {
            "exists": True,
            "format": format.value,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "readable": os.access(file_path, os.R_OK),
            "writable": os.access(file_path, os.W_OK),
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    result = {}

    for config in configs:
        if config:
            _deep_merge_dict(result, config)

    return result


def _deep_merge_dict(target: dict[str, Any], source: dict[str, Any]):
    """Deep merge source dictionary into target dictionary."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge_dict(target[key], value)
        else:
            target[key] = value


def flatten_config(config: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested configuration dictionary."""
    items = []

    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key, sep).items())
        else:
            items.append((new_key, value))

    return dict(items)


def unflatten_config(config: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Unflatten configuration dictionary."""
    result = {}

    for key, value in config.items():
        keys = key.split(sep)
        current = result

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    return result

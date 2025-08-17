"""
Configuration Sources

Configuration format and source management utilities.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class ConfigFormat(Enum):
    """Supported configuration formats."""

    JSON = "json"
    YAML = "yaml"
    ENV = "env"
    DICT = "dict"


class ConfigSourceType(Enum):
    """Configuration source types."""

    FILE = "file"
    ENVIRONMENT = "environment"
    DICT = "dict"
    REMOTE = "remote"


@dataclass
class ConfigSource:
    """Configuration source information."""

    source_type: ConfigSourceType
    location: str
    format: ConfigFormat
    last_modified: datetime | None = None
    checksum: str | None = None

    def __post_init__(self):
        if self.last_modified is None:
            self.last_modified = datetime.now()


def detect_config_format(file_path: str) -> ConfigFormat:
    """Detect configuration format from file extension."""
    suffix = Path(file_path).suffix.lower()

    if suffix in [".json"]:
        return ConfigFormat.JSON
    elif suffix in [".yaml", ".yml"]:
        return ConfigFormat.YAML
    else:
        raise ValueError(f"Cannot detect format for file: {file_path}")


def is_supported_format(file_path: str) -> bool:
    """Check if file format is supported."""
    try:
        detect_config_format(file_path)
        return True
    except ValueError:
        return False


def get_format_extensions(format: ConfigFormat) -> list:
    """Get file extensions for a format."""
    if format == ConfigFormat.JSON:
        return [".json"]
    elif format == ConfigFormat.YAML:
        return [".yaml", ".yml"]
    else:
        return []


def create_file_source(file_path: str, format: ConfigFormat | None = None) -> ConfigSource:
    """Create a file configuration source."""
    if format is None:
        format = detect_config_format(file_path)

    return ConfigSource(source_type=ConfigSourceType.FILE, location=file_path, format=format)


def create_env_source(prefix: str) -> ConfigSource:
    """Create an environment variable configuration source."""
    return ConfigSource(
        source_type=ConfigSourceType.ENVIRONMENT, location=prefix, format=ConfigFormat.ENV
    )


def create_dict_source(name: str = "dict") -> ConfigSource:
    """Create a dictionary configuration source."""
    return ConfigSource(source_type=ConfigSourceType.DICT, location=name, format=ConfigFormat.DICT)


def create_remote_source(url: str, format: ConfigFormat) -> ConfigSource:
    """Create a remote configuration source."""
    return ConfigSource(source_type=ConfigSourceType.REMOTE, location=url, format=format)


def validate_source(source: ConfigSource) -> bool:
    """Validate a configuration source."""
    if source.source_type == ConfigSourceType.FILE:
        # Check if file exists and format is supported
        file_path = Path(source.location)
        return file_path.exists() and is_supported_format(str(file_path))

    elif source.source_type == ConfigSourceType.ENVIRONMENT:
        # Environment sources are always valid
        return True

    elif source.source_type == ConfigSourceType.DICT:
        # Dictionary sources are always valid
        return True

    elif source.source_type == ConfigSourceType.REMOTE:
        # Remote sources would need URL validation
        return source.location.startswith(("http://", "https://"))

    return False

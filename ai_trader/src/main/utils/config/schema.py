"""
Configuration Schema

Configuration schema definition and validation functionality.
"""

from typing import Dict, Any, List, Type, Callable
from dataclasses import dataclass, field


class ConfigValidationError(Exception):
    """Configuration validation error."""
    pass


@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    required_keys: List[str] = field(default_factory=list)
    optional_keys: List[str] = field(default_factory=list)
    type_mapping: Dict[str, Type] = field(default_factory=dict)
    validators: Dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    default_values: Dict[str, Any] = field(default_factory=dict)
    nested_schemas: Dict[str, 'ConfigSchema'] = field(default_factory=dict)
    
    def validate(self, config: Dict[str, Any]):
        """Validate configuration against this schema."""
        # Check required keys
        missing_keys = []
        for key in self.required_keys:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            raise ConfigValidationError(f"Missing required keys: {missing_keys}")
        
        # Check types
        for key, expected_type in self.type_mapping.items():
            if key in config and not isinstance(config[key], expected_type):
                raise ConfigValidationError(
                    f"Invalid type for key '{key}': expected {expected_type.__name__}, "
                    f"got {type(config[key]).__name__}"
                )
        
        # Run custom validators
        for key, validator in self.validators.items():
            if key in config:
                try:
                    if not validator(config[key]):
                        raise ConfigValidationError(f"Validation failed for key '{key}'")
                except Exception as e:
                    raise ConfigValidationError(f"Validator error for key '{key}': {e}")
        
        # Validate nested schemas
        for key, nested_schema in self.nested_schemas.items():
            if key in config and isinstance(config[key], dict):
                nested_schema.validate(config[key])
    
    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to configuration."""
        result = config.copy()
        
        for key, default_value in self.default_values.items():
            if key not in result:
                result[key] = default_value
        
        return result
    
    def add_required_key(self, key: str, key_type: Type = None, validator: Callable = None):
        """Add a required key to the schema."""
        if key not in self.required_keys:
            self.required_keys.append(key)
        
        if key_type:
            self.type_mapping[key] = key_type
        
        if validator:
            self.validators[key] = validator
    
    def add_optional_key(self, key: str, key_type: Type = None, default_value: Any = None, validator: Callable = None):
        """Add an optional key to the schema."""
        if key not in self.optional_keys:
            self.optional_keys.append(key)
        
        if key_type:
            self.type_mapping[key] = key_type
        
        if default_value is not None:
            self.default_values[key] = default_value
        
        if validator:
            self.validators[key] = validator
    
    def add_nested_schema(self, key: str, schema: 'ConfigSchema'):
        """Add a nested schema for a key."""
        self.nested_schemas[key] = schema


def create_config_schema(**kwargs) -> ConfigSchema:
    """Create a configuration schema with fluent interface."""
    return ConfigSchema(**kwargs)


def create_basic_schema() -> ConfigSchema:
    """Create a basic configuration schema."""
    return ConfigSchema(
        required_keys=['name'],
        optional_keys=['description', 'version'],
        type_mapping={
            'name': str,
            'description': str,
            'version': str
        },
        default_values={
            'description': 'No description provided',
            'version': '1.0.0'
        }
    )


def create_database_schema() -> ConfigSchema:
    """Create a database configuration schema."""
    return ConfigSchema(
        required_keys=['host', 'port', 'database'],
        optional_keys=['username', 'password', 'ssl', 'pool_size'],
        type_mapping={
            'host': str,
            'port': int,
            'database': str,
            'username': str,
            'password': str,
            'ssl': bool,
            'pool_size': int
        },
        validators={
            'port': lambda x: 1 <= x <= 65535,
            'pool_size': lambda x: x > 0
        },
        default_values={
            'ssl': False,
            'pool_size': 5
        }
    )


def create_api_schema() -> ConfigSchema:
    """Create an API configuration schema."""
    return ConfigSchema(
        required_keys=['base_url', 'api_key'],
        optional_keys=['timeout', 'retries', 'rate_limit'],
        type_mapping={
            'base_url': str,
            'api_key': str,
            'timeout': (int, float),
            'retries': int,
            'rate_limit': int
        },
        validators={
            'timeout': lambda x: x > 0,
            'retries': lambda x: x >= 0,
            'rate_limit': lambda x: x > 0
        },
        default_values={
            'timeout': 30.0,
            'retries': 3,
            'rate_limit': 100
        }
    )
"""Configuration utilities package."""

# Configuration Optimizer (existing)
from .types import (
    OptimizationStrategy,
    ParameterType,
    ParameterConstraint,
    OptimizationTarget,
    ConfigParameter,
    OptimizationResult
)

from .optimizer import (
    ConfigOptimizer,
    get_global_optimizer
)

from .templates import (
    create_cache_parameters,
    create_database_parameters,
    create_network_parameters,
    create_performance_parameters,
    create_monitoring_parameters
)

# Configuration Wrapper (new)
from .schema import (
    ConfigSchema,
    ConfigValidationError,
    create_config_schema,
    create_basic_schema,
    create_database_schema,
    create_api_schema
)

from .sources import (
    ConfigFormat,
    ConfigSourceType,
    ConfigSource,
    detect_config_format,
    create_file_source,
    create_env_source,
    create_dict_source,
    create_remote_source
)

from .loaders import (
    load_from_file,
    load_from_env,
    load_from_dict,
    merge_configs,
    flatten_config,
    unflatten_config
)

from .wrapper import ConfigurationWrapper

from .persistence import ConfigPersistence

from .global_config import (
    get_global_config,
    set_global_config,
    init_global_config,
    load_config,
    ensure_global_config,
    reset_global_config,
    is_global_config_initialized,
    get_config_value,
    set_config_value,
    has_config_key
)

__all__ = [
    # Config Optimizer Types
    'OptimizationStrategy',
    'ParameterType',
    'ParameterConstraint',
    'OptimizationTarget',
    'ConfigParameter',
    'OptimizationResult',
    
    # Config Optimizer Classes
    'ConfigOptimizer',
    'get_global_optimizer',
    
    # Config Optimizer Templates
    'create_cache_parameters',
    'create_database_parameters',
    'create_network_parameters',
    'create_performance_parameters',
    'create_monitoring_parameters',
    
    # Configuration Wrapper Schema
    'ConfigSchema',
    'ConfigValidationError',
    'create_config_schema',
    'create_basic_schema',
    'create_database_schema',
    'create_api_schema',
    
    # Configuration Wrapper Sources
    'ConfigFormat',
    'ConfigSourceType',
    'ConfigSource',
    'detect_config_format',
    'create_file_source',
    'create_env_source',
    'create_dict_source',
    'create_remote_source',
    
    # Configuration Wrapper Loaders
    'load_from_file',
    'load_from_env',
    'load_from_dict',
    'merge_configs',
    'flatten_config',
    'unflatten_config',
    
    # Configuration Wrapper Core
    'ConfigurationWrapper',
    'ConfigPersistence',
    
    # Configuration Wrapper Global
    'get_global_config',
    'set_global_config',
    'init_global_config',
    'load_config',
    'ensure_global_config',
    'reset_global_config',
    'is_global_config_initialized',
    'get_config_value',
    'set_config_value',
    'has_config_key'
]
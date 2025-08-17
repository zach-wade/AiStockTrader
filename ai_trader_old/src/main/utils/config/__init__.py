"""Configuration utilities package."""

# Configuration Optimizer (existing)
from .global_config import (
    ensure_global_config,
    get_config_value,
    get_global_config,
    has_config_key,
    init_global_config,
    is_global_config_initialized,
    load_config,
    reset_global_config,
    set_config_value,
    set_global_config,
)
from .loaders import (
    flatten_config,
    load_from_dict,
    load_from_env,
    load_from_file,
    merge_configs,
    unflatten_config,
)
from .optimizer import ConfigOptimizer, get_global_optimizer
from .persistence import ConfigPersistence

# Configuration Wrapper (new)
from .schema import (
    ConfigSchema,
    ConfigValidationError,
    create_api_schema,
    create_basic_schema,
    create_config_schema,
    create_database_schema,
)
from .sources import (
    ConfigFormat,
    ConfigSource,
    ConfigSourceType,
    create_dict_source,
    create_env_source,
    create_file_source,
    create_remote_source,
    detect_config_format,
)
from .templates import (
    create_cache_parameters,
    create_database_parameters,
    create_monitoring_parameters,
    create_network_parameters,
    create_performance_parameters,
)
from .types import (
    ConfigParameter,
    OptimizationResult,
    OptimizationStrategy,
    OptimizationTarget,
    ParameterConstraint,
    ParameterType,
)
from .wrapper import ConfigurationWrapper

__all__ = [
    # Config Optimizer Types
    "OptimizationStrategy",
    "ParameterType",
    "ParameterConstraint",
    "OptimizationTarget",
    "ConfigParameter",
    "OptimizationResult",
    # Config Optimizer Classes
    "ConfigOptimizer",
    "get_global_optimizer",
    # Config Optimizer Templates
    "create_cache_parameters",
    "create_database_parameters",
    "create_network_parameters",
    "create_performance_parameters",
    "create_monitoring_parameters",
    # Configuration Wrapper Schema
    "ConfigSchema",
    "ConfigValidationError",
    "create_config_schema",
    "create_basic_schema",
    "create_database_schema",
    "create_api_schema",
    # Configuration Wrapper Sources
    "ConfigFormat",
    "ConfigSourceType",
    "ConfigSource",
    "detect_config_format",
    "create_file_source",
    "create_env_source",
    "create_dict_source",
    "create_remote_source",
    # Configuration Wrapper Loaders
    "load_from_file",
    "load_from_env",
    "load_from_dict",
    "merge_configs",
    "flatten_config",
    "unflatten_config",
    # Configuration Wrapper Core
    "ConfigurationWrapper",
    "ConfigPersistence",
    # Configuration Wrapper Global
    "get_global_config",
    "set_global_config",
    "init_global_config",
    "load_config",
    "ensure_global_config",
    "reset_global_config",
    "is_global_config_initialized",
    "get_config_value",
    "set_config_value",
    "has_config_key",
]

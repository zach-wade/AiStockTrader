"""
Configuration Templates

Predefined parameter templates for common configurations.
"""

from .types import ConfigParameter, ParameterConstraint, ParameterType


def create_cache_parameters() -> list[ConfigParameter]:
    """Create standard cache configuration parameters."""
    return [
        ConfigParameter(
            name="cache_size_mb",
            current_value=512,
            param_type=ParameterType.INTEGER,
            constraint=ParameterConstraint(min_value=64, max_value=4096, step_size=64),
            description="Cache size in megabytes",
            impact_score=0.8,
        ),
        ConfigParameter(
            name="cache_ttl_seconds",
            current_value=300,
            param_type=ParameterType.INTEGER,
            constraint=ParameterConstraint(min_value=60, max_value=3600, step_size=60),
            description="Cache TTL in seconds",
            impact_score=0.6,
        ),
        ConfigParameter(
            name="enable_compression",
            current_value=True,
            param_type=ParameterType.BOOLEAN,
            description="Enable cache compression",
            impact_score=0.4,
        ),
    ]


def create_database_parameters() -> list[ConfigParameter]:
    """Create standard database configuration parameters."""
    return [
        ConfigParameter(
            name="pool_size",
            current_value=20,
            param_type=ParameterType.INTEGER,
            constraint=ParameterConstraint(min_value=5, max_value=100, step_size=5),
            description="Database connection pool size",
            impact_score=0.9,
        ),
        ConfigParameter(
            name="query_timeout",
            current_value=30.0,
            param_type=ParameterType.FLOAT,
            constraint=ParameterConstraint(min_value=5.0, max_value=120.0, step_size=5.0),
            description="Query timeout in seconds",
            impact_score=0.7,
        ),
        ConfigParameter(
            name="enable_query_cache",
            current_value=True,
            param_type=ParameterType.BOOLEAN,
            description="Enable query result caching",
            impact_score=0.5,
        ),
    ]


def create_network_parameters() -> list[ConfigParameter]:
    """Create standard network configuration parameters."""
    return [
        ConfigParameter(
            name="connection_timeout",
            current_value=30.0,
            param_type=ParameterType.FLOAT,
            constraint=ParameterConstraint(min_value=5.0, max_value=120.0, step_size=5.0),
            description="Network connection timeout in seconds",
            impact_score=0.8,
        ),
        ConfigParameter(
            name="max_retries",
            current_value=3,
            param_type=ParameterType.INTEGER,
            constraint=ParameterConstraint(min_value=1, max_value=10, step_size=1),
            description="Maximum number of retries",
            impact_score=0.6,
        ),
        ConfigParameter(
            name="retry_delay",
            current_value=1.0,
            param_type=ParameterType.FLOAT,
            constraint=ParameterConstraint(min_value=0.1, max_value=10.0, step_size=0.1),
            description="Delay between retries in seconds",
            impact_score=0.4,
        ),
    ]


def create_performance_parameters() -> list[ConfigParameter]:
    """Create standard performance configuration parameters."""
    return [
        ConfigParameter(
            name="thread_pool_size",
            current_value=10,
            param_type=ParameterType.INTEGER,
            constraint=ParameterConstraint(min_value=2, max_value=50, step_size=2),
            description="Thread pool size",
            impact_score=0.9,
        ),
        ConfigParameter(
            name="batch_size",
            current_value=100,
            param_type=ParameterType.INTEGER,
            constraint=ParameterConstraint(min_value=10, max_value=1000, step_size=10),
            description="Processing batch size",
            impact_score=0.8,
        ),
        ConfigParameter(
            name="enable_parallel_processing",
            current_value=True,
            param_type=ParameterType.BOOLEAN,
            description="Enable parallel processing",
            impact_score=0.7,
        ),
    ]


def create_monitoring_parameters() -> list[ConfigParameter]:
    """Create standard monitoring configuration parameters."""
    return [
        ConfigParameter(
            name="metrics_interval",
            current_value=60,
            param_type=ParameterType.INTEGER,
            constraint=ParameterConstraint(min_value=10, max_value=300, step_size=10),
            description="Metrics collection interval in seconds",
            impact_score=0.5,
        ),
        ConfigParameter(
            name="alert_threshold",
            current_value=0.8,
            param_type=ParameterType.FLOAT,
            constraint=ParameterConstraint(min_value=0.1, max_value=0.9, step_size=0.1),
            description="Alert threshold (0-1)",
            impact_score=0.6,
        ),
        ConfigParameter(
            name="enable_detailed_logging",
            current_value=False,
            param_type=ParameterType.BOOLEAN,
            description="Enable detailed logging",
            impact_score=0.3,
        ),
    ]

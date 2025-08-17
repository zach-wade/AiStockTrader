"""
Test configurations for data pipeline testing.

Provides various configuration scenarios for testing different
aspects of the data pipeline.
"""

# Third-party imports
from omegaconf import DictConfig


def get_test_config() -> DictConfig:
    """Get basic test configuration."""
    return DictConfig(
        {
            "data_pipeline": {
                "enabled": True,
                "batch_size": 100,
                "collection": {"batch_size": 50, "max_parallel_symbols": 2, "retry_delay": 0.1},
                "resilience": {
                    "max_retries": 2,
                    "initial_delay_seconds": 0.1,
                    "backoff_factor": 2.0,
                    "circuit_breaker_threshold": 3,
                    "recovery_timeout_seconds": 1,
                    "rate_limit_calls": 100,
                    "rate_limit_period": 60,
                },
                "backfill": {
                    "source_priorities": {
                        "market_data": ["alpaca", "polygon", "yahoo"],
                        "news": ["benzinga", "alpaca", "polygon"],
                        "social_sentiment": ["reddit", "twitter"],
                    }
                },
                "storage": {
                    "archive": {
                        "type": "local",
                        "local_path": "/tmp/test_archive",
                        "s3_bucket": None,
                        "compression": "gzip",
                    },
                    "database": {
                        "pool_size": 5,
                        "max_overflow": 10,
                        "pool_timeout": 5,
                        "echo": False,
                    },
                },
                "validation": {
                    "enabled": True,
                    "rules_config_path": None,
                    "monitoring": {"enabled": True, "metrics_prefix": "test_validation"},
                },
                "processing": {"batch_size": 500, "parallel_workers": 2, "timeout_seconds": 30},
            },
            "repository": {
                "batch_size": 1000,
                "cache_ttl": 300,
                "enable_metrics": True,
                "log_operations": False,
                "slow_query_threshold_ms": 100,
            },
        }
    )


def get_minimal_config() -> DictConfig:
    """Get minimal configuration for basic testing."""
    return DictConfig(
        {
            "data_pipeline": {
                "enabled": True,
                "batch_size": 10,
                "resilience": {"max_retries": 1, "initial_delay_seconds": 0.01},
            }
        }
    )


def get_performance_test_config() -> DictConfig:
    """Get configuration optimized for performance testing."""
    config = get_test_config()
    config.data_pipeline.batch_size = 10000
    config.data_pipeline.collection.batch_size = 1000
    config.data_pipeline.collection.max_parallel_symbols = 10
    config.data_pipeline.processing.batch_size = 5000
    config.data_pipeline.processing.parallel_workers = 8
    config.repository.batch_size = 10000
    return config


def get_validation_test_config() -> DictConfig:
    """Get configuration for validation testing."""
    config = get_test_config()
    config.data_pipeline.validation.enabled = True
    config.data_pipeline.validation.monitoring.enabled = True
    config.data_pipeline.validation.rules = DictConfig(
        {
            "market_data": {
                "required_fields": ["open", "high", "low", "close", "volume"],
                "max_null_percentage": 0.0,
                "custom_checks": {
                    "validate_ohlc_relationships": True,
                    "validate_positive_prices": True,
                    "validate_non_negative_volume": True,
                },
            },
            "news": {
                "required_fields": ["headline", "created_at", "symbols"],
                "max_null_percentage": 0.1,
                "custom_checks": {"validate_timestamp": True, "validate_symbols": True},
            },
        }
    )
    return config


def get_error_test_config() -> DictConfig:
    """Get configuration for error handling testing."""
    config = get_test_config()
    config.data_pipeline.resilience.max_retries = 3
    config.data_pipeline.resilience.circuit_breaker_threshold = 2
    config.data_pipeline.resilience.recovery_timeout_seconds = 0.5
    return config

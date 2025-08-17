"""Mock configurations for testing."""

# Standard library imports
import os
import tempfile
from typing import Any

# Third-party imports
import yaml


def create_event_bus_config() -> dict[str, Any]:
    """Create mock configuration for EventBus."""
    return {
        "events": {
            "max_queue_size": 1000,
            "max_workers": 5,
            "enable_history": True,
            "history_retention_seconds": 600,
            "enable_dlq": True,
            "circuit_breaker": {
                "failure_threshold": 3,
                "recovery_timeout": 10,
                "half_open_max_calls": 2,
            },
        }
    }


def create_scanner_bridge_config() -> dict[str, Any]:
    """Create mock configuration for ScannerFeatureBridge."""
    return {
        "scanner_feature_bridge": {
            "batch_size": 10,
            "batch_timeout_seconds": 2.0,
            "max_symbols_per_batch": 20,
            "rate_limit_per_second": 5,
            "dedup_window_seconds": 30,
        }
    }


def create_feature_pipeline_config() -> dict[str, Any]:
    """Create mock configuration for FeaturePipelineHandler."""
    return {
        "feature_pipeline": {
            "event_workers": 3,
            "max_queue_size": 500,
            "request_ttl_seconds": 60,
            "max_requests_per_symbol": 5,
        }
    }


def create_alert_mappings_config() -> dict[str, Any]:
    """Create mock alert to feature mappings configuration."""
    return {
        "alert_mappings": {
            "high_volume": ["volume_features", "price_features"],
            "price_breakout": ["price_features", "trend_features"],
            "volatility_spike": ["volatility_features", "risk_metrics"],
            "momentum_shift": ["momentum_features", "trend_features"],
            "news_sentiment": ["sentiment_features", "price_features"],
            "default": ["price_features"],
        }
    }


def create_priority_config() -> dict[str, Any]:
    """Create mock priority configuration."""
    return {
        "priority_boosts": {
            "alert_type_boosts": {
                "ml_signal": 3,
                "unusual_options": 2,
                "volatility_spike": 1,
                "high_volume": 0,
            },
            "score_threshold": 0.8,
            "high_score_boost": 2,
            "market_hours_boost": 1,
            "max_priority": 10,
            "min_priority": 0,
        }
    }


def create_feature_group_mappings_config() -> dict[str, Any]:
    """Create mock feature group mappings configuration."""
    return {
        "feature_group_mappings": {
            "price_features": ["ohlcv", "returns", "moving_averages"],
            "volume_features": ["volume_profile", "vwap", "volume_momentum"],
            "volatility_features": ["realized_vol", "garch", "vol_surface"],
            "momentum_features": ["rsi", "macd", "stochastic"],
            "trend_features": ["ema_crossover", "trend_strength", "support_resistance"],
        }
    }


def create_complete_test_config() -> dict[str, Any]:
    """Create a complete test configuration combining all configs."""
    config = {}
    config.update(create_event_bus_config())
    config.update(create_scanner_bridge_config())
    config.update(create_feature_pipeline_config())
    return config


def write_test_yaml_configs(base_dir: str = None) -> dict[str, str]:
    """Write test YAML configuration files and return paths."""
    if base_dir is None:
        base_dir = tempfile.mkdtemp()

    config_dir = os.path.join(base_dir, "config", "events")
    os.makedirs(config_dir, exist_ok=True)

    configs = {
        "alert_feature_mappings.yaml": create_alert_mappings_config(),
        "priority_boosts.yaml": create_priority_config(),
        "feature_group_mappings.yaml": create_feature_group_mappings_config(),
    }

    paths = {}
    for filename, config in configs.items():
        filepath = os.path.join(config_dir, filename)
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        paths[filename] = filepath

    return paths


class MockConfig:
    """Mock configuration object for testing."""

    def __init__(self, config_dict: dict[str, Any] = None):
        self._config = config_dict or create_complete_test_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict syntax."""
        return self._config[key]

    def update(self, updates: dict[str, Any]) -> None:
        """Update configuration."""
        self._config.update(updates)


def get_test_config_path() -> str:
    """Get path to test configuration directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "test"
    )

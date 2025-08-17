"""
Feature Configuration Management

Manages configuration for feature calculation, validation, and processing.
Integrates with the main configuration system and provides feature-specific settings.
"""

# Standard library imports
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

# Third-party imports
import yaml

# Local imports
from main.config.config_manager import get_config

logger = logging.getLogger(__name__)


@dataclass
class CalculatorConfig:
    """Configuration for a specific feature calculator."""

    name: str
    enabled: bool = True
    priority: int = 5  # 1-10, higher = more important
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    # Resource limits
    max_memory_mb: int | None = None
    max_execution_time_seconds: int | None = None

    # Caching settings
    cache_enabled: bool = True
    cache_ttl_hours: int = 24

    # Validation settings
    validate_inputs: bool = True
    validate_outputs: bool = True


@dataclass
class FeatureSetConfig:
    """Configuration for a group of related features."""

    name: str
    description: str
    calculators: list[str]
    enabled: bool = True

    # Feature selection
    include_features: list[str] | None = None
    exclude_features: list[str] | None = None

    # Processing settings
    preprocessing: bool = True
    scaling: bool = True

    # Update frequency
    update_frequency: str = "daily"  # 'realtime', 'hourly', 'daily', 'weekly'

    # Dependencies
    required_data_types: list[str] = field(default_factory=lambda: ["market_data"])


@dataclass
class ProcessingConfig:
    """Configuration for feature processing pipeline."""

    # Parallel processing
    max_workers: int = 4
    batch_size: int = 100

    # Memory management
    max_memory_usage_mb: int = 2048
    cleanup_frequency: int = 10  # Every N batches

    # Error handling
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    continue_on_error: bool = True

    # Validation
    validate_results: bool = True
    outlier_detection: bool = True

    # Monitoring
    track_performance: bool = True
    log_execution_times: bool = True


class FeatureConfig:
    """
    Comprehensive feature configuration management.

    Manages configuration for calculators, feature sets, processing settings,
    and integration with the main configuration system.
    """

    def __init__(self, config_path: Path | None = None):
        """
        Initialize feature configuration.

        Args:
            config_path: Optional path to feature-specific config file
        """
        self.config_path = config_path
        self.main_config = get_config()

        # Load feature configuration
        self.feature_config = self._load_feature_config()

        # Initialize calculator configurations
        self.calculator_configs = self._load_calculator_configs()

        # Initialize feature set configurations
        self.feature_set_configs = self._load_feature_set_configs()

        # Initialize processing configuration
        self.processing_config = self._load_processing_config()

        logger.info(f"FeatureConfig initialized with {len(self.calculator_configs)} calculators")

    def _load_feature_config(self) -> dict[str, Any]:
        """Load main feature configuration."""
        # Get feature config from main configuration
        feature_config = self.main_config.get("features", {})

        # Load additional config from file if specified
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    file_config = yaml.safe_load(f)

                # Merge configurations
                feature_config.update(file_config.get("features", {}))
                logger.debug(f"Loaded additional config from {self.config_path}")

            except Exception as e:
                logger.error(f"Failed to load config from {self.config_path}: {e}")

        return feature_config

    def _load_calculator_configs(self) -> dict[str, CalculatorConfig]:
        """Load calculator configurations."""
        calculator_configs = {}

        # Default calculator configurations
        default_calculators = {
            "technical_indicators": {
                "enabled": True,
                "priority": 8,
                "config": {"windows": [10, 20, 50], "include_volume": True},
            },
            "advanced_statistical": {
                "enabled": True,
                "priority": 6,
                "config": {"entropy_windows": [20, 50], "fractal_windows": [10, 20]},
            },
            "cross_asset": {
                "enabled": True,
                "priority": 7,
                "config": {
                    "correlation_windows": [20, 60, 120],
                    "benchmark_symbols": ["SPY", "QQQ", "IWM"],
                },
            },
            "sentiment_features": {
                "enabled": True,
                "priority": 5,
                "dependencies": ["news_features"],
                "config": {
                    "sentiment_windows": [1, 3, 7],
                    "social_platforms": ["twitter", "reddit"],
                },
            },
            "market_regime": {
                "enabled": True,
                "priority": 6,
                "config": {"volatility_windows": [20, 60], "trend_windows": [50, 200]},
            },
        }

        # Get calculator config from main config
        calc_config = self.feature_config.get("calculators", {})

        # Merge with defaults
        for calc_name, default_config in default_calculators.items():
            user_config = calc_config.get(calc_name, {})
            merged_config = {**default_config, **user_config}

            calculator_configs[calc_name] = CalculatorConfig(name=calc_name, **merged_config)

        # Add any additional calculator configs from user config
        for calc_name, user_config in calc_config.items():
            if calc_name not in calculator_configs:
                calculator_configs[calc_name] = CalculatorConfig(name=calc_name, **user_config)

        return calculator_configs

    def _load_feature_set_configs(self) -> dict[str, FeatureSetConfig]:
        """Load feature set configurations."""
        feature_set_configs = {}

        # Default feature sets
        default_sets = {
            "basic_technical": {
                "description": "Basic technical indicators",
                "calculators": ["technical_indicators"],
                "required_data_types": ["market_data"],
            },
            "advanced_analytics": {
                "description": "Advanced statistical and cross-asset features",
                "calculators": ["advanced_statistical", "cross_asset", "market_regime"],
                "required_data_types": ["market_data"],
            },
            "alternative_data": {
                "description": "News and sentiment features",
                "calculators": ["news_features", "sentiment_features"],
                "required_data_types": ["market_data", "news", "social"],
                "update_frequency": "hourly",
            },
            "full_feature_set": {
                "description": "All available features",
                "calculators": [
                    "technical_indicators",
                    "advanced_statistical",
                    "cross_asset",
                    "sentiment_features",
                    "market_regime",
                    "microstructure",
                ],
                "required_data_types": ["market_data", "news", "social"],
            },
        }

        # Get feature sets from config
        sets_config = self.feature_config.get("feature_sets", {})

        # Merge with defaults
        for set_name, default_config in default_sets.items():
            user_config = sets_config.get(set_name, {})
            merged_config = {**default_config, **user_config}

            feature_set_configs[set_name] = FeatureSetConfig(name=set_name, **merged_config)

        # Add any additional feature sets from user config
        for set_name, user_config in sets_config.items():
            if set_name not in feature_set_configs:
                feature_set_configs[set_name] = FeatureSetConfig(name=set_name, **user_config)

        return feature_set_configs

    def _load_processing_config(self) -> ProcessingConfig:
        """Load processing configuration."""
        processing_config = self.feature_config.get("processing", {})

        return ProcessingConfig(**processing_config)

    def get_calculator_config(self, calculator_name: str) -> CalculatorConfig | None:
        """Get configuration for a specific calculator."""
        return self.calculator_configs.get(calculator_name)

    def get_enabled_calculators(self) -> list[str]:
        """Get list of enabled calculator names."""
        return [name for name, config in self.calculator_configs.items() if config.enabled]

    def get_calculators_by_priority(self) -> list[str]:
        """Get calculators sorted by priority (highest first)."""
        enabled_calculators = [
            (name, config) for name, config in self.calculator_configs.items() if config.enabled
        ]

        # Sort by priority (highest first)
        enabled_calculators.sort(key=lambda x: x[1].priority, reverse=True)

        return [name for name, _ in enabled_calculators]

    def get_calculator_dependencies(self, calculator_name: str) -> list[str]:
        """Get dependencies for a calculator."""
        config = self.calculator_configs.get(calculator_name)
        return config.dependencies if config else []

    def resolve_calculator_order(self, requested_calculators: list[str] | None = None) -> list[str]:
        """
        Resolve calculator execution order considering dependencies.

        Args:
            requested_calculators: Specific calculators to include (None for all enabled)

        Returns:
            List of calculator names in execution order
        """
        if requested_calculators is None:
            requested_calculators = self.get_enabled_calculators()

        # Filter to only enabled calculators
        requested_calculators = [
            name
            for name in requested_calculators
            if name in self.calculator_configs and self.calculator_configs[name].enabled
        ]

        # Resolve dependencies using topological sort
        resolved_order = []
        visited = set()
        temp_visited = set()

        def visit(calc_name: str):
            if calc_name in temp_visited:
                logger.warning(f"Circular dependency detected involving {calc_name}")
                return

            if calc_name in visited:
                return

            temp_visited.add(calc_name)

            # Visit dependencies first
            dependencies = self.get_calculator_dependencies(calc_name)
            for dep in dependencies:
                if dep in requested_calculators:
                    visit(dep)

            temp_visited.remove(calc_name)
            visited.add(calc_name)
            resolved_order.append(calc_name)

        # Visit all requested calculators
        for calc_name in requested_calculators:
            visit(calc_name)

        logger.debug(f"Resolved calculator order: {resolved_order}")
        return resolved_order

    def get_feature_set_config(self, set_name: str) -> FeatureSetConfig | None:
        """Get configuration for a feature set."""
        return self.feature_set_configs.get(set_name)

    def get_feature_set_calculators(self, set_name: str) -> list[str]:
        """Get calculators for a feature set."""
        config = self.feature_set_configs.get(set_name)
        return config.calculators if config else []

    def get_available_feature_sets(self) -> list[str]:
        """Get list of available feature set names."""
        return list(self.feature_set_configs.keys())

    def get_processing_config(self) -> ProcessingConfig:
        """Get processing configuration."""
        return self.processing_config

    def validate_configuration(self) -> dict[str, list[str]]:
        """
        Validate the current configuration.

        Returns:
            Dictionary with validation errors by category
        """
        errors = {"calculators": [], "feature_sets": [], "dependencies": [], "processing": []}

        # Validate calculator configurations
        for calc_name, config in self.calculator_configs.items():
            if not config.name:
                errors["calculators"].append(f"Calculator {calc_name} missing name")

            if config.priority < 1 or config.priority > 10:
                errors["calculators"].append(
                    f"Calculator {calc_name} invalid priority: {config.priority}"
                )

        # Validate feature set configurations
        for set_name, config in self.feature_set_configs.items():
            if not config.calculators:
                errors["feature_sets"].append(f"Feature set {set_name} has no calculators")

            # Check if all referenced calculators exist
            for calc_name in config.calculators:
                if calc_name not in self.calculator_configs:
                    errors["feature_sets"].append(
                        f"Feature set {set_name} references unknown calculator: {calc_name}"
                    )

        # Validate dependencies
        for calc_name, config in self.calculator_configs.items():
            for dep in config.dependencies:
                if dep not in self.calculator_configs:
                    errors["dependencies"].append(
                        f"Calculator {calc_name} depends on unknown calculator: {dep}"
                    )

        # Validate processing configuration
        if self.processing_config.max_workers <= 0:
            errors["processing"].append("Max workers must be positive")

        if self.processing_config.batch_size <= 0:
            errors["processing"].append("Batch size must be positive")

        return errors

    def get_config_summary(self) -> dict[str, Any]:
        """Get summary of current configuration."""
        enabled_calculators = self.get_enabled_calculators()

        return {
            "total_calculators": len(self.calculator_configs),
            "enabled_calculators": len(enabled_calculators),
            "calculator_names": enabled_calculators,
            "feature_sets": list(self.feature_set_configs.keys()),
            "processing_config": {
                "max_workers": self.processing_config.max_workers,
                "batch_size": self.processing_config.batch_size,
                "validation_enabled": self.processing_config.validate_results,
            },
        }

    def update_calculator_config(self, calculator_name: str, **kwargs):
        """Update configuration for a calculator."""
        if calculator_name in self.calculator_configs:
            config = self.calculator_configs[calculator_name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown config parameter for {calculator_name}: {key}")
        else:
            logger.error(f"Unknown calculator: {calculator_name}")

    def enable_calculator(self, calculator_name: str):
        """Enable a calculator."""
        if calculator_name in self.calculator_configs:
            self.calculator_configs[calculator_name].enabled = True
            logger.info(f"Enabled calculator: {calculator_name}")
        else:
            logger.error(f"Unknown calculator: {calculator_name}")

    def disable_calculator(self, calculator_name: str):
        """Disable a calculator."""
        if calculator_name in self.calculator_configs:
            self.calculator_configs[calculator_name].enabled = False
            logger.info(f"Disabled calculator: {calculator_name}")
        else:
            logger.error(f"Unknown calculator: {calculator_name}")

    def create_calculator_config(self, calculator_name: str, **kwargs) -> CalculatorConfig:
        """Create configuration for a new calculator."""
        config = CalculatorConfig(name=calculator_name, **kwargs)
        self.calculator_configs[calculator_name] = config
        logger.info(f"Created config for calculator: {calculator_name}")
        return config


# Global feature configuration instance
_feature_config = None


def get_feature_config() -> FeatureConfig:
    """Get global feature configuration instance."""
    global _feature_config
    if _feature_config is None:
        _feature_config = FeatureConfig()
    return _feature_config


def reload_feature_config():
    """Reload global feature configuration."""
    global _feature_config
    _feature_config = None
    _feature_config = FeatureConfig()
    logger.info("Feature configuration reloaded")

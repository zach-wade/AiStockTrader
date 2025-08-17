# tests/unit/test_feature_config.py

# Standard library imports
from pathlib import Path
from unittest.mock import patch

# Third-party imports
import pytest
import yaml

# Local imports
from main.feature_pipeline.feature_config import (
    CalculatorConfig,
    FeatureConfig,
    FeatureSetConfig,
    ProcessingConfig,
    get_feature_config,
    reload_feature_config,
)


# Test Fixtures
@pytest.fixture
def mock_main_config():
    """Mock main configuration."""
    return {
        "features": {
            "calculators": {
                "test_calculator": {"enabled": True, "priority": 7, "config": {"param1": "value1"}}
            },
            "feature_sets": {
                "test_set": {"description": "Test feature set", "calculators": ["test_calculator"]}
            },
            "processing": {"max_workers": 8, "batch_size": 200},
        }
    }


@pytest.fixture
def sample_config_file():
    """Sample configuration file content."""
    return {
        "features": {
            "calculators": {
                "file_calculator": {
                    "enabled": True,
                    "priority": 5,
                    "config": {"file_param": "file_value"},
                }
            },
            "feature_sets": {
                "file_set": {"description": "File feature set", "calculators": ["file_calculator"]}
            },
        }
    }


@pytest.fixture
def temp_config_file(tmp_path, sample_config_file):
    """Create temporary config file."""
    config_file = tmp_path / "feature_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config_file, f)
    return config_file


# Test CalculatorConfig
class TestCalculatorConfig:
    """Test CalculatorConfig dataclass."""

    def test_default_calculator_config(self):
        """Test default CalculatorConfig values."""
        config = CalculatorConfig(name="test_calc")

        assert config.name == "test_calc"
        assert config.enabled is True
        assert config.priority == 5
        assert config.dependencies == []
        assert config.config == {}
        assert config.max_memory_mb is None
        assert config.max_execution_time_seconds is None
        assert config.cache_enabled is True
        assert config.cache_ttl_hours == 24
        assert config.validate_inputs is True
        assert config.validate_outputs is True

    def test_custom_calculator_config(self):
        """Test CalculatorConfig with custom values."""
        config = CalculatorConfig(
            name="custom_calc",
            enabled=False,
            priority=8,
            dependencies=["dep1", "dep2"],
            config={"param": "value"},
            max_memory_mb=1024,
            cache_enabled=False,
        )

        assert config.name == "custom_calc"
        assert config.enabled is False
        assert config.priority == 8
        assert config.dependencies == ["dep1", "dep2"]
        assert config.config == {"param": "value"}
        assert config.max_memory_mb == 1024
        assert config.cache_enabled is False


# Test FeatureSetConfig
class TestFeatureSetConfig:
    """Test FeatureSetConfig dataclass."""

    def test_default_feature_set_config(self):
        """Test default FeatureSetConfig values."""
        config = FeatureSetConfig(
            name="test_set", description="Test set", calculators=["calc1", "calc2"]
        )

        assert config.name == "test_set"
        assert config.description == "Test set"
        assert config.calculators == ["calc1", "calc2"]
        assert config.enabled is True
        assert config.include_features is None
        assert config.exclude_features is None
        assert config.preprocessing is True
        assert config.scaling is True
        assert config.update_frequency == "daily"
        assert config.required_data_types == ["market_data"]

    def test_custom_feature_set_config(self):
        """Test FeatureSetConfig with custom values."""
        config = FeatureSetConfig(
            name="custom_set",
            description="Custom set",
            calculators=["calc1"],
            enabled=False,
            include_features=["feature1", "feature2"],
            update_frequency="hourly",
            required_data_types=["market_data", "news"],
        )

        assert config.name == "custom_set"
        assert config.enabled is False
        assert config.include_features == ["feature1", "feature2"]
        assert config.update_frequency == "hourly"
        assert config.required_data_types == ["market_data", "news"]


# Test ProcessingConfig
class TestProcessingConfig:
    """Test ProcessingConfig dataclass."""

    def test_default_processing_config(self):
        """Test default ProcessingConfig values."""
        config = ProcessingConfig()

        assert config.max_workers == 4
        assert config.batch_size == 100
        assert config.max_memory_usage_mb == 2048
        assert config.cleanup_frequency == 10
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.continue_on_error is True
        assert config.validate_results is True
        assert config.outlier_detection is True
        assert config.track_performance is True
        assert config.log_execution_times is True

    def test_custom_processing_config(self):
        """Test ProcessingConfig with custom values."""
        config = ProcessingConfig(
            max_workers=8, batch_size=200, max_retries=5, continue_on_error=False
        )

        assert config.max_workers == 8
        assert config.batch_size == 200
        assert config.max_retries == 5
        assert config.continue_on_error is False


# Test FeatureConfig Initialization
class TestFeatureConfigInit:
    """Test FeatureConfig initialization."""

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_init_without_config_file(self, mock_get_config, mock_main_config):
        """Test initialization without config file."""
        mock_get_config.return_value = mock_main_config

        config = FeatureConfig()

        assert config.config_path is None
        assert config.main_config == mock_main_config
        assert config.feature_config == mock_main_config["features"]
        assert "test_calculator" in config.calculator_configs
        assert "test_set" in config.feature_set_configs
        assert isinstance(config.processing_config, ProcessingConfig)

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_init_with_config_file(self, mock_get_config, mock_main_config, temp_config_file):
        """Test initialization with config file."""
        mock_get_config.return_value = mock_main_config

        config = FeatureConfig(config_path=temp_config_file)

        assert config.config_path == temp_config_file
        # Should have both main config and file config calculators
        assert "test_calculator" in config.calculator_configs
        assert "file_calculator" in config.calculator_configs

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_init_with_nonexistent_config_file(self, mock_get_config, mock_main_config):
        """Test initialization with non-existent config file."""
        mock_get_config.return_value = mock_main_config
        nonexistent_path = Path("nonexistent.yaml")

        config = FeatureConfig(config_path=nonexistent_path)

        # Should still work with main config only
        assert "test_calculator" in config.calculator_configs

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_init_with_invalid_config_file(
        self, mock_get_config, mock_main_config, tmp_path, caplog
    ):
        """Test initialization with invalid config file."""
        mock_get_config.return_value = mock_main_config

        # Create invalid YAML file
        invalid_file = tmp_path / "invalid.yaml"
        with open(invalid_file, "w") as f:
            f.write("invalid: yaml: content: [")

        config = FeatureConfig(config_path=invalid_file)

        # Should handle error gracefully
        assert "Failed to load config" in caplog.text
        assert "test_calculator" in config.calculator_configs


# Test Calculator Configuration
class TestCalculatorConfiguration:
    """Test calculator configuration management."""

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_load_default_calculator_configs(self, mock_get_config):
        """Test loading default calculator configurations."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Should have default calculators
        assert "technical_indicators" in config.calculator_configs
        assert "advanced_statistical" in config.calculator_configs
        assert "cross_asset" in config.calculator_configs
        assert "sentiment_features" in config.calculator_configs
        assert "market_regime" in config.calculator_configs

        # Check default settings
        tech_config = config.calculator_configs["technical_indicators"]
        assert tech_config.enabled is True
        assert tech_config.priority == 8
        assert "windows" in tech_config.config

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_get_calculator_config(self, mock_get_config):
        """Test getting calculator configuration."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Get existing calculator
        calc_config = config.get_calculator_config("technical_indicators")
        assert calc_config is not None
        assert calc_config.name == "technical_indicators"

        # Get non-existent calculator
        missing_config = config.get_calculator_config("nonexistent")
        assert missing_config is None

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_get_enabled_calculators(self, mock_get_config):
        """Test getting enabled calculators."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        enabled = config.get_enabled_calculators()
        assert "technical_indicators" in enabled
        assert "advanced_statistical" in enabled

        # Disable a calculator and test
        config.disable_calculator("technical_indicators")
        enabled_after = config.get_enabled_calculators()
        assert "technical_indicators" not in enabled_after

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_get_calculators_by_priority(self, mock_get_config):
        """Test getting calculators sorted by priority."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        by_priority = config.get_calculators_by_priority()

        # Should be sorted by priority (highest first)
        priorities = [config.calculator_configs[name].priority for name in by_priority]
        assert priorities == sorted(priorities, reverse=True)

        # Check specific order - technical_indicators has highest priority (8)
        assert by_priority[0] == "technical_indicators"

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_get_calculator_dependencies(self, mock_get_config):
        """Test getting calculator dependencies."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # sentiment_features depends on news_features by default
        deps = config.get_calculator_dependencies("sentiment_features")
        assert "news_features" in deps

        # technical_indicators has no dependencies
        no_deps = config.get_calculator_dependencies("technical_indicators")
        assert no_deps == []

        # Non-existent calculator
        missing_deps = config.get_calculator_dependencies("nonexistent")
        assert missing_deps == []


# Test Dependency Resolution
class TestDependencyResolution:
    """Test calculator dependency resolution."""

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_resolve_calculator_order_no_dependencies(self, mock_get_config):
        """Test resolving order without dependencies."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Request calculators without dependencies
        requested = ["technical_indicators", "advanced_statistical"]
        order = config.resolve_calculator_order(requested)

        assert len(order) == 2
        assert "technical_indicators" in order
        assert "advanced_statistical" in order

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_resolve_calculator_order_with_dependencies(self, mock_get_config):
        """Test resolving order with dependencies."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Create custom calculator with dependency
        config.create_calculator_config("news_features", enabled=True, priority=4)

        # Request calculators where sentiment_features depends on news_features
        requested = ["sentiment_features", "news_features"]
        order = config.resolve_calculator_order(requested)

        # news_features should come before sentiment_features
        news_idx = order.index("news_features")
        sentiment_idx = order.index("sentiment_features")
        assert news_idx < sentiment_idx

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_resolve_calculator_order_circular_dependency(self, mock_get_config, caplog):
        """Test handling circular dependencies."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Create circular dependency: A -> B -> A
        config.create_calculator_config("calc_a", dependencies=["calc_b"])
        config.create_calculator_config("calc_b", dependencies=["calc_a"])

        order = config.resolve_calculator_order(["calc_a", "calc_b"])

        # Should handle gracefully and log warning
        assert "Circular dependency detected" in caplog.text
        assert len(order) <= 2

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_resolve_calculator_order_all_enabled(self, mock_get_config):
        """Test resolving order for all enabled calculators."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        order = config.resolve_calculator_order()

        # Should include all enabled calculators
        enabled = config.get_enabled_calculators()
        assert len(order) == len(enabled)
        for calc in enabled:
            assert calc in order


# Test Feature Set Configuration
class TestFeatureSetConfiguration:
    """Test feature set configuration management."""

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_load_default_feature_sets(self, mock_get_config):
        """Test loading default feature sets."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Should have default feature sets
        assert "basic_technical" in config.feature_set_configs
        assert "advanced_analytics" in config.feature_set_configs
        assert "alternative_data" in config.feature_set_configs
        assert "full_feature_set" in config.feature_set_configs

        # Check default settings
        basic_set = config.feature_set_configs["basic_technical"]
        assert basic_set.enabled is True
        assert "technical_indicators" in basic_set.calculators
        assert "market_data" in basic_set.required_data_types

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_get_feature_set_config(self, mock_get_config):
        """Test getting feature set configuration."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Get existing feature set
        set_config = config.get_feature_set_config("basic_technical")
        assert set_config is not None
        assert set_config.name == "basic_technical"

        # Get non-existent feature set
        missing_config = config.get_feature_set_config("nonexistent")
        assert missing_config is None

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_get_feature_set_calculators(self, mock_get_config):
        """Test getting calculators for a feature set."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Get calculators for existing feature set
        calculators = config.get_feature_set_calculators("basic_technical")
        assert "technical_indicators" in calculators

        # Get calculators for non-existent feature set
        missing_calculators = config.get_feature_set_calculators("nonexistent")
        assert missing_calculators == []

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_get_available_feature_sets(self, mock_get_config):
        """Test getting available feature sets."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        available = config.get_available_feature_sets()
        assert "basic_technical" in available
        assert "advanced_analytics" in available
        assert "alternative_data" in available
        assert "full_feature_set" in available


# Test Processing Configuration
class TestProcessingConfiguration:
    """Test processing configuration."""

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_get_processing_config(self, mock_get_config):
        """Test getting processing configuration."""
        mock_get_config.return_value = {
            "features": {
                "processing": {"max_workers": 8, "batch_size": 200, "validate_results": False}
            }
        }

        config = FeatureConfig()

        proc_config = config.get_processing_config()
        assert proc_config.max_workers == 8
        assert proc_config.batch_size == 200
        assert proc_config.validate_results is False


# Test Configuration Validation
class TestConfigurationValidation:
    """Test configuration validation."""

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_validate_configuration_valid(self, mock_get_config):
        """Test validation with valid configuration."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        errors = config.validate_configuration()

        # Should have no errors for default config
        assert len(errors["calculators"]) == 0
        assert len(errors["feature_sets"]) == 0
        assert len(errors["dependencies"]) == 0
        assert len(errors["processing"]) == 0

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_validate_configuration_invalid_calculator(self, mock_get_config):
        """Test validation with invalid calculator configuration."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Create invalid calculator
        config.create_calculator_config("invalid_calc", name="", priority=15)

        errors = config.validate_configuration()

        assert len(errors["calculators"]) > 0
        assert any("missing name" in error for error in errors["calculators"])
        assert any("invalid priority" in error for error in errors["calculators"])

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_validate_configuration_missing_dependency(self, mock_get_config):
        """Test validation with missing dependencies."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Create calculator with missing dependency
        config.create_calculator_config("test_calc", dependencies=["missing_calc"])

        errors = config.validate_configuration()

        assert len(errors["dependencies"]) > 0
        assert any("unknown calculator: missing_calc" in error for error in errors["dependencies"])

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_validate_configuration_invalid_processing(self, mock_get_config):
        """Test validation with invalid processing configuration."""
        mock_get_config.return_value = {
            "features": {"processing": {"max_workers": -1, "batch_size": 0}}
        }

        config = FeatureConfig()

        errors = config.validate_configuration()

        assert len(errors["processing"]) > 0
        assert any("Max workers must be positive" in error for error in errors["processing"])
        assert any("Batch size must be positive" in error for error in errors["processing"])


# Test Configuration Management
class TestConfigurationManagement:
    """Test configuration management operations."""

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_update_calculator_config(self, mock_get_config, caplog):
        """Test updating calculator configuration."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Update existing calculator
        config.update_calculator_config("technical_indicators", priority=9, enabled=False)

        calc_config = config.get_calculator_config("technical_indicators")
        assert calc_config.priority == 9
        assert calc_config.enabled is False

        # Try to update unknown parameter
        config.update_calculator_config("technical_indicators", unknown_param="value")
        assert "Unknown config parameter" in caplog.text

        # Try to update unknown calculator
        config.update_calculator_config("unknown_calc", priority=5)
        assert "Unknown calculator" in caplog.text

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_enable_disable_calculator(self, mock_get_config, caplog):
        """Test enabling and disabling calculators."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Disable calculator
        config.disable_calculator("technical_indicators")
        calc_config = config.get_calculator_config("technical_indicators")
        assert calc_config.enabled is False
        assert "Disabled calculator: technical_indicators" in caplog.text

        # Enable calculator
        config.enable_calculator("technical_indicators")
        calc_config = config.get_calculator_config("technical_indicators")
        assert calc_config.enabled is True
        assert "Enabled calculator: technical_indicators" in caplog.text

        # Try to enable unknown calculator
        config.enable_calculator("unknown_calc")
        assert "Unknown calculator: unknown_calc" in caplog.text

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_create_calculator_config(self, mock_get_config, caplog):
        """Test creating new calculator configuration."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        # Create new calculator
        new_config = config.create_calculator_config(
            "new_calc", enabled=True, priority=7, dependencies=["technical_indicators"]
        )

        assert new_config.name == "new_calc"
        assert new_config.enabled is True
        assert new_config.priority == 7
        assert "technical_indicators" in new_config.dependencies
        assert "new_calc" in config.calculator_configs
        assert "Created config for calculator: new_calc" in caplog.text

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_get_config_summary(self, mock_get_config):
        """Test getting configuration summary."""
        mock_get_config.return_value = {"features": {}}

        config = FeatureConfig()

        summary = config.get_config_summary()

        assert "total_calculators" in summary
        assert "enabled_calculators" in summary
        assert "calculator_names" in summary
        assert "feature_sets" in summary
        assert "processing_config" in summary

        assert summary["total_calculators"] > 0
        assert len(summary["calculator_names"]) > 0
        assert len(summary["feature_sets"]) > 0


# Test Global Configuration Management
class TestGlobalConfiguration:
    """Test global configuration management."""

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_get_feature_config_singleton(self, mock_get_config):
        """Test global feature config singleton."""
        mock_get_config.return_value = {"features": {}}

        # First call creates instance
        config1 = get_feature_config()
        assert config1 is not None

        # Second call returns same instance
        config2 = get_feature_config()
        assert config1 is config2

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_reload_feature_config(self, mock_get_config, caplog):
        """Test reloading global feature configuration."""
        mock_get_config.return_value = {"features": {}}

        # Get initial config
        config1 = get_feature_config()

        # Reload config
        reload_feature_config()
        assert "Feature configuration reloaded" in caplog.text

        # Get config again - should be new instance
        config2 = get_feature_config()
        assert config1 is not config2


# Test Error Handling
class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_empty_configuration(self, mock_get_config):
        """Test handling empty configuration."""
        mock_get_config.return_value = {}

        config = FeatureConfig()

        # Should still work with defaults
        assert len(config.calculator_configs) > 0
        assert len(config.feature_set_configs) > 0
        assert isinstance(config.processing_config, ProcessingConfig)

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_malformed_configuration(self, mock_get_config):
        """Test handling malformed configuration."""
        mock_get_config.return_value = {
            "features": {
                "calculators": "not_a_dict",  # Should be dict
                "feature_sets": None,  # Should be dict
                "processing": [],  # Should be dict
            }
        }

        config = FeatureConfig()

        # Should handle gracefully with defaults
        assert len(config.calculator_configs) > 0
        assert len(config.feature_set_configs) > 0
        assert isinstance(config.processing_config, ProcessingConfig)

    @patch("main.feature_pipeline.feature_config.get_config")
    def test_config_with_missing_calculator_params(self, mock_get_config):
        """Test configuration with missing calculator parameters."""
        mock_get_config.return_value = {
            "features": {"calculators": {"minimal_calc": {}}}  # Missing most parameters
        }

        config = FeatureConfig()

        # Should create config with defaults
        calc_config = config.get_calculator_config("minimal_calc")
        assert calc_config is not None
        assert calc_config.name == "minimal_calc"
        assert calc_config.enabled is True  # Default
        assert calc_config.priority == 5  # Default


if __name__ == "__main__":
    pytest.main([__file__])

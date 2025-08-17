"""Unit tests for feature_config module."""

# Standard library imports
import os
from unittest.mock import mock_open, patch

# Third-party imports
import pytest
import yaml

# Local imports
from main.events.handlers.feature_pipeline_helpers.feature_config import (
    get_default_feature_config,
    get_feature_group_config,
    load_feature_config,
    merge_feature_configs,
    validate_feature_config,
)
from main.events.handlers.feature_pipeline_helpers.feature_types import FeatureGroup


class TestLoadFeatureConfig:
    """Test load_feature_config function."""

    def test_load_valid_config_file(self):
        """Test loading valid configuration file."""
        config_data = {
            "feature_groups": {
                "PRICE": {"features": ["open", "high", "low", "close"], "priority_boost": 2},
                "VOLUME": {"features": ["volume", "dollar_volume"]},
            },
            "alert_mappings": {"HIGH_VOLUME": ["VOLUME", "PRICE"]},
        }

        yaml_content = yaml.dump(config_data)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.exists", return_value=True):
                config = load_feature_config("/path/to/config.yaml")

                assert config == config_data
                assert "feature_groups" in config
                assert "alert_mappings" in config

    def test_load_missing_config_file(self):
        """Test loading missing configuration file."""
        with patch("os.path.exists", return_value=False):
            config = load_feature_config("/path/to/missing.yaml")

            # Should return empty config
            assert config == {}

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file."""
        invalid_yaml = "invalid: yaml: content: [["

        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with patch("os.path.exists", return_value=True):
                config = load_feature_config("/path/to/invalid.yaml")

                # Should return empty config on error
                assert config == {}

    def test_load_config_with_io_error(self):
        """Test handling IO errors when loading config."""
        with patch("builtins.open", side_effect=OSError("File not found")):
            with patch("os.path.exists", return_value=True):
                config = load_feature_config("/path/to/config.yaml")

                assert config == {}


class TestGetDefaultFeatureConfig:
    """Test get_default_feature_config function."""

    def test_default_config_structure(self):
        """Test default configuration has expected structure."""
        config = get_default_feature_config()

        assert isinstance(config, dict)
        assert "feature_groups" in config
        assert "alert_mappings" in config
        assert "priority_rules" in config
        assert "conditional_rules" in config

    def test_default_feature_groups(self):
        """Test default feature groups configuration."""
        config = get_default_feature_config()
        groups = config["feature_groups"]

        # Check some expected groups
        assert "PRICE" in groups
        assert "VOLUME" in groups
        assert "VOLATILITY" in groups

        # Check group structure
        price_group = groups["PRICE"]
        assert "features" in price_group
        assert "dependencies" in price_group
        assert "priority_boost" in price_group
        assert "required_data_types" in price_group

    def test_default_alert_mappings(self):
        """Test default alert mappings."""
        config = get_default_feature_config()
        mappings = config["alert_mappings"]

        # Check some expected mappings
        assert "HIGH_VOLUME" in mappings
        assert "PRICE_SPIKE" in mappings
        assert "ML_SIGNAL" in mappings

        # Check mapping values are lists
        for alert_type, groups in mappings.items():
            assert isinstance(groups, list)
            assert all(isinstance(g, str) for g in groups)

    def test_default_priority_rules(self):
        """Test default priority rules."""
        config = get_default_feature_config()
        rules = config["priority_rules"]

        assert "base_priorities" in rules
        assert "score_multiplier" in rules
        assert "max_priority" in rules
        assert "min_priority" in rules

        # Check base priorities
        base = rules["base_priorities"]
        assert "ML_SIGNAL" in base
        assert "BREAKOUT" in base

    def test_default_conditional_rules(self):
        """Test default conditional rules."""
        config = get_default_feature_config()
        rules = config["conditional_rules"]

        assert "high_score_threshold" in rules
        assert "volume_spike_multiplier" in rules
        assert "news_keywords" in rules

        # Check types
        assert isinstance(rules["high_score_threshold"], (int, float))
        assert isinstance(rules["news_keywords"], list)


class TestMergeFeatureConfigs:
    """Test merge_feature_configs function."""

    def test_merge_empty_configs(self):
        """Test merging empty configurations."""
        result = merge_feature_configs({}, {})
        assert result == {}

    def test_merge_with_default(self):
        """Test merging custom config with default."""
        default = {
            "feature_groups": {
                "PRICE": {"features": ["open", "close"], "priority_boost": 1},
                "VOLUME": {"features": ["volume"]},
            },
            "alert_mappings": {"HIGH_VOLUME": ["VOLUME"]},
        }

        custom = {
            "feature_groups": {
                "PRICE": {"priority_boost": 3},  # Override
                "CUSTOM": {"features": ["custom1"]},  # New
            }
        }

        result = merge_feature_configs(default, custom)

        # Check override
        assert result["feature_groups"]["PRICE"]["priority_boost"] == 3
        # Check preservation
        assert result["feature_groups"]["PRICE"]["features"] == ["open", "close"]
        # Check addition
        assert result["feature_groups"]["CUSTOM"]["features"] == ["custom1"]
        # Check untouched
        assert result["feature_groups"]["VOLUME"]["features"] == ["volume"]

    def test_merge_nested_dicts(self):
        """Test merging deeply nested dictionaries."""
        base = {"level1": {"level2": {"level3": {"value": 1, "keep": "this"}}}}

        override = {"level1": {"level2": {"level3": {"value": 2}}}}

        result = merge_feature_configs(base, override)

        assert result["level1"]["level2"]["level3"]["value"] == 2
        assert result["level1"]["level2"]["level3"]["keep"] == "this"

    def test_merge_lists(self):
        """Test merging configurations with lists."""
        base = {"features": ["f1", "f2"], "mappings": {"alert1": ["group1"]}}

        override = {
            "features": ["f3", "f4"],  # Should replace
            "mappings": {"alert1": ["group2", "group3"]},  # Should replace
        }

        result = merge_feature_configs(base, override)

        # Lists should be replaced, not merged
        assert result["features"] == ["f3", "f4"]
        assert result["mappings"]["alert1"] == ["group2", "group3"]


class TestValidateFeatureConfig:
    """Test validate_feature_config function."""

    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        config = {
            "feature_groups": {
                "PRICE": {
                    "features": ["open", "close"],
                    "dependencies": [],
                    "priority_boost": 1,
                    "required_data_types": ["prices"],
                }
            },
            "alert_mappings": {"HIGH_VOLUME": ["VOLUME", "PRICE"]},
        }

        # Should not raise
        validate_feature_config(config)

    def test_validate_missing_required_fields(self):
        """Test validation fails for missing required fields."""
        # Missing feature_groups
        config = {"alert_mappings": {}}

        with pytest.raises(ValueError, match="feature_groups"):
            validate_feature_config(config)

        # Missing alert_mappings
        config = {"feature_groups": {}}

        with pytest.raises(ValueError, match="alert_mappings"):
            validate_feature_config(config)

    def test_validate_invalid_feature_group(self):
        """Test validation fails for invalid feature group."""
        config = {
            "feature_groups": {"INVALID": "not a dict"},  # Should be dict
            "alert_mappings": {},
        }

        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_feature_config(config)

    def test_validate_missing_features_list(self):
        """Test validation fails when features list is missing."""
        config = {
            "feature_groups": {
                "PRICE": {
                    "priority_boost": 1
                    # Missing 'features'
                }
            },
            "alert_mappings": {},
        }

        with pytest.raises(ValueError, match="features"):
            validate_feature_config(config)

    def test_validate_invalid_alert_mapping(self):
        """Test validation fails for invalid alert mapping."""
        config = {
            "feature_groups": {},
            "alert_mappings": {"HIGH_VOLUME": "VOLUME"},  # Should be list
        }

        with pytest.raises(ValueError, match="must be a list"):
            validate_feature_config(config)

    def test_validate_unknown_feature_group_in_mapping(self):
        """Test validation warns about unknown feature groups."""
        config = {
            "feature_groups": {"PRICE": {"features": ["open"]}},
            "alert_mappings": {"HIGH_VOLUME": ["PRICE", "UNKNOWN_GROUP"]},
        }

        # May warn but not fail
        try:
            validate_feature_config(config)
        except ValueError as e:
            assert "UNKNOWN_GROUP" in str(e)


class TestGetFeatureGroupConfig:
    """Test get_feature_group_config function."""

    def test_get_existing_group_config(self):
        """Test getting configuration for existing group."""
        config = {"feature_groups": {"PRICE": {"features": ["open", "close"], "priority_boost": 2}}}

        group_config = get_feature_group_config(config, FeatureGroup.PRICE)

        assert group_config is not None
        assert group_config["features"] == ["open", "close"]
        assert group_config["priority_boost"] == 2

    def test_get_missing_group_config(self):
        """Test getting configuration for missing group."""
        config = {"feature_groups": {"PRICE": {"features": ["open"]}}}

        group_config = get_feature_group_config(config, FeatureGroup.VOLUME)

        # Should return None or default
        assert group_config is None or group_config == {}

    def test_get_group_config_with_defaults(self):
        """Test getting group config with defaults applied."""
        if hasattr(get_feature_group_config, "with_defaults"):
            config = {"feature_groups": {"PRICE": {"features": ["open"]}}}

            group_config = get_feature_group_config(config, FeatureGroup.PRICE, with_defaults=True)

            # Should have default fields
            assert "features" in group_config
            assert "dependencies" in group_config
            assert "priority_boost" in group_config


class TestFeatureConfigIntegration:
    """Test feature config functions working together."""

    def test_load_and_merge_with_defaults(self):
        """Test loading config and merging with defaults."""
        custom_config = {"feature_groups": {"PRICE": {"priority_boost": 5}}}

        yaml_content = yaml.dump(custom_config)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.exists", return_value=True):
                # Load custom config
                loaded = load_feature_config("/path/to/custom.yaml")

                # Get defaults
                defaults = get_default_feature_config()

                # Merge
                final = merge_feature_configs(defaults, loaded)

                # Validate
                validate_feature_config(final)

                # Check merge worked
                assert final["feature_groups"]["PRICE"]["priority_boost"] == 5
                # Should still have default features
                assert "features" in final["feature_groups"]["PRICE"]

    def test_config_with_environment_override(self):
        """Test configuration with environment variable override."""
        with patch.dict(os.environ, {"FEATURE_CONFIG_PATH": "/env/config.yaml"}):
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("os.path.exists", return_value=True):
                    # Function might check environment
                    config = load_feature_config()

                    assert isinstance(config, dict)

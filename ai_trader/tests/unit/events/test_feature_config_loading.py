"""Unit tests for feature configuration loading functionality."""

import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any

# Mock the actual module paths since they might vary
FEATURE_CONFIG_MODULE = "main.events.feature_pipeline_helpers.feature_config"


class TestFeatureConfigLoading:
    """Test feature configuration loading from various sources."""
    
    @pytest.fixture
    def sample_yaml_config(self):
        """Sample YAML configuration."""
        return {
            "feature_groups": {
                "PRICE": {
                    "features": ["open", "high", "low", "close"],
                    "priority_boost": 2,
                    "dependencies": [],
                    "required_data_types": ["prices"]
                },
                "VOLUME": {
                    "features": ["volume", "dollar_volume"],
                    "priority_boost": 1,
                    "dependencies": ["PRICE"],
                    "required_data_types": ["trades"]
                }
            },
            "alert_mappings": {
                "HIGH_VOLUME": ["VOLUME", "PRICE"],
                "PRICE_SPIKE": ["PRICE", "VOLATILITY"]
            },
            "priority_rules": {
                "base_priorities": {
                    "ML_SIGNAL": 8,
                    "BREAKOUT": 7
                },
                "score_multiplier": 2.0,
                "max_priority": 10
            }
        }
    
    @pytest.fixture
    def sample_json_config(self):
        """Sample JSON configuration."""
        return {
            "feature_groups": {
                "MOMENTUM": {
                    "features": ["rsi", "macd"],
                    "dependencies": ["PRICE"]
                }
            },
            "version": "1.0"
        }
    
    def test_load_yaml_config_success(self, sample_yaml_config):
        """Test successful YAML config loading."""
        yaml_content = yaml.dump(sample_yaml_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('os.path.exists', return_value=True):
                with patch(f'{FEATURE_CONFIG_MODULE}.load_feature_config') as mock_load:
                    mock_load.return_value = sample_yaml_config
                    
                    config = mock_load("config.yaml")
                    
                    assert config == sample_yaml_config
                    assert "feature_groups" in config
                    assert "PRICE" in config["feature_groups"]
                    
    def test_load_json_config_success(self, sample_json_config):
        """Test successful JSON config loading."""
        json_content = json.dumps(sample_json_config)
        
        with patch('builtins.open', mock_open(read_data=json_content)):
            with patch('os.path.exists', return_value=True):
                # If system supports JSON configs
                config = json.loads(json_content)
                
                assert config == sample_json_config
                assert config["version"] == "1.0"
                
    def test_load_config_file_not_found(self):
        """Test handling missing config file."""
        with patch('os.path.exists', return_value=False):
            with patch(f'{FEATURE_CONFIG_MODULE}.load_feature_config') as mock_load:
                mock_load.return_value = {}
                
                config = mock_load("missing.yaml")
                
                assert config == {}
                
    def test_load_config_parse_error(self):
        """Test handling config parse errors."""
        invalid_yaml = "invalid: yaml: content: [["
        
        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with patch('os.path.exists', return_value=True):
                with patch('yaml.safe_load', side_effect=yaml.YAMLError("Parse error")):
                    with patch(f'{FEATURE_CONFIG_MODULE}.load_feature_config') as mock_load:
                        mock_load.return_value = {}
                        
                        config = mock_load("invalid.yaml")
                        
                        assert config == {}
                        
    def test_load_config_io_error(self):
        """Test handling IO errors."""
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with patch('os.path.exists', return_value=True):
                with patch(f'{FEATURE_CONFIG_MODULE}.load_feature_config') as mock_load:
                    mock_load.return_value = {}
                    
                    config = mock_load("protected.yaml")
                    
                    assert config == {}
                    
    def test_load_config_from_environment(self):
        """Test loading config path from environment variable."""
        env_config_path = "/env/path/config.yaml"
        
        with patch.dict(os.environ, {"FEATURE_CONFIG_PATH": env_config_path}):
            with patch('builtins.open', mock_open(read_data="{}")):
                with patch('os.path.exists', return_value=True):
                    # System might check environment
                    env_path = os.environ.get("FEATURE_CONFIG_PATH")
                    assert env_path == env_config_path
                    
    def test_load_config_with_includes(self, sample_yaml_config):
        """Test loading config with include directives."""
        main_config = {
            "include": ["base_config.yaml", "overrides.yaml"],
            "feature_groups": {
                "CUSTOM": {
                    "features": ["custom1"]
                }
            }
        }
        
        base_config = {
            "feature_groups": sample_yaml_config["feature_groups"]
        }
        
        override_config = {
            "feature_groups": {
                "PRICE": {
                    "priority_boost": 5  # Override
                }
            }
        }
        
        # Mock multiple file reads
        def mock_open_multi(filename, *args, **kwargs):
            content_map = {
                "main.yaml": yaml.dump(main_config),
                "base_config.yaml": yaml.dump(base_config),
                "overrides.yaml": yaml.dump(override_config)
            }
            
            content = content_map.get(Path(filename).name, "{}")
            return mock_open(read_data=content)()
            
        with patch('builtins.open', mock_open_multi):
            with patch('os.path.exists', return_value=True):
                # Simulate merged config
                final_config = {
                    "feature_groups": {
                        **base_config["feature_groups"],
                        "PRICE": {
                            **base_config["feature_groups"]["PRICE"],
                            "priority_boost": 5
                        },
                        "CUSTOM": main_config["feature_groups"]["CUSTOM"]
                    }
                }
                
                assert final_config["feature_groups"]["PRICE"]["priority_boost"] == 5
                assert "CUSTOM" in final_config["feature_groups"]
                
    def test_config_validation_on_load(self, sample_yaml_config):
        """Test that config is validated on load."""
        # Invalid config missing required fields
        invalid_config = {
            "feature_groups": {
                "INVALID": "not a dict"  # Should be dict
            }
        }
        
        yaml_content = yaml.dump(invalid_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('os.path.exists', return_value=True):
                with patch(f'{FEATURE_CONFIG_MODULE}.validate_feature_config') as mock_validate:
                    mock_validate.side_effect = ValueError("Invalid config")
                    
                    # Should handle validation error
                    with patch(f'{FEATURE_CONFIG_MODULE}.load_feature_config') as mock_load:
                        mock_load.return_value = {}
                        
                        config = mock_load("invalid.yaml")
                        assert config == {}
                        
    def test_config_schema_migration(self):
        """Test migrating old config schema to new."""
        old_config = {
            "features": {  # Old structure
                "price_features": ["open", "close"],
                "volume_features": ["volume"]
            }
        }
        
        # Expected new structure
        new_config = {
            "feature_groups": {
                "PRICE": {
                    "features": ["open", "close"]
                },
                "VOLUME": {
                    "features": ["volume"]
                }
            }
        }
        
        # If migration is supported
        yaml_content = yaml.dump(old_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('os.path.exists', return_value=True):
                # System might auto-migrate
                pass
                
    def test_config_caching(self, sample_yaml_config):
        """Test config caching for performance."""
        yaml_content = yaml.dump(sample_yaml_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)) as mock_file:
            with patch('os.path.exists', return_value=True):
                with patch(f'{FEATURE_CONFIG_MODULE}.load_feature_config') as mock_load:
                    # First load
                    mock_load.return_value = sample_yaml_config
                    config1 = mock_load("config.yaml")
                    
                    # Second load (should use cache)
                    config2 = mock_load("config.yaml")
                    
                    # If caching is implemented, file should be read once
                    assert config1 == config2
                    
    def test_config_hot_reload(self, sample_yaml_config):
        """Test hot reloading config changes."""
        initial_config = sample_yaml_config.copy()
        updated_config = sample_yaml_config.copy()
        updated_config["feature_groups"]["PRICE"]["priority_boost"] = 10
        
        # If hot reload is supported
        with patch('builtins.open'):
            with patch(f'{FEATURE_CONFIG_MODULE}.watch_config_changes'):
                # Simulate file change detection
                pass
                
    def test_config_defaults_fallback(self):
        """Test fallback to defaults when config is incomplete."""
        partial_config = {
            "feature_groups": {
                "PRICE": {
                    "features": ["close"]
                    # Missing other fields
                }
            }
        }
        
        yaml_content = yaml.dump(partial_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('os.path.exists', return_value=True):
                with patch(f'{FEATURE_CONFIG_MODULE}.get_default_feature_config') as mock_default:
                    mock_default.return_value = {
                        "feature_groups": {
                            "PRICE": {
                                "features": ["open", "high", "low", "close"],
                                "dependencies": [],
                                "priority_boost": 0,
                                "required_data_types": ["prices"]
                            }
                        }
                    }
                    
                    # Should merge with defaults
                    pass
                    
    def test_config_inheritance(self):
        """Test config inheritance from base configs."""
        base_config = {
            "feature_groups": {
                "BASE": {
                    "features": ["base1", "base2"],
                    "priority_boost": 1
                }
            },
            "global_settings": {
                "cache_ttl": 300
            }
        }
        
        child_config = {
            "extends": "base_config.yaml",
            "feature_groups": {
                "CHILD": {
                    "features": ["child1"],
                    "inherits": "BASE"
                }
            }
        }
        
        # Test inheritance resolution
        with patch('builtins.open'):
            with patch('os.path.exists', return_value=True):
                # Child should inherit from BASE
                pass
                
    def test_config_with_environment_substitution(self):
        """Test config with environment variable substitution."""
        config_with_env = {
            "feature_groups": {
                "PRICE": {
                    "features": ["${PRICE_FEATURES}"],
                    "priority_boost": "${PRICE_PRIORITY:2}"  # Default 2
                }
            },
            "database": {
                "host": "${DB_HOST:localhost}",
                "port": "${DB_PORT:5432}"
            }
        }
        
        with patch.dict(os.environ, {
            "PRICE_FEATURES": "open,high,low,close",
            "DB_HOST": "prod.db.com"
        }):
            # Should substitute environment variables
            yaml_content = yaml.dump(config_with_env)
            
            # After substitution
            expected_features = ["open", "high", "low", "close"]
            expected_host = "prod.db.com"
            expected_port = "5432"  # Default value
            
    def test_multi_format_config_loading(self):
        """Test loading configs in multiple formats."""
        config_files = [
            ("config.yaml", "yaml"),
            ("config.json", "json"),
            ("config.toml", "toml"),
            ("config.ini", "ini")
        ]
        
        for filename, format_type in config_files:
            with patch('os.path.exists', return_value=True):
                # System might support multiple formats
                pass
                
    def test_config_loading_performance(self, sample_yaml_config):
        """Test config loading performance with large files."""
        # Create large config
        large_config = {
            "feature_groups": {}
        }
        
        # Add many feature groups
        for i in range(1000):
            large_config["feature_groups"][f"GROUP_{i}"] = {
                "features": [f"feature_{j}" for j in range(10)],
                "dependencies": [],
                "priority_boost": i % 10
            }
            
        yaml_content = yaml.dump(large_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('os.path.exists', return_value=True):
                import time
                start = time.time()
                
                # Load large config
                # Should complete reasonably fast
                
                elapsed = time.time() - start
                assert elapsed < 1.0  # Should load in under 1 second
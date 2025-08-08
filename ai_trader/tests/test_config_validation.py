"""
Comprehensive tests for the AI Trader configuration validation system.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from pydantic import ValidationError
from main.config.validation_models import (
    AITraderConfig,
    Environment,
    DataProvider,
    PositionSizeType,
    ExecutionAlgorithm,
    validate_config_file,
    get_validation_errors
)
from main.config.validation_utils import (
    ConfigValidator,
    ConfigValidationError,
    validate_startup_config,
    validate_config_and_warn
)


class TestConfigValidationModels:
    """Test configuration validation models."""
    
    def test_valid_minimal_config(self):
        """Test that minimal valid configuration passes validation."""
        config_dict = {
            "system": {
                "environment": "paper"
            },
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            }
        }
        
        config = AITraderConfig(**config_dict)
        assert config.system.environment == Environment.PAPER
        assert config.api_keys.alpaca.key == "test_key"
        assert config.api_keys.alpaca.secret == "test_secret"
        assert config.broker.paper_trading is True
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        config_dict = {
            "system": {
                "environment": "paper"
            }
            # Missing api_keys
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AITraderConfig(**config_dict)
        
        errors = exc_info.value.errors()
        assert any("api_keys" in str(error) for error in errors)
    
    def test_invalid_environment(self):
        """Test that invalid environment values raise ValidationError."""
        config_dict = {
            "system": {
                "environment": "invalid_env"
            },
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AITraderConfig(**config_dict)
        
        errors = exc_info.value.errors()
        assert any("environment" in str(error) for error in errors)
    
    def test_position_size_validation(self):
        """Test position size validation ranges."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            },
            "risk": {
                "position_sizing": {
                    "max_position_size": 15.0  # Too high (>10%)
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AITraderConfig(**config_dict)
        
        errors = exc_info.value.errors()
        assert any("max_position_size" in str(error) for error in errors)
    
    def test_risk_percentage_validation(self):
        """Test risk percentage validation ranges."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            },
            "risk": {
                "circuit_breaker": {
                    "daily_loss_limit": 60.0  # Too high (>50%)
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AITraderConfig(**config_dict)
        
        errors = exc_info.value.errors()
        assert any("daily_loss_limit" in str(error) for error in errors)
    
    def test_strategy_weights_validation(self):
        """Test strategy weights sum to 1.0."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            },
            "strategies": {
                "ensemble": {
                    "weights": {
                        "ml_momentum": 0.6,
                        "mean_reversion": 0.5,  # Sum > 1.0
                        "sentiment": 0.3,
                        "microstructure_alpha": 0.2
                    }
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AITraderConfig(**config_dict)
        
        errors = exc_info.value.errors()
        assert any("weights" in str(error) for error in errors)
    
    def test_macd_periods_validation(self):
        """Test MACD periods validation."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            },
            "features": {
                "technical_indicators": {
                    "macd_fast": 30,  # Should be < slow
                    "macd_slow": 20
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AITraderConfig(**config_dict)
        
        errors = exc_info.value.errors()
        assert any("fast" in str(error) or "slow" in str(error) for error in errors)
    
    def test_environment_consistency_validation(self):
        """Test environment consistency validation."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            },
            "broker": {
                "paper_trading": False  # Inconsistent with paper environment
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AITraderConfig(**config_dict)
        
        errors = exc_info.value.errors()
        assert any("paper" in str(error) for error in errors)
    
    @patch.dict(os.environ, {"TEST_API_KEY": "test_key_value"})
    def test_env_var_substitution(self):
        """Test environment variable substitution."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "${TEST_API_KEY}",
                    "secret": "test_secret"
                }
            }
        }
        
        config = AITraderConfig(**config_dict)
        assert config.api_keys.alpaca.key == "test_key_value"
    
    def test_missing_env_var_validation(self):
        """Test missing environment variable validation."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "${MISSING_API_KEY}",
                    "secret": "test_secret"
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AITraderConfig(**config_dict)
        
        errors = exc_info.value.errors()
        assert any("MISSING_API_KEY" in str(error) for error in errors)
    
    def test_environment_overrides(self):
        """Test environment-specific overrides."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            },
            "risk": {
                "position_sizing": {
                    "max_position_size": 5.0
                }
            },
            "environments": {
                "paper": {
                    "risk": {
                        "position_sizing": {
                            "max_position_size": 10.0
                        }
                    }
                }
            }
        }
        
        config = AITraderConfig(**config_dict)
        env_config = config.get_environment_config()
        
        assert env_config.risk.position_sizing.max_position_size == 10.0
    
    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            },
            "invalid_field": "should_be_rejected"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AITraderConfig(**config_dict)
        
        errors = exc_info.value.errors()
        assert any("extra" in str(error) for error in errors)


class TestConfigValidator:
    """Test configuration validator utilities."""
    
    def test_valid_config_validation(self):
        """Test validation of valid configuration."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            }
        }
        
        validator = ConfigValidator()
        config = validator.validate_dict(config_dict)
        
        assert isinstance(config, AITraderConfig)
        assert config.system.environment == Environment.PAPER
    
    def test_invalid_config_validation(self):
        """Test validation of invalid configuration."""
        config_dict = {
            "system": {"environment": "invalid"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            }
        }
        
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_dict(config_dict)
        
        assert "environment" in str(exc_info.value)
    
    @patch.dict(os.environ, {"TEST_API_KEY": ""})
    def test_empty_env_var_validation(self):
        """Test empty environment variable validation."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "${TEST_API_KEY}",
                    "secret": "test_secret"
                }
            }
        }
        
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_dict(config_dict)
        
        assert "empty" in str(exc_info.value)
    
    def test_system_requirements_check(self):
        """Test system requirements checking."""
        validator = ConfigValidator()
        issues = validator.check_system_requirements()
        
        # Should pass on a properly configured system
        assert isinstance(issues, list)
    
    def test_config_file_validation(self):
        """Test configuration file validation."""
        config_content = """
system:
  environment: paper
api_keys:
  alpaca:
    key: test_key
    secret: test_secret
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            try:
                validator = ConfigValidator()
                config = validator.validate_file(f.name)
                assert isinstance(config, AITraderConfig)
            finally:
                os.unlink(f.name)
    
    def test_nonexistent_file_validation(self):
        """Test validation of nonexistent file."""
        validator = ConfigValidator()
        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_file("/nonexistent/file.yaml")
        
        assert "not found" in str(exc_info.value)
    
    def test_invalid_yaml_validation(self):
        """Test validation of invalid YAML."""
        invalid_yaml = """
system:
  environment: paper
api_keys:
  alpaca:
    key: test_key
    secret: test_secret
  invalid_yaml: [unclosed bracket
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()
            
            try:
                validator = ConfigValidator()
                with pytest.raises(ConfigValidationError) as exc_info:
                    validator.validate_file(f.name)
                assert "YAML" in str(exc_info.value)
            finally:
                os.unlink(f.name)
    
    def test_strict_mode_warnings(self):
        """Test strict mode behavior with warnings."""
        config_dict = {
            "system": {"environment": "live"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            },
            "broker": {
                "paper_trading": True  # Warning in live mode
            }
        }
        
        # Strict mode should raise error on warnings
        validator = ConfigValidator(strict_mode=True)
        with pytest.raises(ConfigValidationError):
            validator.validate_dict(config_dict)
        
        # Non-strict mode should log warnings
        validator = ConfigValidator(strict_mode=False)
        config = validator.validate_dict(config_dict)
        assert len(validator.warnings) > 0


class TestConfigUtilities:
    """Test configuration utility functions."""
    
    def test_get_validation_errors(self):
        """Test get_validation_errors function."""
        config_dict = {
            "system": {"environment": "invalid"},
            "api_keys": {
                "alpaca": {
                    "key": "test_key",
                    "secret": "test_secret"
                }
            }
        }
        
        errors = get_validation_errors(config_dict)
        assert len(errors) > 0
        assert any("environment" in error for error in errors)
    
    def test_validate_config_file_function(self):
        """Test validate_config_file function."""
        config_content = """
system:
  environment: paper
api_keys:
  alpaca:
    key: test_key
    secret: test_secret
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            try:
                config = validate_config_file(f.name)
                assert isinstance(config, AITraderConfig)
            finally:
                os.unlink(f.name)
    
    def test_validate_config_and_warn(self):
        """Test validate_config_and_warn function."""
        config_content = """
system:
  environment: paper
api_keys:
  alpaca:
    key: test_key
    secret: test_secret
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            try:
                config, warnings = validate_config_and_warn(f.name)
                assert config is not None
                assert isinstance(warnings, list)
            finally:
                os.unlink(f.name)
    
    def test_validate_config_and_warn_invalid(self):
        """Test validate_config_and_warn with invalid config."""
        config_content = """
system:
  environment: invalid
api_keys:
  alpaca:
    key: test_key
    secret: test_secret
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            try:
                config, errors = validate_config_and_warn(f.name)
                assert config is None
                assert len(errors) > 0
            finally:
                os.unlink(f.name)


class TestConfigManagerIntegration:
    """Test integration with ConfigManager."""
    
    def test_config_manager_validation_mode(self):
        """Test ConfigManager in validation mode."""
        from main.config.config_manager import ModularConfigManager as ConfigManager
        
        manager = ConfigManager(use_validation=True)
        assert manager.use_validation is True
        assert hasattr(manager, 'validator')
    
    def test_config_manager_legacy_mode(self):
        """Test ConfigManager in legacy mode."""
        from main.config.config_manager import ModularConfigManager as ConfigManager
        
        manager = ConfigManager(use_validation=False)
        assert manager.use_validation is False
    
    def test_get_validated_config(self):
        """Test get_validated_config function."""
        config_content = """
system:
  environment: paper
api_keys:
  alpaca:
    key: test_key
    secret: test_secret
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "unified_config.yaml"
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            from main.config.config_manager import ModularConfigManager as ConfigManager
            manager = ConfigManager(config_dir=tmpdir, use_validation=True)
            
            config = manager.load_validated_config()
            assert isinstance(config, AITraderConfig)
            assert config.system.environment == Environment.PAPER


class TestRealWorldScenarios:
    """Test real-world configuration scenarios."""
    
    def test_production_config_validation(self):
        """Test production-like configuration."""
        config_dict = {
            "system": {"environment": "live"},
            "api_keys": {
                "alpaca": {
                    "key": "AKFZ...",
                    "secret": "xyz123..."
                },
                "polygon": {
                    "key": "poly_key"
                }
            },
            "broker": {
                "paper_trading": False
            },
            "risk": {
                "position_sizing": {
                    "max_position_size": 2.0
                },
                "circuit_breaker": {
                    "daily_loss_limit": 3.0
                }
            },
            "data": {
                "sources": ["alpaca", "polygon"]
            }
        }
        
        config = AITraderConfig(**config_dict)
        assert config.system.environment == Environment.LIVE
        assert config.broker.paper_trading is False
        assert config.risk.position_sizing.max_position_size == 2.0
    
    def test_paper_trading_config_validation(self):
        """Test paper trading configuration."""
        config_dict = {
            "system": {"environment": "paper"},
            "api_keys": {
                "alpaca": {
                    "key": "paper_key",
                    "secret": "paper_secret"
                }
            },
            "broker": {
                "paper_trading": True
            },
            "risk": {
                "position_sizing": {
                    "max_position_size": 10.0
                }
            }
        }
        
        config = AITraderConfig(**config_dict)
        assert config.system.environment == Environment.PAPER
        assert config.broker.paper_trading is True
    
    def test_training_config_validation(self):
        """Test training configuration."""
        config_dict = {
            "system": {"environment": "training"},
            "api_keys": {
                "alpaca": {
                    "key": "training_key",
                    "secret": "training_secret"
                }
            },
            "training": {
                "top_n_symbols_for_training": 100,
                "feature_lookback_days": 365,
                "models": ["xgboost", "lightgbm"]
            }
        }
        
        config = AITraderConfig(**config_dict)
        assert config.system.environment == Environment.TRAINING
        assert config.training.top_n_symbols_for_training == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
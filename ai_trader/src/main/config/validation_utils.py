"""
Configuration validation utilities for AI Trader V3.

This module provides fail-fast validation utilities with specific error messages
for configuration validation failures.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from pydantic import ValidationError
import yaml

from .validation_models import AITraderConfig, get_validation_errors
from main.utils.core import get_logger

logger = get_logger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    
    def __init__(self, message: str, errors: List[str]):
        self.message = message
        self.errors = errors
        super().__init__(self.message)
    
    def __str__(self):
        error_str = "\n".join(f"  - {error}" for error in self.errors)
        return f"{self.message}\n\nValidation Errors:\n{error_str}"


class ConfigValidator:
    """
    Configuration validator with fail-fast validation and detailed error reporting.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize the config validator.
        
        Args:
            strict_mode: If True, fail on warnings. If False, log warnings and continue.
        """
        self.strict_mode = strict_mode
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def validate_file(self, config_path: str) -> AITraderConfig:
        """
        Validate a configuration file with comprehensive error reporting.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Validated AITraderConfig instance
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            # Check if file exists
            if not Path(config_path).exists():
                raise ConfigValidationError(
                    f"Configuration file not found: {config_path}",
                    [f"File does not exist: {config_path}"]
                )
            
            # Load YAML file
            try:
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigValidationError(
                    f"Invalid YAML syntax in configuration file: {config_path}",
                    [f"YAML parsing error: {str(e)}"]
                )
            except Exception as e:
                raise ConfigValidationError(
                    f"Error reading configuration file: {config_path}",
                    [f"File read error: {str(e)}"]
                )
            
            # Validate configuration
            return self.validate_dict(config_dict, config_path)
            
        except ConfigValidationError:
            raise
        except Exception as e:
            raise ConfigValidationError(
                f"Unexpected error validating configuration: {config_path}",
                [f"Unexpected error: {str(e)}"]
            )
    
    def validate_dict(self, config_dict: Dict[str, Any], source: str = "configuration") -> AITraderConfig:
        """
        Validate a configuration dictionary with comprehensive error reporting.
        
        Args:
            config_dict: Configuration dictionary to validate
            source: Source description for error messages
            
        Returns:
            Validated AITraderConfig instance
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Reset validation state
        self.warnings.clear()
        self.errors.clear()
        
        # Perform pre-validation checks
        self._pre_validate_config(config_dict)
        
        # Validate with Pydantic
        try:
            config = AITraderConfig(**config_dict)
        except ValidationError as e:
            validation_errors = self._format_validation_errors(e)
            raise ConfigValidationError(
                f"Configuration validation failed for {source}",
                validation_errors
            )
        
        # Perform post-validation checks
        self._post_validate_config(config)
        
        # Handle warnings
        if self.warnings:
            if self.strict_mode:
                raise ConfigValidationError(
                    f"Configuration validation warnings in strict mode for {source}",
                    self.warnings
                )
            else:
                for warning in self.warnings:
                    logger.warning(warning)
        
        # Check for errors
        if self.errors:
            raise ConfigValidationError(
                f"Configuration validation errors for {source}",
                self.errors
            )
        
        return config
    
    def _pre_validate_config(self, config_dict: Dict[str, Any]) -> None:
        """Perform pre-validation checks."""
        
        # Check for required top-level sections
        required_sections = ['api_keys', 'system']
        for section in required_sections:
            if section not in config_dict:
                self.errors.append(f"Missing required section: {section}")
        
        # Check for critical API keys
        if 'api_keys' in config_dict:
            api_keys = config_dict['api_keys']
            if not isinstance(api_keys, dict):
                self.errors.append("api_keys must be a dictionary")
            else:
                if 'alpaca' not in api_keys:
                    self.errors.append("Missing required API key configuration: alpaca")
                else:
                    alpaca_config = api_keys['alpaca']
                    if not isinstance(alpaca_config, dict):
                        self.errors.append("alpaca API configuration must be a dictionary")
                    else:
                        if 'key' not in alpaca_config:
                            self.errors.append("Missing required field: api_keys.alpaca.key")
                        if 'secret' not in alpaca_config:
                            self.errors.append("Missing required field: api_keys.alpaca.secret")
        
        # Check environment variables
        self._check_environment_variables(config_dict)
    
    def _post_validate_config(self, config: AITraderConfig) -> None:
        """Perform post-validation checks."""
        
        # Check for production safety
        if config.system.environment.value == 'live':
            if config.broker.paper_trading:
                self.warnings.append("Paper trading is enabled in live environment")
            
            # Check risk limits for live trading
            if config.risk.position_sizing.max_position_size > 2.0:
                self.warnings.append(f"High position size limit for live trading: {config.risk.position_sizing.max_position_size}%")
            
            if config.risk.circuit_breaker.daily_loss_limit > 5.0:
                self.warnings.append(f"High daily loss limit for live trading: {config.risk.circuit_breaker.daily_loss_limit}%")
        
        # Check for reasonable configuration values
        if config.trading.starting_cash < 1000:
            self.warnings.append(f"Very low starting cash: ${config.trading.starting_cash}")
        
        if config.data.backfill.max_parallel > 30:
            self.warnings.append(f"High parallel backfill setting may cause rate limiting: {config.data.backfill.max_parallel}")
        
        # Check data source availability
        if len(config.data.sources) == 1 and config.data.sources[0].value == 'alpaca':
            self.warnings.append("Only Alpaca data source configured - consider adding backup sources")
    
    def _check_environment_variables(self, config_dict: Dict[str, Any]) -> None:
        """Check that required environment variables are set."""
        
        def find_env_vars(obj: Any, path: str = "") -> List[str]:
            """Recursively find environment variable references."""
            env_vars = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    env_vars.extend(find_env_vars(value, current_path))
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    current_path = f"{path}[{i}]" if path else f"[{i}]"
                    env_vars.extend(find_env_vars(value, current_path))
            elif isinstance(obj, str):
                import re
                matches = re.findall(r'\$\{([^}]+)\}', obj)
                for match in matches:
                    env_vars.append((match, path))
            return env_vars
        
        env_vars = find_env_vars(config_dict)
        
        for env_var, path in env_vars:
            value = os.getenv(env_var)
            if value is None:
                self.errors.append(f"Environment variable '{env_var}' is not set (used in {path})")
            elif not value.strip():
                self.errors.append(f"Environment variable '{env_var}' is empty (used in {path})")
    
    def _format_validation_errors(self, validation_error: ValidationError) -> List[str]:
        """Format Pydantic validation errors into readable messages."""
        errors = []
        
        for error in validation_error.errors():
            field_path = ".".join(str(loc) for loc in error['loc'])
            error_type = error['type']
            message = error['msg']
            
            # Create more user-friendly error messages
            if error_type == 'value_error.missing':
                errors.append(f"Missing required field: {field_path}")
            elif error_type == 'type_error.none.not_allowed':
                errors.append(f"Field cannot be null: {field_path}")
            elif error_type == 'value_error.number.not_gt':
                errors.append(f"Value must be greater than {error.get('ctx', {}).get('limit_value', 0)}: {field_path}")
            elif error_type == 'value_error.number.not_ge':
                errors.append(f"Value must be greater than or equal to {error.get('ctx', {}).get('limit_value', 0)}: {field_path}")
            elif error_type == 'value_error.number.not_le':
                errors.append(f"Value must be less than or equal to {error.get('ctx', {}).get('limit_value', 0)}: {field_path}")
            elif error_type == 'value_error.str.regex':
                errors.append(f"Invalid format for field: {field_path}")
            elif error_type == 'value_error.url':
                errors.append(f"Invalid URL format: {field_path}")
            elif error_type == 'value_error.enum':
                valid_values = error.get('ctx', {}).get('enum_values', [])
                errors.append(f"Invalid value for {field_path}. Valid values: {valid_values}")
            elif error_type == 'value_error.extra':
                errors.append(f"Unknown field not allowed: {field_path}")
            else:
                errors.append(f"{field_path}: {message}")
        
        return errors
    
    def check_system_requirements(self) -> List[str]:
        """Check system requirements for the AI Trader."""
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8 or higher is required")
        
        # Check required packages
        required_packages = [
            'pydantic',
            'pandas',
            'numpy',
            'alpaca-py',
            'pyyaml',
            'python-dotenv'
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                issues.append(f"Required package not installed: {package}")
        
        return issues


def validate_startup_config(config_path: Optional[str] = None) -> AITraderConfig:
    """
    Validate startup configuration with fail-fast behavior.
    
    This function should be called at application startup to ensure all
    configuration is valid before the system begins trading.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        Validated configuration
        
    Raises:
        SystemExit: If validation fails (fail-fast behavior)
    """
    try:
        # Determine config path
        if config_path is None:
            config_dir = Path(__file__).parent.parent / 'data_pipeline' / 'config'
            config_path = config_dir / 'layer_definitions.yaml'
        
        # Create validator in strict mode
        validator = ConfigValidator(strict_mode=True)
        
        # Check system requirements
        system_issues = validator.check_system_requirements()
        if system_issues:
            print("❌ System Requirements Check Failed:")
            for issue in system_issues:
                print(f"  - {issue}")
            sys.exit(1)
        
        # Validate configuration
        print(f"🔍 Validating configuration: {config_path}")
        config = validator.validate_file(str(config_path))
        
        # Apply environment overrides
        config = config.get_environment_config()
        
        print("✅ Configuration validation successful!")
        print(f"   Environment: {config.system.environment.value}")
        print(f"   Paper Trading: {config.broker.paper_trading}")
        print(f"   Data Sources: {[source.value for source in config.data.sources]}")
        
        return config
        
    except ConfigValidationError as e:
        print(f"❌ Configuration Validation Failed: {e.message}")
        print("\nValidation Errors:")
        for error in e.errors:
            print(f"  - {error}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during configuration validation: {e}")
        sys.exit(1)


def validate_config_and_warn(config_path: str) -> Tuple[Optional[AITraderConfig], List[str]]:
    """
    Validate configuration and return warnings instead of failing.
    
    This function is useful for development and testing where you want to
    see all issues without the application exiting.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (config or None if failed, list of warnings/errors)
    """
    try:
        validator = ConfigValidator(strict_mode=False)
        config = validator.validate_file(config_path)
        return config, validator.warnings
    except ConfigValidationError as e:
        return None, e.errors
    except Exception as e:
        return None, [f"Unexpected error: {str(e)}"]


# Export main functions
__all__ = [
    'ConfigValidationError',
    'ConfigValidator',
    'validate_startup_config',
    'validate_config_and_warn'
]
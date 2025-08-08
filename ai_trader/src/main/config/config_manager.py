"""
Configuration Manager - Clean Implementation

Simplified configuration manager that loads YAML files using OmegaConf
and validates them using Pydantic models.

Key features:
- OmegaConf for YAML loading with interpolation support
- Pydantic validation models for type safety
- Environment variable substitution
- Configuration caching
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig, OmegaConf
import threading
import time
import hashlib

from .validation_models import AITraderConfig
from .validation_utils import ConfigValidator
from .env_loader import ensure_environment_loaded

logger = logging.getLogger(__name__)


class ConfigCache:
    """
    Thread-safe configuration cache with TTL support.
    """
    
    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default TTL
        """
        Initialize configuration cache.
        
        Args:
            ttl_seconds: Time-to-live for cached entries in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # key -> (config, timestamp)
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[DictConfig]:
        """
        Get cached configuration if still valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached configuration or None if expired/missing
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            config, timestamp = self._cache[key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None
            
            return config
    
    def put(self, key: str, config: DictConfig) -> None:
        """
        Cache configuration with current timestamp.
        
        Args:
            key: Cache key
            config: Configuration to cache
        """
        with self._lock:
            self._cache[key] = (config, time.time())
    
    def invalidate(self, key: Optional[str] = None) -> None:
        """
        Invalidate cached entries.
        
        Args:
            key: Specific key to invalidate, or None to clear all
        """
        with self._lock:
            if key is None:
                self._cache.clear()
            elif key in self._cache:
                del self._cache[key]
    
    def clear_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self.ttl_seconds
            ]
            for key in expired_keys:
                del self._cache[key]


class ConfigManager:
    """
    Clean, streamlined configuration manager.
    
    Loads YAML configurations using OmegaConf and validates them using Pydantic.
    Supports environment variable substitution and configuration caching.
    """
    
    def __init__(self, 
                 config_dir: Optional[str] = None, 
                 use_validation: bool = True,
                 strict_validation: bool = False,
                 enable_caching: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            use_validation: Whether to use Pydantic validation (default: True)
            strict_validation: Whether to fail on validation warnings (default: False)
            enable_caching: Whether to enable configuration caching
        """
        # Determine config directory
        if config_dir is None:
            self.config_dir = Path(__file__).parent
        else:
            self.config_dir = Path(config_dir)
        
        self.use_validation = use_validation
        self.strict_validation = strict_validation
        self.enable_caching = enable_caching
        
        # Initialize cache if enabled
        self._cache = ConfigCache() if enable_caching else None
        
        # Ensure environment variables are loaded
        ensure_environment_loaded()
        
        logger.info(f"Initialized config manager for {self.config_dir} (caching: {enable_caching})")
    
    def _load_yaml_config(self, 
                          config_path: Path,
                          overrides: Optional[List[str]] = None) -> DictConfig:
        """
        Internal method to load and process YAML configuration.
        
        Args:
            config_path: Path to the configuration file
            overrides: Optional list of override parameters
            
        Returns:
            Loaded and processed configuration
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML configuration
        cfg = OmegaConf.load(config_path)
        
        # Apply overrides if provided
        if overrides:
            override_cfg = OmegaConf.from_cli(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)
        
        # Resolve environment variables and interpolations
        OmegaConf.resolve(cfg)
        
        return cfg
    
    def _create_cache_key(self, config_name: str, overrides: Optional[List[str]]) -> str:
        """
        Create a reliable cache key that handles overrides properly.
        
        Args:
            config_name: Name of the configuration
            overrides: List of override parameters
            
        Returns:
            Stable cache key string
        """
        if not overrides:
            return config_name
        
        # Create a stable hash of the overrides list
        overrides_str = "|".join(sorted(overrides))  # Sort for consistency
        overrides_hash = hashlib.md5(overrides_str.encode()).hexdigest()[:8]
        
        return f"{config_name}:{overrides_hash}"
    
    def load_config(self, 
                   config_name: str = "layer_definitions",
                   overrides: Optional[List[str]] = None,
                   force_reload: bool = False) -> DictConfig:
        """
        Load configuration from YAML file with full validation and processing.
        
        Args:
            config_name: Name of the main config file
            overrides: List of override parameters
            force_reload: Force reload even if cached
            
        Returns:
            Loaded and processed configuration
        """
        # Special handling for unified_config (backward compatibility)
        if config_name == "unified_config":
            return self._load_unified_config(overrides, force_reload)
        
        # Create proper cache key with hashing for overrides
        cache_key = self._create_cache_key(config_name, overrides)
        
        # Check cache first (unless force reload)
        if not force_reload and self._cache:
            cached_config = self._cache.get(cache_key)
            if cached_config is not None:
                logger.debug(f"Returning cached configuration for {config_name}")
                return cached_config
        
        try:
            config_path = self.config_dir / "yaml" / f"{config_name}.yaml"
            cfg = self._load_yaml_config(config_path, overrides)
            
            # Log summary
            self._log_config_summary(cfg)
            
            # Run configuration validation if enabled
            if self.use_validation:
                self._validate_config(cfg, config_name)
            
            # Cache the result if caching is enabled
            if self._cache:
                self._cache.put(cache_key, cfg)
                logger.debug(f"Cached configuration for {config_name}")
            
            return cfg
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_unified_config(self, 
                             overrides: Optional[List[str]] = None, 
                             force_reload: bool = False) -> DictConfig:
        """
        Load a unified configuration by merging multiple configs.
        This provides backward compatibility for code expecting unified_config.
        
        Args:
            overrides: List of override parameters
            force_reload: Force reload even if cached
            
        Returns:
            Merged configuration that emulates the old unified_config
        """
        import os
        
        # Create cache key for unified config
        cache_key = self._create_cache_key("unified_config", overrides)
        
        # Check cache first
        if not force_reload and self._cache:
            cached_config = self._cache.get(cache_key)
            if cached_config is not None:
                logger.debug("Returning cached unified configuration")
                return cached_config
        
        try:
            # Load base configurations
            configs = []
            
            # Load app context config as base
            app_context_path = self.config_dir / "yaml" / "app_context_config.yaml"
            if app_context_path.exists():
                configs.append(self._load_yaml_config(app_context_path))
            
            # Load layer definitions
            layer_def_path = self.config_dir / "yaml" / "layer_definitions.yaml"
            if layer_def_path.exists():
                configs.append(self._load_yaml_config(layer_def_path))
            
            # Merge all configs
            if configs:
                cfg = OmegaConf.merge(*configs)
            else:
                cfg = OmegaConf.create({})
            
            # Add essential runtime config from environment
            cfg.database = OmegaConf.create({
                "url": os.getenv("DATABASE_URL", ""),
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "name": os.getenv("DB_NAME", "ai_trader"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", "")
            })
            
            cfg.api_keys = OmegaConf.create({
                "alpaca": {
                    "key": os.getenv("ALPACA_API_KEY", ""),
                    "secret": os.getenv("ALPACA_API_SECRET", "")
                },
                "polygon": {
                    "key": os.getenv("POLYGON_API_KEY", "")
                }
            })
            
            # Add minimal defaults for compatibility
            if "trading" not in cfg:
                cfg.trading = OmegaConf.create({
                    "symbols": [],
                    "strategy": "default",
                    "capital": 10000,
                    "risk_management": {
                        "max_risk_per_trade": 0.02,
                        "max_positions": 10
                    },
                    "position_sizing": {
                        "max_position_size": 1000
                    },
                    "starting_cash": 10000
                })
            
            if "monitoring" not in cfg:
                cfg.monitoring = OmegaConf.create({
                    "dashboard": {
                        "port": 8050
                    }
                })
            
            if "system" not in cfg:
                cfg.system = OmegaConf.create({
                    "environment": os.getenv("ENVIRONMENT", "development")
                })
            
            if "broker" not in cfg:
                cfg.broker = OmegaConf.create({
                    "paper_trading": True
                })
            
            # Apply overrides if provided
            if overrides:
                override_cfg = OmegaConf.from_dotlist(overrides)
                cfg = OmegaConf.merge(cfg, override_cfg)
            
            # Cache the result
            if self._cache:
                self._cache.put(cache_key, cfg)
                logger.debug("Cached unified configuration")
            
            logger.info("Loaded unified configuration (backward compatibility mode)")
            return cfg
            
        except Exception as e:
            logger.error(f"Failed to load unified configuration: {e}")
            # Return minimal config to prevent total failure
            return OmegaConf.create({
                "database": {"url": os.getenv("DATABASE_URL", "")},
                "api_keys": {
                    "alpaca": {"key": "", "secret": ""},
                    "polygon": {"key": ""}
                },
                "system": {"environment": "development"},
                "broker": {"paper_trading": True},
                "trading": {"symbols": [], "strategy": "default"},
                "monitoring": {"dashboard": {"port": 8050}}
            })
    
    def load_simple_config(self, config_file: str) -> DictConfig:
        """
        Load a simple configuration file without validation or detailed logging.
        
        Args:
            config_file: Path to configuration file (relative to config_dir/yaml)
            
        Returns:
            Loaded configuration
        """
        config_path = self.config_dir / "yaml" / config_file
        return self._load_yaml_config(config_path)
    
    def _validate_config(self, cfg: DictConfig, source: str) -> None:
        """
        Validate configuration using Pydantic models.
        
        This is the single validation path for all configurations.
        
        Args:
            cfg: Configuration to validate
            source: Source name for error messages
            
        Raises:
            ConfigValidationError: If validation fails in strict mode
        """
        validator = ConfigValidator(strict_mode=self.strict_validation)
        try:
            # Convert OmegaConf to dict for validation
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(config_dict, dict):
                validator.validate_dict(config_dict, source=source)
                logger.info(f"Configuration validation successful for {source}")
        except Exception as validation_error:
            if self.strict_validation:
                # Fail fast in strict mode
                logger.error(f"Configuration validation failed for {source}: {validation_error}")
                raise
            else:
                # Log warning and continue in non-strict mode
                logger.warning(f"Configuration validation failed for {source}: {validation_error}")
                logger.warning("Continuing with unvalidated configuration - consider fixing validation errors")
    
    def validate_config(self, cfg: DictConfig) -> AITraderConfig:
        """
        Validate configuration using Pydantic models.
        
        Args:
            cfg: Configuration to validate
            
        Returns:
            Validated configuration object
        """
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(config_dict, dict):
            raise ValueError("Configuration must be a dictionary")
        
        return AITraderConfig(**config_dict)
    
    def load_environment_config(self, env: str, base_config: Optional[DictConfig] = None) -> DictConfig:
        """
        Load environment-specific configuration.
        
        Args:
            env: Environment name (development, paper, production)
            base_config: Base configuration to merge with
            
        Returns:
            Environment-specific configuration
        """
        env_config_path = self.config_dir / "yaml" / "environments" / f"{env}.yaml"
        
        if env_config_path.exists():
            env_cfg = OmegaConf.load(env_config_path)
            OmegaConf.resolve(env_cfg)
            
            if base_config is not None:
                return self._safe_merge_configs(base_config, env_cfg)
            return env_cfg
        else:
            logger.warning(f"Environment config not found: {env_config_path}")
            return base_config or OmegaConf.create({})
    
    def _safe_merge_configs(self, base_config: DictConfig, env_config: DictConfig) -> DictConfig:
        """
        Safely merge configurations with predictable overlay behavior.
        
        This method provides a simpler, more predictable alternative to OmegaConf's
        deep merge that can sometimes produce unexpected results.
        
        Args:
            base_config: Base configuration
            env_config: Environment configuration to merge
            
        Returns:
            Merged configuration
        """
        try:
            # Use OmegaConf's merge as primary method
            return OmegaConf.merge(base_config, env_config)
        except Exception as e:
            logger.warning(f"OmegaConf merge failed: {e}. Using simple overlay merge.")
            
            # Fallback to simple overlay merge
            try:
                merged = OmegaConf.create(dict(base_config))
                
                # Apply environment overrides section by section
                for key, value in env_config.items():
                    if key in merged and OmegaConf.is_config(merged[key]) and OmegaConf.is_config(value):
                        # Merge nested configuration sections
                        for nested_key, nested_value in value.items():
                            merged[key][nested_key] = nested_value
                    else:
                        # Replace entire section
                        merged[key] = value
                
                return merged
            except Exception as fallback_error:
                logger.error(f"Fallback merge also failed: {fallback_error}")
                # Return base config as last resort
                return base_config
    
    def _log_config_summary(self, cfg: DictConfig):
        """
        Log configuration summary.
        
        Args:
            cfg: Configuration to summarize
        """
        try:
            # Log basic information
            if hasattr(cfg, 'system') and hasattr(cfg.system, 'environment'):
                logger.info(f"Environment: {cfg.system.environment}")
            
            if hasattr(cfg, 'broker') and hasattr(cfg.broker, 'paper_trading'):
                logger.info(f"Paper Trading: {cfg.broker.paper_trading}")
                
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.debug(f"Could not log config summary: {e}")
    
    def clear_cache(self, config_name: Optional[str] = None) -> None:
        """
        Clear configuration cache.
        
        Args:
            config_name: Specific configuration to clear, or None to clear all
        """
        if self._cache:
            if config_name:
                # Clear all cache entries for this config name (thread-safe)
                with self._cache._lock:
                    keys_to_clear = [key for key in self._cache._cache.keys() if key.startswith(f"{config_name}:")]
                for key in keys_to_clear:
                    self._cache.invalidate(key)
                logger.debug(f"Cleared cache for config: {config_name}")
            else:
                self._cache.invalidate()
                logger.debug("Cleared all configuration cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self._cache:
            return {"enabled": False}
        
        with self._cache._lock:
            cache_size = len(self._cache._cache)
            current_time = time.time()
            expired_count = sum(
                1 for _, timestamp in self._cache._cache.values()
                if current_time - timestamp > self._cache.ttl_seconds
            )
            
            return {
                "enabled": True,
                "total_entries": cache_size,
                "valid_entries": cache_size - expired_count,
                "expired_entries": expired_count,
                "ttl_seconds": self._cache.ttl_seconds
            }


# Factory function for easy instantiation
def get_config_manager(config_dir: Optional[str] = None, 
                      use_validation: bool = True,
                      strict_validation: bool = False,
                      enable_caching: bool = True) -> ConfigManager:
    """
    Factory function to create a configuration manager.
    
    Args:
        config_dir: Directory containing configuration files
        use_validation: Whether to enable Pydantic validation (default: True)
        strict_validation: Whether to fail on validation warnings (default: False)
        enable_caching: Whether to enable configuration caching
        
    Returns:
        Configured ConfigManager instance
    """
    return ConfigManager(
        config_dir=config_dir,
        use_validation=use_validation,
        strict_validation=strict_validation,
        enable_caching=enable_caching
    )


def get_production_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """
    Factory function to create a production-ready configuration manager.
    
    This creates a ConfigManager with strict validation enabled for fail-fast
    behavior in production environments.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        ConfigManager with strict validation enabled
    """
    return ConfigManager(
        config_dir=config_dir,
        use_validation=True,
        strict_validation=True,
        enable_caching=True
    )


# Backward compatibility function for existing code
def get_config(config_name: str = "layer_definitions", 
               overrides: Optional[List[str]] = None,
               config_dir: Optional[str] = None) -> DictConfig:
    """
    Load configuration using the config manager.
    
    This function provides backward compatibility for existing code
    that uses get_config() instead of the new manager pattern.
    
    Args:
        config_name: Name of the config file to load
        overrides: List of override parameters
        config_dir: Directory containing configuration files
        
    Returns:
        Loaded configuration as DictConfig
    """
    manager = get_config_manager(config_dir=config_dir, use_validation=False, strict_validation=False)
    return manager.load_config(config_name, overrides)
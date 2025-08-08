# File: risk_management/pre_trade/unified_limit_checker/config.py

"""
Configuration module for the Unified Limit Checker.

Provides configuration classes and default settings for limit checking functionality.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import logging

from main.utils.core import get_logger
from .types import LimitType, LimitAction, ViolationSeverity

logger = get_logger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for threshold settings."""
    soft_threshold_ratio: float = 0.8  # Soft threshold as ratio of hard threshold
    warning_threshold_ratio: float = 0.6  # Warning threshold ratio
    critical_multiplier: float = 1.5  # Multiplier for critical violations
    use_dynamic_thresholds: bool = False  # Enable dynamic threshold adjustment
    threshold_adjustment_rate: float = 0.1  # Rate of threshold adjustment


@dataclass
class ActionConfig:
    """Configuration for violation actions."""
    default_action: LimitAction = LimitAction.ALERT
    severity_action_map: Dict[ViolationSeverity, LimitAction] = field(default_factory=lambda: {
        ViolationSeverity.INFO: LimitAction.LOG_ONLY,
        ViolationSeverity.WARNING: LimitAction.ALERT,
        ViolationSeverity.SOFT_BREACH: LimitAction.ALERT,
        ViolationSeverity.HARD_BREACH: LimitAction.BLOCK_TRADE,
        ViolationSeverity.CRITICAL: LimitAction.EMERGENCY_STOP
    })
    escalation_enabled: bool = True  # Enable action escalation
    escalation_time_window: int = 300  # Time window for escalation (seconds)
    max_violations_before_escalation: int = 3


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and metrics."""
    enable_metrics: bool = True
    metrics_window_seconds: int = 3600  # 1 hour
    violation_history_limit: int = 1000
    check_history_limit: int = 10000
    enable_real_time_alerts: bool = True
    alert_cooldown_seconds: int = 60  # Prevent alert spam
    aggregate_metrics_interval: int = 60  # Aggregate metrics every minute


@dataclass
class LimitTypeConfig:
    """Configuration specific to a limit type."""
    enabled: bool = True
    default_threshold: float = 100.0
    default_action: LimitAction = LimitAction.ALERT
    check_frequency_seconds: Optional[int] = None  # None means check on every request
    cache_duration_seconds: int = 5  # Cache check results
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LimitConfig:
    """Main configuration class for the Unified Limit Checker."""
    # General settings
    enabled: bool = True
    strict_mode: bool = False  # Fail closed on errors
    parallel_checking: bool = True  # Run checks in parallel
    max_parallel_checks: int = 10
    
    # Sub-configurations
    threshold_config: ThresholdConfig = field(default_factory=ThresholdConfig)
    action_config: ActionConfig = field(default_factory=ActionConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Limit type specific configurations
    limit_type_configs: Dict[LimitType, LimitTypeConfig] = field(default_factory=dict)
    
    # Event handling
    enable_events: bool = True
    event_buffer_size: int = 1000
    async_event_handling: bool = True
    
    # Performance settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60
    batch_check_size: int = 50
    
    def get_limit_type_config(self, limit_type: LimitType) -> LimitTypeConfig:
        """Get configuration for a specific limit type."""
        if limit_type not in self.limit_type_configs:
            # Return default config
            return LimitTypeConfig()
        return self.limit_type_configs[limit_type]
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        valid = True
        
        # Validate threshold ratios
        if not 0 < self.threshold_config.soft_threshold_ratio <= 1:
            logger.error("soft_threshold_ratio must be between 0 and 1")
            valid = False
        
        if not 0 < self.threshold_config.warning_threshold_ratio <= 1:
            logger.error("warning_threshold_ratio must be between 0 and 1")
            valid = False
        
        # Validate parallel settings
        if self.max_parallel_checks < 1:
            logger.error("max_parallel_checks must be at least 1")
            valid = False
        
        # Validate cache settings
        if self.cache_ttl_seconds < 0:
            logger.error("cache_ttl_seconds cannot be negative")
            valid = False
        
        return valid


def get_default_config() -> LimitConfig:
    """Get default limit checker configuration."""
    config = LimitConfig()
    
    # Configure specific limit types
    config.limit_type_configs = {
        LimitType.POSITION_SIZE: LimitTypeConfig(
            enabled=True,
            default_threshold=10.0,  # 10% of portfolio
            default_action=LimitAction.BLOCK_TRADE,
            custom_settings={
                "check_notional": True,
                "include_pending_orders": True
            }
        ),
        LimitType.DRAWDOWN: LimitTypeConfig(
            enabled=True,
            default_threshold=20.0,  # 20% max drawdown
            default_action=LimitAction.PAUSE_STRATEGY,
            check_frequency_seconds=60,  # Check every minute
            custom_settings={
                "lookback_days": 30,
                "use_high_water_mark": True
            }
        ),
        LimitType.PORTFOLIO_EXPOSURE: LimitTypeConfig(
            enabled=True,
            default_threshold=95.0,  # 95% max exposure
            default_action=LimitAction.ALERT,
            custom_settings={
                "include_cash": False,
                "net_exposure": True
            }
        ),
        LimitType.TRADING_VELOCITY: LimitTypeConfig(
            enabled=True,
            default_threshold=100,  # 100 trades per hour
            default_action=LimitAction.ALERT,
            check_frequency_seconds=60,
            custom_settings={
                "time_window_minutes": 60,
                "count_modifications": False
            }
        ),
        LimitType.CONCENTRATION: LimitTypeConfig(
            enabled=True,
            default_threshold=30.0,  # 30% max concentration
            default_action=LimitAction.REDUCE_POSITION,
            custom_settings={
                "group_by_sector": True,
                "include_correlated": True
            }
        ),
        LimitType.VOLATILITY: LimitTypeConfig(
            enabled=True,
            default_threshold=50.0,  # 50% annualized vol
            default_action=LimitAction.ALERT,
            check_frequency_seconds=300,  # Every 5 minutes
            custom_settings={
                "lookback_days": 20,
                "use_garch": False
            }
        ),
        LimitType.VAR_UTILIZATION: LimitTypeConfig(
            enabled=True,
            default_threshold=80.0,  # 80% VaR utilization
            default_action=LimitAction.ALERT,
            check_frequency_seconds=900,  # Every 15 minutes
            custom_settings={
                "confidence_level": 0.95,
                "horizon_days": 1
            }
        ),
        LimitType.LEVERAGE: LimitTypeConfig(
            enabled=True,
            default_threshold=2.0,  # 2x max leverage
            default_action=LimitAction.BLOCK_TRADE,
            custom_settings={
                "include_derivatives": True,
                "gross_leverage": True
            }
        )
    }
    
    # Validate before returning
    if not config.validate():
        logger.warning("Default configuration has validation warnings")
    
    return config


def create_test_config() -> LimitConfig:
    """Create configuration suitable for testing."""
    config = get_default_config()
    
    # Disable some features for testing
    config.parallel_checking = False
    config.cache_enabled = False
    config.monitoring_config.enable_real_time_alerts = False
    config.async_event_handling = False
    
    # Lower thresholds for testing
    config.monitoring_config.violation_history_limit = 100
    config.monitoring_config.check_history_limit = 1000
    
    return config
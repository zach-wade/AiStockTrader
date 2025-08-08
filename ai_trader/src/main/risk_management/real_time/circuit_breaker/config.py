"""
Circuit Breaker Configuration Management

Centralized configuration management for all circuit breaker components.
Extracted from monolithic circuit_breaker.py for better organization.

Created: 2025-07-15
"""

from typing import Dict, Any, Optional
from datetime import timedelta
import logging

from .types import BreakerType, BreakerConfiguration

logger = logging.getLogger(__name__)


class BreakerConfig:
    """
    Configuration management for circuit breaker system.
    
    Provides centralized configuration with validation and defaults.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize configuration with validation."""
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_fields = ['volatility_threshold', 'max_drawdown', 'loss_rate_threshold']
        for field in required_fields:
            if field not in self.config:
                logger.warning(f"Missing required config field: {field}, using default")
    
    # Core thresholds
    @property
    def volatility_threshold(self) -> float:
        """Volatility threshold for triggering breaker."""
        return self.config.get('volatility_threshold', 0.05)  # 5% intraday
    
    @property
    def max_drawdown(self) -> float:
        """Maximum allowed drawdown."""
        return self.config.get('max_drawdown', 0.08)  # 8% from peak
    
    @property
    def loss_rate_threshold(self) -> float:
        """Loss rate threshold for rapid loss detection."""
        return self.config.get('loss_rate_threshold', 0.03)  # 3% in 5 min
    
    @property
    def correlation_breakdown(self) -> float:
        """Correlation breakdown threshold."""
        return self.config.get('correlation_breakdown', 0.7)  # vs historical
    
    @property
    def max_positions(self) -> int:
        """Maximum number of positions allowed."""
        return self.config.get('max_positions', 20)
    
    @property
    def max_position_size(self) -> float:
        """Maximum position size as percentage of portfolio."""
        return self.config.get('max_position_size_pct', 0.10)  # 10%
    
    # Enhanced safety thresholds
    @property
    def anomaly_detection_threshold(self) -> float:
        """Anomaly detection threshold in standard deviations."""
        return self.config.get('anomaly_detection_threshold', 3.0)  # 3 sigma
    
    @property
    def vix_emergency_threshold(self) -> float:
        """VIX level for emergency conditions."""
        return self.config.get('vix_emergency_threshold', 40.0)  # VIX above 40
    
    @property
    def data_quality_threshold(self) -> float:
        """Data quality threshold percentage."""
        return self.config.get('data_quality_threshold', 95.0)  # 95% quality
    
    @property
    def model_performance_threshold(self) -> float:
        """Model performance threshold."""
        return self.config.get('model_performance_threshold', 0.60)  # 60% accuracy
    
    @property
    def liquidation_urgency_threshold(self) -> float:
        """Threshold for emergency liquidation."""
        return self.config.get('liquidation_urgency_threshold', 0.15)  # 15% rapid loss
    
    # API endpoints
    @property
    def nyse_api_endpoint(self) -> str:
        """NYSE API endpoint for circuit breaker status."""
        return self.config.get('nyse_api_endpoint', '')
    
    @property
    def nasdaq_api_endpoint(self) -> str:
        """NASDAQ API endpoint for circuit breaker status."""
        return self.config.get('nasdaq_api_endpoint', '')
    
    @property
    def vix_api_endpoint(self) -> str:
        """VIX API endpoint for volatility data."""
        return self.config.get('vix_api_endpoint', '')
    
    # Kill switch configuration
    @property
    def kill_switch_enabled(self) -> bool:
        """Whether kill switch is enabled."""
        return self.config.get('kill_switch_enabled', True)
    
    @property
    def kill_switch_sources(self) -> list:
        """Sources that can trigger kill switch."""
        return self.config.get('kill_switch_sources', ['manual', 'anomaly', 'external'])
    
    # Time windows
    @property
    def loss_rate_window(self) -> timedelta:
        """Time window for loss rate calculation."""
        return timedelta(minutes=self.config.get('loss_rate_window_minutes', 5))
    
    @property
    def volatility_window(self) -> timedelta:
        """Time window for volatility calculation."""
        return timedelta(minutes=self.config.get('volatility_window_minutes', 30))
    
    @property
    def cooldown_period(self) -> timedelta:
        """Cooldown period after breaker trips."""
        return timedelta(minutes=self.config.get('cooldown_period_minutes', 15))
    
    # Trading hours
    @property
    def trading_start(self) -> float:
        """Trading start time (hour as float)."""
        return self.config.get('trading_start_hour', 9.5)  # 9:30 AM
    
    @property
    def trading_end(self) -> float:
        """Trading end time (hour as float)."""
        return self.config.get('trading_end_hour', 16)  # 4:00 PM
    
    # Breaker-specific configurations
    def get_breaker_config(self, breaker_type: BreakerType) -> BreakerConfiguration:
        """Get configuration for specific breaker type."""
        breaker_configs = self.config.get('breaker_configs', {})
        breaker_config = breaker_configs.get(breaker_type.value, {})
        
        # Set default threshold based on breaker type
        default_threshold = self._get_default_threshold(breaker_type)
        
        return BreakerConfiguration(
            breaker_type=breaker_type,
            enabled=breaker_config.get('enabled', True),
            threshold=breaker_config.get('threshold', default_threshold),
            cooldown_minutes=breaker_config.get('cooldown_minutes', 15),
            auto_reset=breaker_config.get('auto_reset', True),
            severity_weight=breaker_config.get('severity_weight', 1.0),
            custom_config=breaker_config.get('custom_config', {})
        )
    
    def _get_default_threshold(self, breaker_type: BreakerType) -> float:
        """Get default threshold for breaker type."""
        defaults = {
            BreakerType.VOLATILITY: self.volatility_threshold,
            BreakerType.DRAWDOWN: self.max_drawdown,
            BreakerType.LOSS_RATE: self.loss_rate_threshold,
            BreakerType.CORRELATION: self.correlation_breakdown,
            BreakerType.ANOMALY_DETECTION: self.anomaly_detection_threshold,
            BreakerType.DATA_FEED_INTEGRITY: self.data_quality_threshold,
            BreakerType.MODEL_PERFORMANCE: self.model_performance_threshold,
            BreakerType.LIQUIDATION_REQUIRED: self.liquidation_urgency_threshold,
        }
        return defaults.get(breaker_type, 0.0)
    
    def get_all_breaker_configs(self) -> Dict[BreakerType, BreakerConfiguration]:
        """Get configurations for all breaker types."""
        return {
            breaker_type: self.get_breaker_config(breaker_type)
            for breaker_type in BreakerType
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self.config.update(updates)
        self._validate_config()
        logger.info(f"Configuration updated: {list(updates.keys())}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get all risk limits in a single dictionary."""
        return {
            'volatility_threshold': self.volatility_threshold,
            'max_drawdown': self.max_drawdown,
            'loss_rate_threshold': self.loss_rate_threshold,
            'correlation_breakdown': self.correlation_breakdown,
            'max_position_size': self.max_position_size,
            'anomaly_detection_threshold': self.anomaly_detection_threshold,
            'vix_emergency_threshold': self.vix_emergency_threshold,
            'data_quality_threshold': self.data_quality_threshold,
            'model_performance_threshold': self.model_performance_threshold,
            'liquidation_urgency_threshold': self.liquidation_urgency_threshold
        }
    
    def validate_limits(self) -> Dict[str, str]:
        """Validate risk limits and return any warnings."""
        warnings = {}
        
        if self.max_drawdown > 0.20:
            warnings['max_drawdown'] = f"High drawdown threshold: {self.max_drawdown:.2%}"
        
        if self.volatility_threshold > 0.10:
            warnings['volatility_threshold'] = f"High volatility threshold: {self.volatility_threshold:.2%}"
        
        if self.loss_rate_threshold > 0.05:
            warnings['loss_rate_threshold'] = f"High loss rate threshold: {self.loss_rate_threshold:.2%}"
        
        if self.max_position_size > 0.20:
            warnings['max_position_size'] = f"High position size limit: {self.max_position_size:.2%}"
        
        return warnings
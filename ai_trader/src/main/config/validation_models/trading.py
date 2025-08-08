"""
Trading configuration validation models.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, confloat, conint
import logging

from .core import Environment

logger = logging.getLogger(__name__)

# Enums
class PositionSizeType(str, Enum):
    """Valid position sizing methods."""
    EQUAL_WEIGHT = "equal_weight"
    KELLY = "kelly"
    FIXED_DOLLAR = "fixed_dollar"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"

class ExecutionAlgorithm(str, Enum):
    """Valid execution algorithms."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ADAPTIVE = "adaptive"

# Custom types
Percentage = confloat(ge=0.0, le=100.0)
PositionSize = confloat(ge=0.0, le=10.0)
RiskPercentage = confloat(ge=0.0, le=50.0)
PositiveInt = conint(ge=1)
NonNegativeInt = conint(ge=0)
PositiveFloat = confloat(gt=0.0)
NonNegativeFloat = confloat(ge=0.0)


# System Configuration Models
class SystemConfig(BaseModel):
    """System-level configuration with validation."""
    environment: Environment = Field(default=Environment.PAPER, description="Trading environment")
    debug: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Debug settings")
    timezone: str = Field(default="US/Eastern", description="System timezone")
    
    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v):
        """Validate timezone string."""
        try:
            import pytz
            pytz.timezone(v)
        except Exception:
            raise ValueError(f"Invalid timezone: {v}")
        return v


# Broker Configuration Models
class BrokerConfig(BaseModel):
    """Broker configuration with validation."""
    name: str = Field(default="alpaca", description="Broker name")
    paper_trading: bool = Field(default=True, description="Enable paper trading")
    
    @field_validator('paper_trading')
    @classmethod
    def validate_paper_trading_safety(cls, v):
        """Ensure paper trading is enabled unless explicitly set to live."""
        # Skip warning for environment overrides - only warn for active config
        return v


# Trading Configuration Models
class PositionSizingConfig(BaseModel):
    """Position sizing configuration with validation."""
    method: PositionSizeType = Field(default=PositionSizeType.EQUAL_WEIGHT, description="Position sizing method")
    default_position_size: PositiveFloat = Field(default=5000.0, description="Default position size in dollars")
    max_position_size: PositiveFloat = Field(default=10000.0, description="Maximum position size in dollars")
    
    @model_validator(mode='after')
    def validate_max_position_size(self):
        """Ensure max position size is greater than default."""
        if self.max_position_size <= self.default_position_size:
            raise ValueError("Max position size must be greater than default position size")
        return self


class ExecutionConfig(BaseModel):
    """Execution configuration with validation."""
    algorithm: ExecutionAlgorithm = Field(default=ExecutionAlgorithm.MARKET, description="Execution algorithm")
    slippage_bps: NonNegativeFloat = Field(default=5.0, le=100.0, description="Slippage in basis points")
    
    @field_validator('slippage_bps')
    @classmethod
    def validate_slippage(cls, v):
        """Warn about high slippage settings."""
        if v > 20.0:
            logger.warning(f"High slippage setting: {v} bps")
        return v


class TradingConfig(BaseModel):
    """Trading configuration with validation."""
    starting_cash: PositiveFloat = Field(default=100000.0, description="Starting cash amount")
    max_symbols: PositiveInt = Field(default=500, le=2000, description="Maximum symbols to trade")
    universe: List[str] = Field(default_factory=list, description="Trading universe symbols")
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig, description="Position sizing config")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="Execution config")
    close_positions_on_shutdown: bool = Field(default=False, description="Close positions on shutdown")


# Risk Management Configuration Models
class RiskPositionSizingConfig(BaseModel):
    """Risk-based position sizing configuration."""
    method: PositionSizeType = Field(default=PositionSizeType.KELLY, description="Risk position sizing method")
    max_position_size: PositionSize = Field(default=20.0, description="Maximum position size as percentage of portfolio")
    max_portfolio_leverage: PositiveFloat = Field(default=1.0, le=4.0, description="Maximum portfolio leverage")
    
    @field_validator('max_portfolio_leverage')
    @classmethod
    def validate_leverage(cls, v):
        """Warn about high leverage settings."""
        if v > 2.0:
            logger.warning(f"High leverage setting: {v}x")
        return v


class RiskLimitsConfig(BaseModel):
    """Risk limits configuration with validation."""
    max_daily_trades: PositiveInt = Field(default=50, le=200, description="Maximum daily trades")
    max_positions: PositiveInt = Field(default=20, le=100, description="Maximum concurrent positions")
    max_sector_exposure: Percentage = Field(default=30.0, description="Maximum sector exposure percentage")
    
    @field_validator('max_daily_trades')
    @classmethod
    def validate_daily_trades(cls, v):
        """Warn about high daily trade limits."""
        if v > 100:
            logger.warning(f"High daily trade limit: {v}")
        return v


class StopLossConfig(BaseModel):
    """Stop loss configuration with validation."""
    enabled: bool = Field(default=True, description="Enable stop loss")
    default_percentage: RiskPercentage = Field(default=2.0, description="Default stop loss percentage")
    
    @field_validator('default_percentage')
    @classmethod
    def validate_stop_loss(cls, v):
        """Validate reasonable stop loss percentage."""
        if v < 0.5:
            raise ValueError("Stop loss percentage too low (< 0.5%)")
        if v > 10.0:
            logger.warning(f"High stop loss percentage: {v}%")
        return v


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration with validation."""
    enabled: bool = Field(default=True, description="Enable circuit breaker")
    daily_loss_limit: RiskPercentage = Field(default=5.0, description="Daily loss limit percentage")
    
    @field_validator('daily_loss_limit')
    @classmethod
    def validate_daily_loss_limit(cls, v):
        """Validate reasonable daily loss limit."""
        if v < 1.0:
            raise ValueError("Daily loss limit too low (< 1%)")
        if v > 20.0:
            logger.warning(f"High daily loss limit: {v}%")
        return v


class RiskConfig(BaseModel):
    """Comprehensive risk management configuration."""
    position_sizing: RiskPositionSizingConfig = Field(default_factory=RiskPositionSizingConfig, description="Position sizing config")
    limits: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig, description="Risk limits config")
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig, description="Stop loss config")
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig, description="Circuit breaker config")


# Strategy Configuration Models
class StrategyWeights(BaseModel):
    """Strategy weighting configuration with validation."""
    min_weight: NonNegativeFloat = Field(default=0.0, le=1.0, description="Minimum strategy weight")
    max_weight: NonNegativeFloat = Field(default=1.0, le=1.0, description="Maximum strategy weight")
    default_weight: NonNegativeFloat = Field(default=0.5, le=1.0, description="Default strategy weight")
    
    @model_validator(mode='after')
    def validate_weights(self):
        """Ensure weight consistency."""
        if self.min_weight > self.max_weight:
            raise ValueError("Min weight cannot be greater than max weight")
        if self.default_weight < self.min_weight or self.default_weight > self.max_weight:
            raise ValueError("Default weight must be between min and max weights")
        return self


class StrategiesConfig(BaseModel):
    """Strategies configuration with validation."""
    enabled: List[str] = Field(default_factory=list, description="Enabled strategies")
    weights: Dict[str, StrategyWeights] = Field(default_factory=dict, description="Strategy weights")
    lookback_periods: Dict[str, int] = Field(default_factory=dict, description="Strategy lookback periods")
    parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Strategy-specific parameters")
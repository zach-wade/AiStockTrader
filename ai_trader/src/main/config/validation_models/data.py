"""
Data and features configuration validation models.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, conint
import logging

logger = logging.getLogger(__name__)

# Enums
class DataProvider(str, Enum):
    """Valid data providers."""
    ALPACA = "alpaca"
    ALPACA_ALT = "alpaca_alt"
    POLYGON = "polygon"
    POLYGON_ALT = "polygon_alt"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alphavantage"
    BENZINGA = "benzinga"
    IEX = "iex"

class TimeFrame(str, Enum):
    """Valid timeframes."""
    MINUTE = "1minute"
    FIVE_MINUTE = "5minute"
    FIFTEEN_MINUTE = "15minute"
    THIRTY_MINUTE = "30minute"
    HOUR = "1hour"
    DAY = "1day"
    WEEK = "1week"
    MONTH = "1month"

class UniverseType(str, Enum):
    """Valid universe types."""
    SP500 = "sp500"
    NASDAQ100 = "nasdaq100"
    RUSSELL2000 = "russell2000"
    CUSTOM = "custom"
    DYNAMIC = "dynamic"

# Custom types
PositiveInt = conint(ge=1)
NonNegativeInt = conint(ge=0)
PositiveFloat = float  # Using float directly instead of confloat


# Data Configuration Models
class UniverseConfig(BaseModel):
    """Universe configuration with validation."""
    provider: DataProvider = Field(default=DataProvider.ALPACA, description="Universe data provider")
    asset_class: str = Field(default="us_equity", description="Asset class")
    include_crypto: bool = Field(default=False, description="Include cryptocurrency")
    max_symbols: PositiveInt = Field(default=2000, le=10000, description="Maximum symbols in universe")


class SourceLimitsConfig(BaseModel):
    """Data source limits configuration."""
    market_data: Dict[TimeFrame, NonNegativeInt] = Field(default_factory=dict, description="Market data limits by timeframe")
    news: NonNegativeInt = Field(default=0, description="News data limit in days")
    corporate_actions: NonNegativeInt = Field(default=0, description="Corporate actions limit in days")
    options: NonNegativeInt = Field(default=0, description="Options data limit in days")


class BackfillStageConfig(BaseModel):
    """Configuration for a single backfill stage."""
    name: str = Field(description="Stage name")
    description: Optional[str] = Field(default=None, description="Stage description")
    sources: List[DataProvider] = Field(description="Data sources for this stage")
    intervals: List[str] = Field(default_factory=list, description="Time intervals")
    lookback_days: PositiveInt = Field(default=30, description="Days to look back")
    lookback_strategy: str = Field(default="days", description="Lookback strategy")
    destination: str = Field(default="data_lake", description="Data destination")
    data_type: str = Field(default="market_data", description="Type of data")
    enabled: bool = Field(default=True, description="Whether stage is enabled")


class BackfillConfig(BaseModel):
    """Data backfill configuration with validation."""
    max_parallel: PositiveInt = Field(default=20, le=50, description="Maximum parallel backfill jobs")
    source_limits: Dict[str, SourceLimitsConfig] = Field(default_factory=dict, description="Source-specific limits")
    stages: List[BackfillStageConfig] = Field(default_factory=list, description="Backfill stages")


class DataConfig(BaseModel):
    """Data configuration with validation."""
    universe: UniverseConfig = Field(default_factory=UniverseConfig, description="Universe configuration")
    sources: List[DataProvider] = Field(default_factory=lambda: [DataProvider.ALPACA], description="Data sources")
    streaming: Dict[str, Any] = Field(default_factory=dict, description="Streaming configuration")
    backfill: BackfillConfig = Field(default_factory=BackfillConfig, description="Backfill configuration")
    
    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v):
        """Ensure at least one data source is configured."""
        if not v:
            raise ValueError("At least one data source must be configured")
        return v


# Feature Configuration Models
class TechnicalIndicatorsConfig(BaseModel):
    """Technical indicators configuration with validation."""
    rsi_period: PositiveInt = Field(default=14, ge=5, le=50, description="RSI period")
    macd_fast: PositiveInt = Field(default=12, ge=5, le=30, description="MACD fast period")
    macd_slow: PositiveInt = Field(default=26, ge=15, le=60, description="MACD slow period")
    macd_signal: PositiveInt = Field(default=9, ge=5, le=20, description="MACD signal period")
    bb_period: PositiveInt = Field(default=20, ge=10, le=50, description="Bollinger Bands period")
    bb_std: PositiveFloat = Field(default=2.0, ge=1.0, le=3.0, description="Bollinger Bands standard deviation")
    
    @model_validator(mode='after')
    def validate_macd_periods(self):
        """Ensure MACD periods are in correct order."""
        if self.macd_fast >= self.macd_slow:
            raise ValueError("MACD fast period must be less than slow period")
        return self


class FeaturesConfig(BaseModel):
    """Features configuration with validation."""
    use_multi_timeframe: bool = Field(default=True, description="Use multi-timeframe features")
    timeframes: List[TimeFrame] = Field(default_factory=lambda: [TimeFrame.MINUTE, TimeFrame.FIVE_MINUTE, TimeFrame.FIFTEEN_MINUTE, TimeFrame.HOUR, TimeFrame.DAY], description="Timeframes to use")
    technical_indicators: TechnicalIndicatorsConfig = Field(default_factory=TechnicalIndicatorsConfig, description="Technical indicators config")
    microstructure: Dict[str, bool] = Field(default_factory=lambda: {"enabled": True}, description="Microstructure features")
    cross_sectional: Dict[str, bool] = Field(default_factory=lambda: {"enabled": True}, description="Cross-sectional features")
    advanced_statistical: Dict[str, bool] = Field(default_factory=lambda: {"enabled": True}, description="Advanced statistical features")
    
    @field_validator('timeframes')
    @classmethod
    def validate_timeframes(cls, v):
        """Ensure at least one timeframe is selected."""
        if not v:
            raise ValueError("At least one timeframe must be selected")
        return v


# Training Configuration Models
class TrainingConfig(BaseModel):
    """Training configuration with validation."""
    symbols: List[str] = Field(default_factory=list, description="Training symbols")
    top_n_symbols_for_training: PositiveInt = Field(default=500, le=2000, description="Top N symbols for training")
    ensure_sector_diversity: bool = Field(default=True, description="Ensure sector diversity")
    min_symbols_per_sector: PositiveInt = Field(default=2, description="Minimum symbols per sector")
    max_symbols_per_sector: PositiveInt = Field(default=50, description="Maximum symbols per sector")
    feature_lookback_days: PositiveInt = Field(default=730, description="Feature lookback days")
    models: List[str] = Field(default_factory=lambda: ["xgboost", "lightgbm", "random_forest", "ensemble"], description="Models to train")
    
    @model_validator(mode='after')
    def validate_symbols_per_sector(self):
        """Ensure max symbols per sector is greater than min."""
        if self.max_symbols_per_sector <= self.min_symbols_per_sector:
            raise ValueError("Max symbols per sector must be greater than min symbols per sector")
        return self


# Universe Configuration Models
class UniverseFiltersConfig(BaseModel):
    """Universe filters configuration with validation."""
    min_price: PositiveFloat = Field(default=10.0, description="Minimum stock price")
    max_price: PositiveFloat = Field(default=500.0, description="Maximum stock price")
    min_volume: PositiveInt = Field(default=1000000, description="Minimum daily volume")
    min_market_cap: PositiveInt = Field(default=100000000, description="Minimum market cap")
    exclude_penny_stocks: bool = Field(default=True, description="Exclude penny stocks")
    exclude_etfs: bool = Field(default=False, description="Exclude ETFs")
    
    @model_validator(mode='after')
    def validate_price_range(self):
        """Ensure max price is greater than min price."""
        if self.max_price <= self.min_price:
            raise ValueError("Max price must be greater than min price")
        return self


class UniverseMainConfig(BaseModel):
    """Main universe configuration with validation."""
    mode: str = Field(default="dynamic", description="Universe mode")
    filters: UniverseFiltersConfig = Field(default_factory=UniverseFiltersConfig, description="Universe filters")


# Paths Configuration Models
class PathsConfig(BaseModel):
    """Paths configuration with validation."""
    features: str = Field(default="data_pipeline/storage/features", description="Features path")
    feature_cache: str = Field(default="data_pipeline/storage/feature_cache", description="Feature cache path")
    models: str = Field(default="models/registry", description="Models path")
    logs: str = Field(default="logs", description="Logs path")
    data: str = Field(default="data_pipeline/storage", description="Data path")
    model_results: str = Field(default="results/hyperopt", description="Model results path")
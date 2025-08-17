"""
Validation models for configuration.
"""

# Import enums from their respective modules
# Import core models (System, Database, API keys)
from .core import (
    AlpacaConfig,
    AlphaVantageConfig,
    ApiKeysConfig,
    BenzingaConfig,
    Environment,
    FinnhubConfig,
    FredConfig,
    NewsApiConfig,
    PolygonConfig,
    RedditConfig,
    TwitterConfig,
    validate_env_var,
)

# Import data models
from .data import (
    BackfillConfig,
    BackfillStageConfig,
    DataConfig,
    DataProvider,
    FeaturesConfig,
    PathsConfig,
    SourceLimitsConfig,
    TechnicalIndicatorsConfig,
    TimeFrame,
    TrainingConfig,
    UniverseConfig,
    UniverseFiltersConfig,
    UniverseMainConfig,
    UniverseType,
)

# Import services models
from .services import (
    CatalystScoringConfig,
    DataPipelineConfig,
    EnvironmentOverrides,
    IntervalsConfig,
    LookbackPeriodsConfig,
    MarketHoursConfig,
    MonitoringConfig,
    OrchestratorConfig,
    OrchestratorFeaturesConfig,
    OrchestratorTrainingConfig,
)

# Import trading models
from .trading import (
    BrokerConfig,
    CircuitBreakerConfig,
    ExecutionAlgorithm,
    ExecutionConfig,
    PositionSizeType,
    PositionSizingConfig,
    RiskConfig,
    RiskLimitsConfig,
    RiskPositionSizingConfig,
    StopLossConfig,
    StrategiesConfig,
    StrategyWeights,
    SystemConfig,
    TradingConfig,
)

# Import validation config separately for convenience
ValidationConfig = DataPipelineConfig.ValidationConfig

# Import main config model
from .main import AITraderConfig, get_validation_errors, validate_config_file

# Define __all__ for explicit exports
__all__ = [
    # Enums
    "PositionSizeType",
    "ExecutionAlgorithm",
    "DataProvider",
    "Environment",
    "TimeFrame",
    "UniverseType",
    # Database/API
    "AlpacaConfig",
    "PolygonConfig",
    "AlphaVantageConfig",
    "BenzingaConfig",
    "FinnhubConfig",
    "FredConfig",
    "NewsApiConfig",
    "RedditConfig",
    "TwitterConfig",
    "ApiKeysConfig",
    "validate_env_var",
    # Trading
    "SystemConfig",
    "BrokerConfig",
    "PositionSizingConfig",
    "ExecutionConfig",
    "TradingConfig",
    "RiskPositionSizingConfig",
    "RiskLimitsConfig",
    "StopLossConfig",
    "CircuitBreakerConfig",
    "RiskConfig",
    "StrategyWeights",
    "StrategiesConfig",
    # Features
    "UniverseConfig",
    "SourceLimitsConfig",
    "BackfillStageConfig",
    "BackfillConfig",
    "DataConfig",
    "TechnicalIndicatorsConfig",
    "FeaturesConfig",
    "TrainingConfig",
    "UniverseFiltersConfig",
    "UniverseMainConfig",
    "PathsConfig",
    # Monitoring
    "MonitoringConfig",
    "EnvironmentOverrides",
    "IntervalsConfig",
    "LookbackPeriodsConfig",
    "MarketHoursConfig",
    "OrchestratorTrainingConfig",
    "OrchestratorFeaturesConfig",
    "CatalystScoringConfig",
    "DataPipelineConfig",
    "ValidationConfig",  # Validation config for data pipeline
    "OrchestratorConfig",
    # Main
    "AITraderConfig",
    "validate_config_file",
    "get_validation_errors",
]

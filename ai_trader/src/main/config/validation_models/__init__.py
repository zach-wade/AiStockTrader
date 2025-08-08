"""
Validation models for configuration.
"""

# Import enums from their respective modules
from .core import Environment
from .trading import PositionSizeType, ExecutionAlgorithm
from .data import DataProvider, TimeFrame, UniverseType

# Import core models (System, Database, API keys)
from .core import (
    AlpacaConfig,
    PolygonConfig,
    AlphaVantageConfig,
    BenzingaConfig,
    FinnhubConfig,
    FredConfig,
    NewsApiConfig,
    RedditConfig,
    TwitterConfig,
    ApiKeysConfig,
    validate_env_var
)

# Import trading models
from .trading import (
    SystemConfig,
    BrokerConfig,
    PositionSizingConfig,
    ExecutionConfig,
    TradingConfig,
    RiskPositionSizingConfig,
    RiskLimitsConfig,
    StopLossConfig,
    CircuitBreakerConfig,
    RiskConfig,
    StrategyWeights,
    StrategiesConfig
)

# Import data models
from .data import (
    UniverseConfig,
    SourceLimitsConfig,
    BackfillStageConfig,
    BackfillConfig,
    DataConfig,
    TechnicalIndicatorsConfig,
    FeaturesConfig,
    TrainingConfig,
    UniverseFiltersConfig,
    UniverseMainConfig,
    PathsConfig
)

# Import services models
from .services import (
    MonitoringConfig,
    EnvironmentOverrides,
    IntervalsConfig,
    LookbackPeriodsConfig,
    MarketHoursConfig,
    OrchestratorTrainingConfig,
    OrchestratorFeaturesConfig,
    CatalystScoringConfig,
    DataPipelineConfig,
    OrchestratorConfig
)

# Import validation config separately for convenience
ValidationConfig = DataPipelineConfig.ValidationConfig

# Import main config model
from .main import (
    AITraderConfig,
    validate_config_file,
    get_validation_errors
)

# Define __all__ for explicit exports
__all__ = [
    # Enums
    'PositionSizeType',
    'ExecutionAlgorithm',
    'DataProvider',
    'Environment',
    'TimeFrame',
    'UniverseType',
    
    # Database/API
    'AlpacaConfig',
    'PolygonConfig',
    'AlphaVantageConfig',
    'BenzingaConfig',
    'FinnhubConfig',
    'FredConfig',
    'NewsApiConfig',
    'RedditConfig',
    'TwitterConfig',
    'ApiKeysConfig',
    'validate_env_var',
    
    # Trading
    'SystemConfig',
    'BrokerConfig',
    'PositionSizingConfig',
    'ExecutionConfig',
    'TradingConfig',
    'RiskPositionSizingConfig',
    'RiskLimitsConfig',
    'StopLossConfig',
    'CircuitBreakerConfig',
    'RiskConfig',
    'StrategyWeights',
    'StrategiesConfig',
    
    # Features
    'UniverseConfig',
    'SourceLimitsConfig',
    'BackfillStageConfig',
    'BackfillConfig',
    'DataConfig',
    'TechnicalIndicatorsConfig',
    'FeaturesConfig',
    'TrainingConfig',
    'UniverseFiltersConfig',
    'UniverseMainConfig',
    'PathsConfig',
    
    # Monitoring
    'MonitoringConfig',
    'EnvironmentOverrides',
    'IntervalsConfig',
    'LookbackPeriodsConfig',
    'MarketHoursConfig',
    'OrchestratorTrainingConfig',
    'OrchestratorFeaturesConfig',
    'CatalystScoringConfig',
    'DataPipelineConfig',
    'ValidationConfig',  # Validation config for data pipeline
    'OrchestratorConfig',
    
    # Main
    'AITraderConfig',
    'validate_config_file',
    'get_validation_errors'
]
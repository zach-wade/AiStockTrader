"""
AI Trader Models Module

This module contains all machine learning models, strategies, and related components:
- Trading strategies (momentum, mean reversion, ML-based)
- Model inference and prediction engines
- Model monitoring and performance tracking
- Outcome classification and labeling
- Specialist models for specific data types
- Training pipelines and orchestration
"""

# Core components
from .common import (
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    Position,
    Order,
    AccountInfo,
    MarketData,
    RiskMetrics,
    Signal
)

# Inference components
from .inference import (
    ModelRegistry,
    PredictionEngine,
    FeaturePipeline,
    ModelAnalyticsService,
    ModelManagementService,
    PredictionEngineService
)

# Monitoring components
from .monitoring import (
    ModelMonitor,
    DriftDetector,
    PerformanceCalculator
)

# Outcome classification
from .outcome_classifier import OutcomeClassifier
from .outcome_classifier_types import (
    OutcomeLabel,
    LabelingConfig,
    OutcomeMetrics
)

# Strategies
from .strategies import (
    BaseStrategy,
    BaseUniverseStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    MLMomentumStrategy,
    RegimeAdaptiveStrategy,
    EnsembleStrategy
)

# Specialists
from .specialists import (
    BaseCatalystSpecialist,
    CatalystPrediction,
    TechnicalSpecialist,
    NewsSpecialist,
    SocialSpecialist,
    OptionsSpecialist,
    EarningsSpecialist,
    CatalystSpecialistEnsemble
)

# Training
from .training import (
    ModelTrainingOrchestrator,
    TrainingPipelineRunner,
    PipelineStages,
    TimeSeriesCV,
    # HyperparameterSearch,  # Requires optuna
    RetrainingScheduler
)

__all__ = [
    # Common types
    'OrderStatus',
    'OrderType', 
    'OrderSide',
    'TimeInForce',
    'Position',
    'Order',
    'AccountInfo',
    'MarketData',
    'RiskMetrics',
    'Signal',
    
    # Inference
    'ModelRegistry',
    'PredictionEngine',
    'FeaturePipeline',
    'ModelAnalyticsService',
    'ModelManagementService',
    'PredictionEngineService',
    
    # Monitoring
    'ModelMonitor',
    'DriftDetector',
    'PerformanceCalculator',
    
    # Outcome classification
    'OutcomeClassifier',
    'OutcomeLabel',
    'OutcomeConfig',
    'OutcomeMetrics',
    
    # Strategies
    'BaseStrategy',
    'BaseUniverseStrategy',
    'BreakoutStrategy',
    'MeanReversionStrategy',
    'MLMomentumStrategy',
    'RegimeAdaptiveStrategy',
    'EnsembleStrategy',
    
    # Specialists
    'BaseSpecialist',
    'TechnicalSpecialist',
    'NewsSpecialist',
    'SocialSpecialist',
    'OptionsSpecialist',
    'EarningsSpecialist',
    'EnsembleSpecialist',
    
    # Training
    'TrainingOrchestrator',
    'PipelineRunner',
    'PipelineStages',
    'CrossValidation',
    'HyperparameterSearch',
    'RetrainingScheduler'
]

# Version info
__version__ = "2.0.0"
__author__ = "AI Trader Team"
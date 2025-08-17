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
    AccountInfo,
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    RiskMetrics,
    Signal,
    TimeInForce,
)

# Inference components
from .inference import (
    FeaturePipeline,
    ModelAnalyticsService,
    ModelManagementService,
    ModelRegistry,
    PredictionEngine,
    PredictionEngineService,
)

# Monitoring components
from .monitoring import DriftDetector, ModelMonitor, PerformanceCalculator

# Outcome classification
from .outcome_classifier import OutcomeClassifier
from .outcome_classifier_types import LabelingConfig, OutcomeLabel, OutcomeMetrics

# Specialists
from .specialists import (
    BaseCatalystSpecialist,
    CatalystPrediction,
    CatalystSpecialistEnsemble,
    EarningsSpecialist,
    NewsSpecialist,
    OptionsSpecialist,
    SocialSpecialist,
    TechnicalSpecialist,
)

# Strategies
from .strategies import (
    BaseStrategy,
    BaseUniverseStrategy,
    BreakoutStrategy,
    EnsembleStrategy,
    MeanReversionStrategy,
    MLMomentumStrategy,
    RegimeAdaptiveStrategy,
)

# Training
from .training import (  # HyperparameterSearch,  # Requires optuna
    ModelTrainingOrchestrator,
    PipelineStages,
    RetrainingScheduler,
    TimeSeriesCV,
    TrainingPipelineRunner,
)

__all__ = [
    # Common types
    "OrderStatus",
    "OrderType",
    "OrderSide",
    "TimeInForce",
    "Position",
    "Order",
    "AccountInfo",
    "MarketData",
    "RiskMetrics",
    "Signal",
    # Inference
    "ModelRegistry",
    "PredictionEngine",
    "FeaturePipeline",
    "ModelAnalyticsService",
    "ModelManagementService",
    "PredictionEngineService",
    # Monitoring
    "ModelMonitor",
    "DriftDetector",
    "PerformanceCalculator",
    # Outcome classification
    "OutcomeClassifier",
    "OutcomeLabel",
    "OutcomeConfig",
    "OutcomeMetrics",
    # Strategies
    "BaseStrategy",
    "BaseUniverseStrategy",
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "MLMomentumStrategy",
    "RegimeAdaptiveStrategy",
    "EnsembleStrategy",
    # Specialists
    "BaseSpecialist",
    "TechnicalSpecialist",
    "NewsSpecialist",
    "SocialSpecialist",
    "OptionsSpecialist",
    "EarningsSpecialist",
    "EnsembleSpecialist",
    # Training
    "TrainingOrchestrator",
    "PipelineRunner",
    "PipelineStages",
    "CrossValidation",
    "HyperparameterSearch",
    "RetrainingScheduler",
]

# Version info
__version__ = "2.0.0"
__author__ = "AI Trader Team"

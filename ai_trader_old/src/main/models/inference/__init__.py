"""
Model inference components for real-time prediction.

This module provides the infrastructure for model deployment and inference:
- Model registry for versioning and deployment
- Prediction engine for real-time inference
- Feature pipeline integration
- Model management and analytics services
- Performance monitoring and optimization
"""

from .feature_pipeline import RealTimeFeaturePipeline as FeaturePipeline
from .model_analytics_service import ModelAnalyticsService

# Service components
from .model_management_service import ModelManagementService

# Core inference components
from .model_registry import ModelRegistry

# Registry enhancements
from .model_registry_enhancements import (
    ModelMetricsTracker,
    ModelRegistryEnhancements,
    ModelVersionManager,
)

# Types
from .model_registry_types import (
    DeploymentStatus,
    ModelDeployment,
    ModelInfo,
    ModelMetrics,
    ModelVersion,
)
from .prediction_engine import PredictionEngine
from .prediction_engine_service import PredictionEngineService

__all__ = [
    # Core components
    "ModelRegistry",
    "PredictionEngine",
    "FeaturePipeline",
    # Services
    "ModelManagementService",
    "ModelAnalyticsService",
    "PredictionEngineService",
    # Enhancements
    "ModelRegistryEnhancements",
    "ModelVersionManager",
    "ModelMetricsTracker",
    # Types
    "ModelInfo",
    "ModelVersion",
    "ModelDeployment",
    "ModelMetrics",
    "DeploymentStatus",
]

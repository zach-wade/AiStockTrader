"""
Validation Framework - Core Components

Core validation orchestration components including the main validation
pipeline, unified validator, and validation factories.

Components:
- validation_pipeline: Main validation pipeline orchestrator
- validation_factory: Factory for creating validation components
- validation_stage_factory: Factory for stage-specific validators
"""

from .validation_factory import (
    ValidationFactory,
    create_unified_validator,
    create_validation_pipeline,
)
from .validation_pipeline import ValidationContext, ValidationPipeline, ValidationResult
from .validation_stage_factory import (
    StageConfiguredValidator,
    ValidationStageFactory,
    create_feature_ready_validator,
    create_ingest_validator,
    create_post_etl_validator,
)

__all__ = [
    # Core components
    "ValidationPipeline",
    "ValidationContext",
    "ValidationResult",
    "ValidationFactory",
    "ValidationStageFactory",
    "StageConfiguredValidator",
    # Factory functions
    "create_validation_pipeline",
    "create_unified_validator",
    "create_ingest_validator",
    "create_post_etl_validator",
    "create_feature_ready_validator",
]

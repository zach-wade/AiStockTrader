"""
Data Pipeline Orchestration

Layer-based orchestration using focused coordinators that compose
existing services, following service-oriented architecture principles.
"""

# Import focused coordinators
from .coordinators import (
    DataFetchCoordinator,
    DataStage,
    FetchRequest,
    LayerCoordinator,
    StorageCoordinator,
    StorageRequest,
)
from .event_coordinator import EventCoordinator
from .layer_manager import LayerManager
from .retention_manager import RetentionManager

# Import unified pipeline
from .unified_pipeline import PipelineMode, PipelineRequest, PipelineResult, UnifiedPipeline

__all__ = [
    # Existing managers
    "LayerManager",
    "RetentionManager",
    "EventCoordinator",
    # Unified pipeline
    "UnifiedPipeline",
    "PipelineMode",
    "PipelineRequest",
    "PipelineResult",
    # Focused coordinators
    "LayerCoordinator",
    "DataFetchCoordinator",
    "StorageCoordinator",
    # Data types
    "DataStage",
    "FetchRequest",
    "StorageRequest",
]

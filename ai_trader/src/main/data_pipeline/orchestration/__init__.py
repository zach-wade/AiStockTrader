"""
Data Pipeline Orchestration

Layer-based orchestration using focused coordinators that compose
existing services, following service-oriented architecture principles.
"""

from .layer_manager import LayerManager
from .retention_manager import RetentionManager
from .event_coordinator import EventCoordinator

# Import focused coordinators
from .coordinators import (
    LayerCoordinator,
    DataFetchCoordinator,
    StorageCoordinator,
    DataStage,
    FetchRequest,
    StorageRequest
)

# Import unified pipeline
from .unified_pipeline import (
    UnifiedPipeline,
    PipelineMode,
    PipelineRequest,
    PipelineResult
)

__all__ = [
    # Existing managers
    'LayerManager',
    'RetentionManager',
    'EventCoordinator',
    
    # Unified pipeline
    'UnifiedPipeline',
    'PipelineMode',
    'PipelineRequest',
    'PipelineResult',
    
    # Focused coordinators
    'LayerCoordinator',
    'DataFetchCoordinator',
    'StorageCoordinator',
    
    # Data types
    'DataStage',
    'FetchRequest',
    'StorageRequest',
]
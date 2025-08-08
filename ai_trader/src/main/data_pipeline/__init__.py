"""
Data Pipeline Module - Clean Architecture Implementation

This module provides a unified, layer-based data pipeline architecture that replaces
the previous tier-based system. It implements clean separation of concerns,
event-driven processing, and comprehensive data management.

Key Features:
- Layer-based architecture (0-3) replacing tier system
- Event-driven automatic backfill triggers
- Unified orchestration replacing multiple orchestrators
- World-class validation framework
- Dual storage patterns with hot/cold retention policies
- Security-hardened with no pickle usage
- Comprehensive utils integration

Architecture Overview:
- core/: Base classes, enums, exceptions (uses existing /interfaces/)
- orchestration/: Unified orchestrator with layer management
- ingestion/: Standardized client framework
- storage/: Repositories and bulk loaders (migrated excellent components)
- processing/: Data transformation and standardization
- validation/: Comprehensive validation framework (migrated)
- historical/: Service-oriented historical management
- events/: Event-driven coordination (uses existing /interfaces/events/)
- monitoring/: Health and performance monitoring
"""

from .core.enums import DataLayer, DataType, ProcessingPriority
from .core.exceptions import DataPipelineError, ValidationError, StorageError

__version__ = "2.0.0"
__author__ = "AI Trader - Clean Architecture Implementation"

__all__ = [
    # Core enums
    'DataLayer',
    'DataType', 
    'ProcessingPriority',
    
    # Core exceptions
    'DataPipelineError',
    'ValidationError',
    'StorageError'
]
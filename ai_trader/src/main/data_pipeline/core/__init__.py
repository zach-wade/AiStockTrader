"""
Data Pipeline Core Components

Provides foundational components for the data pipeline including enums,
exceptions, and base classes. Interfaces are handled by the existing
/interfaces/ directory structure.
"""

from .enums import DataLayer, DataType, ProcessingPriority, RetentionPolicy
from .exceptions import (
    DataPipelineError,
    ValidationError, 
    StorageError,
    IngestionError,
    ProcessingError,
    OrchestrationError
)

__all__ = [
    # Enums
    'DataLayer',
    'DataType',
    'ProcessingPriority', 
    'RetentionPolicy',
    
    # Exceptions
    'DataPipelineError',
    'ValidationError',
    'StorageError', 
    'IngestionError',
    'ProcessingError',
    'OrchestrationError'
]
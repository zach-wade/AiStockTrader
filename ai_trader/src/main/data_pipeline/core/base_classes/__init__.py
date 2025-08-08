"""
Data Pipeline Base Classes

Provides abstract base classes that define common functionality and patterns
for data pipeline components. These work with the interfaces defined in
/interfaces/data_pipeline/ to provide a complete architecture.
"""

from .base_processor import BaseProcessor
from .base_manager import BaseManager
from .base_service import BaseService

__all__ = [
    'BaseProcessor',
    'BaseManager', 
    'BaseService'
]
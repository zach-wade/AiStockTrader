"""
Base Repository Implementation

Legacy compatibility layer that imports the new modular repository architecture.
For new code, use BaseRepositoryCoordinator directly.
"""

# Import the new modular implementation
from .base_repository_coordinator import BaseRepositoryCoordinator

# For backward compatibility, alias the new coordinator as BaseRepository
BaseRepository = BaseRepositoryCoordinator

# Re-export all the supporting classes and types
from .repository_core_operations import RepositoryCoreOperations
from .repository_patterns import CacheMixin, DualStorageMixin, MetricsMixin, RepositoryMixin
from .repository_query_builder import RepositoryQueryBuilder
from .repository_query_processor import RepositoryQueryProcessor

__all__ = [
    "BaseRepository",
    "BaseRepositoryCoordinator",
    "RepositoryMixin",
    "DualStorageMixin",
    "CacheMixin",
    "MetricsMixin",
    "RepositoryQueryBuilder",
    "RepositoryCoreOperations",
    "RepositoryQueryProcessor",
]

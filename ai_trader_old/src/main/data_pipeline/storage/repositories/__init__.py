"""
Storage Repositories

Interface-based repository implementations for data storage operations.
"""

from .base_repository import BaseRepository
from .company_repository import CompanyRepository
from .feature_repository import FeatureRepository
from .financials_repository import FinancialsRepository

# Core repositories
from .market_data_repository import MarketDataRepository
from .news_repository import NewsRepository
from .repository_factory import RepositoryFactory, create_repository, get_repository_factory
from .repository_patterns import CacheMixin, DualStorageMixin, MetricsMixin, RepositoryMixin
from .repository_provider import (
    IRepositoryProvider,
    RepositoryProvider,
    RepositoryServiceLocator,
    create_repository_provider,
    initialize_repository_service_locator,
)
from .repository_types import (
    OperationResult,
    QueryFilter,
    RepositoryConfig,
    TimeRange,
    TransactionStrategy,
    ValidationLevel,
    create_time_range,
)
from .scanner_data_repository import ScannerDataRepository

# Specialized repositories
from .specialized_repositories import (
    DividendsRepository,
    GuidanceRepository,
    RatingsRepository,
    SentimentRepository,
    SocialSentimentRepository,
)

__all__ = [
    # Types
    "RepositoryConfig",
    "QueryFilter",
    "OperationResult",
    "ValidationLevel",
    "TransactionStrategy",
    "TimeRange",
    "create_time_range",
    # Patterns
    "RepositoryMixin",
    "DualStorageMixin",
    "CacheMixin",
    "MetricsMixin",
    # Core infrastructure
    "BaseRepository",
    "RepositoryFactory",
    "get_repository_factory",
    "create_repository",
    "RepositoryProvider",
    "RepositoryServiceLocator",
    "IRepositoryProvider",
    "create_repository_provider",
    "initialize_repository_service_locator",
    # Core repositories
    "MarketDataRepository",
    "CompanyRepository",
    "FeatureRepository",
    "ScannerDataRepository",
    "NewsRepository",
    "FinancialsRepository",
    # Specialized repositories
    "SentimentRepository",
    "RatingsRepository",
    "DividendsRepository",
    "SocialSentimentRepository",
    "GuidanceRepository",
]

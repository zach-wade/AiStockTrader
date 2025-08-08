"""
Storage Repositories

Interface-based repository implementations for data storage operations.
"""

from .repository_types import (
    RepositoryConfig,
    QueryFilter,
    OperationResult,
    ValidationLevel,
    TransactionStrategy,
    TimeRange,
    create_time_range
)

from .repository_patterns import (
    RepositoryMixin,
    DualStorageMixin,
    CacheMixin,
    MetricsMixin
)

from .base_repository import BaseRepository
from .repository_factory import RepositoryFactory, get_repository_factory, create_repository
from .repository_provider import (
    RepositoryProvider, 
    RepositoryServiceLocator,
    IRepositoryProvider,
    create_repository_provider,
    initialize_repository_service_locator
)

# Core repositories
from .market_data_repository import MarketDataRepository
from .company_repository import CompanyRepository
from .feature_repository import FeatureRepository
from .scanner_data_repository import ScannerDataRepository
from .news_repository import NewsRepository
from .financials_repository import FinancialsRepository

# Specialized repositories
from .specialized_repositories import (
    SentimentRepository,
    RatingsRepository,
    DividendsRepository,
    SocialSentimentRepository,
    GuidanceRepository
)

__all__ = [
    # Types
    'RepositoryConfig',
    'QueryFilter', 
    'OperationResult',
    'ValidationLevel',
    'TransactionStrategy',
    'TimeRange',
    'create_time_range',
    
    # Patterns
    'RepositoryMixin',
    'DualStorageMixin',
    'CacheMixin',
    'MetricsMixin',
    
    # Core infrastructure
    'BaseRepository',
    'RepositoryFactory',
    'get_repository_factory',
    'create_repository',
    'RepositoryProvider',
    'RepositoryServiceLocator',
    'IRepositoryProvider',
    'create_repository_provider',
    'initialize_repository_service_locator',
    
    # Core repositories
    'MarketDataRepository',
    'CompanyRepository',
    'FeatureRepository',
    'ScannerDataRepository',
    'NewsRepository',
    'FinancialsRepository',
    
    # Specialized repositories
    'SentimentRepository',
    'RatingsRepository',
    'DividendsRepository',
    'SocialSentimentRepository',
    'GuidanceRepository'
]
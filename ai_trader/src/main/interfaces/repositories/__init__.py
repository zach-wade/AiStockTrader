"""
Repository Interfaces

Comprehensive interface definitions for all data repositories
in the data pipeline system.
"""

from .base import (
    IRepository,
    IRepositoryFactory,
    IRepositoryProvider,
    RepositoryConfig,
    QueryFilter,
    OperationResult
)

from .market_data import IMarketDataRepository
from .company import ICompanyRepository
from .feature import IFeatureRepository
from .sentiment import ISentimentRepository
from .financials import (
    IFinancialsRepository,
    IGuidanceRepository,
    IRatingsRepository,
    IDividendsRepository
)
from .scanner import IScannerDataRepository
from .social import ISocialSentimentRepository
from .news import INewsRepository

__all__ = [
    # Base interfaces
    'IRepository',
    'IRepositoryFactory',
    'IRepositoryProvider',
    'RepositoryConfig',
    'QueryFilter',
    'OperationResult',
    
    # Specialized repositories
    'IMarketDataRepository',
    'ICompanyRepository',
    'IFeatureRepository',
    'ISentimentRepository',
    'IFinancialsRepository',
    'IGuidanceRepository',
    'IRatingsRepository',
    'IDividendsRepository',
    'IScannerDataRepository',
    'ISocialSentimentRepository',
    'INewsRepository'
]
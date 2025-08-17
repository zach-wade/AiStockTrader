"""
Repository Interfaces

Comprehensive interface definitions for all data repositories
in the data pipeline system.
"""

from .base import (
    IRepository,
    IRepositoryFactory,
    IRepositoryProvider,
    OperationResult,
    QueryFilter,
    RepositoryConfig,
)
from .company import ICompanyRepository
from .feature import IFeatureRepository
from .financials import (
    IDividendsRepository,
    IFinancialsRepository,
    IGuidanceRepository,
    IRatingsRepository,
)
from .market_data import IMarketDataRepository
from .news import INewsRepository
from .scanner import IScannerDataRepository
from .sentiment import ISentimentRepository
from .social import ISocialSentimentRepository

__all__ = [
    # Base interfaces
    "IRepository",
    "IRepositoryFactory",
    "IRepositoryProvider",
    "RepositoryConfig",
    "QueryFilter",
    "OperationResult",
    # Specialized repositories
    "IMarketDataRepository",
    "ICompanyRepository",
    "IFeatureRepository",
    "ISentimentRepository",
    "IFinancialsRepository",
    "IGuidanceRepository",
    "IRatingsRepository",
    "IDividendsRepository",
    "IScannerDataRepository",
    "ISocialSentimentRepository",
    "INewsRepository",
]

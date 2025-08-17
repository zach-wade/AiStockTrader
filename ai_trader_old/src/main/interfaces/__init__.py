"""
Interfaces Module

This module defines contracts and protocols used across the AI Trader system
to ensure clean architecture and prevent circular dependencies.

Key interfaces:
- Calculator interfaces for feature engineering
- Data source interfaces for data pipeline
- Trading interfaces for order execution
"""

from .backtesting import (
    BacktestConfig,
    BacktestMode,
    BacktestResult,
    IBacktestEngine,
    IBacktestEngineFactory,
    IPerformanceMetrics,
    IStrategy,
)
from .calculators import (
    CalculatorConfig,
    FeatureResult,
    ICalculatorFactory,
    IFeatureCalculator,
    ISentimentCalculator,
    ITechnicalCalculator,
)
from .database import IAsyncDatabase, IDatabase, IDatabaseFactory, IDatabasePool
from .ingestion import (
    BulkLoadConfig,
    BulkLoadResult,
    IBulkLoader,
    IBulkLoaderFactory,
    IDataTransformer,
    IIngestionClient,
    IIngestionOrchestrator,
    LoadStrategy,
)
from .repositories import IFeatureRepository, IMarketDataRepository, IRepository, IRepositoryFactory

__all__ = [
    # Calculator interfaces
    "IFeatureCalculator",
    "ITechnicalCalculator",
    "ISentimentCalculator",
    "ICalculatorFactory",
    "CalculatorConfig",
    "FeatureResult",
    # Repository interfaces
    "IRepository",
    "IMarketDataRepository",
    "IFeatureRepository",
    "IRepositoryFactory",
    # Backtesting interfaces
    "BacktestMode",
    "BacktestConfig",
    "BacktestResult",
    "IBacktestEngine",
    "IBacktestEngineFactory",
    "IStrategy",
    "IPerformanceMetrics",
    # Database interfaces
    "IDatabase",
    "IAsyncDatabase",
    "IDatabasePool",
    "IDatabaseFactory",
    # Ingestion interfaces
    "IBulkLoader",
    "IBulkLoaderFactory",
    "IIngestionClient",
    "IIngestionOrchestrator",
    "IDataTransformer",
    "BulkLoadConfig",
    "BulkLoadResult",
    "LoadStrategy",
]

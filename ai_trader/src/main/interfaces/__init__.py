"""
Interfaces Module

This module defines contracts and protocols used across the AI Trader system
to ensure clean architecture and prevent circular dependencies.

Key interfaces:
- Calculator interfaces for feature engineering
- Data source interfaces for data pipeline
- Trading interfaces for order execution
"""

from .calculators import (
    IFeatureCalculator,
    ITechnicalCalculator,
    ISentimentCalculator,
    ICalculatorFactory,
    CalculatorConfig,
    FeatureResult
)

from .repositories import (
    IRepository,
    IMarketDataRepository,
    IFeatureRepository,
    IRepositoryFactory
)

from .backtesting import (
    BacktestMode,
    BacktestConfig,
    BacktestResult,
    IBacktestEngine,
    IBacktestEngineFactory,
    IStrategy,
    IPerformanceMetrics
)

from .database import (
    IDatabase,
    IAsyncDatabase,
    IDatabasePool,
    IDatabaseFactory
)

from .ingestion import (
    IBulkLoader,
    IBulkLoaderFactory,
    IIngestionClient,
    IIngestionOrchestrator,
    IDataTransformer,
    BulkLoadConfig,
    BulkLoadResult,
    LoadStrategy
)

__all__ = [
    # Calculator interfaces
    'IFeatureCalculator',
    'ITechnicalCalculator', 
    'ISentimentCalculator',
    'ICalculatorFactory',
    'CalculatorConfig',
    'FeatureResult',
    
    # Repository interfaces
    'IRepository',
    'IMarketDataRepository',
    'IFeatureRepository',
    'IRepositoryFactory',
    
    # Backtesting interfaces
    'BacktestMode',
    'BacktestConfig',
    'BacktestResult',
    'IBacktestEngine',
    'IBacktestEngineFactory',
    'IStrategy',
    'IPerformanceMetrics',
    
    # Database interfaces
    'IDatabase',
    'IAsyncDatabase',
    'IDatabasePool',
    'IDatabaseFactory',
    
    # Ingestion interfaces
    'IBulkLoader',
    'IBulkLoaderFactory',
    'IIngestionClient',
    'IIngestionOrchestrator',
    'IDataTransformer',
    'BulkLoadConfig',
    'BulkLoadResult',
    'LoadStrategy'
]
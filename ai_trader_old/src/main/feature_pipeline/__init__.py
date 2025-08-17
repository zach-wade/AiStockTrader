"""
Feature Pipeline Module

This module handles all feature engineering and calculation for the AI Trader system.
Provides comprehensive feature calculation capabilities across technical, fundamental,
sentiment, and alternative data sources.

Components:
- feature_orchestrator: Main orchestration logic for coordinating feature calculations
- unified_feature_engine: Unified interface for all feature calculations
- feature_store: Feature storage and retrieval with PostgreSQL backend
- feature_adapter: Bridge between features and trading strategies
- data_preprocessor: Data cleaning and normalization utilities
- dataloader: Unified data loading interface
- feature_config: Configuration management for feature calculations
- calculators/: Specialized feature calculation modules
"""

from .data_preprocessor import DataPreprocessor
from .dataloader import DataLoader
from .feature_adapter import FeatureAdapter
from .feature_config import FeatureConfig
from .feature_orchestrator import FeatureOrchestrator
from .feature_store import FeatureStoreRepository

# Import compatibility wrapper for legacy code
from .feature_store_compat import FeatureStore

# Import calculator module for access to all calculators
from . import calculators
from .unified_feature_engine import UnifiedFeatureEngine

__all__ = [
    # Core components
    "FeatureOrchestrator",
    "UnifiedFeatureEngine",
    "FeatureStoreRepository",
    "FeatureStore",  # Compatibility wrapper
    "FeatureAdapter",
    # Data handling
    "DataPreprocessor",
    "DataLoader",
    "FeatureConfig",
    # Calculator modules
    "calculators",
    # Legacy compatibility
    "feature_orchestrator",
    "unified_feature_engine",
    "feature_store",
    "feature_adapter",
]

# Legacy aliases for backward compatibility
feature_orchestrator = FeatureOrchestrator
unified_feature_engine = UnifiedFeatureEngine
feature_store = FeatureStoreRepository
feature_adapter = FeatureAdapter

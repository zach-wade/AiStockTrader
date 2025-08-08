"""
Data Pipeline Core Enumerations

Defines the layer-based architecture (0-3) that replaces the previous tier system,
along with other core enumerations for data types, priorities, and policies.

Key Architectural Concepts:
- DataLayer (0-3): API rate limiting and data collection frequency layers
- Storage Tiers (hot/cold): Data temperature management for recent vs archived data
- Cache Tiers (memory/redis): Backend selection for caching infrastructure

These are separate, orthogonal concepts that work together in the unified architecture.
"""

from enum import Enum, IntEnum
from typing import Dict, List, Any
from main.utils.core import get_logger

logger = get_logger(__name__)


class DataLayer(IntEnum):
    """
    Layer-based architecture replacing the tier system.
    
    Each layer defines data retention policies, update frequencies,
    and processing priorities based on symbol importance and trading activity.
    """
    
    # Layer 0: Basic Tradable Symbols (~10,000 symbols)
    BASIC = 0
    
    # Layer 1: Liquid Symbols (~2,000 symbols) 
    LIQUID = 1
    
    # Layer 2: Catalyst-Driven Symbols (~500 symbols)
    CATALYST = 2
    
    # Layer 3: Active Trading Symbols (~50 symbols)
    ACTIVE = 3
    
    @property
    def retention_days(self) -> int:
        """Get retention period in days for this layer."""
        retention_map = {
            DataLayer.BASIC: 30,      # 1 month
            DataLayer.LIQUID: 365,    # 1 year
            DataLayer.CATALYST: 730,  # 2 years
            DataLayer.ACTIVE: 1825    # 5 years
        }
        return retention_map[self]
    
    @property
    def hot_storage_days(self) -> int:
        """Get hot storage period in days for this layer."""
        hot_storage_map = {
            DataLayer.BASIC: 7,       # 1 week
            DataLayer.LIQUID: 30,     # 1 month
            DataLayer.CATALYST: 60,   # 2 months
            DataLayer.ACTIVE: 90      # 3 months
        }
        return hot_storage_map[self]
    
    @property
    def supported_intervals(self) -> List[str]:
        """Get supported data intervals for this layer."""
        interval_map = {
            DataLayer.BASIC: ['daily'],
            DataLayer.LIQUID: ['daily', 'hourly', '5min'],
            DataLayer.CATALYST: ['daily', 'hourly', '5min', '1min'],
            DataLayer.ACTIVE: ['daily', 'hourly', '5min', '1min', 'tick']
        }
        return interval_map[self]
    
    @property
    def max_symbols(self) -> int:
        """Get maximum symbols for this layer."""
        symbol_limits = {
            DataLayer.BASIC: 10000,
            DataLayer.LIQUID: 2000,
            DataLayer.CATALYST: 500,
            DataLayer.ACTIVE: 50
        }
        return symbol_limits[self]
    
    @property
    def description(self) -> str:
        """Get human-readable description of this layer."""
        descriptions = {
            DataLayer.BASIC: "Basic Tradable - Daily data, basic retention",
            DataLayer.LIQUID: "Liquid Symbols - Hourly data, extended retention", 
            DataLayer.CATALYST: "Catalyst-Driven - High frequency data, long retention",
            DataLayer.ACTIVE: "Active Trading - All intervals, maximum retention"
        }
        return descriptions[self]


class DataType(Enum):
    """Data types supported by the pipeline."""
    
    MARKET_DATA = "market_data"
    NEWS = "news"
    SOCIAL = "social"
    FINANCIALS = "financials"
    CORPORATE_ACTIONS = "corporate_actions"
    RATINGS = "ratings"
    DIVIDENDS = "dividends"
    REALTIME = "realtime"
    
    @property
    def table_name(self) -> str:
        """Get database table name for this data type."""
        table_map = {
            DataType.MARKET_DATA: "market_data_1h",
            DataType.NEWS: "news_data",
            DataType.SOCIAL: "social_data", 
            DataType.FINANCIALS: "financials_data",
            DataType.CORPORATE_ACTIONS: "corporate_actions_data",
            DataType.RATINGS: "ratings_data",
            DataType.DIVIDENDS: "dividends_data",
            DataType.REALTIME: "realtime_data"
        }
        return table_map[self]
    
    @property
    def supports_intervals(self) -> bool:
        """Check if this data type supports time intervals."""
        interval_types = {
            DataType.MARKET_DATA, 
            DataType.REALTIME
        }
        return self in interval_types


class ProcessingPriority(IntEnum):
    """Processing priority levels for data pipeline operations."""
    
    LOW = 1
    NORMAL = 2  
    HIGH = 3
    CRITICAL = 4
    
    @property
    def description(self) -> str:
        """Get description of this priority level."""
        descriptions = {
            ProcessingPriority.LOW: "Low priority - batch processing",
            ProcessingPriority.NORMAL: "Normal priority - standard processing",
            ProcessingPriority.HIGH: "High priority - expedited processing", 
            ProcessingPriority.CRITICAL: "Critical priority - immediate processing"
        }
        return descriptions[self]


class RetentionPolicy(Enum):
    """Data retention policies for different storage tiers."""
    
    HOT_ONLY = "hot_only"        # Keep in hot storage only
    HOT_COLD = "hot_cold"        # Move to cold storage after hot period
    ARCHIVE = "archive"          # Move to archive after retention period
    PERMANENT = "permanent"      # Keep permanently
    
    @property
    def description(self) -> str:
        """Get description of this retention policy."""
        descriptions = {
            RetentionPolicy.HOT_ONLY: "Hot storage only - deleted after hot period",
            RetentionPolicy.HOT_COLD: "Hot then cold storage - dual tier retention",
            RetentionPolicy.ARCHIVE: "Archive after retention - compressed storage",
            RetentionPolicy.PERMANENT: "Permanent retention - never deleted"
        }
        return descriptions[self]


class IngestionStatus(Enum):
    """Status values for data ingestion operations."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ValidationStatus(Enum):
    """Status values for data validation operations."""
    
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    SKIPPED = "skipped"
    FAILED = "failed"


class StorageStatus(Enum):
    """Status values for data storage operations."""
    
    STORED = "stored"
    ARCHIVED = "archived"
    FAILED = "failed"
    DUPLICATE = "duplicate"
    QUARANTINED = "quarantined"


def get_layer_config(layer: DataLayer) -> Dict[str, Any]:
    """
    Get comprehensive configuration for a data layer.
    
    Args:
        layer: The data layer to get configuration for
        
    Returns:
        Dictionary containing layer configuration
    """
    return {
        'layer': layer.value,
        'name': layer.name.lower(),
        'description': layer.description,
        'retention_days': layer.retention_days,
        'hot_storage_days': layer.hot_storage_days,
        'supported_intervals': layer.supported_intervals,
        'max_symbols': layer.max_symbols
    }


def get_all_layer_configs() -> Dict[int, Dict[str, Any]]:
    """Get configuration for all data layers."""
    return {
        layer.value: get_layer_config(layer) 
        for layer in DataLayer
    }
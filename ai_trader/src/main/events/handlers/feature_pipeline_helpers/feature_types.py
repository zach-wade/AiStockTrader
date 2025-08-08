"""
Feature type definitions for the feature pipeline.

This module contains the core data types used throughout the feature
computation pipeline including enums, dataclasses, and type definitions.
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from main.events.types import AlertType


class FeatureGroup(Enum):
    """Feature groups available for computation."""
    # Market data features
    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    TREND = "trend"
    
    # Technical indicators
    TECHNICAL_BASIC = "technical_basic"
    TECHNICAL_ADVANCED = "technical_advanced"
    
    # Market microstructure
    MICROSTRUCTURE = "microstructure"
    ORDER_FLOW = "order_flow"
    
    # Sentiment features
    NEWS_SENTIMENT = "news_sentiment"
    SOCIAL_SENTIMENT = "social_sentiment"
    
    # Risk features
    RISK_METRICS = "risk_metrics"
    CORRELATION = "correlation"
    
    # Event-driven features
    EARNINGS = "earnings"
    CORPORATE_ACTIONS = "corporate_actions"
    
    # Machine learning features
    ML_SIGNALS = "ml_signals"
    EMBEDDINGS = "embeddings"


@dataclass
class FeatureGroupConfig:
    """Configuration for a feature group."""
    name: str
    description: str
    required_data: List[str] = field(default_factory=list)
    dependencies: List[FeatureGroup] = field(default_factory=list)
    computation_params: Dict[str, Any] = field(default_factory=dict)
    priority_boost: int = 0
    
    def __post_init__(self):
        """Validate configuration."""
        if self.priority_boost < -10 or self.priority_boost > 10:
            raise ValueError(f"Priority boost must be between -10 and 10, got {self.priority_boost}")


@dataclass
class FeatureRequest:
    """Request for feature computation."""
    symbol: str
    feature_groups: List[FeatureGroup]
    alert_type: AlertType
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if not self.feature_groups:
            raise ValueError("At least one feature group must be specified")
        if self.priority < 0 or self.priority > 10:
            raise ValueError(f"Priority must be between 0 and 10, got {self.priority}")
    
    def get_all_required_groups(self, configs: Dict[FeatureGroup, FeatureGroupConfig]) -> Set[FeatureGroup]:
        """Get all groups including dependencies."""
        all_groups = set()
        visited = set()
        
        def add_deps(group: FeatureGroup):
            if group in visited:
                return
            visited.add(group)
            all_groups.add(group)
            
            if group in configs:
                for dep in configs[group].dependencies:
                    add_deps(dep)
        
        for group in self.feature_groups:
            add_deps(group)
            
        return all_groups
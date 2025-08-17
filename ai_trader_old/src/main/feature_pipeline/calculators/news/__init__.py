"""
News Analysis Module

Provides comprehensive news-based feature calculators for market sentiment,
event detection, and information flow analysis. Each calculator focuses on
specific aspects of news data and its impact on trading decisions.
"""

# Standard library imports
from typing import List, Optional, Tuple

# Third-party imports
import pandas as pd

# Base class and configuration
from .base_news import BaseNewsCalculator
from .credibility_calculator import NewsCredibilityCalculator as CredibilityCalculator
from .event_calculator import NewsEventCalculator as EventCalculator
from .monetary_calculator import MonetaryImpactCalculator
from .news_config import (
    EventType,
    NewsConfig,
    SourceTier,
    create_comprehensive_news_config,
    create_default_news_config,
    create_fast_news_config,
)

# Unified news feature facade
from .news_feature_facade import NewsFeatureFacade

# Specialized news calculators
from .sentiment_calculator import SentimentCalculator
from .topic_calculator import NewsTopicCalculator as TopicCalculator
from .volume_calculator import VolumeCalculator

# Alias for backward compatibility
NewsFeatureCalculator = NewsFeatureFacade

__all__ = [
    # Base classes and configuration
    "BaseNewsCalculator",
    "NewsConfig",
    "SourceTier",
    "EventType",
    # Configuration factories
    "create_default_news_config",
    "create_fast_news_config",
    "create_comprehensive_news_config",
    # Specialized calculators
    "SentimentCalculator",
    "VolumeCalculator",
    "MonetaryImpactCalculator",
    "TopicCalculator",
    "EventCalculator",
    "CredibilityCalculator",
    # Facade
    "NewsFeatureFacade",
    "NewsFeatureCalculator",  # Backward compatibility alias
]

# Registry for news calculators
NEWS_CALCULATOR_REGISTRY = {
    "sentiment": SentimentCalculator,
    "volume": VolumeCalculator,
    "monetary": MonetaryImpactCalculator,
    "topic": TopicCalculator,
    "event": EventCalculator,
    "credibility": CredibilityCalculator,
    "facade": NewsFeatureFacade,
}

# Feature counts for each calculator
FEATURE_COUNTS = {
    "sentiment": 48,  # Sentiment scores, polarity, subjectivity, time-weighted
    "volume": 35,  # News count, velocity, acceleration, coverage metrics
    "monetary": 42,  # Price targets, estimate changes, impact scores
    "topic": 50,  # Dynamic topics, categories, entity mentions
    "event": 40,  # Event detection, breaking news, event clustering
    "credibility": 32,  # Source credibility, consensus, diversity metrics
}

# Total features across all calculators
TOTAL_NEWS_FEATURES = sum(FEATURE_COUNTS.values())  # 247 total features

# News metric categories
NEWS_CATEGORIES = {
    "sentiment_analysis": ["sentiment", "monetary"],
    "information_flow": ["volume", "credibility"],
    "content_analysis": ["topic", "event"],
}


def get_news_calculator(calculator_name: str, config: dict = None):
    """
    Get news calculator instance by name.

    Args:
        calculator_name: Name of calculator from registry
        config: Optional configuration dictionary

    Returns:
        Calculator instance

    Raises:
        ValueError: If calculator name not found
    """
    if calculator_name not in NEWS_CALCULATOR_REGISTRY:
        available = list(NEWS_CALCULATOR_REGISTRY.keys())
        raise ValueError(f"Calculator '{calculator_name}' not found. Available: {available}")

    calc_class = NEWS_CALCULATOR_REGISTRY[calculator_name]
    return calc_class(config)


def get_all_news_calculators(config: dict = None):
    """
    Get instances of all news calculators.

    Args:
        config: Optional configuration dictionary

    Returns:
        Dictionary mapping calculator names to instances
    """
    calculators = {}

    for name, calc_class in NEWS_CALCULATOR_REGISTRY.items():
        if name != "facade":  # Skip facade to avoid duplication
            try:
                calculators[name] = calc_class(config)
            except Exception as e:
                print(f"Warning: Failed to initialize {name}: {e}")
                continue

    return calculators


def get_news_feature_summary():
    """
    Get summary of features provided by each calculator.

    Returns:
        Dictionary with feature counts and descriptions
    """
    summary = {
        "total_features": TOTAL_NEWS_FEATURES,
        "calculator_counts": FEATURE_COUNTS.copy(),
        "descriptions": {
            "sentiment": "News sentiment analysis and emotional indicators (48 features)",
            "volume": "News flow volume and velocity metrics (35 features)",
            "monetary": "Monetary impact and analyst estimate analysis (42 features)",
            "topic": "Topic modeling and content categorization (50 features)",
            "event": "Event detection and breaking news analysis (40 features)",
            "credibility": "Source credibility and information quality (32 features)",
        },
        "categories": NEWS_CATEGORIES,
    }

    return summary


def calculate_news_impact_score(news_data: pd.DataFrame, config: dict | None = None):
    """
    Calculate composite news impact score.

    Args:
        news_data: DataFrame with news articles
        config: Optional configuration

    Returns:
        Series with impact scores
    """
    # Use facade for comprehensive calculation
    facade = NewsFeatureFacade(config)

    # Calculate features
    features = facade.calculate(news_data)

    # Composite score (weighted average of key metrics)
    impact_components = {
        "sentiment_impact": features.get("sentiment_composite_score", 0),
        "volume_impact": features.get("news_intensity_score", 0),
        "event_impact": features.get("event_composite_score", 0),
        "credibility_weight": features.get("credibility_weighted_score", 1),
    }

    # Calculate weighted impact
    impact_score = (
        impact_components["sentiment_impact"] * 0.3
        + impact_components["volume_impact"] * 0.2
        + impact_components["event_impact"] * 0.3
    ) * impact_components["credibility_weight"]

    return impact_score


# News data validation utilities
def validate_news_data(news_df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate news DataFrame has required columns and format.

    Args:
        news_df: News DataFrame to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    required_columns = {"timestamp", "headline", "source"}
    errors = []

    # Check required columns
    missing_cols = required_columns - set(news_df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check data types
    if "timestamp" in news_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(news_df["timestamp"]):
            errors.append("'timestamp' column must be datetime type")

    # Check for empty data
    if len(news_df) == 0:
        errors.append("News DataFrame is empty")

    # Check for null values in critical columns
    for col in required_columns:
        if col in news_df.columns and news_df[col].isnull().any():
            null_count = news_df[col].isnull().sum()
            errors.append(f"Column '{col}' has {null_count} null values")

    return len(errors) == 0, errors


# Source credibility presets
SOURCE_CREDIBILITY_TIERS = {
    "tier_1": {  # Most credible
        "reuters": 1.0,
        "bloomberg": 1.0,
        "wsj": 0.95,
        "ft": 0.95,
        "nyt": 0.9,
        "economist": 0.9,
    },
    "tier_2": {  # Credible
        "cnbc": 0.8,
        "marketwatch": 0.8,
        "barrons": 0.8,
        "forbes": 0.75,
        "businessinsider": 0.7,
    },
    "tier_3": {  # Moderate credibility
        "seekingalpha": 0.6,
        "yahoo": 0.6,
        "benzinga": 0.5,
        "investorplace": 0.5,
    },
    "tier_4": {"reddit": 0.3, "twitter": 0.3, "stocktwits": 0.3, "other": 0.4},  # Lower credibility
}


def get_source_credibility(source: str) -> float:
    """
    Get credibility score for a news source.

    Args:
        source: News source name

    Returns:
        Credibility score between 0 and 1
    """
    source_lower = source.lower()

    # Check each tier
    for tier, sources in SOURCE_CREDIBILITY_TIERS.items():
        for src, score in sources.items():
            if src in source_lower:
                return score

    # Default for unknown sources
    return 0.4


# Version information
__version__ = "2.0.0"
__author__ = "AI Trader News Analysis Team"

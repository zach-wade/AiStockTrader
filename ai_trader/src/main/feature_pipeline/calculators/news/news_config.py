"""
News Analysis Configuration

Centralized configuration for all news-based feature calculations.
Contains parameters for time windows, sentiment analysis, topic modeling,
event detection, and source credibility weighting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class SourceTier(Enum):
    """News source credibility tiers."""
    TIER1 = "tier1"  # Major financial news sources (Bloomberg, Reuters, WSJ)
    TIER2 = "tier2"  # Secondary sources (MarketWatch, Seeking Alpha)
    TIER3 = "tier3"  # Blogs and smaller sources


class EventType(Enum):
    """Types of news events."""
    EARNINGS = "earnings"
    MERGER = "merger"
    REGULATORY = "regulatory"
    PRODUCT = "product"
    MANAGEMENT = "management"
    MARKET = "market"
    OTHER = "other"


@dataclass
class NewsConfig:
    """Configuration for news feature calculations."""
    
    # Time window parameters
    time_windows: List[int] = field(default_factory=lambda: [1, 6, 24, 72, 168])  # Hours: 1h, 6h, 1d, 3d, 1w
    
    # News decay parameters
    decay_half_life: float = 24.0  # Hours - news relevance decay
    
    # Volume and velocity parameters
    volume_spike_threshold: float = 3.0  # Multiple of average for spike detection
    velocity_smoothing_window: int = 3  # Points for velocity smoothing
    
    # Sentiment analysis parameters
    sentiment_smoothing: int = 3  # Number of articles for sentiment smoothing
    sentiment_threshold_positive: float = 0.1  # Threshold for positive sentiment
    sentiment_threshold_negative: float = -0.1  # Threshold for negative sentiment
    subjectivity_threshold: float = 0.5  # Threshold for subjective content
    
    # Topic modeling parameters
    n_topics: int = 10  # Number of topics to extract
    min_topic_words: int = 5  # Minimum words per topic
    topic_min_docs: int = 3  # Minimum documents for topic
    tfidf_max_features: int = 100  # Maximum TF-IDF features
    tfidf_min_df: int = 1  # Minimum document frequency
    tfidf_max_df: float = 0.95  # Maximum document frequency
    ngram_range: tuple = (1, 2)  # N-gram range for TF-IDF
    
    # Event detection parameters
    event_threshold: float = 2.0  # Standard deviations for event detection
    event_min_articles: int = 3  # Minimum articles for event
    event_time_cluster: int = 2  # Hours for clustering events
    anomaly_detection_window: int = 24  # Hours for anomaly detection baseline
    
    # Breaking news parameters
    breaking_news_keywords: List[str] = field(default_factory=lambda: [
        'breaking', 'urgent', 'alert', 'emergency', 'crash', 'surge', 
        'plunge', 'spike', 'halt', 'suspend', 'investigation', 'acquisition',
        'merger', 'bankruptcy', 'earnings', 'guidance', 'outlook'
    ])
    breaking_news_recency: int = 1  # Hours for breaking news classification
    breaking_impact_decay: float = 6.0  # Hours for breaking news impact decay
    
    # Monetary impact parameters
    price_correlation_window: int = 24  # Hours for price correlation analysis
    momentum_calculation_periods: List[int] = field(default_factory=lambda: [1, 6, 24])  # Hours
    impact_measurement_delay: int = 2  # Hours delay for measuring news impact
    
    # Source credibility parameters
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        # Tier 1 - Premium financial sources
        'reuters': 1.0,
        'bloomberg': 1.0,
        'dow jones': 1.0,
        'financial times': 1.0,
        
        # Tier 2 - Major financial media
        'wsj': 0.9,
        'wall street journal': 0.9,
        'cnbc': 0.85,
        'marketwatch': 0.8,
        'yahoo finance': 0.75,
        
        # Tier 3 - Financial news sites
        'benzinga': 0.7,
        'investing.com': 0.65,
        'seekingalpha': 0.6,
        'motley fool': 0.55,
        'zacks': 0.5,
        
        # Tier 4 - Social/Alternative sources
        'reddit': 0.4,
        'stocktwits': 0.35,
        'twitter': 0.3,
        'discord': 0.25,
        
        # Default for unknown sources
        'unknown': 0.5
    })
    
    # Credibility calculation parameters
    credibility_recency_weight: float = 0.3  # Weight for recent source performance
    credibility_volume_weight: float = 0.2  # Weight for source volume
    credibility_accuracy_weight: float = 0.5  # Weight for historical accuracy
    credibility_update_window: int = 168  # Hours for credibility updates (1 week)
    
    # Text processing parameters
    text_min_length: int = 10  # Minimum text length for processing
    text_max_length: int = 5000  # Maximum text length (truncate longer)
    stop_words_custom: List[str] = field(default_factory=lambda: [
        'stock', 'stocks', 'share', 'shares', 'price', 'market', 'trading',
        'investor', 'investors', 'company', 'companies', 'business'
    ])
    
    # Performance optimization parameters
    batch_size: int = 1000  # Batch size for processing large news datasets
    cache_sentiment_results: bool = True  # Cache sentiment analysis results
    parallel_processing: bool = False  # Enable parallel processing (experimental)
    max_news_articles_per_window: int = 500  # Limit articles per time window
    
    # Feature engineering parameters
    feature_lag_periods: List[int] = field(default_factory=lambda: [0, 1, 6, 24])  # Hours for lagged features
    rolling_statistics_windows: List[int] = field(default_factory=lambda: [6, 24, 72])  # Hours for rolling stats
    percentile_levels: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])  # Percentiles
    
    # Quality control parameters
    min_articles_for_analysis: int = 1  # Minimum articles needed for analysis
    max_missing_data_ratio: float = 0.5  # Maximum ratio of missing data allowed
    outlier_detection_threshold: float = 3.0  # Z-score threshold for outlier detection
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate time windows are positive
        if not all(w > 0 for w in self.time_windows):
            raise ValueError("All time windows must be positive")
        
        # Validate thresholds are reasonable
        if not -1.0 <= self.sentiment_threshold_negative <= self.sentiment_threshold_positive <= 1.0:
            raise ValueError("Sentiment thresholds must be between -1 and 1, with negative <= positive")
        
        # Validate source weights are non-negative
        if not all(w >= 0 for w in self.source_weights.values()):
            raise ValueError("Source weights must be non-negative")
        
        # Ensure time windows are sorted
        self.time_windows = sorted(self.time_windows)
        
        # Validate credibility weights sum to 1
        total_credibility_weight = (
            self.credibility_recency_weight + 
            self.credibility_volume_weight + 
            self.credibility_accuracy_weight
        )
        if abs(total_credibility_weight - 1.0) > 1e-6:
            raise ValueError("Credibility weights must sum to 1.0")
    
    def get_time_window_names(self) -> List[str]:
        """Get human-readable names for time windows."""
        names = []
        for hours in self.time_windows:
            if hours < 24:
                names.append(f"{hours}h")
            elif hours < 168:
                days = hours // 24
                names.append(f"{days}d")
            else:
                weeks = hours // 168
                names.append(f"{weeks}w")
        return names
    
    def get_source_weight(self, source: str) -> float:
        """Get weight for a specific news source."""
        return self.source_weights.get(source.lower(), self.source_weights.get('unknown', 0.5))
    
    def is_breaking_news(self, text: str, timestamp_hours_ago: float) -> bool:
        """Determine if news qualifies as breaking news."""
        if timestamp_hours_ago > self.breaking_news_recency:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.breaking_news_keywords)
    
    def calculate_time_decay_weight(self, hours_ago: float) -> float:
        """Calculate time decay weight for news article."""
        return max(0.0, 2.0 ** (-hours_ago / self.decay_half_life))
    
    def get_sentiment_category(self, polarity: float) -> str:
        """Categorize sentiment based on polarity score."""
        if polarity >= self.sentiment_threshold_positive:
            return 'positive'
        elif polarity <= self.sentiment_threshold_negative:
            return 'negative'
        else:
            return 'neutral'

# Factory functions for creating configurations
def create_default_news_config() -> NewsConfig:
    """Create default news configuration."""
    return NewsConfig()

def create_fast_news_config() -> NewsConfig:
    """Create configuration optimized for speed."""
    config = NewsConfig()
    config.time_windows = [1, 24]  # Only 1h and 1d
    config.n_topics = 5  # Fewer topics
    config.tfidf_max_features = 50  # Fewer features
    return config

def create_comprehensive_news_config() -> NewsConfig:
    """Create comprehensive configuration for detailed analysis."""
    config = NewsConfig()
    config.time_windows = [1, 3, 6, 12, 24, 48, 72, 168]  # More time windows
    config.n_topics = 20  # More topics
    config.tfidf_max_features = 200  # More features
    return config

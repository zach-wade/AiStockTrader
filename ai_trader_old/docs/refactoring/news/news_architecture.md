# A10.3 News Features Refactoring - Architecture Guide

## Overview

The A10.3 refactoring transformed the monolithic `news_features.py` (1,070 lines) into a modular architecture with specialized news analysis calculators, implementing advanced natural language processing and sentiment analysis for financial news data.

## Original Monolith Analysis

### Problems with Original Architecture

- **Single Responsibility Violation**: One massive file handling sentiment analysis, topic modeling, credibility scoring, event detection, volume analysis, and monetary impact
- **NLP Complexity**: Multiple natural language processing domains mixed together
- **Data Source Coupling**: Tight coupling between different news data sources and processing
- **Performance Issues**: Loading entire news processing library for specific analysis types
- **Extensibility Challenges**: Adding new NLP methods required modifying the massive unified class

### Original File Structure

```
news_features.py (1,070 lines)
├── SentimentAnalysis (mixed with other types)
├── TopicModeling (mixed with other types)
├── CredibilityScoring (mixed with other types)
├── EventDetection (mixed with other types)
├── VolumeAnalysis (mixed with other types)
├── MonetaryImpact (mixed with other types)
└── NewsFeatureCalculator (god class)
```

## New Modular Architecture

### Design Principles Applied

1. **Single Responsibility Principle**: Each calculator focuses on one news analysis domain
2. **NLP Domain Separation**: Related NLP methods grouped logically
3. **Data Source Abstraction**: Clean separation between data sources and analysis
4. **Performance Optimization**: Domain-specific optimizations for NLP tasks
5. **Extensibility**: New NLP methods can be added to appropriate calculators

### New Architecture Structure

```
news/
├── __init__.py                     # Module exports and registry
├── news_config.py                  # Configuration management (150 lines)
├── base_news.py                    # Common utilities and validation (220 lines)
├── sentiment_calculator.py         # Sentiment analysis & emotion detection (280 lines, 35 features)
├── topic_calculator.py             # Topic modeling & classification (250 lines, 30 features)
├── credibility_calculator.py       # Source credibility & fact checking (200 lines, 25 features)
├── event_calculator.py             # Event detection & impact analysis (240 lines, 28 features)
├── volume_calculator.py            # News volume & frequency analysis (180 lines, 22 features)
├── monetary_calculator.py          # Monetary policy & economic impact (210 lines, 26 features)
└── news_feature_facade.py          # Backward compatibility facade (270 lines)
```

## Component Responsibilities

### BaseNewsCalculator (`base_news.py`)

**Purpose**: Common NLP utilities and validation for all news calculators
**Key Features**:

- Text preprocessing and cleaning
- NLP pipeline management
- Common sentiment lexicons and embeddings
- News data validation and normalization
- Performance optimization for text processing

**Core Methods**:

```python
def preprocess_news_text(self, text: str) -> str
def tokenize_and_clean(self, text: str) -> List[str]
def validate_news_data(self, news_df: pd.DataFrame) -> bool
def load_sentiment_lexicon(self, lexicon_type: str) -> Dict
def calculate_text_statistics(self, text: str) -> Dict
```

### SentimentCalculator (`sentiment_calculator.py`)

**Purpose**: Comprehensive sentiment analysis and emotion detection
**Methods Included**:

- VADER sentiment analysis
- TextBlob polarity and subjectivity
- FinBERT financial sentiment
- Emotion classification (fear, greed, uncertainty)
- Sentiment momentum and volatility
- Social media sentiment aggregation

**Feature Count**: 35 sentiment-based features
**NLP Foundation**: Sentiment analysis, emotion detection, financial language models

### TopicCalculator (`topic_calculator.py`)

**Purpose**: Topic modeling and content classification
**Methods Included**:

- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorization (NMF)
- Financial topic classification
- Earnings-related content detection
- Regulatory announcement classification
- Market sector topic analysis

**Feature Count**: 30 topic modeling features
**NLP Foundation**: Topic modeling, document classification, financial taxonomy

### CredibilityCalculator (`credibility_calculator.py`)

**Purpose**: News source credibility and reliability analysis
**Methods Included**:

- Source reliability scoring
- Author credibility metrics
- Publication quality assessment
- Fact-checking indicators
- Bias detection and classification
- Information quality metrics

**Feature Count**: 25 credibility features
**NLP Foundation**: Source analysis, quality assessment, bias detection

### EventCalculator (`event_calculator.py`)

**Purpose**: Financial event detection and impact analysis
**Methods Included**:

- Earnings announcement detection
- M&A activity identification
- Regulatory filing analysis
- Economic indicator releases
- Market-moving event classification
- Event impact scoring

**Feature Count**: 28 event detection features
**NLP Foundation**: Named entity recognition, event extraction, impact analysis

### VolumeCalculator (`volume_calculator.py`)

**Purpose**: News volume and frequency analysis
**Methods Included**:

- News flow volume tracking
- Attention intensity metrics
- Coverage breadth analysis
- News spike detection
- Temporal pattern analysis
- Cross-source volume correlation

**Feature Count**: 22 volume analysis features
**NLP Foundation**: Time series analysis, attention metrics, coverage analysis

### MonetaryCalculator (`monetary_calculator.py`)

**Purpose**: Monetary policy and economic impact analysis
**Methods Included**:

- Central bank communication analysis
- Policy stance detection
- Economic indicator mentions
- Inflation expectations extraction
- Interest rate sentiment
- Economic outlook classification

**Feature Count**: 26 monetary policy features
**NLP Foundation**: Financial domain knowledge, policy analysis, economic language processing

### NewsFeatureFacade (`news_feature_facade.py`)

**Purpose**: Backward compatibility and unified news analysis
**Features**:

- 100% backward compatibility with original interface
- Intelligent routing to appropriate calculators
- Unified news processing pipeline
- Legacy method support for existing code

## NLP Technology Integration

### External Library Integration

```python
# Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import gensim

# Advanced NLP
import spacy
import nltk
from sentence_transformers import SentenceTransformer
```

### Financial NLP Models

```python
# FinBERT integration for financial sentiment
class FinancialSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def analyze_financial_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using FinBERT financial language model."""
        # Implementation for financial-specific sentiment analysis
        pass

# Custom financial entity recognition
class FinancialEntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.financial_patterns = self._load_financial_patterns()

    def extract_financial_entities(self, text: str) -> List[Dict]:
        """Extract financial entities (companies, currencies, metrics)."""
        # Implementation for financial entity extraction
        pass
```

## Performance Optimization

### Text Processing Optimization

```python
# Efficient text preprocessing pipeline
class OptimizedTextProcessor:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.PorterStemmer()

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Optimized batch text preprocessing."""
        # Vectorized text processing for better performance
        pass

# Caching for expensive NLP operations
from functools import lru_cache

class CachedNLPProcessor:
    @lru_cache(maxsize=10000)
    def cached_sentiment_analysis(self, text_hash: str, text: str) -> Dict:
        """Cache sentiment analysis results for repeated texts."""
        pass
```

### Parallel Processing for News Analysis

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def parallel_news_analysis(news_data: pd.DataFrame) -> pd.DataFrame:
    """Process news data using parallel NLP calculators."""

    calculators = {
        'sentiment': SentimentCalculator(),
        'topic': TopicCalculator(),
        'credibility': CredibilityCalculator(),
        'events': EventCalculator()
    }

    def process_calculator(calc_pair):
        name, calc = calc_pair
        return name, calc.calculate(news_data)

    # Parallel NLP processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = dict(executor.map(process_calculator, calculators.items()))

    # Combine NLP features
    all_features = pd.concat(results.values(), axis=1)
    return all_features
```

## Configuration Management

### NewsConfig (`news_config.py`)

```python
@dataclass
class NewsConfig:
    # Sentiment analysis parameters
    sentiment_lexicon: str = 'vader'
    financial_sentiment_model: str = 'finbert'
    emotion_detection: bool = True
    sentiment_window_hours: int = 24

    # Topic modeling parameters
    num_topics: int = 10
    topic_algorithm: str = 'lda'
    min_document_frequency: int = 2
    max_document_frequency: float = 0.8

    # Credibility scoring parameters
    source_whitelist: List[str] = field(default_factory=list)
    credibility_threshold: float = 0.7
    bias_detection: bool = True

    # Event detection parameters
    event_keywords: Dict[str, List[str]] = field(default_factory=dict)
    impact_scoring: bool = True
    event_window_hours: int = 48

    # Volume analysis parameters
    volume_aggregation: str = 'hourly'
    spike_threshold: float = 2.0
    baseline_window_days: int = 7

    # Monetary policy parameters
    central_banks: List[str] = field(default_factory=lambda: ['FED', 'ECB', 'BOJ'])
    policy_keywords: List[str] = field(default_factory=list)
    economic_indicators: List[str] = field(default_factory=list)
```

## Feature Examples

### Sentiment Features

```python
# SentimentCalculator features
{
    'sentiment_score_vader': 0.25,        # VADER compound score
    'sentiment_positive': 0.15,           # Positive sentiment ratio
    'sentiment_negative': 0.05,           # Negative sentiment ratio
    'sentiment_neutral': 0.80,            # Neutral sentiment ratio
    'sentiment_subjectivity': 0.60,       # TextBlob subjectivity
    'sentiment_polarity': 0.20,           # TextBlob polarity
    'finbert_positive': 0.70,             # FinBERT positive probability
    'finbert_negative': 0.10,             # FinBERT negative probability
    'finbert_neutral': 0.20,              # FinBERT neutral probability
    'emotion_fear': 0.15,                 # Fear emotion score
    'emotion_greed': 0.30,                # Greed emotion score
    'emotion_uncertainty': 0.25,          # Uncertainty emotion score
    'sentiment_momentum': 0.05,           # Sentiment change rate
    'sentiment_volatility': 0.40,         # Sentiment volatility
    'social_sentiment_avg': 0.35          # Social media sentiment average
}
```

### Topic Features

```python
# TopicCalculator features
{
    'topic_earnings': 0.60,               # Earnings-related content probability
    'topic_ma': 0.15,                     # M&A content probability
    'topic_regulatory': 0.25,             # Regulatory content probability
    'topic_economic': 0.40,               # Economic content probability
    'topic_diversity': 0.75,              # Topic diversity score
    'sector_technology': 0.30,            # Technology sector mentions
    'sector_finance': 0.50,               # Finance sector mentions
    'sector_healthcare': 0.20,            # Healthcare sector mentions
    'content_complexity': 0.65,           # Content complexity score
    'topic_coherence': 0.80               # Topic model coherence
}
```

### Event Features

```python
# EventCalculator features
{
    'event_earnings_detected': 1,         # Earnings event detected (binary)
    'event_ma_detected': 0,               # M&A event detected (binary)
    'event_regulatory_detected': 1,       # Regulatory event detected (binary)
    'event_impact_score': 0.75,           # Predicted impact score
    'event_urgency': 0.60,                # Event urgency level
    'event_market_relevance': 0.85,       # Market relevance score
    'entity_mentions_count': 15,          # Number of entity mentions
    'executive_mentions': 3,              # Executive name mentions
    'financial_metrics_count': 8,         # Financial metrics mentioned
    'forward_looking_statements': 0.40    # Forward-looking content ratio
}
```

## Integration Examples

### Basic Usage

```python
from ai_trader.feature_pipeline.calculators.news import SentimentCalculator

# Load news data
news_data = pd.DataFrame({
    'timestamp': timestamps,
    'title': news_titles,
    'content': news_content,
    'source': news_sources
})

# Calculate sentiment features
sentiment_calc = SentimentCalculator()
sentiment_features = sentiment_calc.calculate(news_data)

print(f"Average sentiment: {sentiment_features['sentiment_score_vader'].mean():.3f}")
```

### Multi-Calculator Analysis

```python
from ai_trader.feature_pipeline.calculators.news import (
    SentimentCalculator,
    EventCalculator,
    VolumeCalculator
)

# Comprehensive news analysis
sentiment_calc = SentimentCalculator()
event_calc = EventCalculator()
volume_calc = VolumeCalculator()

sentiment_features = sentiment_calc.calculate(news_data)
event_features = event_calc.calculate(news_data)
volume_features = volume_calc.calculate(news_data)

# Combine for trading signal
combined_features = pd.concat([
    sentiment_features,
    event_features,
    volume_features
], axis=1)

# Generate trading signal
bullish_signal = (
    (combined_features['sentiment_score_vader'] > 0.1) &
    (combined_features['event_impact_score'] > 0.7) &
    (combined_features['news_volume_spike'] == 1)
)
```

## Architecture Benefits

### 1. NLP Specialization

- **Domain Expertise**: Each calculator implements specialized NLP techniques
- **Model Optimization**: Domain-specific models and preprocessing
- **Accuracy**: Focused analysis for better prediction accuracy

### 2. Performance

- **Selective Processing**: Process only required NLP analysis types
- **Parallel Execution**: Run multiple NLP calculators simultaneously
- **Caching**: Intelligent caching of expensive NLP operations

### 3. Maintainability

- **Clear Boundaries**: Well-defined responsibilities for each NLP domain
- **Independent Updates**: Update NLP models without affecting other components
- **Testing**: Focused testing for each type of news analysis

### 4. Extensibility

- **New NLP Methods**: Easy to add new analysis methods to appropriate calculators
- **Model Updates**: Simple integration of improved NLP models
- **Custom Analysis**: Support for domain-specific news analysis

---

**Last Updated**: 2025-07-15
**Architecture Version**: 2.0
**NLP Technology**: State-of-the-art financial language processing
**Status**: Production Ready

"""
Text Processing Service

Handles text cleaning, truncation, sentiment extraction, and keyword extraction
for news and other text-based data.
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import html

from main.utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class TextProcessingConfig:
    """Configuration for text processing."""
    max_text_length: int = 10000
    max_title_length: int = 500
    max_keywords: int = 20
    max_symbols_per_article: int = 10
    clean_html: bool = True
    extract_sentiment: bool = True
    normalize_whitespace: bool = True


class TextProcessingService:
    """
    Service for processing text content in news and other textual data.
    
    Handles cleaning, truncation, sentiment analysis, and keyword extraction.
    """
    
    def __init__(self, config: Optional[TextProcessingConfig] = None):
        """
        Initialize the text processing service.
        
        Args:
            config: Service configuration
        """
        self.config = config or TextProcessingConfig()
        
        # Compile regex patterns for efficiency
        self._html_tag_pattern = re.compile(r'<[^>]+>')
        self._whitespace_pattern = re.compile(r'\s+')
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._symbol_pattern = re.compile(r'\$([A-Z]{1,5})\b')
        
        logger.info(f"TextProcessingService initialized with config: {self.config}")
    
    def clean_text(self, text: Optional[str]) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags if configured
        if self.config.clean_html:
            text = self._html_tag_pattern.sub(' ', text)
        
        # Remove URLs
        text = self._url_pattern.sub(' ', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self._whitespace_pattern.sub(' ', text)
            text = text.strip()
        
        # Truncate if needed
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length] + "..."
            logger.debug(f"Truncated text to {self.config.max_text_length} characters")
        
        return text
    
    def clean_title(self, title: Optional[str]) -> str:
        """
        Clean and normalize title text.
        
        Args:
            title: Raw title to clean
            
        Returns:
            Cleaned title
        """
        if not title:
            return ""
        
        # Basic cleaning
        title = html.unescape(title)
        title = self._whitespace_pattern.sub(' ', title).strip()
        
        # Truncate if needed
        if len(title) > self.config.max_title_length:
            title = title[:self.config.max_title_length] + "..."
        
        return title
    
    def extract_symbols(self, text: str) -> List[str]:
        """
        Extract stock symbols from text.
        
        Looks for patterns like $AAPL, $MSFT, etc.
        
        Args:
            text: Text to extract symbols from
            
        Returns:
            List of unique symbols found
        """
        if not text:
            return []
        
        # Find all symbol mentions
        symbols = self._symbol_pattern.findall(text)
        
        # Deduplicate while preserving order
        seen = set()
        unique_symbols = []
        for symbol in symbols:
            symbol = symbol.upper()
            if symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)
        
        # Limit number of symbols
        if len(unique_symbols) > self.config.max_symbols_per_article:
            unique_symbols = unique_symbols[:self.config.max_symbols_per_article]
            logger.debug(
                f"Limited symbols to {self.config.max_symbols_per_article} "
                f"(found {len(symbols)})"
            )
        
        return unique_symbols
    
    def extract_keywords(self, text: str, title: str = "") -> List[str]:
        """
        Extract keywords from text and title.
        
        Simple keyword extraction based on important terms.
        
        Args:
            text: Main text content
            title: Title text
            
        Returns:
            List of keywords
        """
        if not text and not title:
            return []
        
        # Combine title and text for keyword extraction
        combined = f"{title} {text}".lower()
        
        # Simple keyword extraction based on important financial terms
        financial_terms = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'merger', 'acquisition', 'ipo', 'bankruptcy', 'dividend',
            'buyback', 'guidance', 'forecast', 'upgrade', 'downgrade',
            'beat', 'miss', 'outlook', 'expansion', 'layoff', 'ceo',
            'lawsuit', 'settlement', 'fda', 'approval', 'recall',
            'investigation', 'sec', 'fine', 'penalty', 'restructuring'
        ]
        
        keywords = []
        for term in financial_terms:
            if term in combined:
                keywords.append(term)
        
        # Also extract mentioned company names/symbols as keywords
        symbols = self.extract_symbols(combined)
        keywords.extend(symbols)
        
        # Limit keywords
        if len(keywords) > self.config.max_keywords:
            keywords = keywords[:self.config.max_keywords]
        
        return keywords
    
    def extract_sentiment(self, text: str, title: str = "") -> Dict[str, float]:
        """
        Extract sentiment scores from text.
        
        Simple rule-based sentiment analysis for financial text.
        
        Args:
            text: Main text content
            title: Title text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.config.extract_sentiment:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        combined = f"{title} {text}".lower()
        
        # Positive indicators
        positive_words = [
            'beat', 'exceed', 'surpass', 'profit', 'gain', 'rise', 'surge',
            'jump', 'soar', 'rally', 'upgrade', 'buy', 'strong', 'growth',
            'expand', 'improve', 'breakthrough', 'success', 'win', 'positive',
            'optimistic', 'bullish', 'record', 'high', 'best'
        ]
        
        # Negative indicators
        negative_words = [
            'miss', 'loss', 'decline', 'fall', 'drop', 'plunge', 'crash',
            'sink', 'slump', 'downgrade', 'sell', 'weak', 'concern', 'worry',
            'fear', 'risk', 'lawsuit', 'investigation', 'recall', 'bankruptcy',
            'layoff', 'cut', 'reduce', 'negative', 'pessimistic', 'bearish',
            'worst', 'low', 'fail'
        ]
        
        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in combined)
        negative_count = sum(1 for word in negative_words if word in combined)
        
        # Calculate simple scores
        total = positive_count + negative_count
        if total == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        positive_score = positive_count / total
        negative_score = negative_count / total
        
        # Determine overall sentiment
        if positive_score > negative_score + 0.2:
            sentiment = "positive"
        elif negative_score > positive_score + 0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": 1.0 - positive_score - negative_score,
            "overall": sentiment
        }
    
    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete news article.
        
        Args:
            article: Raw article data
            
        Returns:
            Processed article with cleaned text and extracted features
        """
        # Clean text fields
        title = self.clean_title(article.get('title', ''))
        content = self.clean_text(article.get('content', article.get('description', '')))
        
        # Extract features
        symbols = self.extract_symbols(f"{title} {content}")
        keywords = self.extract_keywords(content, title)
        sentiment = self.extract_sentiment(content, title)
        
        # Build processed article
        processed = {
            'title': title,
            'content': content,
            'symbols': symbols,
            'keywords': keywords,
            'sentiment_positive': sentiment.get('positive', 0.0),
            'sentiment_negative': sentiment.get('negative', 0.0),
            'sentiment_neutral': sentiment.get('neutral', 1.0),
            'sentiment_overall': sentiment.get('overall', 'neutral')
        }
        
        # Preserve other fields
        for key in ['url', 'published_at', 'source', 'author', 'id']:
            if key in article:
                processed[key] = article[key]
        
        return processed
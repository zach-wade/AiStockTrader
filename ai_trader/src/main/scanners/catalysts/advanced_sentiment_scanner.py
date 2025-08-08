# File: ai_trader/scanners/advanced_sentiment_scanner.py
"""
Advanced Sentiment Scanner

A specialized scanner that uses transformer models (e.g., FinBERT) to analyze
the sentiment and intent of recent news and social media content.

Now uses the repository pattern with hot/cold storage awareness to
efficiently access textual data for NLP analysis.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# All model-related imports are now self-contained in this module
try:
    from transformers import pipeline, Pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from omegaconf import DictConfig
from main.interfaces.scanners import IScanner, IScannerRepository
from main.interfaces.events import IEventBus
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.utils.scanners import (
    ScannerDataAccess,
    ScannerCacheManager,
    ScannerMetricsCollector
)
from main.utils.core import timer
from main.scanners.base_scanner import BaseScanner
from main.scanners.catalyst_scanner_base import CatalystScannerBase
from main.events.types import ScanAlert, AlertType
import pandas as pd

logger = logging.getLogger(__name__)

class AdvancedSentimentScanner(CatalystScannerBase):
    """
    A specialized scanner that uses transformer models (e.g., FinBERT) to analyze
    the sentiment and intent of recent news and social media content.
    
    Now uses the repository pattern with hot/cold storage awareness to
    efficiently access textual data for NLP analysis.
    """

    def __init__(
        self,
        config: DictConfig,
        repository: IScannerRepository,
        event_bus: Optional[IEventBus] = None,
        metrics_collector: Optional[ScannerMetricsCollector] = None,
        cache_manager: Optional[ScannerCacheManager] = None
    ):
        """
        Initializes the AdvancedSentimentScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "AdvancedSentimentScanner",
            config,
            repository,
            event_bus,
            metrics_collector,
            cache_manager
        )
        
        self.params = self.config.get('scanners.advanced_sentiment', {})
        self.lookback_hours = self.params.get('lookback_hours', 24)
        self.use_cache = self.params.get('use_cache', True)
        self.min_sentiment_threshold = self.params.get('min_sentiment_threshold', 0.5)
        
        # Transformer models (loaded lazily)
        self.finbert: Optional[Pipeline] = None
        self.intent_classifier: Optional[Pipeline] = None
        self._models_loaded = False
        
        # Track initialization
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize scanner resources."""
        if self._initialized:
            return
        
        logger.info(f"Initializing {self.name}")
        
        # Load models on first use
        if TRANSFORMERS_AVAILABLE and not self._models_loaded:
            self.finbert, self.intent_classifier = self._initialize_transformer_models()
            self._models_loaded = True
        
        self._initialized = True
    
    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        logger.info(f"Cleaning up {self.name}")
        self._initialized = False
        
        # Clean up models
        self.finbert = None
        self.intent_classifier = None
        self._models_loaded = False

    def _initialize_transformer_models(self) -> tuple[Optional['Pipeline'], Optional['Pipeline']]:
        """Loads and initializes the FinBERT and Zero-Shot models."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not installed. Advanced sentiment analysis is disabled.")
            return None, None

        try:
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading transformer models on device: {'GPU' if device == 0 else 'CPU'}")

            # Financial BERT for sentiment
            finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=device
            )
            
            # Zero-shot for intent classification
            intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device
            )
            logger.info("✅ Transformer models (FinBERT, BART) loaded successfully.")
            return finbert, intent_classifier
        except Exception as e:
            logger.error(f"❌ Failed to load transformer models: {e}", exc_info=True)
            return None, None

    async def scan(self, symbols: List[str], **kwargs) -> List[ScanAlert]:
        """
        Scan for advanced sentiment signals using transformer models.
        
        Uses repository pattern for efficient textual data access with hot storage
        for recent news and social media content.
        
        Args:
            symbols: List of symbols to scan
            **kwargs: Additional parameters
            
        Returns:
            List of ScanAlert objects
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.finbert or not self.intent_classifier:
            logger.warning("Transformer models not available. Skipping advanced sentiment analysis.")
            return []
            
        with timer() as t:
            logger.info(f"🧠 Advanced Sentiment Scanner: Analyzing {len(symbols)} symbols...")
            
            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(timezone.utc)
            
            try:
                # Check cache if enabled
                if self.cache and self.use_cache:
                    cache_key = f"advanced_sentiment:{','.join(sorted(symbols[:10]))}:{self.lookback_hours}"
                    cached_alerts = await self.cache.get_cached_result(
                        self.name,
                        "batch",
                        cache_key
                    )
                    if cached_alerts is not None:
                        logger.info(f"Using cached results for advanced sentiment scan")
                        return cached_alerts
                
                # Build query filter for recent data
                query_filter = QueryFilter(
                    symbols=symbols,
                    start_date=datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours),
                    end_date=datetime.now(timezone.utc)
                )
                
                # Get news and social data from repository
                # Both will primarily use hot storage for recent data
                news_data = await self.repository.get_news_sentiment(symbols, query_filter)
                social_data = await self.repository.get_social_sentiment(symbols, query_filter)
                
                # Combine textual data
                textual_data = self._combine_textual_data(news_data, social_data)
                
                if textual_data.empty:
                    logger.info("No recent textual data found to analyze.")
                    return []
                
                # Analyze sentiment and intent for each piece of content
                analyzed_data = self._analyze_content(textual_data)
                
                # Aggregate scores and generate alerts
                alerts = self._generate_alerts(analyzed_data)
                
                # Cache results if enabled
                if self.cache and self.use_cache and alerts:
                    await self.cache.cache_result(
                        self.name,
                        "batch",
                        cache_key,
                        alerts,
                        ttl_seconds=1800  # 30 minute TTL for NLP results
                    )
                
                # Deduplicate alerts
                alerts = self.deduplicate_alerts(alerts)
                
                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)
                
                logger.info(
                    f"✅ Advanced Sentiment Scanner: Found {len(alerts)} symbols with strong sentiment signals "
                    f"in {t.elapsed_ms:.2f}ms"
                )
                
                # Record metrics
                if self.metrics:
                    self.metrics.record_scan_duration(
                        self.name,
                        t.elapsed_ms,
                        len(symbols)
                    )
                
                return alerts
                
            except Exception as e:
                logger.error(f"❌ Error in Advanced Sentiment Scanner: {e}", exc_info=True)
                
                # Record error metric
                if self.metrics:
                    self.metrics.record_scan_error(
                        self.name,
                        type(e).__name__,
                        str(e)
                    )
                
                return []
    
    async def run(self, universe: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Legacy method for backward compatibility.
        
        Args:
            universe: The list of symbols to scan.

        Returns:
            A dictionary mapping symbols to a catalyst signal if significant sentiment is detected.
        """
        # Use the new scan method
        alerts = await self.scan(universe)
        
        # Convert to legacy format
        catalyst_signals = {}
        for alert in alerts:
            catalyst_signals[alert.symbol] = {
                'score': alert.metadata.get('raw_score', alert.score * 5.0),  # Convert from 0-1 to 0-5 scale
                'reason': alert.metadata.get('reason', ''),
                'signal_type': 'advanced_sentiment',
                'metadata': {
                    'avg_sentiment': alert.metadata.get('avg_sentiment', 0)
                }
            }
        
        return catalyst_signals

    def _combine_textual_data(self, news_data: Dict[str, List[Dict]], social_data: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Combine news and social data into a single DataFrame for analysis."""
        rows = []
        
        # Process news data
        for symbol, news_items in news_data.items():
            for item in news_items:
                rows.append({
                    'symbol': symbol,
                    'content': item.get('headline', item.get('title', '')),
                    'source': 'news',
                    'timestamp': item.get('timestamp')
                })
        
        # Process social data
        for symbol, social_items in social_data.items():
            for item in social_items:
                rows.append({
                    'symbol': symbol,
                    'content': item.get('content', ''),
                    'source': 'social',
                    'timestamp': item.get('timestamp')
                })
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Filter out empty content
        df = df[df['content'].str.strip() != '']
        
        return df

    def _analyze_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs transformer models on the text content."""
        if df.empty:
            return df

        # Check transformer model cache for each piece of content
        cache_key_prefix = "transformer_analysis"
        
        # Analyze Sentiment with FinBERT
        sentiments = []
        for idx, text in enumerate(df['content']):
            # Try to get from cache first
            if self.cache and self.use_cache:
                content_hash = hash(text) % 10000000  # Simple hash for cache key
                cache_key = f"{cache_key_prefix}:sentiment:{content_hash}"
                cached_sentiment = self.cache._get_from_cache(cache_key)
                if cached_sentiment:
                    sentiments.append(cached_sentiment)
                    continue
            
            # Run model if not in cache
            sentiment = self.finbert(text)[0]
            sentiments.append(sentiment)
            
            # Cache the result
            if self.cache and self.use_cache:
                self.cache._add_to_cache(cache_key, sentiment, ttl=3600)  # 1 hour TTL
        
        df['sentiment_label'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['score'] * (1 if s['label'] == 'positive' else -1 if s['label'] == 'negative' else 0) for s in sentiments]

        # Analyze Intent with Zero-Shot Classifier
        candidate_labels = self.params.get('intent_labels', ['positive catalyst', 'negative catalyst', 'general discussion'])
        intents = self.intent_classifier(list(df['content']), candidate_labels=candidate_labels)
        df['intent_label'] = [i['labels'][0] for i in intents]
        df['intent_score'] = [i['scores'][0] for i in intents]
        
        return df

    def _generate_alerts(self, df: pd.DataFrame) -> List[ScanAlert]:
        """Generate alerts from analyzed sentiment data."""
        alerts = []
        
        # Calculate a weighted average sentiment score per symbol
        sentiment_by_symbol = df.groupby('symbol')['sentiment_score'].agg(['mean', 'count']).to_dict('index')
        
        for symbol, stats in sentiment_by_symbol.items():
            avg_sentiment = stats['mean']
            content_count = stats['count']
            
            if abs(avg_sentiment) >= self.min_sentiment_threshold:
                # Score is based on the magnitude of the sentiment and content volume
                base_score = abs(avg_sentiment)
                volume_boost = min(content_count / 10, 1.0)  # Boost for more content
                score = base_score * (1 + volume_boost * 0.5)
                
                # Normalize score to 0-1 range
                normalized_score = min(score, 1.0)
                
                direction = "positive" if avg_sentiment > 0 else "negative"
                
                # Determine alert type
                alert_type = AlertType.SENTIMENT_SURGE if avg_sentiment > 0 else AlertType.SENTIMENT_SURGE
                
                # Get dominant intent for this symbol
                symbol_df = df[df['symbol'] == symbol]
                dominant_intent = symbol_df.groupby('intent_label').size().idxmax() if not symbol_df.empty else 'unknown'
                
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=alert_type,
                    score=normalized_score,
                    metadata={
                        'avg_sentiment': avg_sentiment,
                        'content_count': content_count,
                        'direction': direction,
                        'dominant_intent': dominant_intent,
                        'reason': f"Strong {direction} FinBERT sentiment ({avg_sentiment:.2f}) detected across {content_count} items",
                        'catalyst_type': 'advanced_sentiment',
                        'raw_score': score * 5.0  # Legacy scale
                    }
                )
                alerts.append(alert)
                
                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name,
                        alert_type,
                        symbol,
                        normalized_score
                    )
        
        return alerts
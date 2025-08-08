# File: ai_trader/scanners/news_scanner.py

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import hashlib

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
from ..catalyst_scanner_base import CatalystScannerBase
from main.events.types import AlertType, ScanAlert

logger = logging.getLogger(__name__)

class NewsScanner(CatalystScannerBase):
    """
    Scans a universe of symbols for recent, high-impact news catalysts.
    
    Now uses the repository pattern with hot/cold storage awareness to
    efficiently access recent news from hot storage and historical patterns
    from cold storage.
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
        Initializes the NewsScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "NewsScanner",
            config,
            repository,
            event_bus,
            metrics_collector,
            cache_manager
        )
        
        # Scanner-specific parameters
        self.params = self.config.get('scanners.news', {})
        self.lookback_hours = self.params.get('lookback_hours', 24)
        self.min_sentiment_score = self.params.get('min_sentiment_score', 0.7)
        self.use_cache = self.params.get('use_cache', True)
        self.dedup_window_hours = self.params.get('dedup_window_hours', 48)
        
        # News impact weights
        self.impact_weights = self.params.get('impact_weights', {
            'upgrade': 0.9,
            'downgrade': 0.8,
            'earnings_beat': 0.85,
            'earnings_miss': 0.8,
            'acquisition': 0.9,
            'partnership': 0.7,
            'product_launch': 0.6,
            'regulatory': 0.8,
            'general': 0.5
        })
        
        # Keywords for classification
        self.classification_keywords = self.params.get('classification_keywords', {
            'upgrade': ['upgrade', 'raised', 'buy rating', 'outperform'],
            'downgrade': ['downgrade', 'lowered', 'sell rating', 'underperform'],
            'earnings': ['earnings', 'revenue', 'eps', 'beat', 'miss', 'guidance'],
            'acquisition': ['acquisition', 'merger', 'buyout', 'acquired'],
            'partnership': ['partnership', 'collaboration', 'agreement', 'deal'],
            'product': ['launch', 'release', 'announce', 'introduce', 'unveil'],
            'regulatory': ['fda', 'sec', 'regulatory', 'approval', 'investigation']
        })
        
        # Seen articles cache for deduplication
        self._seen_articles = set()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize scanner resources."""
        if self._initialized:
            return
        
        logger.info(f"Initializing {self.name}")
        self._initialized = True
    
    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        logger.info(f"Cleaning up {self.name}")
        self._seen_articles.clear()
        self._initialized = False

    async def scan(self, symbols: List[str], **kwargs) -> List[ScanAlert]:
        """
        Scan symbols for recent high-impact news.
        
        Uses repository for efficient news data access with hot storage
        for recent news (real-time alerts) and cold storage for historical
        sentiment patterns.
        
        Args:
            symbols: List of stock symbols to scan
            **kwargs: Additional scanner-specific parameters
            
        Returns:
            List of ScanAlert objects for news signals
        """
        if not self._initialized:
            await self.initialize()
            
        with timer() as t:
            logger.info(f"📰 News Scanner: Analyzing {len(symbols)} symbols for news from the last {self.lookback_hours} hours...")
            
            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(timezone.utc)
            
            try:
                # Check cache if enabled
                if self.cache and self.use_cache:
                    cache_key = f"news_scan:{','.join(sorted(symbols[:10]))}:{self.lookback_hours}"
                    cached_alerts = await self.cache.get_cached_result(
                        self.name,
                        "batch",
                        cache_key
                    )
                    if cached_alerts is not None:
                        logger.info(f"Using cached results for news scan")
                        return cached_alerts
                
                # Build query filter for news data
                query_filter = QueryFilter(
                    symbols=symbols,
                    start_date=datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours),
                    end_date=datetime.now(timezone.utc)
                )
                
                # Get news and sentiment data from repository
                # This will use hot storage for recent news (< 30 days)
                news_sentiment_data = await self.repository.get_news_sentiment(
                    symbols=symbols,
                    query_filter=query_filter
                )
                
                # Check if we got data
                if not news_sentiment_data:
                    logger.warning("No news sentiment data returned from repository")
                    return []
                
                alerts = []
                for symbol, news_items in news_sentiment_data.items():
                    if not news_items:
                        continue
                    
                    # Process news items for this symbol
                    symbol_alerts = await self._process_symbol_news(
                        symbol,
                        news_items
                    )
                    alerts.extend(symbol_alerts)
                
                # Deduplicate alerts
                alerts = self._deduplicate_alerts(alerts)
                
                # Cache results if enabled
                if self.cache and self.use_cache and alerts:
                    await self.cache.cache_result(
                        self.name,
                        "batch",
                        cache_key,
                        alerts,
                        ttl_seconds=300  # 5 minute TTL for news
                    )
                
                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)
                
                logger.info(
                    f"✅ News Scanner: Found {len(alerts)} news alerts "
                    f"in {t.elapsed * 1000:.2f}ms"
                )
                
                # Record metrics
                if self.metrics:
                    self.metrics.record_scan_duration(
                        self.name,
                        t.elapsed * 1000,
                        len(symbols)
                    )
                
                return alerts
                
            except Exception as e:
                logger.error(f"❌ Error in News Scanner: {e}", exc_info=True)
                
                # Record error metric
                if self.metrics:
                    self.metrics.record_scan_error(
                        self.name,
                        type(e).__name__,
                        str(e)
                    )
                
                return []
    
    async def run(self, universe: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Legacy method for backward compatibility.
        
        Args:
            universe: The list of symbols to scan.

        Returns:
            A dictionary mapping symbols to a list of their news catalyst signals.
        """
        # Use the new scan method
        alerts = await self.scan(universe)
        
        # Convert to legacy format
        catalyst_signals = defaultdict(list)
        for alert in alerts:
            signal = {
                'score': alert.metadata.get('raw_score', alert.score * 5.0),  # Convert from 0-1 to 0-5 scale
                'reason': alert.metadata.get('reason', ''),
                'signal_type': 'news',
                'metadata': {
                    'headline': alert.metadata.get('headline'),
                    'source': alert.metadata.get('source'),
                    'timestamp': alert.metadata.get('timestamp'),
                    'url': alert.metadata.get('url')
                }
            }
            catalyst_signals[alert.symbol].append(signal)
        
        return dict(catalyst_signals)

    async def _process_symbol_news(
        self, 
        symbol: str, 
        news_items: List[Dict[str, Any]]
    ) -> List[ScanAlert]:
        """Process news items for a symbol and generate alerts."""
        alerts = []
        
        for item in news_items:
            # Skip if already seen (deduplication)
            article_hash = self._get_article_hash(item)
            if article_hash in self._seen_articles:
                continue
            self._seen_articles.add(article_hash)
            
            # Extract key information
            headline = item.get('headline', '').lower()
            content = item.get('content', '').lower()
            sentiment_score = item.get('sentiment_score', 0.5)
            published_at = item.get('published_at', datetime.now(timezone.utc))
            
            # Skip if sentiment too low
            if sentiment_score < self.min_sentiment_score and sentiment_score > -self.min_sentiment_score:
                continue
            
            # Classify news type
            news_type, impact_score = self._classify_news(headline, content)
            
            # Skip low impact news
            if impact_score < 0.5:
                continue
            
            # Determine alert type based on classification
            alert_type = self._get_alert_type(news_type)
            
            # Calculate final score combining sentiment and impact
            final_score = (abs(sentiment_score) + impact_score) / 2
            final_score = min(final_score, 1.0)
            
            # Create alert
            alert = self.create_alert(
                symbol=symbol,
                alert_type=alert_type,
                score=final_score,
                metadata={
                    'headline': item.get('headline', ''),
                    'source': item.get('source', 'unknown'),
                    'published_at': published_at,
                    'url': item.get('url', ''),
                    'sentiment_score': sentiment_score,
                    'news_type': news_type,
                    'impact_score': impact_score,
                    'reason': f"{news_type.title()}: {item.get('headline', '')[:75]}..."
                }
            )
            alerts.append(alert)
            
            # Record metric
            if self.metrics:
                self.metrics.record_alert_generated(
                    self.name,
                    alert_type,
                    symbol,
                    final_score
                )
        
        return alerts
    
    def _classify_news(self, headline: str, content: str) -> tuple[str, float]:
        """Classify news type and calculate impact score."""
        text = f"{headline} {content}".lower()
        
        # Check each category
        best_match = ('general', self.impact_weights.get('general', 0.5))
        
        for category, keywords in self.classification_keywords.items():
            if any(keyword in text for keyword in keywords):
                impact = self.impact_weights.get(category, 0.5)
                if impact > best_match[1]:
                    best_match = (category, impact)
        
        return best_match
    
    def _get_alert_type(self, news_type: str) -> str:
        """Map news type to alert type."""
        type_mapping = {
            'earnings': AlertType.EARNINGS_ANNOUNCEMENT,
            'upgrade': AlertType.ANALYST_RATING,
            'downgrade': AlertType.ANALYST_RATING,
            'acquisition': AlertType.CORPORATE_ACTION,
            'regulatory': AlertType.REGULATORY,
            'partnership': AlertType.BREAKING_NEWS,
            'product': AlertType.BREAKING_NEWS,
            'general': AlertType.NEWS_CATALYST
        }
        
        return type_mapping.get(news_type, AlertType.NEWS_CATALYST)
    
    def _get_article_hash(self, article: Dict[str, Any]) -> str:
        """Generate hash for article deduplication."""
        # Use headline and source for deduplication
        key = f"{article.get('headline', '')}:{article.get('source', '')}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _deduplicate_alerts(self, alerts: List[ScanAlert]) -> List[ScanAlert]:
        """Deduplicate alerts based on symbol and news content."""
        seen = set()
        deduped = []
        
        for alert in alerts:
            # Create dedup key from symbol and headline
            key = f"{alert.symbol}:{alert.metadata.get('headline', '')[:50]}"
            if key not in seen:
                seen.add(key)
                deduped.append(alert)
        
        return deduped
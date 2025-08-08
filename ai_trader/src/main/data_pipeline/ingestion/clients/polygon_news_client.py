"""
Polygon News Client - Refactored

Simplified client for fetching financial news articles from Polygon.io API.
Uses PolygonApiHandler for common functionality.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
import hashlib

from main.utils.core import get_logger, ensure_utc
from main.utils.monitoring import timer, record_metric, MetricType
from main.data_pipeline.core.enums import DataLayer
from main.data_pipeline.services.ingestion import TextProcessingService
from main.data_pipeline.services.ingestion.polygon_api_handler import PolygonApiHandler
from .base_client import BaseIngestionClient, ClientConfig, FetchResult


class PolygonNewsClient(BaseIngestionClient[List[Dict[str, Any]]]):
    """
    Simplified client for fetching news from Polygon.io.
    
    Delegates common functionality to PolygonApiHandler.
    """
    
    def __init__(
        self,
        api_key: str,
        layer: DataLayer = DataLayer.BASIC,
        config: Optional[ClientConfig] = None,
        text_processor: Optional[TextProcessingService] = None
    ):
        """Initialize the Polygon news client."""
        self.api_handler = PolygonApiHandler()
        
        # Create config using handler with layer-based configuration
        # News uses custom cache TTL regardless of layer
        config = self.api_handler.create_polygon_config(
            api_key=api_key,
            layer=layer,
            config=config,
            cache_ttl_seconds=600  # Cache news for 10 minutes
        )
        
        super().__init__(config)
        self.layer = layer
        self.text_processor = text_processor or TextProcessingService()
        self._seen_articles: Set[str] = set()  # For deduplication
        
        self.logger = get_logger(__name__)
        self.logger.info(f"PolygonNewsClient initialized with layer: {layer.name}")
    
    def get_base_url(self) -> str:
        """Get the base URL for Polygon API."""
        return self.config.base_url
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return self.api_handler.get_standard_headers(self.config.api_key)
    
    async def validate_response(self, response) -> bool:
        """Validate Polygon API response."""
        return await self.api_handler.validate_http_response(response)
    
    async def parse_response(self, response) -> List[Dict[str, Any]]:
        """Parse Polygon API response into standardized format."""
        results = await self.api_handler.parse_polygon_response(response)
        
        parsed_articles = []
        for article in results:
            parsed = self._normalize_article(article)
            if parsed:
                parsed_articles.append(parsed)
        
        return parsed_articles
    
    async def fetch_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100,
        include_related: bool = True
    ) -> FetchResult[List[Dict[str, Any]]]:
        """Fetch news articles for a symbol."""
        # Build date params using handler
        params = self.api_handler.build_date_params(
            start_date, end_date, date_field='published_utc'
        )
        
        # Add news-specific params
        params.update({
            'ticker': symbol.upper(),
            'limit': str(min(limit, 1000)),  # Max 1000 per request
            'sort': 'published_utc'
        })
        
        # Track API call performance
        with timer("polygon.news.fetch", tags={"symbol": symbol}):
            # Use handler for pagination
            endpoint = 'v2/reference/news'
            result = await self.api_handler.fetch_with_pagination(
                self, endpoint, params, limit=limit, max_pages=10
            )
        
        if result.success and result.data:
            # Track raw articles fetched
            record_metric("polygon.news.raw_articles", len(result.data), MetricType.COUNTER,
                         tags={"symbol": symbol})
            
            # Deduplicate articles
            unique_articles = self._deduplicate_articles(result.data)
            
            # Track deduplication effectiveness
            if len(result.data) > 0:
                dedup_rate = (len(result.data) - len(unique_articles)) / len(result.data)
                record_metric("polygon.news.dedup_rate", dedup_rate, MetricType.GAUGE,
                             tags={"symbol": symbol})
            
            # Filter by relevance if not including related
            if not include_related:
                before_filter = len(unique_articles)
                unique_articles = [
                    article for article in unique_articles
                    if symbol.upper() in article.get('tickers', [])
                ]
                # Track filtering impact
                filtered_out = before_filter - len(unique_articles)
                if filtered_out > 0:
                    record_metric("polygon.news.filtered_related", filtered_out, MetricType.COUNTER,
                                 tags={"symbol": symbol})
            
            result.data = unique_articles
            
            # Track final article count
            record_metric("polygon.news.articles", len(unique_articles), MetricType.COUNTER,
                         tags={"symbol": symbol})
        else:
            # Track API errors
            record_metric("polygon.api.errors", 1, MetricType.COUNTER,
                         tags={"data_type": "news", "symbol": symbol,
                               "error": result.error or "unknown"})
        
        return result
    
    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        max_concurrent: int = 5
    ) -> Dict[str, FetchResult[List[Dict[str, Any]]]]:
        """Fetch news for multiple symbols concurrently using the handler."""
        async def fetch_symbol(symbol: str) -> FetchResult:
            return await self.fetch_news(symbol, start_date, end_date)
        
        # Track batch operation
        batch_start = datetime.now()
        gauge("polygon.news.batch_symbols", len(symbols))
        
        # Use handler's batch_fetch
        results = await self.api_handler.batch_fetch(
            fetch_symbol,
            symbols,
            batch_size=50,
            max_concurrent=max_concurrent
        )
        
        # Calculate batch metrics
        batch_duration = (datetime.now() - batch_start).total_seconds()
        successful = sum(1 for r in results.values() if r.success)
        total_articles = sum(len(r.data) for r in results.values() if r.success and r.data)
        
        # Record batch metrics
        record_metric("polygon.news.batch_duration", batch_duration, MetricType.HISTOGRAM,
                     tags={"symbols": len(symbols)})
        record_metric("polygon.news.batch_success_rate", successful / len(symbols) if symbols else 0,
                     MetricType.GAUGE)
        record_metric("polygon.news.batch_total_articles", total_articles, MetricType.COUNTER)
        
        if len(symbols) - successful > 0:
            self.logger.warning(f"News batch fetch failed for {len(symbols) - successful}/{len(symbols)} symbols")
        
        return results
    
    def _normalize_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize article data to standard format."""
        try:
            # Extract and process title
            title = article.get('title', '').strip()
            if not title:
                return None
            
            # Parse published date
            published_str = article.get('published_utc', '')
            if published_str:
                published_at = datetime.fromisoformat(
                    published_str.replace('Z', '+00:00')
                )
            else:
                published_at = datetime.now(timezone.utc)
            
            # Extract tickers
            tickers = article.get('tickers', [])
            if isinstance(tickers, str):
                tickers = [tickers]
            tickers = [t.upper() for t in tickers]
            
            return {
                'id': article.get('id', self._generate_article_id(article)),
                'title': title,
                'author': article.get('author'),
                'published_at': published_at,
                'url': article.get('article_url', ''),
                'tickers': tickers,
                'description': article.get('description'),
                'keywords': article.get('keywords', []),
                'publisher': article.get('publisher', {}).get('name', ''),
                'publisher_url': article.get('publisher', {}).get('homepage_url'),
                'amp_url': article.get('amp_url'),
                'image_url': article.get('image_url'),
                'sentiment': self._process_text_if_available(article)
            }
        except Exception as e:
            self.logger.debug(f"Error normalizing article: {e}")
            raise
    
    def _generate_article_id(self, article: Dict[str, Any]) -> str:
        """Generate unique ID for article if not provided."""
        content = f"{article.get('title', '')}{article.get('published_utc', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on ID."""
        unique_articles = []
        for article in articles:
            article_id = article.get('id')
            if article_id and article_id not in self._seen_articles:
                self._seen_articles.add(article_id)
                unique_articles.append(article)
        return unique_articles
    
    def _process_text_if_available(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process article text for sentiment if text processor is available."""
        if not self.text_processor:
            return None
        
        text = f"{article.get('title', '')} {article.get('description', '')}"
        if text.strip():
            with timer("polygon.news.sentiment_analysis"):
                sentiment = self.text_processor.analyze_sentiment(text)
            
            # Track sentiment distribution
            if sentiment and 'score' in sentiment:
                record_metric("polygon.news.sentiment_score", sentiment['score'], MetricType.HISTOGRAM)
            
            return sentiment
        return None
    
    def clear_cache(self):
        """Clear seen articles cache."""
        self._seen_articles.clear()
        self.logger.debug("Article cache cleared")
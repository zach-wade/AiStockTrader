"""
News Repository

Repository for news data management and search.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import json
import time

from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import (
    RepositoryConfig,
    QueryFilter,
    OperationResult
)

from .base_repository import BaseRepository
from .helpers import (
    QueryBuilder,
    BatchProcessor,
    CrudExecutor,
    RepositoryMetricsCollector
)

from main.utils.core import get_logger, ensure_utc

logger = get_logger(__name__)


class NewsRepository(BaseRepository):
    """
    Repository for news articles and sentiment analysis.
    
    Handles news storage, retrieval, search, and sentiment tracking.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ):
        """Initialize the NewsRepository."""
        super().__init__(
            db_adapter,
            type('NewsArticle', (), {'__tablename__': 'news_data'}),
            config
        )
        
        self.crud_executor = CrudExecutor(
            db_adapter,
            'news_data',
            transaction_strategy=config.transaction_strategy if config else None
        )
        self.batch_processor = BatchProcessor(
            batch_size=config.batch_size if config else 100
        )
        self.metrics = RepositoryMetricsCollector('NewsRepository')
        
        logger.info("NewsRepository initialized")
    
    def get_required_fields(self) -> List[str]:
        """Get required fields for news data."""
        return ['symbol', 'published_at', 'title', 'url']
    
    def validate_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate news record."""
        errors = []
        
        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate sentiment scores if present
        for field in ['sentiment_score', 'relevance_score']:
            if field in record and record[field] is not None:
                value = record[field]
                if not isinstance(value, (int, float)) or value < -1 or value > 1:
                    errors.append(f"{field} must be between -1 and 1")
        
        return errors
    
    async def get_news(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        min_relevance: Optional[float] = None
    ) -> pd.DataFrame:
        """Get news articles for symbols in date range."""
        start_time = time.time()
        
        try:
            # Build query
            placeholders = [f"${i+1}" for i in range(len(symbols))]
            conditions = [
                f"symbol IN ({','.join(placeholders)})",
                f"published_at >= ${len(symbols) + 1}",
                f"published_at <= ${len(symbols) + 2}"
            ]
            
            params = [self._normalize_symbol(s) for s in symbols]
            params.extend([ensure_utc(start_date), ensure_utc(end_date)])
            
            if min_relevance is not None:
                conditions.append(f"relevance_score >= ${len(params) + 1}")
                params.append(min_relevance)
            
            query = f"""
                SELECT * FROM news_data
                WHERE {' AND '.join(conditions)}
                ORDER BY published_at DESC
            """
            
            results = await self.db_adapter.fetch_all(query, *params)
            df = pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()
            
            # Record metrics
            duration = time.time() - start_time
            await self.metrics.record_operation(
                'get_news', duration, True, len(df)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting news: {e}")
            await self.metrics.record_operation(
                'get_news', time.time() - start_time, False
            )
            return pd.DataFrame()
    
    async def store_news(
        self,
        articles: List[Dict[str, Any]]
    ) -> OperationResult:
        """Store news articles."""
        start_time = time.time()
        
        try:
            # Prepare records
            records = []
            for article in articles:
                record = {
                    'symbol': self._normalize_symbol(article['symbol']),
                    'published_at': ensure_utc(article['published_at']),
                    'title': article['title'],
                    'url': article['url'],
                    'summary': article.get('summary'),
                    'content': article.get('content'),
                    'author': article.get('author'),
                    'source': article.get('source'),
                    'sentiment_score': article.get('sentiment_score'),
                    'relevance_score': article.get('relevance_score'),
                    'keywords': json.dumps(article.get('keywords', [])),
                    'entities': json.dumps(article.get('entities', {})),
                    'created_at': datetime.now(timezone.utc)
                }
                records.append(record)
            
            # Process in batches
            async def store_batch(batch: List[Dict[str, Any]]) -> Any:
                for record in batch:
                    await self._upsert_article(record)
                return len(batch)
            
            result = await self.batch_processor.process_batch(
                records,
                store_batch
            )
            
            return OperationResult(
                success=result['success'],
                records_affected=result['statistics']['succeeded'],
                records_created=result['statistics']['succeeded'],
                duration_seconds=time.time() - start_time,
                metadata=result['statistics']
            )
            
        except Exception as e:
            logger.error(f"Error storing news: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time
            )
    
    async def get_latest_news(
        self,
        symbol: str,
        limit: int = 10
    ) -> pd.DataFrame:
        """Get latest news for a symbol."""
        try:
            query = """
                SELECT * FROM news_data
                WHERE symbol = $1
                ORDER BY published_at DESC
                LIMIT $2
            """
            
            results = await self.db_adapter.fetch_all(
                query,
                self._normalize_symbol(symbol),
                limit
            )
            
            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting latest news: {e}")
            return pd.DataFrame()
    
    async def search_news(
        self,
        keywords: List[str],
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Search news by keywords."""
        try:
            conditions = []
            params = []
            param_count = 1
            
            # Add keyword search
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.extend([
                    f"title ILIKE ${param_count}",
                    f"summary ILIKE ${param_count}",
                    f"content ILIKE ${param_count}"
                ])
                params.append(f"%{keyword}%")
                param_count += 1
            
            if keyword_conditions:
                conditions.append(f"({' OR '.join(keyword_conditions)})")
            
            # Add symbol filter
            if symbols:
                placeholders = [f"${i}" for i in range(param_count, param_count + len(symbols))]
                conditions.append(f"symbol IN ({','.join(placeholders)})")
                params.extend([self._normalize_symbol(s) for s in symbols])
                param_count += len(symbols)
            
            # Add date filters
            if start_date:
                conditions.append(f"published_at >= ${param_count}")
                params.append(ensure_utc(start_date))
                param_count += 1
            
            if end_date:
                conditions.append(f"published_at <= ${param_count}")
                params.append(ensure_utc(end_date))
                param_count += 1
            
            where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            
            query = f"""
                SELECT * FROM news_data
                {where_clause}
                ORDER BY published_at DESC
                LIMIT 1000
            """
            
            results = await self.db_adapter.fetch_all(query, *params)
            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error searching news: {e}")
            return pd.DataFrame()
    
    async def get_sentiment_summary(
        self,
        symbol: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get sentiment summary for a symbol."""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            query = """
                SELECT 
                    COUNT(*) as article_count,
                    AVG(sentiment_score) as avg_sentiment,
                    MIN(sentiment_score) as min_sentiment,
                    MAX(sentiment_score) as max_sentiment,
                    STDDEV(sentiment_score) as sentiment_stddev,
                    COUNT(CASE WHEN sentiment_score > 0.2 THEN 1 END) as positive_count,
                    COUNT(CASE WHEN sentiment_score < -0.2 THEN 1 END) as negative_count,
                    COUNT(CASE WHEN sentiment_score BETWEEN -0.2 AND 0.2 THEN 1 END) as neutral_count
                FROM news_data
                WHERE symbol = $1
                AND published_at >= $2
                AND published_at <= $3
                AND sentiment_score IS NOT NULL
            """
            
            result = await self.db_adapter.fetch_one(
                query,
                self._normalize_symbol(symbol),
                start_date,
                end_date
            )
            
            if result:
                return {
                    'symbol': symbol,
                    'period_days': days,
                    'article_count': result['article_count'],
                    'avg_sentiment': float(result['avg_sentiment']) if result['avg_sentiment'] else 0,
                    'min_sentiment': float(result['min_sentiment']) if result['min_sentiment'] else 0,
                    'max_sentiment': float(result['max_sentiment']) if result['max_sentiment'] else 0,
                    'sentiment_stddev': float(result['sentiment_stddev']) if result['sentiment_stddev'] else 0,
                    'positive_count': result['positive_count'],
                    'negative_count': result['negative_count'],
                    'neutral_count': result['neutral_count']
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {}
    
    async def get_trending_topics(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 1
    ) -> List[Dict[str, Any]]:
        """Get trending topics from recent news."""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            conditions = [
                "published_at >= $1",
                "published_at <= $2"
            ]
            params = [start_date, end_date]
            
            if symbols:
                placeholders = [f"${i+3}" for i in range(len(symbols))]
                conditions.append(f"symbol IN ({','.join(placeholders)})")
                params.extend([self._normalize_symbol(s) for s in symbols])
            
            # This is a simplified version - in production would analyze keywords/entities
            query = f"""
                SELECT 
                    symbol,
                    COUNT(*) as mention_count,
                    AVG(sentiment_score) as avg_sentiment
                FROM news_data
                WHERE {' AND '.join(conditions)}
                GROUP BY symbol
                ORDER BY mention_count DESC
                LIMIT 20
            """
            
            results = await self.db_adapter.fetch_all(query, *params)
            
            trending = []
            for row in results:
                trending.append({
                    'symbol': row['symbol'],
                    'mention_count': row['mention_count'],
                    'avg_sentiment': float(row['avg_sentiment']) if row['avg_sentiment'] else 0
                })
            
            return trending
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []
    
    async def cleanup_old_news(
        self,
        days_to_keep: int = 365
    ) -> OperationResult:
        """Clean up old news articles."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            query = "DELETE FROM news_data WHERE published_at < $1"
            result = await self.crud_executor.execute_delete(query, [cutoff_date])
            
            logger.info(f"Cleaned up {result.records_deleted} old news articles")
            
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning up old news: {e}")
            return OperationResult(success=False, error=str(e))
    
    # Private helper methods
    async def _upsert_article(self, record: Dict[str, Any]) -> None:
        """Upsert a news article."""
        query = """
            INSERT INTO news_data 
            (symbol, published_at, title, url, summary, content, author, source,
             sentiment_score, relevance_score, keywords, entities, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (symbol, url) DO UPDATE
            SET published_at = EXCLUDED.published_at,
                title = EXCLUDED.title,
                summary = EXCLUDED.summary,
                content = EXCLUDED.content,
                author = EXCLUDED.author,
                source = EXCLUDED.source,
                sentiment_score = EXCLUDED.sentiment_score,
                relevance_score = EXCLUDED.relevance_score,
                keywords = EXCLUDED.keywords,
                entities = EXCLUDED.entities
        """
        
        await self.db_adapter.execute_query(query, *list(record.values()))
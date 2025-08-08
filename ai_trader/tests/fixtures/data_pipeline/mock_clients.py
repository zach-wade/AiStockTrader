"""
Mock API clients for data pipeline testing.

Provides mock implementations of data source clients for testing
without making actual API calls.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import asyncio

from main.data_pipeline.ingestion.base_source import BaseSource
from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.types import RawDataRecord

from .mock_data import (
    create_market_data_batch,
    create_news_record,
    create_social_sentiment_record,
    ALPACA_MARKET_DATA_SAMPLE,
    POLYGON_MARKET_DATA_SAMPLE
)


class MockAlpacaMarketClient(BaseSource):
    """Mock Alpaca market data client."""
    
    def __init__(self, config=None, fail_on_fetch=False):
        self.config = config
        self.fail_on_fetch = fail_on_fetch
        self.fetch_count = 0
        
    def can_fetch(self, data_type: str) -> bool:
        """Check if this source can fetch the requested data type."""
        return data_type == 'market_data'
    
    async def fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[RawDataRecord]:
        """Fetch mock market data."""
        self.fetch_count += 1
        
        if self.fail_on_fetch:
            raise Exception("Mock fetch failure")
        
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        records = []
        for symbol in symbols:
            # Create raw record matching Alpaca format
            record = RawDataRecord(
                source='alpaca',
                data_type='market_data',
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                data=ALPACA_MARKET_DATA_SAMPLE,
                metadata={'fetch_count': self.fetch_count}
            )
            records.append(record)
        
        return records
    
    async def fetch_and_archive(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        archive: DataArchive,
        **kwargs
    ) -> Dict[str, Any]:
        """Fetch data and archive it."""
        records = await self.fetch_data(data_type, symbols, start_date, end_date, **kwargs)
        
        # Mock archiving
        archived_count = 0
        for record in records:
            # In real implementation, this would call archive methods
            archived_count += 1
        
        return {
            'records_processed': archived_count,
            'data': records
        }
    
    async def fetch_market_data(
        self, 
        symbol: str, 
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1day'
    ) -> List[Dict[str, Any]]:
        """Fetch market data for a single symbol."""
        self.fetch_count += 1
        
        if self.fail_on_fetch:
            raise Exception("Mock fetch failure")
        
        # Return mock market data
        return create_market_data_batch(symbol, start_date, end_date, timeframe=timeframe)
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        # Return mock price based on symbol
        base_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2500.0,
            'TSLA': 200.0
        }
        return base_prices.get(symbol, 100.0)
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current quote for a symbol."""
        price = await self.get_current_price(symbol)
        if price:
            return {
                'symbol': symbol,
                'bid': price - 0.01,
                'ask': price + 0.01,
                'bid_size': 100,
                'ask_size': 100,
                'last': price,
                'volume': 1000000
            }
        return None


class MockPolygonMarketClient(BaseSource):
    """Mock Polygon market data client."""
    
    def __init__(self, config=None, response_delay=0.1):
        self.config = config
        self.response_delay = response_delay
        
    def can_fetch(self, data_type: str) -> bool:
        """Check if this source can fetch the requested data type."""
        return data_type in ['market_data', 'options']
    
    async def fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[RawDataRecord]:
        """Fetch mock data."""
        await asyncio.sleep(self.response_delay)
        
        records = []
        for symbol in symbols:
            record = RawDataRecord(
                source='polygon',
                data_type=data_type,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                data=POLYGON_MARKET_DATA_SAMPLE,
                metadata={'api_version': '2'}
            )
            records.append(record)
        
        return records
    
    async def fetch_and_archive(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        archive: DataArchive,
        **kwargs
    ) -> Dict[str, Any]:
        """Fetch data and archive it."""
        records = await self.fetch_data(data_type, symbols, start_date, end_date, **kwargs)
        return {
            'records_processed': len(records),
            'data': records
        }
    
    async def fetch_market_data(
        self, 
        symbol: str, 
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1day'
    ) -> List[Dict[str, Any]]:
        """Fetch market data for a single symbol."""
        # Return mock market data from Polygon
        return create_market_data_batch(symbol, start_date, end_date, timeframe=timeframe)
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        # Return mock price based on symbol
        base_prices = {
            'AAPL': 150.5,
            'MSFT': 300.5,
            'GOOGL': 2500.5,
            'TSLA': 200.5
        }
        return base_prices.get(symbol, 100.5)
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current quote for a symbol."""
        price = await self.get_current_price(symbol)
        if price:
            return {
                'symbol': symbol,
                'bid': price - 0.02,
                'ask': price + 0.02,
                'bid_size': 200,
                'ask_size': 200,
                'last': price,
                'volume': 2000000
            }
        return None


class MockNewsClient(BaseSource):
    """Mock news data client."""
    
    def __init__(self, config=None, source_name='mock_news'):
        self.config = config
        self.source_name = source_name
        
    def can_fetch(self, data_type: str) -> bool:
        """Check if this source can fetch the requested data type."""
        return data_type == 'news'
    
    async def fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[RawDataRecord]:
        """Fetch mock news data."""
        records = []
        
        for symbol in symbols:
            # Create 3 news items per symbol
            for i in range(3):
                news_data = create_news_record(symbol, datetime.now(timezone.utc), self.source_name)
                record = RawDataRecord(
                    source=self.source_name,
                    data_type='news',
                    symbol=symbol,
                    timestamp=news_data['created_at'],
                    data=news_data,
                    metadata={'article_index': i}
                )
                records.append(record)
        
        return records
    
    async def fetch_and_archive(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        archive: DataArchive,
        **kwargs
    ) -> Dict[str, Any]:
        """Fetch data and archive it."""
        records = await self.fetch_data(data_type, symbols, start_date, end_date, **kwargs)
        return {
            'records_processed': len(records),
            'data': records
        }
    
    async def fetch_market_data(
        self, 
        symbol: str, 
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1day'
    ) -> List[Dict[str, Any]]:
        """News clients don't fetch market data."""
        return []
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """News clients don't provide price data."""
        return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """News clients don't provide quote data."""
        return None


class MockSocialSentimentClient(BaseSource):
    """Mock social sentiment client."""
    
    def __init__(self, config=None, platform='reddit'):
        self.config = config
        self.platform = platform
        
    def can_fetch(self, data_type: str) -> bool:
        """Check if this source can fetch the requested data type."""
        return data_type == 'social_sentiment'
    
    async def fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[RawDataRecord]:
        """Fetch mock social sentiment data."""
        records = []
        
        for symbol in symbols:
            # Create 5 social posts per symbol
            for i in range(5):
                social_data = create_social_sentiment_record(
                    symbol, 
                    datetime.now(timezone.utc),
                    self.platform
                )
                record = RawDataRecord(
                    source=self.platform,
                    data_type='social_sentiment',
                    symbol=symbol,
                    timestamp=social_data['timestamp'],
                    data=social_data,
                    metadata={'post_index': i}
                )
                records.append(record)
        
        return records
    
    async def fetch_and_archive(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        archive: DataArchive,
        **kwargs
    ) -> Dict[str, Any]:
        """Fetch data and archive it."""
        records = await self.fetch_data(data_type, symbols, start_date, end_date, **kwargs)
        return {
            'records_processed': len(records),
            'data': records
        }
    
    async def fetch_market_data(
        self, 
        symbol: str, 
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1day'
    ) -> List[Dict[str, Any]]:
        """Social sentiment clients don't fetch market data."""
        return []
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Social sentiment clients don't provide price data."""
        return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Social sentiment clients don't provide quote data."""
        return None


def create_mock_clients_dict(config=None) -> Dict[str, BaseSource]:
    """Create a dictionary of mock clients for testing."""
    return {
        'alpaca_market': MockAlpacaMarketClient(config),
        'polygon_market': MockPolygonMarketClient(config),
        'alpaca_news': MockNewsClient(config, 'alpaca_news'),
        'benzinga_news': MockNewsClient(config, 'benzinga'),
        'reddit_social': MockSocialSentimentClient(config, 'reddit'),
        'twitter_social': MockSocialSentimentClient(config, 'twitter')
    }
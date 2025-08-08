"""
Mock services for integration testing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import random  # DEPRECATED - use secure_random
from main.utils.core import secure_uniform, secure_randint, secure_choice, secure_sample, secure_shuffle

from main.utils.exceptions import APIError, RateLimitError


class MockAlpacaClient:
    """Mock Alpaca API client for testing."""
    
    def __init__(self, fail_rate: float = 0.0, rate_limit: bool = False):
        self.fail_rate = fail_rate
        self.rate_limit = rate_limit
        self.call_count = 0
        self.rate_limit_threshold = 10
    
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1Day'
    ) -> List[Dict]:
        """Mock market data bars."""
        self.call_count += 1
        
        # Simulate rate limiting
        if self.rate_limit and self.call_count > self.rate_limit_threshold:
            raise RateLimitError(
                "Rate limit exceeded",
                status_code=429,
                api_name="alpaca"
            )
        
        # Simulate random failures
        if random.random() < self.fail_rate:
            raise APIError(
                "Mock API failure",
                status_code=500,
                api_name="alpaca"
            )
        
        # Generate mock data
        bars = []
        current = start
        base_price = 100.0
        
        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                price_change = secure_uniform(-0.02, 0.02)
                open_price = base_price * (1 + price_change)
                high_price = open_price * secure_uniform(1.0, 1.02)
                low_price = open_price * secure_uniform(0.98, 1.0)
                close_price = secure_uniform(low_price, high_price)
                
                bars.append({
                    'timestamp': current,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': secure_randint(1000000, 5000000),
                    'trade_count': secure_randint(1000, 5000),
                    'vwap': (open_price + close_price) / 2,
                })
                
                base_price = close_price
            
            current += timedelta(days=1)
        
        return bars
    
    async def get_latest_quote(self, symbol: str) -> Dict:
        """Mock latest quote data."""
        base_price = 100.0
        spread = 0.01
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'bid_price': base_price - spread/2,
            'ask_price': base_price + spread/2,
            'bid_size': secure_randint(100, 1000),
            'ask_size': secure_randint(100, 1000),
            'conditions': ['Normal'],
        }
    
    async def get_trades(
        self,
        symbol: str,
        start: datetime,
        limit: int = 100
    ) -> List[Dict]:
        """Mock trade data."""
        trades = []
        current_time = start
        base_price = 100.0
        
        for i in range(limit):
            price_change = secure_uniform(-0.001, 0.001)
            price = base_price * (1 + price_change)
            
            trades.append({
                'symbol': symbol,
                'timestamp': current_time + timedelta(seconds=i),
                'price': price,
                'size': secure_randint(100, 1000),
                'conditions': ['Normal'],
                'trade_id': f"trade_{i}",
            })
            
            base_price = price
        
        return trades


class MockNewsClient:
    """Mock news API client for testing."""
    
    def __init__(self):
        self.headlines = [
            "{} Reports Strong Quarterly Earnings",
            "Analysts Upgrade {} to Buy Rating",
            "{} Announces Strategic Partnership",
            "{} Faces Regulatory Scrutiny",
            "{} CEO Discusses Growth Strategy",
        ]
        
        self.sources = ['Reuters', 'Bloomberg', 'WSJ', 'CNBC', 'MarketWatch']
    
    async def get_news(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        limit: int = 50
    ) -> List[Dict]:
        """Mock news articles."""
        articles = []
        
        for symbol in symbols:
            num_articles = secure_randint(1, min(10, limit))
            
            for i in range(num_articles):
                # Random timestamp within range
                time_offset = secure_uniform(0, (end_date - start_date).total_seconds())
                timestamp = start_date + timedelta(seconds=time_offset)
                
                # Generate sentiment based on headline
                headline = secure_choice(self.headlines).format(symbol)
                if any(word in headline.lower() for word in ['strong', 'upgrade', 'growth']):
                    sentiment = secure_uniform(0.1, 0.5)
                elif any(word in headline.lower() for word in ['scrutiny', 'faces', 'concerns']):
                    sentiment = secure_uniform(-0.5, -0.1)
                else:
                    sentiment = secure_uniform(-0.2, 0.2)
                
                articles.append({
                    'id': f"news_{symbol}_{i}",
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'headline': headline,
                    'summary': f"Summary for {headline}",
                    'source': secure_choice(self.sources),
                    'url': f"https://example.com/news/{symbol}/{i}",
                    'sentiment': sentiment,
                    'relevance': secure_uniform(0.5, 1.0),
                })
        
        # Sort by timestamp
        articles.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return articles[:limit]


class MockDatabasePool:
    """Mock database connection pool for testing."""
    
    def __init__(self):
        self.data = {}
        self.query_count = 0
        self.transaction_active = False
    
    async def acquire(self):
        """Acquire mock connection."""
        return MockConnection(self)
    
    async def close(self):
        """Close mock pool."""
        pass


class MockConnection:
    """Mock database connection."""
    
    def __init__(self, pool: MockDatabasePool):
        self.pool = pool
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def execute(self, query: str, *args):
        """Execute mock query."""
        self.pool.query_count += 1
        return "OK"
    
    async def fetch(self, query: str, *args):
        """Fetch mock results."""
        self.pool.query_count += 1
        
        # Return mock data based on query patterns
        if "COUNT(*)" in query:
            return [{'count': secure_randint(0, 100)}]
        elif "companies" in query.lower():
            return [
                {'symbol': 'AAPL', 'name': 'Apple Inc.'},
                {'symbol': 'MSFT', 'name': 'Microsoft Corp.'},
            ]
        else:
            return []
    
    async def fetchrow(self, query: str, *args):
        """Fetch single row."""
        results = await self.fetch(query, *args)
        return results[0] if results else None
    
    async def fetchval(self, query: str, *args):
        """Fetch single value."""
        row = await self.fetchrow(query, *args)
        return list(row.values())[0] if row else None


class MockCache:
    """Mock cache for testing."""
    
    def __init__(self):
        self.data = {}
        self.hit_count = 0
        self.miss_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.data:
            self.hit_count += 1
            return self.data[key]
        else:
            self.miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        self.data[key] = value
    
    async def delete(self, key: str):
        """Delete value from cache."""
        if key in self.data:
            del self.data[key]
    
    async def clear(self):
        """Clear all cache data."""
        self.data.clear()
    
    async def close(self):
        """Close cache connection."""
        pass
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'size': len(self.data),
        }
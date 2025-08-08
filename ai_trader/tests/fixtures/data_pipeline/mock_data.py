"""
Mock data fixtures for data pipeline tests.

Provides sample data for various data types including market data,
news, social sentiment, etc.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import random  # DEPRECATED - use secure_random
from main.utils.core import secure_uniform, secure_randint, secure_choice, secure_sample, secure_shuffle


def create_market_data_record(
    symbol: str,
    timestamp: datetime,
    price_base: float = 100.0
) -> Dict[str, Any]:
    """Create a single market data record."""
    # Add some randomness to prices
    variation = secure_uniform(0.95, 1.05)
    
    return {
        'symbol': symbol,
        'timestamp': timestamp,
        'open': price_base * variation,
        'high': price_base * variation * 1.02,
        'low': price_base * variation * 0.98,
        'close': price_base * variation * 1.01,
        'volume': secure_randint(1000000, 5000000),
        'vwap': price_base * variation * 1.005,
        'trade_count': secure_randint(5000, 20000)
    }


def create_market_data_batch(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    interval_minutes: int = 60
) -> List[Dict[str, Any]]:
    """Create a batch of market data records."""
    records = []
    current_time = start_date
    
    while current_time <= end_date:
        for symbol in symbols:
            records.append(create_market_data_record(symbol, current_time))
        current_time += timedelta(minutes=interval_minutes)
    
    return records


def create_news_record(
    symbol: str,
    timestamp: datetime,
    source: str = 'test_source'
) -> Dict[str, Any]:
    """Create a single news record."""
    headlines = [
        f"{symbol} Reports Strong Quarterly Earnings",
        f"Analysts Upgrade {symbol} to Buy",
        f"{symbol} Announces New Product Launch",
        f"Market Update: {symbol} Trading Higher",
        f"{symbol} CEO Discusses Growth Strategy"
    ]
    
    return {
        'id': f"news_{symbol}_{timestamp.timestamp()}",
        'symbols': [symbol],
        'headline': secure_choice(headlines),
        'summary': f"Sample news summary for {symbol}",
        'author': 'Test Author',
        'created_at': timestamp,
        'updated_at': timestamp,
        'url': f"https://example.com/news/{symbol}",
        'source': source,
        'sentiment_score': secure_uniform(-1, 1)
    }


def create_social_sentiment_record(
    symbol: str,
    timestamp: datetime,
    platform: str = 'reddit'
) -> Dict[str, Any]:
    """Create a single social sentiment record."""
    return {
        'id': f"social_{platform}_{symbol}_{timestamp.timestamp()}",
        'symbol': symbol,
        'platform': platform,
        'timestamp': timestamp,
        'content': f"Sample {platform} post about {symbol}",
        'author': f"test_user_{secure_randint(1, 100)}",
        'score': secure_randint(1, 1000),
        'num_comments': secure_randint(0, 500),
        'sentiment_score': secure_uniform(-1, 1),
        'reach': secure_randint(100, 10000)
    }


def create_validation_test_data() -> Dict[str, List[Dict[str, Any]]]:
    """Create data with various validation scenarios."""
    now = datetime.now(timezone.utc)
    
    return {
        'valid_data': [
            create_market_data_record('AAPL', now),
            create_market_data_record('MSFT', now)
        ],
        'missing_fields': [
            {
                'symbol': 'AAPL',
                'timestamp': now,
                'open': 150.0,
                # Missing required fields: high, low, close, volume
            }
        ],
        'invalid_values': [
            {
                'symbol': 'AAPL',
                'timestamp': now,
                'open': 150.0,
                'high': 140.0,  # High < open (invalid)
                'low': 160.0,   # Low > high (invalid)
                'close': 155.0,
                'volume': -1000  # Negative volume (invalid)
            }
        ],
        'null_values': [
            {
                'symbol': 'AAPL',
                'timestamp': now,
                'open': None,
                'high': None,
                'low': None,
                'close': None,
                'volume': 1000000
            }
        ]
    }


def create_bulk_test_data(num_records: int = 10000) -> List[Dict[str, Any]]:
    """Create large dataset for performance testing."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
    start_date = datetime.now(timezone.utc) - timedelta(days=30)
    
    records = []
    for i in range(num_records):
        symbol = secure_choice(symbols)
        timestamp = start_date + timedelta(minutes=i)
        records.append(create_market_data_record(symbol, timestamp))
    
    return records


# Raw data samples matching actual API responses
ALPACA_MARKET_DATA_SAMPLE = {
    'bars': [
        {
            't': '2024-01-15T14:30:00Z',
            'o': 150.25,
            'h': 151.50,
            'l': 149.75,
            'c': 151.00,
            'v': 1234567,
            'n': 5432,
            'vw': 150.75
        }
    ],
    'symbol': 'AAPL',
    'next_page_token': None
}

POLYGON_MARKET_DATA_SAMPLE = {
    'status': 'OK',
    'results': [
        {
            't': 1705332600000,  # Unix timestamp in milliseconds
            'o': 150.25,
            'h': 151.50,
            'l': 149.75,
            'c': 151.00,
            'v': 1234567,
            'n': 5432,
            'vw': 150.75
        }
    ],
    'ticker': 'AAPL'
}

YAHOO_MARKET_DATA_SAMPLE = {
    'timestamp': [1705332600],
    'indicators': {
        'quote': [
            {
                'open': [150.25],
                'high': [151.50],
                'low': [149.75],
                'close': [151.00],
                'volume': [1234567]
            }
        ]
    }
}
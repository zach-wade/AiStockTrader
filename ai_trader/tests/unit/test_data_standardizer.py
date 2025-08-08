# File: tests/unit/test_data_standardizer.py

import pytest
import pandas as pd
from datetime import datetime
import logging

# The class we are testing
from main.data_pipeline.processing.standardizer import DataStandardizer

# --- Test Fixtures ---

@pytest.fixture
def standardizer_config():
    """Provides a mock config dictionary for the standardizer."""
    return {
        "data": {
            "column_mappings": {
                "alpaca": {
                    't': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 
                    'c': 'close', 'v': 'volume', 'vw': 'vwap', 'n': 'trades'
                },
                "benzinga": { # Assuming we might standardize Benzinga market data one day
                    # Add mappings if needed
                }
            }
        }
    }

@pytest.fixture
def data_standardizer(standardizer_config):
    """Provides an initialized DataStandardizer instance."""
    return DataStandardizer(config=standardizer_config)

@pytest.fixture
def raw_alpaca_df():
    """Provides a sample raw DataFrame as returned by the Alpaca API."""
    data = {
        'o': [150.0, 151.0],
        'h': [152.0, 151.5],
        'l': [149.5, 150.5],
        'c': [151.5, 150.8],
        'v': [10000, 12000],
        'n': [100, 120],
        'vw': [151.0, 150.9]
    }
    # Alpaca uses nanosecond precision timestamps
    index = pd.to_datetime(['2025-06-23 14:30:00', '2025-06-23 14:31:00'], utc=True)
    return pd.DataFrame(data, index=index)

@pytest.fixture
def raw_benzinga_news_list():
    """Provides a sample list of raw news from main.Benzinga."""
    return [
        {
            "id": 98765,
            "title": "Quantum Inc. Announces Breakthrough",
            "author": "Futuristic Wire",
            "created": "Mon, 23 Jun 2025 15:00:00 +0000",
            "updated": "Mon, 23 Jun 2025 15:01:00 +0000",
            "teaser": "Shares of Quantum Inc. (QTM) are soaring after a major breakthrough.",
            "url": "https://example.com/news/98765",
            "stocks": [{"name": "QTM"}],
            "tags": [{"name": "Technology"}]
        }
    ]


# --- Test Cases ---

def test_standardize_market_data_success(data_standardizer, raw_alpaca_df):
    """Tests successful standardization of an Alpaca DataFrame."""
    test_data = raw_alpaca_df.copy()
    test_data.reset_index(inplace=True)  # Move index to column
    test_data.rename(columns={'index': 'timestamp'}, inplace=True)
    # Act
    df = data_standardizer.standardize_market_data(raw_alpaca_df, source='alpaca', symbol='TEST')

    # Assert
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz.zone == 'UTC'
    
    expected_columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades']
    assert all(col in df.columns for col in expected_columns)
    
    assert df['symbol'].iloc[0] == 'TEST'
    assert df['open'].iloc[0] == 150.0
    assert df['trades'].iloc[0] == 100
    assert not df.empty

def test_standardize_market_data_no_timestamp(data_standardizer, raw_alpaca_df, caplog):
    """Tests failure when the timestamp column cannot be identified."""
    # Arrange: Create a dataframe without a recognizable timestamp column
    bad_df = raw_alpaca_df.copy().reset_index(drop=True)
    caplog.set_level(logging.ERROR)
    
    # Act
    df = data_standardizer.standardize_market_data(bad_df, source='alpaca', symbol='TEST')
    
    # Assert
    assert df.empty
    assert "Standardization failed for TEST: 'timestamp' column not found" in caplog.text

def test_standardize_benzinga_news(data_standardizer, raw_benzinga_news_list):
    """Tests successful standardization of a Benzinga news item."""
    # Act
    articles = [data_standardizer.standardize_benzinga_news_item(item) for item in raw_benzinga_news_list]
    
    # Assert
    assert len(articles) == 1
    article = articles[0]
    
    expected_keys = ['source_id', 'headline', 'summary', 'author', 'created_at', 'symbols', 'source']
    assert all(key in article for key in expected_keys)
    assert article['source_id'] == 98765
    assert article['source'] == 'benzinga'
    assert article['symbols'] == ['QTM']
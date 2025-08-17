# Social Sentiment Data Integration Guide

## Overview

This document outlines the work completed and remaining tasks for integrating Reddit and Twitter sentiment data into the AI Trader pipeline.

## ✅ Completed Work

### 1. Orchestrator Integration

- Added imports for `RedditClient` and `TwitterClient` in `orchestrator.py`
- Added Reddit and Twitter to the client registry initialization
- Created `run_social_sentiment_collection()` method for collecting social data
- Imported `SocialSentimentRepository` for data storage

### 2. Social Sentiment Collection Method

The new method in `orchestrator.py`:

- Collects data from Reddit (wallstreetbets, stocks, investing, StockMarket)
- Collects data from Twitter using cashtag searches
- Processes posts/tweets to extract mentions of stock symbols
- Formats data with metadata (scores, comments, timestamps, etc.)
- Stores data using the social sentiment repository

## ❌ Remaining Work

### 1. Fix Abstract Method Implementation

The Reddit and Twitter clients inherit from `BaseSource` but don't implement required abstract methods:

- `fetch_news()`
- `fetch_raw_data()`
- `get_all_tradable_assets()`

**Solution**: Add these methods to both clients (can be empty implementations if not applicable)

### 2. Create SocialSentimentData Model

Add to `data_pipeline/storage/database.py`:

```python
class SocialSentimentData(Base):
    __tablename__ = 'social_sentiment'

    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)  # 'reddit' or 'twitter'
    platform_id = Column(String, nullable=False)
    author = Column(String)
    content = Column(Text, nullable=False)
    url = Column(String)
    score = Column(Integer, default=0)
    num_comments = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False)
    subreddit = Column(String)  # For Reddit posts
    metadata = Column(JSON)  # For additional platform-specific data
    sentiment_score = Column(Float)  # For future sentiment analysis

    __table_args__ = (
        Index('idx_social_sentiment_symbol_created', 'symbol', 'created_at'),
        UniqueConstraint('source', 'platform_id', name='uq_social_sentiment_source_id')
    )
```

### 3. Update Repository Initialization

In `orchestrator.py`, change:

```python
from ..database import SocialSentimentData
# ...
self.social_sentiment_repo = SocialSentimentRepository(self.db_adapter, SocialSentimentData)
```

### 4. Create Database Migration

Create a migration to add the `social_sentiment` table to the database.

### 5. Connect to Feature Pipeline

Update `feature_pipeline/calculators/sentiment_calculator.py` to:

- Query social sentiment data from the repository
- Calculate sentiment scores per symbol
- Add features like:
  - `social_sentiment_score`
  - `reddit_mention_count`
  - `twitter_mention_count`
  - `social_momentum` (change in mentions over time)

### 6. Update Configuration

Add API keys to `.env`:

```
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=ai_trader_bot/1.0

TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
```

### 7. Schedule Collection

Add to the data collection schedule:

- Run social sentiment collection every hour
- Focus on high-volume symbols and trending stocks
- Implement rate limiting to respect API limits

## Testing

### Test Script Available

`scripts/test_social_sentiment_integration.py` - Tests the integration once the above fixes are implemented.

### Manual Testing Steps

1. Ensure API keys are configured
2. Run: `python scripts/test_social_sentiment_integration.py`
3. Check database for stored sentiment data
4. Verify feature pipeline generates sentiment features

## Benefits Once Complete

1. **Enhanced Sentiment Analysis**: Real-time social media sentiment integrated with news sentiment
2. **Meme Stock Detection**: Identify unusual social media activity spikes
3. **Improved Predictions**: Social sentiment features for ML models
4. **Risk Management**: Detect potential volatility from social media trends

## Priority

This is marked as HIGH priority in the refactor log because:

- Sentiment data is already expected by existing strategies
- The `FinalSentimentStrategy` requires social sentiment features
- Yahoo fundamentals and Benzinga news are already integrated, completing the multi-source data picture

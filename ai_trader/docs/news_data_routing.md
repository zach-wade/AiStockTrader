# News Data Routing Documentation

## Overview

The AI Trader system uses a dual storage architecture with intelligent routing for optimal performance and cost efficiency. This document explains how news data is stored and accessed, particularly for backtesting scenarios.

## Storage Architecture

### Hot Storage (PostgreSQL Database)
- **Purpose**: Fast access to recent and frequently accessed data
- **Default Retention**: 30 days (configurable)
- **Schema**: Structured with full news_data table including:
  - Sentiment analysis fields (score, label, magnitude)
  - Full content and metadata
  - JSONB fields for flexible data (symbols, keywords, insights)
  - Proper indexing for performance

### Cold Storage (Data Lake)
- **Purpose**: Long-term archival of historical data
- **Format**: Raw JSON files as received from providers
- **Structure**: `/data_lake/raw/alternative_data/news/symbol={symbol}/date={date}/`
- **Schema**: Raw API responses, requires transformation for use

## Routing Logic

### StorageRouter Component

The `StorageRouter` class determines which storage tier to use based on:

1. **Repository-specific overrides** (highest priority)
2. **Query type overrides**
3. **Time-based routing** (default)

### Current Configuration

```yaml
storage:
  routing:
    # News repository override
    repository_overrides:
      news:
        force_tier: hot
        reason: "New schema with sentiment analysis only in hot storage"
    
    # Query type defaults
    query_type_overrides:
      real_time: hot       # Live trading queries
      analysis: hot        # Backtesting and analysis
      feature_calc: hot    # Feature calculations
      bulk_export: cold    # Large data exports
      admin: both         # Administrative queries
```

### News Data Routing

Currently, **all news queries use hot storage (PostgreSQL)** because:

1. The new schema with sentiment analysis is only available in hot storage
2. Cold storage contains raw JSON that would need transformation
3. Repository override forces hot storage for all news queries

## Backtesting Data Access

### Data Flow for Backtesting

```
BacktestEngine
    ↓
DataLoader (UnifiedDataLoader)
    ↓
AlternativeDataSource (for news)
    ↓
StorageRouter.execute_query()
    ↓
Decision: Use HOT storage (PostgreSQL)
    ↓
NewsRepository.get_by_filter()
    ↓
Return structured news data with sentiment
```

### Key Points for Backtesting

1. **Backtesting uses QueryType.ANALYSIS** which defaults to hot storage
2. **News repository override** ensures hot storage regardless of query type
3. **Fallback mechanism** exists but rarely needed since hot storage is primary
4. **Performance**: Fast access since all queries hit indexed PostgreSQL tables

## Future Improvements

### Cold Storage Transformation (TODO)

To enable cold storage usage for historical news:

1. Implement transformation layer in `ColdStorageQueryEngine`
2. Map raw JSON fields to structured schema
3. Calculate sentiment scores if missing
4. Cache transformed results for performance

### Hybrid Approach

For very large historical backtests:

1. Use hot storage for recent data (< 30 days)
2. Use cold storage with transformation for historical data
3. Merge results transparently in DataLoader

## Configuration Changes

To modify routing behavior:

1. Edit `/src/main/config/dual_storage.yaml`
2. Update the `storage.routing` section
3. Repository overrides take precedence over query type defaults

### Example: Enable Cold Storage for News

```yaml
storage:
  routing:
    repository_overrides:
      news:
        force_tier: null  # Remove override
        # Now uses standard time-based routing
```

## Monitoring and Debugging

### Check Current Routing

```python
from main.data_pipeline.storage.storage_router import StorageRouter, QueryType
from main.data_pipeline.storage.repositories.repository_types import QueryFilter

router = StorageRouter()
decision = router.route_query(
    query_filter=QueryFilter(start_date=..., end_date=...),
    query_type=QueryType.ANALYSIS,
    repository_name='news'
)
print(f"Tier: {decision.primary_tier.value}, Reason: {decision.reason}")
```

### Routing Statistics

The StorageRouter tracks usage statistics:
- `hot_queries`: Number of queries routed to hot storage
- `cold_queries`: Number of queries routed to cold storage
- `fallback_used`: Number of times fallback tier was used

## Best Practices

1. **Keep hot storage optimized**: Only recent/frequently accessed data
2. **Use repository overrides sparingly**: Only when schema differences exist
3. **Monitor performance**: Track query times for both storage tiers
4. **Plan migrations carefully**: When moving between storage tiers

## Summary

The current configuration ensures that backtesting always accesses news data from the hot storage (PostgreSQL database) which provides:
- Consistent structured schema
- Pre-calculated sentiment analysis
- Fast indexed queries
- No transformation overhead

This approach prioritizes performance and data quality over storage cost optimization for news data.
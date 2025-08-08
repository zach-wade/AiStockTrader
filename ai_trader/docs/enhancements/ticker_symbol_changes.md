# Ticker Symbol Change Enhancement Proposals

## Overview
This document outlines proposed enhancements for handling ticker symbol changes, corporate actions, and related data continuity issues in the AI Trader system. These proposals arose from investigating the CMRC/BIGC ticker change on August 1, 2025.

## Current Issue
When companies change their ticker symbols (e.g., BigCommerce from BIGC to CMRC), the system cannot retrieve historical data under the new symbol, leading to:
- "No aggregates returned from Polygon" warnings
- Data discontinuity
- Loss of historical analysis capabilities
- Confusion about data availability

## Proposed Enhancements

### 1. Ticker Symbol Change Tracking System

#### 1.1 Database Schema
Create new tables to track symbol changes:

```sql
-- Track all ticker symbol changes
CREATE TABLE ticker_changes (
    id SERIAL PRIMARY KEY,
    old_symbol VARCHAR(10) NOT NULL,
    new_symbol VARCHAR(10) NOT NULL,
    change_date DATE NOT NULL,
    company_name VARCHAR(255),
    exchange VARCHAR(50),
    reason VARCHAR(100), -- 'rebrand', 'merger', 'spinoff', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(old_symbol, new_symbol, change_date)
);

-- Track current active symbols and their history
CREATE TABLE symbol_lineage (
    current_symbol VARCHAR(10) PRIMARY KEY,
    symbol_history JSONB, -- Array of {symbol, start_date, end_date}
    company_id VARCHAR(50), -- Permanent company identifier
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 1.2 Implementation Features
- Automatic detection of symbol changes from data providers
- Manual override capability for corrections
- API endpoint for symbol lookups and history

### 2. Data Fetching Enhancements

#### 2.1 Smart Historical Data Retrieval
Modify `polygon_market_client.py` to:
```python
async def fetch_with_symbol_history(self, symbol: str, start_date: datetime, end_date: datetime):
    """Fetch data considering symbol changes"""
    # Check if symbol has history
    symbol_history = await self.get_symbol_history(symbol)
    
    # Fetch data for each historical symbol in the date range
    all_data = []
    for hist in symbol_history:
        if hist['start_date'] <= end_date and hist['end_date'] >= start_date:
            data = await self.fetch_data(hist['symbol'], 
                                       max(start_date, hist['start_date']),
                                       min(end_date, hist['end_date']))
            all_data.append(data)
    
    return self.merge_historical_data(all_data)
```

#### 2.2 Fallback Mechanisms
- If no data found for current symbol, automatically check previous symbols
- Configurable behavior: auto-fallback vs. explicit user confirmation
- Cache symbol mappings for performance

### 3. Data Migration Tools

#### 3.1 Bulk Symbol Update Utility
```python
class SymbolMigrator:
    """Handle bulk updates of ticker symbols in the database"""
    
    async def migrate_symbol(self, old_symbol: str, new_symbol: str, 
                           change_date: date, preserve_history: bool = True):
        """Migrate all data from old to new symbol"""
        # Options:
        # 1. Update all records to new symbol
        # 2. Keep historical data under old symbol, new data under new
        # 3. Duplicate records with both symbols for transition period
```

#### 3.2 Data Reconciliation
- Tools to identify and merge duplicate data
- Validation of data continuity across symbol changes
- Reporting on affected date ranges and data types

### 4. User Interface Enhancements

#### 4.1 Symbol Change Notifications
- Alert users when querying symbols with recent changes
- Suggest using historical symbols for specific date ranges
- Display symbol history in UI components

#### 4.2 Transparent Data Access
```python
# Example API response with symbol metadata
{
    "symbol": "CMRC",
    "data": [...],
    "metadata": {
        "symbol_history": [
            {"symbol": "BIGC", "period": "2020-08-05 to 2025-07-31"},
            {"symbol": "CMRC", "period": "2025-08-01 to present"}
        ],
        "data_sources": {
            "2020-08-05 to 2025-07-31": "BIGC via Polygon",
            "2025-08-01 to present": "CMRC via Polygon"
        }
    }
}
```

### 5. Corporate Actions Integration

#### 5.1 Comprehensive Corporate Events Tracking
Extend beyond simple ticker changes to track:
- Mergers and acquisitions
- Spin-offs and split-offs
- Delisting and relisting
- Exchange transfers
- Name changes without ticker changes

#### 5.2 Data Provider Integration
- Polygon corporate actions API
- Yahoo Finance events feed
- Manual data entry interface for corrections

### 6. Advanced Features

#### 6.1 Symbol Prediction
- ML model to predict potential ticker changes based on:
  - Corporate news sentiment
  - SEC filing patterns
  - Industry consolidation trends

#### 6.2 Automated Data Healing
- Background jobs to detect and fix data gaps
- Automatic backfill when symbol changes are detected
- Self-healing data pipelines

#### 6.3 Cross-Reference Database
- Map symbols across different data providers
- Handle provider-specific symbol conventions
- Support for international symbol formats

### 7. Error Handling Improvements

#### 7.1 Enhanced Error Messages
Replace generic "No aggregates returned" with:
```python
if not data and await self.is_new_symbol(symbol):
    ticker_info = await self.get_ticker_info(symbol)
    if ticker_info.previous_symbols:
        logger.warning(
            f"No data found for {symbol}. This symbol started trading on "
            f"{ticker_info.start_date}. Historical data may be available "
            f"under previous symbol(s): {', '.join(ticker_info.previous_symbols)}"
        )
```

#### 7.2 Automatic Recovery Suggestions
- Suggest alternative symbols or date ranges
- Provide direct links to fetch historical data
- Option to auto-retry with suggested parameters

### 8. Performance Optimizations

#### 8.1 Symbol Mapping Cache
- In-memory cache of symbol mappings
- Redis cache for distributed systems
- Periodic refresh from authoritative source

#### 8.2 Batch Processing
- Bulk symbol resolution for large universes
- Parallel fetching for multi-symbol histories
- Optimized database queries for symbol lookups

### 9. Testing and Validation

#### 9.1 Test Data Sets
- Create test cases for common symbol change scenarios
- Historical symbol change data for backtesting
- Edge cases: multiple changes, circular changes, etc.

#### 9.2 Data Quality Metrics
- Monitor data continuity scores
- Alert on unexpected gaps after symbol changes
- Regular audits of symbol mapping accuracy

### 10. Documentation and Training

#### 10.1 User Guide
- How to handle symbol changes in queries
- Best practices for historical analysis
- Troubleshooting guide for common issues

#### 10.2 Developer Documentation
- API changes and new endpoints
- Database schema updates
- Integration examples

## Implementation Priority

1. **High Priority** (1-2 months)
   - Basic ticker change tracking table
   - Manual symbol mapping capability
   - Enhanced error messages

2. **Medium Priority** (3-4 months)
   - Automatic historical data fetching
   - Symbol change detection
   - Basic UI notifications

3. **Low Priority** (6+ months)
   - Advanced ML predictions
   - Comprehensive corporate actions
   - Full automation

## Estimated Impact

- **User Experience**: Significant improvement in data continuity
- **Data Quality**: Reduced gaps and missing data issues  
- **Maintenance**: Lower support burden for symbol-related issues
- **Scalability**: Better handling of growing symbol universe

## Next Steps

1. Review and prioritize enhancements with team
2. Create detailed technical specifications
3. Estimate development effort for each component
4. Begin with high-priority database schema changes
5. Implement basic symbol mapping functionality

## Related Issues

- Archive system limitations with symbol changes
- Backfill process assumptions about symbol stability
- Feature calculation continuity across symbol changes
- Portfolio tracking with changing symbols

---

*Document created: August 2025*  
*Last updated: August 2025*  
*Author: AI Trading System Team*
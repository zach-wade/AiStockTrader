# Backfill System Enhancement Roadmap

## Executive Summary

The current backfill system uses a complex tier-based architecture that was originally designed around market cap classifications. With the introduction of the scanner layer system (Layers 0-3), there's now a mismatch between how symbols are classified for trading versus data retention. This document outlines a roadmap to simplify and improve the backfill system.

### Current State
- **Tier System**: Symbols are categorized into Priority/Active/Standard/Archive tiers based on market cap
- **Scanner Layers**: Symbols are qualified through Layers 0-3 based on liquidity and trading characteristics
- **Issue**: Tier system overrides user-specified date ranges, causing data gaps
- **Complexity**: Two competing classification systems create confusion

### Key Problems Identified
1. Archive tier forcing 7-day lookback on Layer 0 symbols
2. Redundant classification systems (tiers vs scanner layers)
3. Complex parameter passing and configuration
4. Lack of visibility into data retention decisions

## Immediate Enhancements (1-2 Weeks)

### 1. Simplify Tier System Usage
**Goal**: Make the tier system optional and transparent

```python
# Add configuration option
backfill:
  orchestration:
    use_symbol_tiers: false  # Disable by default
    always_respect_user_days: true  # Never override --days parameter
```

**Implementation**:
- Add bypass logic in orchestrator to skip tier categorization
- Use simple batching based on symbol count
- Maintain tier system only for parallel processing limits

### 2. Scanner Layer Integration
**Goal**: Update companies table with scanner qualification data

```sql
-- Add missing columns
ALTER TABLE companies 
ADD COLUMN layer1_qualified BOOLEAN DEFAULT FALSE,
ADD COLUMN layer2_qualified BOOLEAN DEFAULT FALSE,
ADD COLUMN layer3_qualified BOOLEAN DEFAULT FALSE,
ADD COLUMN liquidity_score DECIMAL(10,2),
ADD COLUMN layer1_updated TIMESTAMP,
ADD COLUMN layer2_updated TIMESTAMP,
ADD COLUMN layer3_updated TIMESTAMP;
```

**Implementation**:
- Update Layer1LiquidityFilter to set layer1_qualified and liquidity_score
- Update Layer2CatalystOrchestrator to set layer2_qualified
- Update Layer3 scanners to set layer3_qualified
- Add batch update methods for performance

### 3. Improve Error Handling
**Goal**: Better visibility into backfill failures

```python
# Enhanced error reporting
class BackfillResult:
    def add_symbol_error(self, symbol: str, error: str, stage: str):
        self.symbol_errors[symbol] = {
            'error': error,
            'stage': stage,
            'timestamp': datetime.now()
        }
```

## Medium-term Improvements (1-2 Months)

### 1. Unified Data Retention Policy
**Goal**: Single source of truth for data retention based on scanner layers

```yaml
data_retention:
  layer_based:
    layer_0:
      daily_bars: 30  # Minimum for technical analysis
      intraday_bars: 0  # Not needed
      tick_data: 0
    layer_1:
      daily_bars: 60  # Extended for backtesting
      intraday_bars: 5  # Recent intraday only
      tick_data: 0
    layer_2:
      daily_bars: 90  # Full quarter for catalyst analysis
      intraday_bars: 30  # Month of intraday
      tick_data: 0
    layer_3:
      daily_bars: 120  # Full history for active trading
      intraday_bars: 60  # Two months intraday
      tick_data: 5  # Recent tick data
```

**Benefits**:
- Clear, predictable data retention
- Aligned with trading strategy needs
- Easy to modify per layer

### 2. Smart Backfill Scheduling
**Goal**: Optimize API usage and reduce redundant requests

```python
class SmartBackfillScheduler:
    async def get_missing_ranges(self, symbol: str, interval: str) -> List[DateRange]:
        """Identify gaps in existing data"""
        existing = await self.get_existing_ranges(symbol, interval)
        requested = DateRange(start_date, end_date)
        return requested.subtract(existing)
    
    async def schedule_backfill(self, symbols: List[str]) -> BackfillPlan:
        """Create optimized backfill plan"""
        # Group symbols by similar data gaps
        # Prioritize based on scanner layer
        # Respect rate limits
```

### 3. Progress Tracking Enhancement
**Goal**: Persistent, queryable progress tracking

```sql
-- Backfill progress table
CREATE TABLE backfill_progress (
    id SERIAL PRIMARY KEY,
    job_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    interval VARCHAR(20),
    start_date DATE,
    end_date DATE,
    status VARCHAR(20) NOT NULL,
    records_loaded INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id, symbol, data_type, interval)
);
```

## Long-term Architecture (3-6 Months)

### 1. Event-Driven Backfill System
**Goal**: Decouple backfill from monolithic orchestrator

```python
# Event-based architecture
class BackfillEvents:
    SYMBOL_QUALIFIED = "scanner.symbol.qualified"  # Trigger backfill
    DATA_GAP_DETECTED = "data.gap.detected"  # Fill specific gaps
    LAYER_PROMOTED = "scanner.layer.promoted"  # Extend retention
    
class BackfillWorker:
    async def handle_symbol_qualified(self, event: Event):
        symbol = event.data['symbol']
        layer = event.data['layer']
        retention_days = self.get_retention_for_layer(layer)
        await self.queue_backfill(symbol, retention_days)
```

**Benefits**:
- Automatic backfill on symbol qualification
- Reactive to data needs
- Scalable worker pattern

### 2. Microservice Architecture
**Goal**: Separate concerns for better scalability

```yaml
services:
  backfill-api:
    # REST API for backfill requests
    # Status endpoints
    # Admin interface
    
  backfill-scheduler:
    # Intelligent scheduling
    # Rate limit management
    # Priority queuing
    
  backfill-worker:
    # Actual data fetching
    # Parallel processing
    # Error recovery
    
  data-quality-monitor:
    # Gap detection
    # Data validation
    # Alerting
```

### 3. Real-time Integration
**Goal**: Seamless transition from backfill to real-time data

```python
class UnifiedDataPipeline:
    def __init__(self):
        self.backfill_manager = BackfillManager()
        self.realtime_manager = RealtimeManager()
        self.gap_detector = GapDetector()
    
    async def ensure_data_continuity(self, symbol: str):
        """Ensure no gaps between historical and real-time data"""
        latest_historical = await self.get_latest_historical(symbol)
        first_realtime = await self.get_first_realtime(symbol)
        
        if gap := self.detect_gap(latest_historical, first_realtime):
            await self.backfill_manager.fill_gap(symbol, gap)
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. ✅ Fix tier override issue (COMPLETED)
2. ✅ Clean up unused code (COMPLETED)
3. 🔄 Add scanner qualification updates
4. 🔄 Implement configuration-based tier bypass

### Phase 2: Enhancement (Months 1-2)
1. 📋 Design unified retention policy
2. 📋 Implement smart scheduling
3. 📋 Add comprehensive progress tracking
4. 📋 Create admin dashboard

### Phase 3: Architecture (Months 3-6)
1. 📋 Design event-driven system
2. 📋 Prototype microservice split
3. 📋 Implement real-time integration
4. 📋 Migration and deployment

## Success Metrics

### Short-term
- ✅ Layer 0 symbols get requested data (30 days)
- 📊 Backfill success rate > 95%
- ⏱️ Average backfill time < 5 minutes for 100 symbols

### Medium-term
- 📊 API usage reduction by 30% (smart scheduling)
- 🎯 Data gaps < 1% for active symbols
- ⚡ Backfill latency < 1 minute for single symbol

### Long-term
- 🔄 Zero-gap data continuity
- 📈 Horizontal scalability
- 🤖 Fully automated data management

## Configuration Examples

### Simplified Configuration (Immediate)
```yaml
backfill:
  # Simple, clear options
  respect_user_overrides: true
  use_scanner_layers: true
  parallel_symbols: 10
  
  # No complex tier configuration needed
```

### Layer-based Configuration (Medium-term)
```yaml
data_management:
  retention_policy: "scanner_layer_based"
  
  scanner_layers:
    layer_0:
      retention_days: 30
      intervals: ["1day"]
      priority: "low"
      
    layer_1:
      retention_days: 60
      intervals: ["1day", "1hour"]
      priority: "medium"
      
    layer_2:
      retention_days: 90
      intervals: ["1day", "1hour", "5min"]
      priority: "high"
      
    layer_3:
      retention_days: 120
      intervals: ["1day", "1hour", "5min", "1min"]
      priority: "critical"
```

## Conclusion

The backfill system evolution should focus on:
1. **Simplification**: Remove unnecessary complexity
2. **Alignment**: Match data retention to trading strategy
3. **Automation**: Reduce manual intervention
4. **Scalability**: Prepare for growth

By following this roadmap, the backfill system will transform from a complex, tier-based system to a streamlined, scanner-aligned, event-driven architecture that serves the trading strategy effectively.
# Four-Layer Funnel Integration Analysis

## How the Four-Layer Funnel Maps to Existing Project Structure

### Current Components That Can Be Leveraged

#### 1. Universe Management (Layer 0 & 1)

- **Existing**: `data_pipeline/universe_manager.py`
- **Current Implementation**:
  - Uses companies table with market_cap, avg_volume filters
  - Already filters by quote_type (EQUITY, ETF)
- **Required Changes**:
  - Add dollar volume calculation (avg_volume × current_price)
  - Implement quarterly static universe refresh
  - Add exchange filtering (NASDAQ, NYSE only)
  - Tighten price range filter ($5-$500)

#### 2. Data Collection Infrastructure (Layer 2)

- **Existing Components**:
  - `data_pipeline/sources/yahoo_client.py` - Price data, news
  - `data_pipeline/sources/benzinga_client.py` - News & events
  - `data_pipeline/sources/reddit_client.py` - Social sentiment
  - `data_pipeline/news_manager.py` - News processing
- **Current Capabilities**:
  - Already collecting news, social sentiment
  - Have earnings dates in company data
  - Technical indicators in feature pipeline
- **Required Additions**:
  - Pre-market gap scanner
  - Real-time analyst ratings monitor
  - Social velocity calculator (std dev from average)
  - Volatility squeeze detector

#### 3. Feature Engineering (Layer 2 Technical)

- **Existing**: `feature_pipeline/calculators/`
  - `technical_indicators.py` - Has Bollinger Bands
  - `microstructure.py` - Has VWAP, spread analysis
  - `enhanced_correlation.py` - Cross-asset analysis
- **Required Additions**:
  - 52-week high/low proximity calculator
  - Bollinger Band squeeze detector (6-month minimum width)
  - Pre-market gap percentage calculator

#### 4. Real-time Processing (Layer 3)

- **Existing**:
  - `data_pipeline/stream_processor.py` - Real-time data streaming
  - `trading_engine/brokers/alpaca_broker.py` - Pre-market data access
- **Required Additions**:
  - Pre-market RVOL calculator
  - Pattern recognition for consolidation structures
  - Real-time spread monitoring
  - Pre-market session handler (8:30-9:25 AM)

### New Components Needed

#### 1. Convergence Scanner Engine

```python
# Proposed location: analysis/convergence_scanner.py
class ConvergenceScanner:
    def __init__(self):
        self.layer0_universe = []  # Static universe
        self.layer1_tradable = []  # Liquidity filtered
        self.layer2_catalysts = []  # Catalyst driven
        self.layer3_final = []     # Pre-market confirmed

    def run_layer0_quarterly(self):
        # Build static universe

    def run_layer1_nightly(self):
        # Apply liquidity filters

    def run_layer2_scans(self):
        # Run parallel catalyst scans

    def run_layer3_premarket(self):
        # Real-time pre-market confirmation
```

#### 2. Catalyst Scoring System

```python
# Proposed location: analysis/catalyst_scorer.py
class CatalystScorer:
    def __init__(self):
        self.scoring_weights = {
            'earnings_today': 5,
            'analyst_upgrade': 4,
            'news_m&a': 6,
            'gap_up_4pct': 3,
            'near_52wk_high': 2,
            'volatility_squeeze': 3,
            'social_3std_dev': 4,
            'premarket_rvol_5': 5
        }
```

#### 3. Pre-Market Monitor

```python
# Proposed location: monitoring/premarket_monitor.py
class PreMarketMonitor:
    def __init__(self):
        self.start_time = "08:30"
        self.end_time = "09:25"

    async def monitor_symbols(self, catalyst_list):
        # Real-time pre-market monitoring
```

### Integration Points

1. **Replace Current Symbol Selection**:
   - Current: `app/run_symbol_selection.py` uses simple volume/correlation
   - New: Use four-layer funnel output instead

2. **Enhance Universe Manager**:
   - Add methods for each layer
   - Store intermediate results for analysis
   - Track why symbols were filtered at each stage

3. **Schedule Integration**:
   - Layer 0: Quarterly cron job
   - Layer 1: Nightly at 7 PM EST
   - Layer 2: Nightly at 8 PM EST + Pre-market at 7 AM EST
   - Layer 3: Live from 8:30-9:25 AM EST

4. **Database Schema Additions**:
   - `symbol_universe` table for Layer 0 results
   - `daily_liquidity` table for Layer 1 results
   - `catalyst_scores` table for Layer 2 results
   - `premarket_confirmations` table for Layer 3

### Implementation Priority

1. **Phase 1**: Build Layer 0 & 1 (Foundation)
   - Enhance universe_manager.py
   - Add dollar volume calculations
   - Create quarterly refresh job

2. **Phase 2**: Implement Layer 2 (Catalysts)
   - Build parallel scanning infrastructure
   - Create catalyst scoring system
   - Integrate existing news/social feeds

3. **Phase 3**: Add Layer 3 (Pre-market)
   - Build pre-market monitoring system
   - Add RVOL calculations
   - Implement pattern recognition

4. **Phase 4**: Full Integration
   - Replace existing symbol selection
   - Add monitoring dashboards
   - Performance tracking

### Expected Improvements

- **Current**: 500 symbols selected by simple filters
- **New**: 20-50 high-conviction symbols with multiple catalysts
- **Benefit**: 10-25x reduction in symbols to monitor
- **Result**: Higher win rate, better resource utilization

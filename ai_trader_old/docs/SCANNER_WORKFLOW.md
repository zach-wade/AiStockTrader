# Scanner Workflow: Layer 0-3 Qualification System

## Overview

The AI Trading System implements a sophisticated four-layer filtering funnel that progressively narrows down ~10,000+ market symbols to 20-50 high-conviction trading candidates. This document explains when and how to run each layer.

## Layer Progression Overview

```
Layer 0: Universe Discovery     →  ~10,000+ symbols (ALL tradeable assets)
Layer 1: Liquidity Filtering    →  ~2,000-3,000 symbols (liquid, tradeable)
Layer 1.5: Sector Diversification →  ~500-1,000 symbols (sector balanced)
Layer 2: Technical Analysis     →  ~100-200 symbols (momentum candidates)
Layer 3: Fundamental Analysis   →  20-50 symbols (high-conviction picks)
```

## Layer Specifications

### Layer 0: Universe Discovery

**Purpose**: Discover all tradeable assets from broker APIs

**Data Sources**:

- Alpaca Markets API (US equities)
- Polygon.io API (comprehensive universe)
- Static symbol lists for crypto/forex

**Qualification Criteria**:

- Asset must be tradeable through configured brokers
- Basic symbol validation (format, exchange listing)
- Active trading status (not delisted/suspended)

**Expected Output**: 10,000+ symbols
**CLI Command**: `python ai_trader.py universe --populate`
**Frequency**: Daily (pre-market)

### Layer 1: Liquidity Filtering

**Purpose**: Filter for liquid, institutionally tradeable symbols

**Qualification Criteria**:

- Average daily volume > 1,000,000 shares (20-day average)
- Share price > $1.00 (penny stock filter)
- Market capitalization > $100M
- Bid-ask spread < 2% of mid-price
- Options availability (institutional requirement)

**Data Requirements**:

- 20 days of price/volume history
- Real-time quote data for spread calculation
- Market cap data from financials

**Expected Output**: 2,000-3,000 symbols
**CLI Command**: Currently automatic via scheduler
**Frequency**: Daily (6:00 AM EST, pre-market)

### Layer 1.5: Sector Diversification

**Purpose**: Ensure balanced representation across market sectors

**Qualification Criteria**:

- Must pass Layer 1 qualification
- Sector classification via GICS/SIC codes
- Top 30% performers within each sector (by volume-weighted momentum)
- Maximum 15% allocation to any single sector

**Sector Categories**:

- Technology, Healthcare, Financial Services
- Consumer Discretionary, Industrials, Energy
- Materials, Utilities, Real Estate, Telecommunications

**Expected Output**: 500-1,000 symbols
**CLI Command**: Currently automatic via scheduler
**Frequency**: Daily (after Layer 1 completion)

### Layer 2: Technical Analysis

**Purpose**: Technical screening for momentum and pattern recognition

**Qualification Criteria**:

- RSI(14) between 30-70 (avoid overbought/oversold)
- Volume surge: 3-day average > 1.5x 20-day average
- Price momentum: 5-day return between -2% and +8%
- Moving average alignment: Price > SMA(20) > SMA(50)
- Bollinger Band position: Not touching extreme bands
- MACD signal: Bullish crossover within 5 days

**Technical Indicators Required**:

- RSI, MACD, Bollinger Bands
- Simple Moving Averages (20, 50, 200)
- Volume patterns and anomalies

**Expected Output**: 100-200 symbols
**CLI Command**: Currently automatic via scheduler
**Frequency**: Intraday (every 15 minutes during market hours)

### Layer 3: Fundamental Analysis

**Purpose**: Final fundamental screening for high-conviction picks

**Qualification Criteria**:

- P/E ratio between 8-25 (reasonable valuation)
- Debt-to-equity ratio < 0.6 (financial stability)
- Revenue growth > 5% (trailing 12 months)
- Positive earnings trend (last 2 quarters)
- News sentiment score > 0.3 (positive news flow)
- Analyst rating average > 3.0/5.0

**Data Requirements**:

- Quarterly financial statements
- Analyst estimates and ratings
- News sentiment analysis
- Corporate actions history

**Expected Output**: 20-50 symbols
**CLI Command**: Currently automatic via scheduler
**Frequency**: Daily (4:30 PM EST, post-market)

## Daily Operational Schedule

### Pre-Market (6:00 AM EST)

```bash
# 1. Refresh universe discovery
python ai_trader.py universe --populate

# 2. Layer 1 liquidity filtering (automatic)
# Runs via master scheduler, analyzes previous day's data

# 3. Layer 1.5 sector diversification (automatic)
# Balances sector allocation based on Layer 1 results
```

### Market Hours (9:30 AM - 4:00 PM EST)

```bash
# Layer 2 technical scanning runs automatically every 15 minutes
# Updates qualification status based on real-time price action
# Results available via: python ai_trader.py universe --layer 2
```

### Post-Market (4:30 PM EST)

```bash
# Layer 3 fundamental analysis (automatic)
# Incorporates end-of-day financials and news sentiment
# Results available via: python ai_trader.py universe --layer 3
```

## Manual Operations

### Check Current Layer Status

```bash
# View all active companies (Layer 0)
python ai_trader.py universe --layer 0

# View liquid symbols (Layer 1)
python ai_trader.py universe --layer 1 --limit 100

# View technical candidates (Layer 2)
python ai_trader.py universe --layer 2

# View high-conviction picks (Layer 3)
python ai_trader.py universe --layer 3
```

### Monitor Layer Health

```bash
# Check universe statistics
python ai_trader.py universe --stats

# Check system health
python ai_trader.py universe --health

# View qualification progression
python ai_trader.py status
```

### Force Layer Re-qualification

```bash
# Note: Individual layer re-runs not yet implemented in CLI
# Currently handled via master scheduler or manual module execution

# Development/debug approach:
cd src && python -m ai_trader.scanners.layers.layer1_liquidity_filter --reprocess-all
cd src && python -m ai_trader.scanners.layers.layer2_technical_filter --reprocess-all
```

## Configuration Files

### Layer Configuration Locations

- **Layer 0**: `config/scanners/layer0_universe.yml`
- **Layer 1**: `config/scanners/layer1_liquidity.yml`
- **Layer 1.5**: `config/scanners/layer15_diversification.yml`
- **Layer 2**: `config/scanners/layer2_technical.yml`
- **Layer 3**: `config/scanners/layer3_fundamental.yml`

### Key Configurable Parameters

#### Layer 1 Liquidity Filters

```yaml
filters:
  min_volume: 1000000        # Minimum daily volume
  min_price: 1.00           # Minimum share price
  min_market_cap: 100000000 # Minimum market cap ($100M)
  max_spread_pct: 2.0       # Maximum bid-ask spread %
```

#### Layer 2 Technical Filters

```yaml
technical_criteria:
  rsi_min: 30               # RSI lower bound
  rsi_max: 70               # RSI upper bound
  volume_surge_threshold: 1.5 # Volume surge multiplier
  momentum_min: -0.02       # Minimum 5-day return
  momentum_max: 0.08        # Maximum 5-day return
```

#### Layer 3 Fundamental Filters

```yaml
fundamental_criteria:
  pe_ratio_min: 8           # Minimum P/E ratio
  pe_ratio_max: 25          # Maximum P/E ratio
  debt_equity_max: 0.6      # Maximum debt-to-equity
  revenue_growth_min: 0.05  # Minimum revenue growth
  sentiment_min: 0.3        # Minimum news sentiment
```

## Performance Metrics

### Expected Qualification Rates

- **Layer 0 → Layer 1**: ~25% (2,500 / 10,000)
- **Layer 1 → Layer 1.5**: ~30% (750 / 2,500)
- **Layer 1.5 → Layer 2**: ~20% (150 / 750)
- **Layer 2 → Layer 3**: ~30% (45 / 150)

### Success Criteria

- **Layer 0**: Successfully discovers 8,000+ active symbols
- **Layer 1**: Maintains 2,000+ liquid candidates
- **Layer 2**: Produces 50-200 technical candidates daily
- **Layer 3**: Maintains 20-50 high-conviction picks

### Warning Thresholds

- **Layer 0**: < 8,000 symbols (potential data source issues)
- **Layer 1**: < 1,500 symbols (market liquidity concerns)
- **Layer 2**: < 30 symbols (overly restrictive technical criteria)
- **Layer 3**: < 15 symbols (insufficient trading candidates)

## Integration with Trading Pipeline

### Data Flow Architecture

```
Layer 0 Discovery → Backfill Pipeline (ALL symbols)
       ↓
Layer 1-3 Qualification → Feature Pipeline (qualified symbols)
       ↓
Layer 3 Results → Model Training (high-conviction focus)
       ↓
Trading Engine → Live Trading (Layer 3 symbols)
```

### Event-Driven Updates

- **ScannerFeatureBridge**: Automatically updates feature store when layer qualifications change
- **EventBus Integration**: Triggers dependent processes when layer completion events fire
- **Model Retraining**: Automatically triggered when Layer 3 composition changes >20%

## Troubleshooting

### Common Issues

#### Layer 0: No symbols discovered

```bash
# Check data source connectivity
python ai_trader.py universe --health

# Verify API credentials in config
# Check network connectivity to Alpaca/Polygon APIs
```

#### Layer 1: Too few symbols qualified

```bash
# Check market conditions (bear market = lower liquidity)
# Review volume thresholds in layer1_liquidity.yml
# Verify price/volume data availability
```

#### Layer 2: No technical candidates

```bash
# Review technical criteria - may be too restrictive
# Check if market is trending (reduces momentum candidates)
# Verify technical indicator calculations
```

#### Layer 3: Fundamental data missing

```bash
# Check financial data provider connectivity
# Verify news sentiment analysis is functioning
# Review analyst data feed status
```

### Debug Commands

```bash
# Detailed layer statistics
python ai_trader.py universe --stats --verbose

# Check individual symbol qualification status
python ai_trader.py universe --symbol AAPL --layer-details

# Review qualification history
python ai_trader.py universe --qualification-history --days 7
```

---

**Next**: See [SCANNER_INTEGRATION.md](SCANNER_INTEGRATION.md) for detailed integration with backfill and training processes.

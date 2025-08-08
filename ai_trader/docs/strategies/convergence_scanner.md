# Convergence Scanner for Trading Selection

## The "Convergence" Philosophy

The smartest way to find the best symbols is to look for a convergence of catalysts. You build multiple independent scanners, each looking for a different type of signal. The highest-quality candidates are the symbols that appear on multiple lists simultaneously.

## Key Signal Categories

### 1. Price/Volume Action (The "What")

This is the most direct signal of market interest. Go beyond simple "high volume" and look for unusual behavior.

#### Relative Volume (RVOL) is King
- **Importance**: Single most important indicator for finding stocks "in play"
- **Implementation**: Compare current volume over last X minutes to average volume over same X-minute period for last 30 days
- **Thresholds**: 
  - RVOL > 3: Highly active
  - RVOL > 10: Exceptional

#### Pre-Market Gappers
- **Time**: Scan pre-market from ~4:00 AM EST
- **Criteria**: Stocks gapping up/down significantly (> +/-4%) on high volume
- **Signal**: Gap indicates powerful overnight catalyst

#### Volatility Contraction & Expansion
- **Metric**: Bollinger Band Squeeze
- **Scan**: Identify stocks where Bollinger Bands narrowed to multi-month low
- **Opportunity**: Price breakout from narrow range often triggers volatile move
- **Benefit**: Find stocks before their big move

#### Proximity to Key Levels
- **Scan**: Stocks trading near critical price points (attract order flow)
- **Levels**: 
  - 52-week high/low
  - Multi-month high/low
  - Yesterday's high/low

### 2. News & Events (The "Why")

Provides fundamental reason for stock to be in play. System should ingest and flag news in real-time.

#### Earnings & Guidance
- **Priority**: #1 fundamental catalyst
- **Implementation**: 
  - Track earnings calendar
  - Flag companies reporting after close yesterday or before open today
  - Market reaction to report = prime trading opportunity

#### Analyst Ratings
- **Signal**: Surprise upgrade/downgrade from major firm
- **Impact**: Can move stock for full session

#### Sector-Specific Catalysts
- **Biotech**: FDA approval/rejection dates, clinical trial data releases
- **Tech**: Major product announcements, conference presentation dates
- **M&A**: Scan keywords: "merger," "acquire," "acquisition," "takeover"

### 3. Alternative & Social Data (The "Who")

Tracks where retail and institutional attention is focused.

#### Social Media Velocity
- **Metric**: Rate of change, not just mention count
- **Implementation**: 
  - Build scanner for X (Twitter) and Reddit forums (r/stocks, r/investing, r/wallstreetbets)
  - Alert when hourly mention count significantly higher than daily average
  - Sudden spike = powerful signal

#### Unusual Options Activity (UOA)
- **Importance**: 'Holy grail' signal (shows where big money places bets)
- **Look for**:
  - Single large block trades of calls/puts
  - High volume in short-dated out-of-the-money options
  - High call-to-put volume ratio
  - Often precedes major news or sharp price move

#### Insider Activity
- **Signal**: Cluster of recent buys by multiple executives
- **Nature**: Often longer-term but adds weight to short-term setup

## Multi-Factor Funnel Architecture

```mermaid
graph TD
    subgraph "Top of Funnel: Broad Scanning"
        A[Price/Volume Scanner<br/>(RVOL, Gaps, Volatility)]
        B[News/Event Scanner<br/>(Earnings, Ratings, M&A)]
        C[Social/Alt-Data Scanner<br/>(Mentions, Options Activity)]
    end

    subgraph "Mid-Funnel: Convergence Engine"
        A --> D{Symbol Scoring & Convergence}
        B --> D
        C --> D
    end

    subgraph "Bottom of Funnel: Final Watchlist"
        D --> E[High-Priority Watchlist<br/>(e.g., 'AAPL: High RVOL + Analyst Upgrade')]
    end

    E --> F[Execute ML Analysis]

    style E fill:#d5f5e3,stroke:#333,stroke-width:2px
```

## Implementation Approach

The existing SymbolSelectionPipeline is well-suited to be the engine for this. Modifications needed:

1. **Parallel Scanning**: Run different types of scans in parallel
2. **Scoring System**: Create point-based system
   - +3 points for gapping up
   - +2 points for high RVOL
   - +4 points for earnings beat
   - etc.
3. **Convergence Selection**: Final list consists of symbols with highest convergence scores

## Benefits

This multi-faceted approach ensures:
- Not just finding statistically noisy patterns
- Focusing powerful ML model on handful of stocks with multiple, independent reasons to be in motion
- Higher quality trading opportunities through catalyst convergence
- Better risk/reward setups through multi-factor confirmation
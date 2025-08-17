# The Four-Layer Filtering Funnel

This engine is designed to take the entire market and methodically distill it down to a small group of stocks with the highest probability of significant movement today.

## Layer 0: The Static Universe (Run Quarterly)

**Purpose**: To establish a baseline of all theoretically tradable instruments. This is a one-time setup that rarely changes.

**Process**:

- Ingest a list of all symbols from US exchanges (~10,000+)
- Filter out everything that isn't a common stock or a highly liquid ETF
- Remove warrants, preferred shares, low-volume ETFs, SPACs before they de-SPAC, etc.
- Filter for symbols listed on major exchanges only (NASDAQ, NYSE)

**Outcome**: A master list of ~8,000 "tradable" symbols. This list is the foundation and prevents you from ever analyzing junk.

## Layer 1: The Liquidity & Tradability Filter (Run Nightly)

**Purpose**: To ensure every symbol on your potential list can be traded at scale without significant slippage. This is the most important mechanical filter.

**Process**: Run this on the "Static Universe" from Layer 0.

### Key Filters

- **Average Dollar Volume**: This is the critical metric
  - Calculate 30-day average daily dollar volume (avg_volume × avg_price)
  - Filter for symbols with > $20 million in daily dollar volume
- **Price Range**: Filter out stocks that are too cheap or too expensive
  - Common range: Share Price between $5 and $500
  - Penny stocks have different dynamics
  - High-priced stocks can tie up capital

**Outcome**: The "Potentially Tradable Universe" - ~1,500 symbols that are liquid and in a good price range. All subsequent, more intensive scans are run only on this list.

## Layer 2: The Catalyst Scan (Run Nightly & Pre-Market)

**Purpose**: To find a fundamental or technical reason for a stock to move today. This is where you find the "why." This layer consists of multiple, independent scans run in parallel on the "Potentially Tradable Universe."

### Scan 2A: News & Events (Fundamental)

- **Earnings**: Any stock that reported earnings in the last 18 hours
- **Analyst Ratings**: Any stock with a new rating or significant upgrade/downgrade
- **News Desk**: Use NLP to scan news headlines for high-impact keywords:
  - "FDA", "clinical trial", "partnership", "acquisition", "takeover", "guidance", "new contract"

### Scan 2B: Technical Setups (Price-Based)

- **Gappers**: Identify all stocks gapping up or down > 4% from yesterday's close in pre-market
- **Breakouts**: Find stocks trading within 2% of:
  - 52-week high (for long candidates)
  - 52-week low (for short candidates)
- **Volatility Squeeze**: Identify stocks whose Bollinger Bands are at their narrowest point in the last 6 months
  - These are coiled springs, ready for a volatile move

### Scan 2C: Social & Alternative Data (Sentiment)

- **Social Velocity**: Find stocks where:
  - 24-hour social media mention count (X, Reddit)
  - Is more than 3 standard deviations above its 30-day average
  - This finds abnormal chatter

**Outcome**: Multiple lists of "interesting" symbols. The key is **convergence**:

- Create a "Catalyst Score" for each stock
- A stock gets points for every list it appears on
- Result: "Catalyst-Driven List" of ~100-200 symbols with highest scores

## Layer 3: Pre-Market Confirmation (Run from 8:30 AM - 9:25 AM EST)

**Purpose**: To validate that the theoretical catalysts from Layer 2 are translating into real institutional interest right before the market opens.

**Process**: This is an intense, real-time scan of only the ~150 stocks on the "Catalyst-Driven List."

### Key Metrics

- **Pre-Market Relative Volume (RVOL)**:
  - Compare traded volume so far this pre-market to average for same time window
  - Filter for stocks with RVOL > 5
  - This confirms the interest today is real
- **Clear Structure**:
  - Is the stock forming a clean consolidation pattern? (e.g., tight flag or range)
  - Or is it chaotic and messy?
  - Prioritize stocks with clear, actionable pre-market levels
- **Spread & Liquidity Check**:
  - Ensure bid-ask spread is not widening abnormally
  - This confirms pre-market activity is liquid

**Outcome**: The **Final, High-Potency Watchlist**

- Hyper-focused list of 20-50 symbols
- Passed every layer of scrutiny
- They are liquid, have a catalyst, and have confirmed institutional interest

## Implementation Benefits

This final list is the input you feed into your real-time monitoring and ML trading system. By using this funnel, you dramatically increase the signal-to-noise ratio, ensuring your system focuses its resources on the highest-probability opportunities of the day.

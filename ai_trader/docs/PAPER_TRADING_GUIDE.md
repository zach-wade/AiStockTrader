# AI Trader - Paper Trading Quick Start Guide

This guide will help you start the AI Trader system in paper trading mode with live data and monitoring dashboards.

## Prerequisites

1. **Environment Variables** - Create a `.env` file in the project root:
```bash
# API Keys
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
POLYGON_API_KEY=your_polygon_api_key  # Optional but recommended

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_trader
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# Paper Trading Endpoint (Alpaca)
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

2. **Database Setup** - Ensure PostgreSQL is running and the database exists:
```bash
createdb ai_trader
```

3. **Python Dependencies** - Install required packages:
```bash
pip install -r requirements.txt
```

## Quick Start Methods

### Method 1: Using the Startup Script (Recommended)

```bash
# Basic startup with default symbols (AAPL, MSFT, GOOGL, AMZN, TSLA)
./start_paper_trading.sh

# With custom symbols
./start_paper_trading.sh "SPY,QQQ,IWM"

# With ML trading enabled
./start_paper_trading.sh "AAPL,MSFT" --enable-ml
```

### Method 2: Using Python Helper

```bash
# Basic startup
python start_paper_trading.py

# With custom symbols
python start_paper_trading.py AAPL,MSFT,GOOGL

# With ML trading enabled
python start_paper_trading.py AAPL,MSFT --enable-ml
```

### Method 3: Direct Command

```bash
# Full control with all options
python ai_trader.py trade \
    --mode paper \
    --symbols AAPL,MSFT,GOOGL \
    --enable-monitoring \
    --enable-streaming \
    --dashboard-port 8080 \
    --websocket-port 8081
```

## System Components

When you start the system, the following components are initialized:

1. **Database Connection** - Connection pool for data storage
2. **Data Sources** - Alpaca, Polygon, Yahoo Finance clients
3. **Stream Processor** - Real-time market data streaming
4. **Event Bus** - Component communication system
5. **ML Orchestrator** - Trading logic and model execution
6. **Paper Broker** - Simulated trading execution
7. **Dashboard Server** - Web-based monitoring interface

## Monitoring Dashboard

Once the system is running, access the monitoring dashboard at:
- **Main Dashboard**: http://localhost:8080
- **WebSocket Feed**: ws://localhost:8081

The dashboard provides:
- Real-time position tracking
- P&L monitoring
- Trade activity log
- System performance metrics
- Market data visualization
- Alert notifications

## First-Time Setup

If this is your first time running the system:

1. **Populate the Universe** (recommended):
```bash
python ai_trader.py universe --populate
```

2. **Backfill Historical Data** (optional):
```bash
# Quick backfill for testing (7 days)
python ai_trader.py backfill --symbols AAPL,MSFT --days 7

# Full backfill for ML training (1 year)
python ai_trader.py backfill --symbols AAPL,MSFT --days 365
```

3. **Train ML Models** (if using ML):
```bash
python ai_trader.py train --symbols AAPL,MSFT --models xgboost
```

## Configuration Options

### Trading Symbols
- Default: AAPL, MSFT, GOOGL, AMZN, TSLA
- Custom: Pass comma-separated list via `--symbols`
- Universe: Use all symbols from universe database

### Trading Strategies
- Available: momentum, mean_reversion, ml_ensemble
- Configure in `unified_config.yaml` under `strategies.active`

### Risk Management
- Position sizing: Configured in `trading.position_sizing`
- Risk limits: Set in `trading.risk_management`

## Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
pg_isready

# Test connection
psql -h localhost -U your_user -d ai_trader -c "SELECT 1"
```

### API Key Issues
- Ensure Alpaca API keys are for paper trading account
- Verify keys are correctly set in `.env` file
- Check API rate limits haven't been exceeded

### Port Conflicts
If ports 8080/8081 are in use:
```bash
# Use different ports
python ai_trader.py trade --mode paper \
    --dashboard-port 8090 \
    --websocket-port 8091
```

### Missing Data
If you see "No data available" errors:
1. Run universe population: `python ai_trader.py universe --populate`
2. Backfill recent data: `python ai_trader.py backfill --days 7`

## Stopping the System

To gracefully shutdown:
1. Press `Ctrl+C` in the terminal
2. The system will close all connections and save state
3. Wait for "✅ Shutdown completed" message

## Next Steps

After successfully running paper trading:

1. **Monitor Performance** - Watch the dashboard for trade signals
2. **Adjust Configuration** - Tune parameters in `unified_config.yaml`
3. **Analyze Results** - Review trades and performance metrics
4. **Train Models** - Use collected data to improve ML models
5. **Go Live** - Switch to live trading when ready (use with caution!)

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review configuration in `src/main/config/unified_config.yaml`
- Consult API documentation for data providers
- File issues on GitHub with error logs

Happy Trading! 🚀📈
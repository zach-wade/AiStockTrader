# Trading Dashboard V2 Documentation

## Overview

The Trading Dashboard V2 is a comprehensive web-based monitoring interface for real-time trading system oversight. Built with Dash and Plotly, it provides interactive visualizations and real-time updates of trading activity, market conditions, portfolio performance, and system alerts.

**URL**: <http://localhost:8080>
**Technology Stack**: Python, Dash, Plotly, AsyncIO
**Update Frequency**: 5 seconds (configurable)

## Architecture

### Dashboard Structure

```
TradingDashboardV2
├── Tab 1: Trading Overview
├── Tab 2: Market Analysis
├── Tab 3: Portfolio Analytics
└── Tab 4: Alerts & Activity
```

### Data Flow

1. **Database Connection**: Uses `DatabasePool` for efficient connection management
2. **Async Data Fetching**: Thread pool executor prevents blocking during data retrieval
3. **Real-time Updates**: Dash interval component triggers updates every 5 seconds
4. **Data Processing**: In-memory caching and pandas DataFrames for efficient visualization

## Tab Details

### Tab 1: Trading Overview

#### Components

1. **Key Metrics Cards**
   - Total P&L (Daily/All-time)
   - Active Positions Count
   - Win Rate Percentage
   - Average Trade Duration

2. **P&L Chart**
   - Time series visualization of cumulative P&L
   - Interactive zoom and hover details
   - Data Source: `trading_pnl` table
   - Query: Last 30 days of P&L history

3. **Active Positions Table**
   - Real-time position monitoring
   - Columns: Symbol, Quantity, Entry Price, Current Price, P&L, Duration
   - Data Source: `positions` table (status='OPEN')
   - Color coding: Green (profit), Red (loss)

4. **Recent Trades**
   - Last 10 executed trades
   - Shows execution details and outcomes
   - Data Source: `trades` table

#### Data Sources

```sql
-- P&L History Query
SELECT date, cumulative_pnl FROM trading_pnl
WHERE date >= NOW() - INTERVAL '30 days'
ORDER BY date;

-- Active Positions Query
SELECT * FROM positions
WHERE status = 'OPEN'
ORDER BY opened_at DESC;
```

### Tab 2: Market Analysis

#### Components

1. **Market Indices**
   - Real-time tracking of major indices (SPY, QQQ, DIA, IWM)
   - Percentage changes and trends
   - Data Source: `market_indices` table

2. **VIX Indicator**
   - Volatility gauge with historical context
   - Color-coded risk levels:
     - Green: VIX < 20 (Low volatility)
     - Yellow: VIX 20-30 (Moderate)
     - Red: VIX > 30 (High volatility)
   - Data Source: `market_data` table (symbol='VIX')

3. **Sector Performance Heatmap**
   - 11 major sectors performance visualization
   - Color intensity shows performance strength
   - Data Source: `sector_performance` table
   - Updates: Every 24 hours

4. **Market Breadth Indicators**
   - Advance/Decline ratio
   - New highs/lows count
   - Market momentum indicators
   - Data Source: `market_breadth` table

#### Data Processing

- Sector data aggregated from last 24 hours
- VIX data fetched with 1-minute granularity
- Indices updated in real-time from market data feeds

### Tab 3: Portfolio Analytics

#### Components

1. **Portfolio Composition Pie Chart**
   - Asset allocation by market value
   - Interactive drill-down capability
   - Data Source: Aggregated from `positions` table

2. **Risk Metrics Dashboard**
   - Portfolio Beta
   - Sharpe Ratio
   - Maximum Drawdown
   - Value at Risk (VaR)
   - Data Source: Calculated from `portfolio_metrics` table

3. **Performance Attribution**
   - Breakdown by strategy
   - Sector contribution analysis
   - Time-based performance trends
   - Data Source: `performance_attribution` table

4. **Correlation Matrix**
   - Position correlations heatmap
   - Diversification analysis
   - Data Source: Calculated from historical price data

### Tab 4: Alerts & Activity

#### Components

1. **Active Alerts Panel**
   - Real-time system alerts
   - Severity levels: Critical, Warning, Info
   - Alert types: Risk, System, Market, Execution
   - Data Source: `alerts` table (status='ACTIVE')

2. **Alert History**
   - Searchable alert log
   - Filter by type, severity, date range
   - Data Source: `alerts` table (all records)

3. **Activity Feed**
   - Real-time system activity log
   - Trade executions, system events, data updates
   - Data Source: `activity_log` table
   - Limit: Last 100 activities

4. **System Health Indicators**
   - Connection status (Alpaca, Database, Market Data)
   - Data feed latency
   - Last update timestamps

## Data Refresh Mechanism

### Async Data Fetching

```python
def _run_async(self, coro):
    """Run async coroutine in thread pool to avoid blocking."""
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    future = self._executor.submit(run_in_thread)
    return future.result()
```

### Update Cycle

1. **Interval Trigger**: Every 5 seconds
2. **Data Fetch**: Async queries to database
3. **Processing**: Data transformation and aggregation
4. **UI Update**: Dash callback updates components
5. **Error Handling**: Graceful degradation on fetch failures

## Database Schema Dependencies

### Required Tables

- `positions`: Active trading positions
- `trades`: Executed trade history
- `trading_pnl`: P&L tracking
- `market_data`: Real-time market prices
- `market_indices`: Index tracking
- `sector_performance`: Sector analytics
- `alerts`: System alerts
- `activity_log`: System activity tracking
- `portfolio_metrics`: Risk and performance metrics

### Key Relationships

```
positions -> trades (position_id)
trades -> trading_pnl (trade_id)
positions -> market_data (symbol)
alerts -> activity_log (alert_id)
```

## Configuration

### Environment Variables

- `DB_HOST`: Database host (default: localhost)
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name (default: ai_trader)
- `DB_USER`: Database user
- `DB_PASSWORD`: Database password

### Dashboard Settings

- Port: 8080 (configurable)
- Update Interval: 5 seconds (configurable)
- Max Positions Display: 50
- Max Activities: 100
- Chart History: 30 days

## Performance Considerations

### Optimization Strategies

1. **Connection Pooling**: Reuses database connections
2. **Data Caching**: In-memory storage for frequently accessed data
3. **Batch Queries**: Combines multiple queries where possible
4. **Async Operations**: Non-blocking data fetches
5. **Incremental Updates**: Only fetches changed data

### Resource Usage

- Memory: ~100-200MB typical
- CPU: <5% during updates
- Database Connections: 1-4 active
- Network: Minimal (local database)

## Error Handling

### Common Issues

1. **Database Connection Errors**
   - Automatic retry with exponential backoff
   - Fallback to cached data
   - User notification via alert banner

2. **Data Fetch Timeouts**
   - 30-second timeout per query
   - Partial data display on timeout
   - Error logged to system

3. **Invalid Data**
   - Data validation before display
   - Default values for missing fields
   - Error boundaries prevent crashes

## Security Considerations

1. **Authentication**: Currently relies on system-level access control
2. **Database Credentials**: Masked in logs
3. **Input Validation**: All user inputs sanitized
4. **XSS Protection**: Dash framework built-in protections

## Future Enhancements

### Planned Features

1. **Enhanced Visualizations**
   - 3D portfolio visualization
   - Advanced candlestick charts
   - Real-time order book depth

2. **Additional Analytics**
   - Monte Carlo simulations
   - Options Greeks display
   - Pairs trading dashboard
   - ML model performance tracking

3. **Interactive Features**
   - Trade execution from dashboard
   - Alert configuration UI
   - Custom dashboard layouts
   - Export functionality

4. **Performance Improvements**
   - WebSocket real-time updates
   - Server-side data aggregation
   - Redis caching layer
   - Pagination for large datasets

5. **Integration Enhancements**
   - Multiple broker support
   - News feed integration
   - Social sentiment analysis
   - Economic calendar overlay

### Technical Improvements

1. **Architecture**
   - Microservice separation
   - GraphQL API layer
   - React frontend option
   - Mobile-responsive design

2. **Monitoring**
   - Prometheus metrics export
   - Grafana integration
   - Custom alert webhooks
   - Audit trail enhancement

3. **Deployment**
   - Docker containerization
   - Kubernetes orchestration
   - CI/CD pipeline
   - Multi-environment support

## Troubleshooting

### Common Problems

1. **Dashboard Won't Start**

   ```bash
   # Check database connection
   psql -h localhost -U your_user -d ai_trader -c "SELECT 1;"

   # Verify environment variables
   echo $DB_HOST $DB_PORT $DB_NAME
   ```

2. **No Data Displayed**
   - Check database tables exist
   - Verify data population scripts
   - Review dashboard logs for errors

3. **Slow Updates**
   - Check database query performance
   - Review network latency
   - Consider increasing update interval

### Debug Mode

Enable debug mode for detailed logging:

```python
dashboard.run(debug=True)
```

## Maintenance

### Regular Tasks

1. **Daily**: Check alert logs
2. **Weekly**: Review performance metrics
3. **Monthly**: Database optimization
4. **Quarterly**: Update dependencies

### Monitoring Checklist

- [ ] Database connection health
- [ ] Update latency < 1 second
- [ ] Memory usage stable
- [ ] No critical alerts active
- [ ] All data sources updating

## Support

For issues or questions:

1. Check system logs in `logs/dashboard/`
2. Review database connection settings
3. Verify all required tables exist
4. Ensure proper permissions on database

---
*Last Updated: 2025-07-30*
*Version: 2.0*

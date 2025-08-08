# AI Trader Dashboard System Documentation

## Overview

The AI Trader Dashboard System provides comprehensive real-time monitoring and visualization for the trading platform. Built on a modern, scalable architecture, it consists of two primary dashboards that work together to provide complete system oversight.

## Dashboard Components

### 1. [Trading Dashboard V2](./trading-dashboard-v2.md)
- **Purpose**: Monitor trading activity, positions, and market conditions
- **URL**: http://localhost:8080
- **Focus**: Business metrics and trading performance

### 2. [System Dashboard V2](./system-dashboard-v2.md)
- **Purpose**: Monitor system health, infrastructure, and data pipelines
- **URL**: http://localhost:8052
- **Focus**: Technical operations and system performance

## Architecture Overview

### Technology Stack
- **Backend**: Python 3.8+
- **Framework**: Dash (built on Flask)
- **Visualization**: Plotly
- **Database**: PostgreSQL with connection pooling
- **Async Operations**: AsyncIO with ThreadPoolExecutor
- **System Monitoring**: psutil
- **Process Management**: subprocess with isolated Python processes

### Key Design Principles

1. **Process Isolation**
   - Each dashboard runs in its own process
   - Prevents blocking and resource contention
   - Enables independent scaling and restart

2. **Async Data Operations**
   - Non-blocking database queries
   - Thread pool for async operations in Dash callbacks
   - Efficient resource utilization

3. **Real-time Updates**
   - Configurable update intervals
   - Incremental data fetching
   - Client-side caching

4. **Fault Tolerance**
   - Graceful error handling
   - Automatic reconnection
   - Fallback to cached data

## Getting Started

### Prerequisites
```bash
# Required Python packages
pip install dash plotly pandas numpy psutil asyncpg

# Database setup
PostgreSQL 12+ with ai_trader database
```

### Starting the Dashboards

#### Option 1: Via AI Trader CLI (Recommended)
```bash
python ai_trader.py trade --enable-monitoring
```

#### Option 2: Standalone Dashboard Scripts
```bash
# Start Trading Dashboard
python src/main/monitoring/dashboards/v2/run_trading_dashboard.py \
  --config '{"host":"localhost","port":5432,"database":"ai_trader","user":"your_user","password":"your_pass"}' \
  --port 8080

# Start System Dashboard  
python src/main/monitoring/dashboards/v2/run_system_dashboard.py \
  --config '{"host":"localhost","port":5432,"database":"ai_trader","user":"your_user","password":"your_pass"}' \
  --port 8052
```

#### Option 3: Using Dashboard Manager
```python
from main.monitoring.dashboards.v2 import DashboardManager

db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'ai_trader',
    'user': 'your_user',
    'password': 'your_password'
}

manager = DashboardManager(db_config)
await manager.start_all()
```

## Configuration

### Environment Variables
```bash
# Database Configuration
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=ai_trader
export DB_USER=your_user
export DB_PASSWORD=your_password

# Dashboard Settings
export TRADING_DASHBOARD_PORT=8080
export SYSTEM_DASHBOARD_PORT=8052
export DASHBOARD_UPDATE_INTERVAL=5
```

### Configuration File
Create `config/dashboards.yaml`:
```yaml
dashboards:
  trading:
    port: 8080
    update_interval: 5
    max_positions: 50
    chart_history_days: 30
    
  system:
    port: 8052
    update_interval: 5
    metrics_retention: 3600  # seconds
    process_filter: ['python', 'postgres', 'redis']
```

## Database Schema Requirements

### Core Tables
- `positions` - Active trading positions
- `trades` - Trade execution history
- `trading_pnl` - Profit/loss tracking
- `market_data` - Real-time price data
- `alerts` - System alerts and notifications
- `pipeline_status` - Data pipeline health
- `model_performance` - ML model metrics

See individual dashboard documentation for detailed schema requirements.

## Development Guide

### Adding New Dashboard Features

1. **Create New Tab**
```python
# In dashboard class __init__
self.app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='New Feature', value='new-feature'),
        # ... existing tabs
    ])
])

# Add callback
@self.app.callback(
    Output('new-feature-content', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_new_feature(n):
    # Fetch and process data
    data = self._run_async(self.fetch_new_data())
    return create_feature_layout(data)
```

2. **Add Data Source**
```python
async def fetch_new_data(self):
    """Fetch data for new feature."""
    async with self.db_pool.acquire() as conn:
        query = """
        SELECT * FROM new_table
        WHERE created_at > NOW() - INTERVAL '1 hour'
        """
        rows = await conn.fetch(query)
        return [dict(row) for row in rows]
```

3. **Create Visualization**
```python
def create_feature_chart(data):
    """Create Plotly chart for feature."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['x_values'],
        y=data['y_values'],
        mode='lines+markers'
    ))
    return dcc.Graph(figure=fig)
```

### Testing Dashboards

1. **Unit Tests**
```python
import pytest
from main.monitoring.dashboards.v2 import TradingDashboardV2

@pytest.fixture
def mock_db_pool():
    # Create mock database pool
    pass

def test_dashboard_creation(mock_db_pool):
    dashboard = TradingDashboardV2(mock_db_pool)
    assert dashboard.port == 8080
    assert dashboard.app is not None
```

2. **Integration Tests**
```bash
# Start test database
docker run -d -p 5432:5432 postgres:13

# Run dashboard tests
pytest tests/dashboards/ -v
```

3. **Performance Testing**
```python
import time
import requests

# Test dashboard response time
start = time.time()
response = requests.get('http://localhost:8080')
assert response.status_code == 200
assert time.time() - start < 1.0  # Should load in under 1 second
```

## Deployment

### Production Considerations

1. **Security**
   - Use HTTPS with SSL certificates
   - Implement authentication middleware
   - Restrict database access
   - Sanitize all inputs

2. **Performance**
   - Enable production mode (debug=False)
   - Use CDN for static assets
   - Implement Redis caching
   - Configure nginx reverse proxy

3. **Monitoring**
   - Set up health check endpoints
   - Configure log aggregation
   - Enable performance monitoring
   - Set up alerting

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8080 8052

CMD ["python", "-m", "main.monitoring.dashboards.v2.dashboard_manager"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-dashboard
spec:
  replicas: 2
  selector:
    matchLabels:
      app: trading-dashboard
  template:
    metadata:
      labels:
        app: trading-dashboard
    spec:
      containers:
      - name: dashboard
        image: ai-trader/dashboard:v2
        ports:
        - containerPort: 8080
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
```

## Troubleshooting

### Common Issues

1. **Dashboard Won't Start**
   - Check database connectivity
   - Verify environment variables
   - Review logs in `logs/dashboards/`
   - Ensure ports are available

2. **No Data Displayed**
   - Verify database tables exist
   - Check data population jobs
   - Review browser console for errors
   - Inspect network requests

3. **Performance Issues**
   - Monitor database query times
   - Check system resources
   - Review update intervals
   - Enable performance profiling

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run dashboard in debug mode
dashboard.run(debug=True)
```

### Health Checks
```bash
# Check dashboard process
ps aux | grep dashboard

# Test dashboard endpoint
curl http://localhost:8080/_dash-layout

# Check database connection
psql -h localhost -U user -d ai_trader -c "SELECT 1;"
```

## Best Practices

### Data Management
1. Implement data retention policies
2. Use appropriate indexes on frequently queried columns
3. Batch database operations where possible
4. Cache static or slowly changing data

### UI/UX Guidelines
1. Keep update intervals reasonable (5-30 seconds)
2. Provide loading indicators for slow operations
3. Include error messages for failed data fetches
4. Use consistent color schemes and layouts

### Security
1. Never expose sensitive data in dashboards
2. Implement row-level security in database
3. Use prepared statements for all queries
4. Log all access and modifications

## Roadmap

### Q1 2025
- [ ] WebSocket real-time updates
- [ ] Multi-user support with roles
- [ ] Mobile-responsive design
- [ ] Export functionality

### Q2 2025
- [ ] Advanced analytics dashboards
- [ ] Integration with external monitoring tools
- [ ] Custom dashboard builder
- [ ] Performance optimization

### Q3 2025
- [ ] Machine learning insights dashboard
- [ ] Automated anomaly detection
- [ ] Predictive analytics
- [ ] Cost optimization dashboard

## Contributing

### Code Style
- Follow PEP 8 for Python code
- Use type hints for function parameters
- Document all public methods
- Write unit tests for new features

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## Support

### Resources
- [Dash Documentation](https://dash.plotly.com/)
- [Plotly Python](https://plotly.com/python/)
- [AsyncIO Guide](https://docs.python.org/3/library/asyncio.html)

### Getting Help
1. Check the troubleshooting guide
2. Review system logs
3. Search existing issues
4. Contact support team

---
*Last Updated: 2025-07-30*
*Version: 2.0*
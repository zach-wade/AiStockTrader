# ML Trading Integration Guide

This guide explains how to integrate and use ML models for live trading with the AI Trader system.

## Overview

The ML Trading Integration allows you to:
- Use trained ML models for generating trading signals
- Integrate predictions with the live trading engine
- Monitor model performance in real-time
- Automatically manage position sizing and risk

## Architecture

```
Market Data → Feature Pipeline → ML Model → Prediction → Strategy → Signal → Execution
     ↓                                                                              ↓
  Monitoring ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← Outcomes
```

### Key Components

1. **MLTradingService** - Orchestrates ML model integration
2. **RealTimeFeaturePipeline** - Calculates features from streaming data
3. **PredictionEngine** - Generates predictions from models
4. **MLRegressionStrategy** - Converts predictions to trading signals
5. **UnifiedSignalHandler** - Routes signals to execution engine
6. **ModelMonitor** - Tracks performance and detects drift

## Setup

### 1. Train and Register a Model

First, ensure you have a trained model:

```bash
# Train AAPL model (if not already done)
python scripts/train_aapl_model.py
```

### 2. Deploy Model to Production

Deploy the trained model for live trading:

```bash
# List available models
python scripts/deploy_ml_model.py --list

# Deploy AAPL model to production
python scripts/deploy_ml_model.py --deploy aapl_xgboost

# Or deploy from file
python scripts/deploy_ml_model.py --from-file models/aapl_xgboost --model-id aapl_xgboost
```

### 3. Configure ML Trading

Edit `config/ml_trading_config.yaml`:

```yaml
ml_trading:
  enabled: true
  
  models:
    - model_id: "aapl_xgboost"
      symbol: "AAPL"
      min_confidence: 0.6
      position_size: 0.02  # 2% of portfolio
      enabled: true
  
  risk_limits:
    max_ml_positions: 5
    daily_ml_loss_limit: 0.02  # 2% daily loss limit
```

### 4. Run ML Trading

#### Paper Trading Example

```bash
# Run ML trading in paper mode
python examples/ml_trading_example.py
```

#### Integration with Main Trading System

```python
from main.models.ml_trading_integration import create_ml_trading_integration

# In your main trading application
ml_integration = await create_ml_trading_integration(
    execution_engine=execution_engine,
    signal_handler=signal_handler,
    alert_manager=alert_manager,
    config=config
)

# Start ML trading
await ml_integration.start_ml_trading()
```

## Configuration Options

### Model Configuration

```yaml
models:
  - model_id: "model_name"      # Unique model identifier
    symbol: "AAPL"              # Trading symbol
    strategy: "ml_regression"   # Strategy type
    min_confidence: 0.6         # Minimum confidence threshold
    position_size: 0.02         # Position size (fraction of portfolio)
    enabled: true               # Enable/disable model
```

### Feature Pipeline

```yaml
feature_pipeline:
  buffer_size: 500              # Data points to keep in memory
  cache_ttl_seconds: 5          # Feature cache TTL
  lookback_periods:
    rsi: 14                     # RSI period
    sma_20: 20                  # SMA period
    # ... other indicators
```

### Risk Management

```yaml
risk_limits:
  max_ml_position_size: 0.05    # Max 5% per position
  max_ml_positions: 5           # Max concurrent positions
  daily_ml_loss_limit: 0.02     # 2% daily loss limit
```

## Signal Flow

1. **Market Data Reception**
   - Real-time price updates received
   - Buffered in RealTimeDataBuffer

2. **Feature Calculation**
   - Technical indicators calculated
   - Features cached for performance

3. **Prediction Generation**
   - Model makes prediction
   - Confidence score calculated

4. **Signal Creation**
   - MLRegressionStrategy converts prediction to signal
   - Position sizing applied
   - Risk checks performed

5. **Signal Routing**
   - UnifiedSignalHandler aggregates signals
   - Deduplication and prioritization
   - Routed to ExecutionEngine

6. **Order Execution**
   - Orders created from signals
   - Submitted to broker
   - Position tracking updated

## Monitoring

### Real-time Monitoring

The system continuously monitors:
- Prediction accuracy
- Model drift
- Performance metrics
- Risk exposure

### Performance Metrics

Access ML trading metrics:

```python
# Get ML status
ml_status = ml_integration.get_ml_status()

# Get model monitoring summary
monitoring_summary = model_monitor.get_monitoring_summary()
```

### Alerts

The system generates alerts for:
- Performance degradation (>20% MAE increase)
- Prediction drift detected
- Risk limit breaches
- Model failures

## Best Practices

1. **Start Small**
   - Begin with small position sizes (1-2%)
   - Use paper trading first
   - Monitor closely during initial deployment

2. **Model Selection**
   - Use models with consistent out-of-sample performance
   - Ensure sufficient training data (>1 year)
   - Validate on recent market conditions

3. **Risk Management**
   - Set conservative position limits
   - Use stop losses
   - Monitor correlation between ML positions

4. **Performance Monitoring**
   - Track prediction accuracy daily
   - Monitor for regime changes
   - Set up alerts for anomalies

5. **Model Updates**
   - Retrain models regularly (monthly/quarterly)
   - A/B test new versions
   - Keep fallback models ready

## Troubleshooting

### Model Not Loading

```bash
# Check model registry
python scripts/deploy_ml_model.py --list

# Verify model files exist
ls models/trained/
```

### No Signals Generated

1. Check model confidence thresholds
2. Verify market data is flowing
3. Check feature calculation logs
4. Ensure model is in production status

### Performance Issues

1. Monitor feature cache hit rate
2. Check prediction latency
3. Verify buffer sizes are appropriate
4. Consider reducing prediction frequency

## Advanced Topics

### Custom Strategies

Create custom ML strategies by extending MLRegressionStrategy:

```python
class CustomMLStrategy(MLRegressionStrategy):
    def _generate_signal_from_prediction(self, ...):
        # Custom signal generation logic
        pass
```

### Ensemble Models

Deploy multiple models for the same symbol:

```yaml
models:
  - model_id: "aapl_xgboost"
    symbol: "AAPL"
    weight: 0.5
  - model_id: "aapl_lstm"
    symbol: "AAPL"
    weight: 0.5
```

### Feature Engineering

Add custom features to the pipeline:

```python
class CustomFeatureCalculator:
    def calculate(self, data):
        # Custom feature logic
        return features
```

## Support

For issues or questions:
1. Check logs in `logs/ml_trading.log`
2. Review model metrics in monitoring dashboard
3. Consult the troubleshooting section
4. Open an issue on GitHub
# ML Trading Quick Start Guide

This guide shows how to get ML-powered trading up and running quickly.

## Prerequisites

1. Trained AAPL model (already completed)
2. Configuration files set up
3. Paper trading account (for testing)

## Quick Start

### 1. Deploy the AAPL Model

First, ensure the trained AAPL model is deployed:

```bash
# Check if model is deployed
python scripts/deploy_ml_model.py --list

# Deploy if not already done
python scripts/deploy_ml_model.py --deploy aapl_xgboost
```

### 2. Run ML Trading

The easiest way to start ML trading:

```bash
# Run ML trading in paper mode
python ai_trader.py trade --mode paper --symbols AAPL --enable-ml
```

Or use the example script:

```bash
./examples/run_ml_trading.sh
```

### 3. Check System Status

In another terminal, check the trading system status:

```bash
python ai_trader.py status
```

## Command Line Options

### Basic ML Trading
```bash
python ai_trader.py trade --mode paper --enable-ml
```

### With Specific Symbols
```bash
python ai_trader.py trade --mode paper --symbols AAPL,MSFT,GOOGL --enable-ml
```

### Live Trading (Use with Caution!)
```bash
python ai_trader.py trade --mode live --symbols AAPL --enable-ml
```

## Testing

### Run End-to-End Test
```bash
python scripts/test_ml_trading.py
```

### Test Specific Components
```bash
# Test just the ML prediction pipeline
python examples/ml_trading_example.py
```

## Configuration

ML trading is configured in `config/ml_trading.yaml`:

```yaml
ml_trading:
  enabled: true
  
  models:
    - model_id: "aapl_xgboost"
      symbol: "AAPL"
      min_confidence: 0.6
      position_size: 0.02  # 2% of portfolio
```

## Monitoring

### View Logs
```bash
# Follow ML trading logs
tail -f logs/ml_trading.log

# Follow all logs
tail -f logs/ai_trader_*.log
```

### Performance Metrics

The system tracks:
- Prediction accuracy
- Signal generation rate
- Position P&L
- Model drift

Access metrics through the status command or monitoring dashboard.

## Troubleshooting

### Model Not Found
```bash
# Redeploy the model
python scripts/train_aapl_model.py
python scripts/deploy_ml_model.py --deploy aapl_xgboost
```

### No Signals Generated
1. Check model confidence threshold in config
2. Verify market data is flowing
3. Check logs for prediction errors

### Connection Issues
```bash
# Test with paper trading first
python ai_trader.py trade --mode paper --enable-ml
```

## Next Steps

1. **Add More Models**: Train models for other symbols
2. **Customize Strategies**: Modify `ml_regression_strategy.py`
3. **Risk Management**: Adjust position sizing in config
4. **Performance Monitoring**: Set up the monitoring dashboard

## Architecture Overview

```
Market Data → Feature Pipeline → ML Model → Prediction
                                              ↓
Execution ← Signal Router ← Trading Signal ← Signal Adapter
```

The ML trading system integrates seamlessly with the existing trading infrastructure:
- **MLOrchestrator**: Coordinates all components
- **MLTradingService**: Manages ML models
- **SignalAdapter**: Converts predictions to signals
- **ExecutionEngine**: Executes trades

## Safety Features

- Confidence thresholds prevent low-quality trades
- Position size limits prevent over-exposure
- Stop loss/take profit for risk management
- Paper trading mode for testing
- Real-time monitoring and alerts

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Run diagnostic test: `python scripts/test_ml_trading.py`
3. Review configuration in `config/ml_trading.yaml`
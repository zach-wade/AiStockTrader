# Backtesting Integration into Training Pipeline

**Date:** July 1, 2025  
**Status:** Implemented

## Overview

Backtesting has been integrated directly into the model training pipeline to provide automatic validation of trained models before they are registered and deployed. This ensures models perform well not just on training metrics but also in simulated trading conditions.

## Key Changes

### 1. Training Orchestrator Enhancement
- Added `validate_with_backtest()` method to `ModelTrainingOrchestrator`
- Backtesting runs automatically after model training if enabled
- Results are included in model metrics and registration

### 2. Configuration
Added to `unified_config.yaml`:
```yaml
models:
  training:
    backtest_validation: true  # Enable post-training backtesting
    min_sharpe_ratio: 0.5     # Minimum Sharpe ratio to accept model
    min_win_rate: 0.45        # Minimum win rate to accept model
```

### 3. Integration Flow
1. **Train Models** → Standard ML model training
2. **Backtest Validation** → Each model is wrapped in MLMomentumStrategy and backtested
3. **Metrics Collection** → Backtest metrics (Sharpe, return, drawdown, win rate) collected
4. **Model Registration** → Backtest metrics included in model registry
5. **Model Selection** → Best model selected based on backtest Sharpe ratio

### 4. Key Features
- **Automatic Validation**: No manual intervention needed
- **Strategy Wrapper**: Models tested with MLMomentumStrategy wrapper
- **Comprehensive Metrics**: Total return, Sharpe ratio, max drawdown, win rate
- **Registry Integration**: Backtest results stored with model metadata
- **Smart Selection**: Model deployment prioritizes backtest performance

## Usage

### Running Training with Backtesting
```bash
python app/run_training.py
```

The training pipeline will:
1. Calculate features for selected symbols
2. Train ML models (XGBoost, LightGBM, Random Forest, Ensemble)
3. Run backtesting on each trained model
4. Register models with both training and backtest metrics
5. Select best model based on backtest Sharpe ratio

### Disabling Backtesting
To disable backtesting (for faster iteration):
```yaml
models:
  training:
    backtest_validation: false
```

### Output Example
```
================================================================================
Running Backtesting Validation
================================================================================
Backtesting xgboost model...
xgboost backtest complete: Return=15.32%, Sharpe=1.24

Backtesting lightgbm model...
lightgbm backtest complete: Return=18.45%, Sharpe=1.56

xgboost Backtest Results:
  Sharpe Ratio: 1.2400
  Total Return: 15.32%
  Max Drawdown: -3.45%
```

## Benefits

1. **Quality Assurance**: Models validated in realistic trading conditions
2. **Risk Awareness**: Drawdown and risk metrics captured before deployment
3. **Automated Workflow**: No separate backtesting step needed
4. **Better Model Selection**: Models chosen based on trading performance, not just ML metrics
5. **Traceability**: All metrics stored in model registry for audit

## Technical Details

### Backtest Engine Configuration
- Initial capital: $100,000
- Commission: 0.1%
- Slippage: 0.05%
- Strategy: MLMomentumStrategy wrapper
- Data: Features from feature store

### Metrics Captured
- `backtest_total_return`: Total portfolio return
- `backtest_sharpe_ratio`: Risk-adjusted return
- `backtest_max_drawdown`: Maximum peak-to-trough loss
- `backtest_win_rate`: Percentage of profitable trades
- `backtest_total_trades`: Number of trades executed
- `backtest_validated`: Boolean flag indicating validation status

## Future Enhancements

1. **Strategy Variety**: Test models with multiple strategy wrappers
2. **Parameter Optimization**: Optimize strategy parameters during backtest
3. **Walk-Forward Analysis**: Implement rolling window backtesting
4. **Multi-Asset Testing**: Validate on different asset classes
5. **Transaction Costs**: More sophisticated cost modeling
6. **Risk Limits**: Enforce position sizing and risk constraints

## Migration Notes

- Existing models in registry remain unchanged
- New models automatically get backtest validation
- Can re-train existing models to add backtest metrics
- Backward compatible - system works with non-backtested models
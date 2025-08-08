# Scanner Integration: Backfill & Training Pipeline

## Overview

This document explains how the Layer 0-3 scanning system integrates with the backfill and training processes, including data flow architecture, event-driven updates, and operational workflows.

## Integration Architecture

### Data Flow Pipeline
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Layer 0   │───▶│   Backfill   │───▶│   Feature   │───▶│   Training   │
│ (Universe)  │    │  (All Data)  │    │  Pipeline   │    │  Pipeline    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
       │                   ▲                   ▲                   ▲
       ▼                   │                   │                   │
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ Layer 1-3   │    │  Selective   │    │ Qualified   │    │   Model      │
│ (Qualified) │───▶│  Backfill    │───▶│ Features    │───▶│ Retraining   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### Component Integration Map

| Component | Layer 0 Usage | Layer 1-3 Usage | Integration Point |
|-----------|---------------|------------------|-------------------|
| **Backfill** | Primary data collection | Prioritized updates | `BackfillOrchestrator._get_universe_symbols()` |
| **Feature Store** | Raw data ingestion | Qualified feature generation | `ScannerFeatureBridge` |
| **Training** | Background corpus | Focus candidates | `TrainingOrchestrator.get_training_universe()` |
| **Trading** | Risk management | Primary candidates | `TradingEngine.get_eligible_symbols()` |

## Backfill Integration

### Layer 0: Comprehensive Data Collection

**Purpose**: Collect historical data for ALL tradeable symbols
```bash
# Layer 0 drives comprehensive backfill
python ai_trader.py universe --populate  # Discover ~10,000 symbols
python ai_trader.py backfill --days 90   # Backfill ALL symbols
```

**Integration Details**:
- **File**: `src/main/app/run_backfill.py:42-79`
- **Method**: `BackfillOrchestrator._get_universe_symbols()`
- **Behavior**: 
  - Queries Layer 0 universe (all active companies)
  - No symbol limit (`fallback_limit=None`)
  - Auto-populates universe if empty
  - Falls back to config only if database unavailable

**Data Coverage**:
- **Historical Market Data**: OHLCV for all Layer 0 symbols
- **News Data**: Corporate news for all symbols  
- **Financial Data**: Quarterly reports for all symbols
- **Options Data**: Options chains for optionable symbols

### Layer 1-3: Prioritized Updates

**Purpose**: Prioritize real-time updates for qualified symbols
```bash
# Get current Layer 3 candidates for priority backfill
SYMBOLS=$(python ai_trader.py universe --layer 3 --limit 50)
python ai_trader.py backfill --symbols $SYMBOLS --days 5 --force
```

**Event-Driven Updates**:
- **New Layer 3 Symbol**: Triggers immediate 30-day backfill
- **Layer 3 Disqualification**: Reduces update frequency to daily
- **Layer Graduation**: Upgrades symbol to higher-frequency collection

### Backfill Configuration

#### Universe-Driven Settings
```yaml
# config/data_pipeline/backfill.yml
backfill:
  layer_priorities:
    layer_0: 
      frequency: "daily"
      lookback_days: 2
    layer_1:
      frequency: "every_6_hours" 
      lookback_days: 5
    layer_2:
      frequency: "every_2_hours"
      lookback_days: 7
    layer_3:
      frequency: "every_30_minutes"
      lookback_days: 10
```

## Training Integration

### Training Universe Selection

**Layer-Based Training Strategy**:
```bash
# Train on high-conviction Layer 3 symbols (focused models)
python ai_trader.py train --symbols $(python ai_trader.py universe --layer 3) --models ensemble

# Train on broader Layer 1 universe (generalization)  
python ai_trader.py train --symbols $(python ai_trader.py universe --layer 1 --limit 500) --models xgboost

# Full universe training (research/backtesting)
python ai_trader.py train --symbols $(python ai_trader.py universe --layer 0 --limit 2000) --models research
```

### Training Data Weighting

**Integration Point**: `src/main/models/training/training_orchestrator.py`

```python
def get_training_weights(self, symbols: List[str]) -> Dict[str, float]:
    """Assign training weights based on layer qualification"""
    weights = {}
    for symbol in symbols:
        layer = self.universe_manager.get_symbol_layer(symbol)
        weights[symbol] = {
            "0": 0.5,    # Base weight for all symbols
            "1": 1.0,    # Standard weight for liquid symbols  
            "2": 2.0,    # Higher weight for technical candidates
            "3": 4.0     # Highest weight for fundamental picks
        }.get(layer, 0.1)
    return weights
```

### Model Architecture Integration

#### Multi-Layer Model Training
```python
# Layer-specific model specialization
models = {
    "layer1_liquidity": LiquidityPredictor(symbols=layer1_symbols),
    "layer2_technical": TechnicalMomentumModel(symbols=layer2_symbols), 
    "layer3_fundamental": FundamentalValueModel(symbols=layer3_symbols),
    "ensemble": EnsembleModel(layer3_symbols)  # Primary trading model
}
```

#### Feature Engineering by Layer
- **Layer 0**: Basic OHLCV features, macro indicators
- **Layer 1**: Volume patterns, liquidity metrics
- **Layer 2**: Technical indicators, momentum signals
- **Layer 3**: Fundamental ratios, sentiment scores, analyst data

## Event-Driven Architecture

### EventBus Integration

**Event Types**:
```python
# Layer completion events
LayerCompletionEvent(layer=0, timestamp=now, symbol_count=10000)
LayerCompletionEvent(layer=3, timestamp=now, symbol_count=45)

# Symbol qualification events  
SymbolQualifiedEvent(symbol="AAPL", from_layer=2, to_layer=3)
SymbolDisqualifiedEvent(symbol="XYZ", from_layer=3, reason="liquidity")

# Universe update events
UniverseUpdateEvent(layer=0, added=["NEW1", "NEW2"], removed=["OLD1"])
```

**Event Handlers**:
- **BackfillHandler**: Triggers priority backfill for newly qualified symbols
- **FeatureHandler**: Updates feature store for layer changes
- **TrainingHandler**: Schedules model retraining when Layer 3 changes >20%
- **TradingHandler**: Updates eligible symbol lists for live trading

### ScannerFeatureBridge

**Purpose**: Automatically propagate scanner results to feature pipeline

**Integration Point**: `src/main/feature_pipeline/scanner_bridge.py`

```python
class ScannerFeatureBridge:
    """Bridges scanner qualification results into feature store"""
    
    async def on_layer_completion(self, event: LayerCompletionEvent):
        """Update features when layer qualification completes"""
        layer_features = {
            "layer_0_qualified": event.layer >= 0,
            "layer_1_qualified": event.layer >= 1, 
            "layer_2_qualified": event.layer >= 2,
            "layer_3_qualified": event.layer >= 3,
            "qualification_timestamp": event.timestamp
        }
        await self.feature_store.update_symbol_features(
            symbols=event.symbols, 
            features=layer_features
        )
```

### Automatic Retraining Triggers

**Layer 3 Composition Changes**:
```python
# Trigger retraining when Layer 3 universe changes significantly
if layer3_change_rate > 0.20:  # 20% threshold
    training_orchestrator.schedule_retraining(
        priority="high",
        models=["ensemble", "layer3_fundamental"],
        symbols=new_layer3_symbols
    )
```

## Operational Workflows

### Daily Production Workflow

#### 6:00 AM EST - Pre-Market Setup
```bash
# 1. Update universe and check health
python ai_trader.py universe --populate
python ai_trader.py universe --health

# 2. Priority backfill for Layer 3 symbols (previous day's close)
python ai_trader.py backfill --stage daily --days 1

# 3. Layer 1 qualification runs automatically
# (Analyzes previous day's volume/liquidity data)
```

#### 9:30 AM EST - Market Open
```bash
# 1. Real-time data collection starts
# Layer 2 technical scans run every 15 minutes

# 2. Monitor qualification changes
python ai_trader.py universe --stats

# 3. Trading engine uses Layer 3 symbols
python ai_trader.py trade --mode live
```

#### 4:30 PM EST - Post-Market
```bash
# 1. Layer 3 fundamental analysis runs automatically
# 2. Check for significant universe changes
python ai_trader.py universe --layer 3

# 3. Trigger training if needed
if [ "$(python ai_trader.py universe --layer 3 | wc -l)" -ne "$PREV_LAYER3_COUNT" ]; then
    python ai_trader.py train --models ensemble --symbols $(python ai_trader.py universe --layer 3)
fi
```

### Weekly Maintenance

#### Sunday Evening - Full Refresh
```bash
# 1. Complete universe refresh
python ai_trader.py universe --populate --force

# 2. Extended historical backfill
python ai_trader.py backfill --days 30 --force

# 3. Full model retraining
python ai_trader.py train --models all --lookback-days 365
```

## Configuration Integration

### Universe-Aware Configuration

#### Dynamic Symbol Lists
```yaml
# config/trading/universe.yml
trading:
  universe:
    primary: "layer_3"           # Primary trading candidates
    fallback: "layer_2"         # Fallback if Layer 3 < 20 symbols
    min_symbols: 15             # Minimum trading universe size
    max_symbols: 50             # Maximum for position sizing
    
  allocation:
    layer_3_weight: 0.80        # 80% allocation to Layer 3
    layer_2_weight: 0.20        # 20% allocation to Layer 2
```

#### Model Training Configuration
```yaml
# config/models/training.yml
training:
  universe_strategy: "layered"
  
  layer_configs:
    layer_0:
      sample_size: 2000         # Random sample for broad training
      weight: 0.5
    layer_1: 
      sample_size: 1000
      weight: 1.0
    layer_2:
      sample_size: 200
      weight: 2.0  
    layer_3:
      sample_size: 50           # All Layer 3 symbols
      weight: 4.0
```

## Monitoring and Observability

### Key Metrics

#### Universe Health Metrics
```bash
# Layer progression rates
python ai_trader.py universe --stats
# Expected: L0→L1: 25%, L1→L2: 20%, L2→L3: 30%

# Data coverage metrics  
python ai_trader.py status --component data
# Expected: >95% historical data coverage for Layer 3 symbols
```

#### Integration Health Checks
```bash
# Backfill coverage by layer
python ai_trader.py validate --component data --layer-breakdown

# Feature store synchronization
python ai_trader.py validate --component features --scanner-sync

# Model training universe alignment
python ai_trader.py validate --component models --universe-alignment
```

### Alerting Thresholds

#### Universe Degradation
- **Layer 3 < 15 symbols**: Critical - insufficient trading candidates
- **Layer 2 < 50 symbols**: Warning - may need looser technical criteria  
- **Layer 1 < 1500 symbols**: Warning - market liquidity concerns

#### Integration Failures
- **Backfill lag > 6 hours**: Critical - data freshness issue
- **Feature sync lag > 1 hour**: Warning - potential trading delays
- **Training universe mismatch > 10%**: Warning - model drift risk

## Performance Optimization

### Efficient Data Access

#### Layered Data Storage
```python
# Prioritized storage tiers based on layer qualification
storage_tiers = {
    "layer_3": "hot_storage",      # SSD, Redis cache
    "layer_2": "warm_storage",     # SSD, minimal cache  
    "layer_1": "cold_storage",     # HDD, batch access
    "layer_0": "archive_storage"   # Compressed, S3/archive
}
```

#### Selective Feature Computation
```python
# Compute expensive features only for qualified symbols
if symbol in layer_3_symbols:
    features.update(compute_advanced_sentiment(symbol))
    features.update(compute_options_flow(symbol))
    
if symbol in layer_2_symbols:
    features.update(compute_technical_indicators(symbol))
```

### Batch Processing Optimization

#### Layer-Aware Batch Sizes
```yaml
# Optimize batch processing by layer priority
batch_configs:
  layer_3:
    batch_size: 10              # Small batches, frequent updates
    max_latency: "30s"
  layer_2:  
    batch_size: 50              # Medium batches
    max_latency: "5m"
  layer_1:
    batch_size: 200             # Large batches
    max_latency: "1h"
  layer_0:
    batch_size: 1000            # Very large batches  
    max_latency: "24h"
```

---

**Next**: See [SCANNER_OPERATIONS.md](SCANNER_OPERATIONS.md) for operational procedures and troubleshooting guides.
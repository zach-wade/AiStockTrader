# AI Trader Complete System Overview

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Flow Overview](#data-flow-overview)
3. [Project Structure](#project-structure)
4. [Core Components](#core-components)
5. [Detailed Module Breakdown](#detailed-module-breakdown)

## System Architecture

The AI Trader system is a comprehensive algorithmic trading platform built with Python that implements:
- Multi-source data ingestion and processing
- Multi-layer scanning system for opportunity detection
- Feature engineering pipeline
- Multiple trading strategies with ensemble learning
- Risk management and position sizing
- Real-time execution engine
- Comprehensive monitoring and alerting

### Key Design Principles
- **Event-Driven Architecture**: Loose coupling between components using pub-sub pattern
- **Multi-Stage Validation**: Data quality assurance at every processing stage
- **Hot/Cold Data Management**: Automatic lifecycle management for storage optimization
- **Modular Design**: Each component can function independently
- **Async-First**: Built for concurrent processing and real-time operations

## Data Flow Overview

```
1. DATA INGESTION
   ├── Market Data Sources (Alpaca, Polygon, Yahoo)
   ├── News Sources (Benzinga, Reddit, Twitter)
   └── Alternative Data (Options flow, Insider trading)
           ↓
2. VALIDATION STAGE 1: INGEST
   └── Raw data validation
           ↓
3. DATA STANDARDIZATION & STORAGE
   ├── Hot Storage (PostgreSQL) - Recent 30 days
   └── Cold Storage (Data Lake) - Historical
           ↓
4. VALIDATION STAGE 2: POST-ETL
   └── Transformation validation
           ↓
5. SCANNER SYSTEM (3 Layers)
   ├── Layer 1: Basic pattern detection
   ├── Layer 1.5: Universe filtering
   └── Layer 2: Catalyst detection
           ↓
6. EVENT BUS
   └── Scanner alerts trigger feature computation
           ↓
7. FEATURE PIPELINE
   ├── Technical indicators
   ├── Market microstructure
   ├── Sentiment analysis
   └── Cross-asset correlations
           ↓
8. VALIDATION STAGE 3: PRE-FEATURE
   └── Feature-ready validation
           ↓
9. TRADING STRATEGIES
   ├── Mean Reversion
   ├── ML Momentum
   ├── Pairs Trading
   ├── Regime Adaptive
   └── Ensemble Meta-Learning
           ↓
10. RISK MANAGEMENT
    ├── Position sizing
    ├── Circuit breakers
    └── Drawdown control
           ↓
11. EXECUTION ENGINE
    ├── Order management
    ├── Broker integration
    └── Portfolio tracking
           ↓
12. MONITORING & ALERTS
    ├── Performance logging
    ├── System health monitoring
    └── Alert notifications
```

## Project Structure

```
ai_trader/
├── src/main/              # Main source code
│   ├── __init__.py            # Package initialization with exports
│   ├── config/                # Configuration management
│   ├── data_pipeline/         # Data ingestion and processing
│   ├── scanners/              # Multi-layer scanning system
│   ├── feature_pipeline/      # Feature engineering
│   ├── models/                # Trading strategies
│   ├── trading_engine/        # Execution and portfolio management
│   ├── risk_management/       # Risk controls
│   ├── monitoring/            # Logging and alerts
│   ├── backtesting/          # Historical testing
│   ├── orchestration/        # System coordination
│   ├── events/               # Event-driven architecture
│   └── utils/                # Shared utilities
├── tests/                     # Test suite
├── docs/                      # Documentation
├── configs/                   # Configuration files
├── scripts/                   # Utility scripts
└── requirements.txt          # Dependencies
```

## Core Components

### 1. Configuration (`config/`)
- **ConfigManager**: Centralized configuration management
- **Config Models**: Pydantic models for type-safe configuration

### 2. Data Pipeline (`data_pipeline/`)
- **Ingestion**: Multi-source data collection
- **Validation**: 3-stage validation system
- **Storage**: Hot/cold data lifecycle management
- **Transformation**: Data standardization and ETL

### 3. Scanners (`scanners/`)
- **Layer 1**: Basic pattern detection (volume, price, momentum)
- **Layer 1.5**: Universe filtering and prioritization
- **Layer 2**: Advanced catalyst detection

### 4. Feature Pipeline (`feature_pipeline/`)
- **Technical Features**: Price, volume, volatility indicators
- **Market Microstructure**: Order flow, bid-ask analysis
- **Sentiment Features**: News and social media analysis
- **Cross-Asset Features**: Correlations and regime detection

### 5. Trading Strategies (`models/strategies/`)
- Individual strategy implementations
- Ensemble meta-learning system
- Signal generation and aggregation

### 6. Execution Engine (`trading_engine/`)
- Order management and routing
- Broker integration (Alpaca)
- Portfolio tracking and P&L calculation

### 7. Risk Management (`risk_management/`)
- Position sizing algorithms
- Real-time risk monitoring
- Circuit breakers and limits

### 8. Event System (`events/`)
- Event bus for pub-sub communication
- Scanner-feature pipeline bridge
- Asynchronous event processing

## Detailed Module Breakdown

### `/src/main/__init__.py`
Main package initialization that exports key components:
- Scanner components (ScanAlert, AlertType, Layer2CatalystOrchestrator)
- Data pipeline components (ValidationPipeline, DataLifecycleManager)
- Utils (get_logger, ScanAlert)

### `/src/main/config/`

#### `config_manager.py`
- **ConfigManager**: Loads and manages configuration from YAML/JSON files
- **get_config()**: Global config accessor function
- **Config**: Main configuration class with nested sections

#### `config_models.py`
- **DatabaseConfig**: Database connection settings
- **APIKeysConfig**: API credentials for data sources
- **TradingConfig**: Trading parameters and limits
- **RiskConfig**: Risk management settings

### `/src/main/data_pipeline/`

#### `ingestion/`
Data source client implementations:

##### `base_api_client.py`
- **BaseAPIClient**: Abstract base for API clients with rate limiting
- **RateLimitConfig**: Rate limit configuration
- **AuthMethod**: Authentication method enum

##### `alpaca_client.py`
- **AlpacaClient**: Alpaca data and trading API integration
  - `fetch_market_data()`: Get historical price data
  - `fetch_news()`: Get news articles
  - `get_tradable_assets()`: Get list of tradable symbols
  - Implements validation on ingest

##### `polygon_client.py`
- **PolygonClient**: Polygon.io data integration
  - `fetch_market_data()`: Historical and real-time data
  - `fetch_news()`: News and sentiment data
  - `get_market_calendar()`: Trading calendar

##### `yahoo_client.py`
- **YahooClient**: Yahoo Finance integration
  - `fetch_market_data()`: Free historical data
  - `fetch_fundamentals()`: Company financials
  - Handles data back to 1960s

##### `benzinga_client.py`
- **BenzingaClient**: Benzinga news and analytics
  - `fetch_news()`: Financial news
  - `fetch_ratings()`: Analyst ratings
  - `fetch_economics()`: Economic indicators

#### `validation/`
Multi-stage validation system:

##### `validation_pipeline.py`
- **ValidationPipeline**: Orchestrates 3-stage validation
- **ValidationStage**: Enum (INGEST, POST_ETL, PRE_FEATURE)
- **ValidationResult**: Validation outcome with issues and metrics
- **IngestValidator**: Stage 1 - raw data validation
- **PostETLValidator**: Stage 2 - post-transformation validation
- **PreFeatureValidator**: Stage 3 - pre-feature engineering validation

##### `data_validator.py`
- **DataValidator**: Legacy validator wrapper for compatibility
  - `validate_market_data()`: OHLCV validation
  - `validate_data_quality()`: Quality metrics

#### `storage/`
Data storage and lifecycle management:

##### `partition_manager.py`
- **DataLakePartitionManager**: Manages data lake partitions
  - `discover_partitions()`: Find all data partitions
  - `analyze_fragmentation()`: Identify consolidation opportunities
  - `consolidate_partitions()`: Merge fragmented partitions
  - `run_lifecycle_automation()`: Automated hot/cold management

##### `data_lifecycle_manager.py`
- **DataLifecycleManager**: Manages hot/cold data transitions
  - `apply_policies()`: Apply lifecycle policies
  - `archive_to_cold()`: Move old data to cold storage
  - `restore_from_cold()`: Restore data to hot storage

#### `transform/`
Data transformation and standardization:

##### `data_standardizer.py`
- **DataStandardizer**: Standardizes data from different sources
  - `standardize_market_data()`: OHLCV standardization
  - `standardize_news_data()`: News format standardization
  - Source-specific methods for each data provider

#### `staging/`
ETL and batch processing:

##### `etl_processor.py`
- **ColdPathProcessor**: Batch ETL for historical data
  - `process_nightly_etl()`: Daily ETL pipeline
  - `_process_daily_aggregation()`: Aggregate minute to daily bars
  - `_process_feature_engineering()`: Generate technical features
  - `_process_ml_training_prep()`: Prepare ML datasets

### `/src/main/scanners/`

#### Base Components

##### `__init__.py`
- **ScanAlert**: Standardized alert dataclass
- **AlertType**: Alert type enumeration

##### `scan_alert.py`
- **ScanAlert**: Core alert structure
  - symbol, alert_type, timestamp, score, metadata
  - source_scanner tracking

#### Layer 1 Scanners (`layers/layer1/`)

##### `layer1_pattern_detector.py`
- **Layer1PatternDetector**: Basic pattern detection
  - `high_volume_scanner`: Unusual volume detection
  - `price_breakout_scanner`: Price breakout patterns
  - `momentum_scanner`: Momentum shifts

#### Layer 1.5 (`layers/`)

##### `layer1_5_universe_filter.py`
- **Layer1_5UniverseFilter**: Universe filtering and prioritization
  - Liquidity filtering
  - Volatility-based ranking
  - Combines Layer 1 signals

#### Layer 2 (`layers/`)

##### `layer2_catalyst_orchestrator.py`
- **Layer2CatalystOrchestrator**: Advanced catalyst detection
  - Orchestrates multiple catalyst scanners
  - Aggregates signals
  - Publishes events to event bus
  - `_publish_catalyst_events()`: Event publishing

#### Catalyst Scanners (`catalysts/`)

##### `insider_scanner.py`
- **InsiderScanner**: Insider trading pattern detection

##### `options_scanner.py`
- **OptionsScanner**: Options flow analysis

##### `sentiment_scanner.py`
- **AdvancedSentimentScanner**: News/social sentiment

##### `intermarket_scanner.py`
- **InterMarketScanner**: Cross-asset correlations

##### `sector_scanner.py`
- **SectorScanner**: Sector rotation patterns

### `/src/main/feature_pipeline/`

#### `orchestrator.py`
- **FeatureOrchestrator**: Coordinates feature computation
  - `compute_features()`: Main feature computation
  - Manages feature dependencies
  - Caches computed features

#### `calculators/`
Individual feature calculators:

##### `technical_indicators.py`
- **TechnicalIndicatorCalculator**: Standard technical indicators
  - Moving averages (SMA, EMA)
  - RSI, MACD, Bollinger Bands
  - ATR, Stochastic

##### `microstructure_features.py`
- **MicrostructureCalculator**: Market microstructure features
  - Bid-ask spread analysis
  - Order flow imbalance
  - Price impact measures

##### `sentiment_features.py`
- **SentimentCalculator**: Sentiment analysis features
  - News sentiment scores
  - Social media sentiment
  - Sentiment momentum

##### `cross_asset_features.py`
- **CrossAssetCalculator**: Cross-asset relationships
  - Correlations with indices
  - Sector relationships
  - Currency impacts

##### `regime_features.py`
- **RegimeCalculator**: Market regime detection
  - Volatility regimes
  - Trend regimes
  - Risk-on/risk-off indicators

### `/src/main/models/`

#### `strategies/`
Trading strategy implementations:

##### `mean_reversion_strategy.py`
- **MeanReversionStrategy**: Mean reversion trading
  - Z-score based signals
  - Dynamic thresholds
  - Multiple timeframe analysis

##### `ml_momentum_strategy.py`
- **MLMomentumStrategy**: Machine learning momentum
  - XGBoost/LightGBM models
  - Feature importance tracking
  - Online learning updates

##### `pairs_trading_strategy.py`
- **PairsTradingStrategy**: Statistical arbitrage
  - Cointegration testing
  - Spread trading signals
  - Dynamic pair selection

##### `regime_adaptive_strategy.py`
- **RegimeAdaptiveStrategy**: Regime-based trading
  - Regime detection
  - Strategy switching
  - Risk adjustment by regime

##### `ensemble_meta_strategy.py`
- **EnsembleMetaLearningStrategy**: Combines all strategies
  - Dynamic weight allocation
  - Performance tracking
  - Meta-learning optimization

### `/src/main/trading_engine/`

#### `core/`

##### `execution_engine.py`
- **ExecutionEngine**: Order execution management
  - `execute_signals()`: Convert signals to orders
  - `route_order()`: Smart order routing
  - Order tracking and updates

##### `portfolio_manager.py`
- **PortfolioManager**: Portfolio tracking
  - Position management
  - P&L calculation
  - Risk metrics computation

##### `order_manager.py`
- **OrderManager**: Order lifecycle management
  - Order validation
  - Status tracking
  - Fill management

#### `brokers/`

##### `broker_interface.py`
- **BrokerInterface**: Abstract broker interface
  - Standard methods for all brokers
  - Event callbacks

##### `alpaca_broker.py`
- **AlpacaBroker**: Alpaca brokerage integration
  - `submit_order()`: Place orders
  - `get_positions()`: Position tracking
  - `get_account_info()`: Account data
  - Real-time data streaming

### `/src/main/risk_management/`

#### `real_time/`

##### `circuit_breaker.py`
- **CircuitBreaker**: Trading halts and limits
  - Daily loss limits
  - Position limits
  - Volatility-based halts

##### `drawdown_control.py`
- **DrawdownController**: Drawdown management
  - Maximum drawdown limits
  - Recovery rules
  - Position reduction

##### `position_sizer.py`
- **PositionSizer**: Position sizing algorithms
  - Kelly Criterion
  - Fixed fractional
  - Volatility-based sizing

### `/src/main/events/`
Event-driven architecture components:

#### `event_bus.py`
- **EventBus**: Central pub-sub event system
  - `publish()`: Publish events
  - `subscribe()`: Subscribe to event types
  - Event history and replay
- **Event**: Base event class
- **EventType**: Event type enumeration
- **ScannerAlertEvent**: Scanner alert events
- **FeatureRequestEvent**: Feature computation requests

#### `scanner_feature_bridge.py`
- **ScannerFeatureBridge**: Connects scanners to features
  - Batches scanner alerts
  - Maps alerts to required features
  - Priority-based processing

#### `feature_pipeline_handler.py`
- **FeaturePipelineHandler**: Handles feature requests
  - Priority queue processing
  - Worker pool management
  - Feature computation coordination

### `/src/main/monitoring/`

#### `logging/`

##### `performance_logger.py`
- **PerformanceLogger**: Performance tracking
  - Trade logging
  - P&L tracking
  - Strategy performance metrics

#### `alerts/`

##### `unified_alerts.py`
- **UnifiedAlertSystem**: Alert management
  - Email notifications
  - Slack integration
  - Alert aggregation and throttling

### `/src/main/orchestration/`

#### `main_orchestrator.py`
- **UnifiedOrchestrator**: Main system coordinator
  - `start()`: System startup sequence
  - `_run_trading_mode()`: Main trading loop
  - `_data_collection_loop()`: Data ingestion
  - `_strategy_execution_loop()`: Strategy execution
  - `_risk_monitoring_loop()`: Risk monitoring
  - `_lifecycle_management_loop()`: Data lifecycle
  - Event system integration
  - Graceful shutdown handling

### `/src/main/utils/`
Shared utilities:

#### `logging_config.py`
- **get_logger()**: Configured logger factory
- Centralized logging configuration

#### `db_utils.py`
- **DatabaseManager**: Database connection management
- Connection pooling
- Query helpers

#### `market_utils.py`
- **MarketCalendar**: Trading calendar
- **is_market_open()**: Market hours check
- Trading day calculations

## Key Design Patterns

### 1. Event-Driven Architecture
- Loose coupling between scanners and feature pipeline
- Asynchronous event processing
- Pub-sub pattern for scalability

### 2. Strategy Pattern
- Common interface for all trading strategies
- Easy addition of new strategies
- Strategy composition through ensemble

### 3. Factory Pattern
- Data source client creation
- Feature calculator instantiation
- Strategy initialization

### 4. Observer Pattern
- Market data updates
- Order status updates
- System health monitoring

### 5. Pipeline Pattern
- Data processing pipeline
- Feature engineering pipeline
- Validation pipeline

## Configuration Files

### `configs/config.yaml`
Main configuration file with sections:
- Database connections
- API credentials
- Trading parameters
- Risk limits
- Scanner thresholds
- Feature pipeline settings

### `configs/universe.yaml`
Trading universe definition:
- Symbol lists
- Sector classifications
- Liquidity thresholds

## Testing Structure

### Unit Tests (`tests/unit/`)
- Individual component testing
- Mocked dependencies
- Fast execution

### Integration Tests (`tests/integration/`)
- Multi-component testing
- Real data sources
- End-to-end scenarios

### Performance Tests (`tests/performance/`)
- Load testing
- Latency measurements
- Scalability testing

## Deployment Considerations

### System Requirements
- Python 3.8+
- PostgreSQL for hot storage
- Sufficient disk space for data lake
- Network connectivity for data sources

### Monitoring
- Health check endpoints
- Performance metrics
- Alert thresholds
- Log aggregation

### Scalability
- Horizontal scaling through event system
- Database partitioning
- Distributed processing capability

## Future Enhancements

### Planned Features
1. Additional data sources (IEX, Quandl)
2. More trading strategies
3. Advanced ML models
4. Real-time dashboard
5. Mobile notifications
6. Cloud deployment templates

### Architecture Improvements
1. Kubernetes deployment
2. Message queue integration (Kafka/RabbitMQ)
3. Distributed computing (Dask/Ray)
4. Time-series database integration
5. GraphQL API layer
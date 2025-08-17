# AI Trader Configuration Architecture

## Overview

The AI Trader configuration system is built on a clean, simple architecture that prioritizes reliability, maintainability, and thread safety. The system uses OmegaConf for YAML processing, Pydantic for validation, and implements proper caching with thread safety.

## Architecture Principles

1. **Simplicity Over Complexity**: No over-engineering or unnecessary abstractions
2. **Single Responsibility**: Each component has a clear, focused purpose
3. **Thread Safety**: All operations are safe for concurrent access
4. **Type Safety**: Comprehensive Pydantic validation models
5. **Environment Flexibility**: Clean environment variable support and overrides

## Directory Structure

```
config/
├── __init__.py                    # Clean exports - main public API
├── config_manager.py              # Core ConfigManager with thread-safe caching
├── field_mappings.py              # Data source field mappings (Polygon, Alpaca, etc.)
├── env_loader.py                  # Environment variable loading and validation
├── validation_utils.py            # Configuration validation utilities
├── validation_models/             # Pydantic validation models
│   ├── __init__.py               # Model exports
│   ├── core.py                   # API keys, environment enums
│   ├── trading.py                # Trading and risk management models
│   ├── data.py                   # Data pipeline and features models
│   ├── services.py               # Monitoring and orchestrator models
│   └── main.py                   # Main AITraderConfig model
├── yaml/                         # Configuration files
│   ├── layer_definitions.yaml    # Symbol layer definitions
│   ├── app_context_config.yaml   # Application context
│   ├── event_config.yaml         # Event system configuration
│   ├── startup_validation.yaml   # Startup validation rules
│   ├── universe_definitions.yaml # Trading universe definitions
│   ├── environments/            # Environment-specific overrides
│   │   ├── development.yaml
│   │   ├── paper.yaml
│   │   └── production.yaml
│   ├── services/                # Service-specific configurations
│   │   ├── alerting.yaml
│   │   ├── backfill.yaml
│   │   ├── dashboard.yaml
│   │   ├── ml_trading.yaml
│   │   ├── models.yaml
│   │   └── scanners.yaml
│   └── defaults/                # Default configuration values
│       ├── data.yaml
│       ├── lifecycle.yaml
│       ├── network.yaml
│       ├── rate_limits.yaml
│       ├── risk.yaml
│       ├── storage.yaml
│       ├── system.yaml
│       └── trading.yaml
├── docs/                        # Documentation
│   ├── CONFIG_ARCHITECTURE.md   # This file
│   ├── CONFIG_FIXES_SUMMARY.md  # Summary of fixes applied
│   └── README.md                 # Quick start guide
└── monitoring/                  # Monitoring configurations
    ├── grafana_dashboard.json
    └── prometheus_alerts.yml
```

## Core Components

### 1. ConfigManager (config_manager.py)

The heart of the configuration system, providing:

- **Thread-safe caching** with TTL support using `threading.RLock()`
- **Reliable cache key generation** using MD5 hashing for overrides
- **Single validation path** with comprehensive error handling
- **Environment variable resolution** through OmegaConf
- **YAML processing** with interpolation support

```python
from main.config import get_config_manager

# Standard usage
manager = get_config_manager()
config = manager.load_config("layer_definitions")

# With validation enabled (default)
manager = get_config_manager(use_validation=True)
config = manager.load_config("layer_definitions")

# Production mode (strict validation)
from main.config import get_production_config_manager
manager = get_production_config_manager()
config = manager.load_config("layer_definitions")  # Will fail fast on errors
```

### 2. ConfigCache (config_manager.py)

Thread-safe configuration cache with:

- **TTL-based expiration** (default 5 minutes)
- **Thread-safe operations** using RLock
- **Automatic cleanup** of expired entries
- **Cache statistics** and monitoring support

### 3. Validation Models (validation_models/)

Comprehensive Pydantic models providing:

#### Core Models (core.py)

- `Environment` enum (PAPER, LIVE, TRAINING, BACKTEST)
- API key configurations (Alpaca, Polygon, Alpha Vantage, etc.)
- Environment variable validation with `validate_env_var()`

#### Trading Models (trading.py)

- `SystemConfig` - System-level settings
- `BrokerConfig` - Broker configuration
- `TradingConfig` - Trading parameters
- `RiskConfig` - Risk management settings
- Position sizing, execution, and strategy configurations

#### Data Models (data.py)

- `DataConfig` - Data source configuration
- `BackfillConfig` - Historical data backfill settings
- `FeaturesConfig` - Feature engineering configuration
- `TrainingConfig` - Model training parameters

#### Services Models (services.py)

- `MonitoringConfig` - System monitoring
- `OrchestratorConfig` - Orchestrator settings
- `EnvironmentOverrides` - Environment-specific overrides

#### Main Model (main.py)

- `AITraderConfig` - Complete system configuration
- Environment consistency validation
- Risk configuration validation
- Configuration merging and override support

### 4. Field Mappings (field_mappings.py)

Data source field standardization:

```python
from main.config import get_field_mapping_config

mapping_config = get_field_mapping_config()
polygon_mapping = mapping_config.get_mapping('polygon')
# Returns: {'t': 'timestamp', 'o': 'open', 'h': 'high', ...}
```

### 5. Environment Loader (env_loader.py)

Environment variable management:

```python
from main.config import ensure_environment_loaded, is_production

ensure_environment_loaded()  # Load .env file
is_prod = is_production()    # Check environment type
```

### 6. Validation Utils (validation_utils.py)

Configuration validation utilities:

```python
from main.config import ConfigValidator, validate_config_and_warn

# Strict validation
validator = ConfigValidator(strict_mode=True)
config = validator.validate_file('config.yaml')

# Non-strict validation with warnings
config, warnings = validate_config_and_warn('config.yaml')
```

## Usage Patterns

### Basic Configuration Loading

```python
from main.config import get_config_manager

# Load main configuration
manager = get_config_manager()
config = manager.load_config("layer_definitions")

# Access configuration values
api_key = config.api_keys.alpaca.key
paper_trading = config.broker.paper_trading
max_position = config.risk.position_sizing.max_position_size
```

### Configuration with Overrides

```python
# Override specific values
config = manager.load_config(
    "layer_definitions",
    overrides=["system.environment=live", "broker.paper_trading=false"]
)
```

### Environment-Specific Configuration

```python
# Load environment-specific overrides
env_config = manager.load_environment_config("production", base_config=config)
```

### Backward Compatibility

```python
from main.config import get_config

# Legacy usage still supported
config = get_config("layer_definitions")
value = config.get('api_keys.alpaca.key')
```

### Cache Management

```python
# Clear specific configuration cache
manager.clear_cache("layer_definitions")

# Clear all cache
manager.clear_cache()

# Get cache statistics
stats = manager.get_cache_stats()
print(f"Cache entries: {stats['total_entries']}")
```

## Configuration Files

### Main Configuration (layer_definitions.yaml)

Defines symbol layers and trading universe:

```yaml
layers:
  layer_0:
    name: "Universe"
    description: "All tradable symbols (~9,800)"
    criteria:
      exchanges: ["NYSE", "NASDAQ", "AMEX"]
      min_market_cap: 100_000_000
      min_price: 1.0
    retention:
      hot_storage_days: 30
      data_types: ["1hour", "1day"]
```

### Environment Overrides

Environment-specific settings in `environments/`:

```yaml
# environments/production.yaml
system:
  environment: live
broker:
  paper_trading: false
risk:
  position_sizing:
    max_position_size: 2.0  # More conservative for live trading
  circuit_breaker:
    daily_loss_limit: 2.0   # Stricter limits
```

### Service Configurations

Service-specific settings in `services/`:

```yaml
# services/backfill.yaml
backfill:
  max_parallel: 20
  stages:
    - name: market_data
      sources: [polygon]
      intervals: [1day, 1hour]
      lookback_days: 365
```

## Environment Variables

The system supports environment variable substitution:

```yaml
api_keys:
  alpaca:
    key: ${ALPACA_API_KEY}
    secret: ${ALPACA_SECRET_KEY}
  polygon:
    key: ${POLYGON_API_KEY}
```

Required environment variables:

- `ALPACA_API_KEY` - Alpaca API key
- `ALPACA_SECRET_KEY` - Alpaca secret key
- `POLYGON_API_KEY` - Polygon API key
- `DATABASE_URL` - Database connection string

## Thread Safety

The configuration system is fully thread-safe:

1. **ConfigCache** uses `threading.RLock()` for all operations
2. **Cache key generation** is atomic and consistent
3. **Configuration loading** can be called concurrently
4. **Cache clearing** is properly synchronized

## Error Handling

Comprehensive error handling with:

1. **Validation errors** with detailed messages
2. **Environment variable validation** with clear error reporting
3. **File not found** errors with helpful suggestions
4. **YAML parsing errors** with line number information
5. **Cache operation errors** with fallback behavior

## Performance

- **Caching** reduces repeated YAML parsing and validation
- **TTL-based expiration** balances performance and freshness
- **Thread-safe operations** without performance penalties
- **Lazy loading** of validation models
- **Efficient cache key generation** using MD5 hashing

## Best Practices

1. **Use the factory functions**: `get_config_manager()`, `get_production_config_manager()`
2. **Enable validation in production**: Use strict validation for fail-fast behavior
3. **Cache management**: Clear cache when configurations change
4. **Environment variables**: Store sensitive data in environment variables, not YAML
5. **Configuration validation**: Always validate configurations before deployment
6. **Error handling**: Handle `ConfigValidationError` appropriately in your application

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   - Check file paths relative to the config directory
   - Ensure YAML files are in the correct location

2. **Environment variable not set**
   - Check your `.env` file in the project root
   - Verify environment variable names match YAML references

3. **Validation errors**
   - Check Pydantic model constraints
   - Review error messages for specific field issues

4. **Cache issues**
   - Clear cache after configuration changes
   - Check cache statistics to verify operation

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

manager = get_config_manager()
config = manager.load_config("layer_definitions")
```

## Future Enhancements

Potential improvements for the configuration system:

1. **Configuration schema validation** for YAML files
2. **Configuration reload capabilities** without restart
3. **Configuration diff tools** for comparing environments
4. **Metrics integration** for cache hit rates and performance
5. **Configuration hot-reloading** during development

## References

- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Python Threading Documentation](https://docs.python.org/3/library/threading.html)

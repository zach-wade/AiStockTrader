# Configuration Directory

## Quick Start

The AI Trader configuration system provides a clean, thread-safe way to manage application settings with comprehensive validation and environment support.

```python
from main.config import get_config_manager

# Load configuration
manager = get_config_manager()
config = manager.load_config("layer_definitions")

# Access configuration values  
api_key = config.api_keys.alpaca.key
paper_trading = config.broker.paper_trading
```

## Directory Structure

```
config/
├── __init__.py                    # Main public API exports
├── config_manager.py              # Core ConfigManager with caching
├── field_mappings.py              # Data source field mappings
├── env_loader.py                  # Environment variable handling
├── validation_utils.py            # Configuration validation
├── validation_models/             # Pydantic validation models
│   ├── __init__.py
│   ├── core.py                   # API keys, environments
│   ├── trading.py                # Trading configurations
│   ├── data.py                   # Data pipeline settings
│   ├── services.py               # Monitoring & orchestrator
│   └── main.py                   # Main AITraderConfig
├── yaml/                         # Configuration files
│   ├── layer_definitions.yaml    # Main configuration
│   ├── app_context_config.yaml
│   ├── event_config.yaml
│   ├── universe_definitions.yaml
│   ├── environments/            # Environment overrides
│   │   ├── development.yaml
│   │   ├── paper.yaml
│   │   └── production.yaml
│   ├── services/                # Service configurations
│   │   ├── alerting.yaml
│   │   ├── backfill.yaml
│   │   ├── dashboard.yaml
│   │   └── scanners.yaml
│   └── defaults/                # Default values
│       ├── data.yaml
│       ├── risk.yaml
│       ├── storage.yaml
│       └── trading.yaml
└── docs/                        # Documentation
    ├── CONFIG_ARCHITECTURE.md   # Detailed architecture docs
    ├── CONFIG_FIXES_SUMMARY.md  # Summary of applied fixes
    └── README.md                 # This file
```

## Core Components

### ConfigManager
Thread-safe configuration manager with caching and validation.

```python
from main.config import get_config_manager, get_production_config_manager

# Standard usage
manager = get_config_manager()
config = manager.load_config("layer_definitions")

# Production mode (strict validation, fail-fast)
prod_manager = get_production_config_manager()
config = prod_manager.load_config("layer_definitions")
```

### Validation Models
Comprehensive Pydantic models for type-safe configuration:

- **Core**: API keys, environments, system settings
- **Trading**: Broker, risk management, position sizing
- **Data**: Data sources, backfill, feature configuration
- **Services**: Monitoring, orchestrator, alerting

### Field Mappings
Standardize field names across data sources:

```python
from main.config import get_field_mapping_config

mapping = get_field_mapping_config()
polygon_fields = mapping.get_mapping('polygon')
# {'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
```

### Environment Loader
Handle environment variables and .env files:

```python
from main.config import ensure_environment_loaded, is_production

ensure_environment_loaded()  # Load .env file
if is_production():
    # Production-specific logic
    pass
```

## Usage Examples

### Basic Configuration Access

```python
from main.config import get_config_manager

manager = get_config_manager()
config = manager.load_config("layer_definitions")

# Access nested values
api_key = config.api_keys.alpaca.key
max_position = config.risk.position_sizing.max_position_size
data_sources = config.data.sources
```

### Configuration with Overrides

```python
# Override values at runtime
config = manager.load_config(
    "layer_definitions",
    overrides=[
        "system.environment=live",
        "broker.paper_trading=false",
        "risk.position_sizing.max_position_size=5000"
    ]
)
```

### Environment-Specific Configuration

```python
# Load base configuration
config = manager.load_config("layer_definitions")

# Apply environment-specific overrides
env_config = manager.load_environment_config("production", config)
```

### Cache Management

```python
# Clear specific configuration cache
manager.clear_cache("layer_definitions")

# Clear all cached configurations
manager.clear_cache()

# Get cache statistics
stats = manager.get_cache_stats()
print(f"Cache entries: {stats['total_entries']}")
print(f"Cache hit rate: {stats['valid_entries']}/{stats['total_entries']}")
```

### Validation

```python
from main.config import ConfigValidator

# Strict validation (production)
validator = ConfigValidator(strict_mode=True)
try:
    config = validator.validate_file("config.yaml")
except ConfigValidationError as e:
    print(f"Validation failed: {e}")
    for error in e.errors:
        print(f"  - {error}")

# Non-strict validation (development)
validator = ConfigValidator(strict_mode=False)
config = validator.validate_file("config.yaml")  # Logs warnings, doesn't fail
```

## Configuration Files

### Main Configuration (layer_definitions.yaml)

Primary configuration file defining trading layers and system settings:

```yaml
layers:
  layer_1:
    name: "Liquid"
    description: "High liquidity symbols"
    criteria:
      min_avg_dollar_volume: 5_000_000
      min_price: 1.0
    retention:
      hot_storage_days: 90
      data_types: ["1min", "1hour", "1day"]
```

### Environment Overrides

Environment-specific configurations in `environments/`:

```yaml
# environments/production.yaml
system:
  environment: live
broker:
  paper_trading: false
risk:
  position_sizing:
    max_position_size: 2.0
  circuit_breaker:
    daily_loss_limit: 2.0
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
      intervals: [1day, 1hour, 15min]
      lookback_days: 365
```

## Environment Variables

Set up your `.env` file in the project root:

```bash
# Required API keys
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
POLYGON_API_KEY=your_polygon_key_here

# Database
DATABASE_URL=postgresql://user:pass@localhost/aitrader

# Optional
ENVIRONMENT=paper
TRADING_MODE=paper
DEBUG=true
```

Reference environment variables in YAML:

```yaml
api_keys:
  alpaca:
    key: ${ALPACA_API_KEY}
    secret: ${ALPACA_SECRET_KEY}
  polygon:
    key: ${POLYGON_API_KEY}
```

## Backward Compatibility

Legacy usage patterns are still supported:

```python
from main.config import get_config

# Old style access
config = get_config("layer_definitions")
value = config.get('api_keys.alpaca.key')

# Still works with dot notation
paper_trading = config.get('broker.paper_trading', True)
```

## Features

### Thread Safety
- All operations are thread-safe using `threading.RLock()`
- Safe for concurrent access across multiple threads
- Proper synchronization for cache operations

### Caching
- TTL-based caching (default 5 minutes)
- Automatic cache expiration and cleanup
- Cache statistics and monitoring
- Reliable cache key generation using MD5 hashing

### Validation
- Comprehensive Pydantic models for type safety
- Environment variable validation
- Configuration consistency checks
- Detailed error reporting

### Performance
- Efficient YAML parsing with OmegaConf
- Cached configurations reduce repeated parsing
- Lazy loading of validation models
- Minimal overhead for cache operations

## Best Practices

1. **Use factory functions**: Always use `get_config_manager()` instead of direct instantiation
2. **Enable validation**: Use validation in production for fail-fast behavior
3. **Environment variables**: Store secrets in environment variables, not YAML files
4. **Cache management**: Clear cache after configuration changes
5. **Error handling**: Properly handle `ConfigValidationError` in your application

## Troubleshooting

### Common Issues

**Configuration file not found**
```python
# Ensure file exists in yaml/ directory
config = manager.load_config("layer_definitions")  # Looks for yaml/layer_definitions.yaml
```

**Environment variable not set**
```bash
# Check your .env file
cat .env | grep ALPACA_API_KEY
```

**Validation errors**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Cache issues**
```python
# Clear cache and reload
manager.clear_cache()
config = manager.load_config("layer_definitions", force_reload=True)
```

## API Reference

### ConfigManager Methods

- `load_config(config_name, overrides=None, force_reload=False)` - Load configuration
- `load_simple_config(config_file)` - Load without validation
- `load_environment_config(env, base_config=None)` - Load environment overrides
- `clear_cache(config_name=None)` - Clear configuration cache
- `get_cache_stats()` - Get cache statistics

### Factory Functions

- `get_config_manager()` - Standard configuration manager
- `get_production_config_manager()` - Strict validation manager
- `get_config()` - Legacy compatibility function

### Field Mappings

- `get_field_mapping_config()` - Get field mapping configuration
- `FieldMappingConfig.get_mapping(source)` - Get mappings for data source

### Environment Utilities

- `ensure_environment_loaded()` - Load .env file
- `get_environment_info()` - Get environment information
- `is_development()` - Check if development environment
- `is_production()` - Check if production environment

For detailed architecture information, see [CONFIG_ARCHITECTURE.md](CONFIG_ARCHITECTURE.md).
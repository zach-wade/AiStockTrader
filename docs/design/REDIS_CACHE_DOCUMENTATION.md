# Redis Caching Layer Documentation

## Overview

The Redis caching layer provides a comprehensive, production-ready caching solution for the AI trading system. It's designed to handle high-frequency operations with sub-millisecond latency requirements, supporting up to 1000 orders/second performance targets.

## Key Features

### Core Capabilities

- **Async Redis Operations** - Full async/await support with connection pooling
- **Trading Object Serialization** - Automatic handling of Decimal, datetime, and domain objects
- **Connection Pooling** - Optimized Redis connections with configurable limits
- **Batch Operations** - Efficient multi-key operations for high throughput
- **TTL Management** - Flexible expiration policies for different data types
- **Namespace Support** - Logical separation of cache data by functionality

### Performance Features

- **Sub-millisecond Latency** - Optimized for high-frequency trading requirements
- **Compression** - Automatic compression for large values to reduce memory usage
- **Circuit Breaker Integration** - Resilient operation when Redis is unavailable
- **Metrics Collection** - Comprehensive performance monitoring and alerting
- **Background Health Checks** - Automatic monitoring of Redis connection health

### Trading-Specific Features

- **Market Data Caching** - Specialized operations for real-time market data
- **Portfolio Calculations** - Caching of expensive portfolio computations
- **Risk Metrics** - Efficient storage and retrieval of risk calculations
- **Session Management** - User session caching with TTL extension
- **Cache Invalidation** - Intelligent invalidation patterns for data consistency

## Architecture

### Component Structure

```
src/infrastructure/cache/
├── __init__.py              # Public API exports
├── redis_cache.py           # Low-level Redis operations
├── cache_manager.py         # High-level business logic
├── config.py               # Configuration management
├── decorators.py           # Automatic caching decorators
├── serializers.py          # Object serialization utilities
└── exceptions.py           # Cache-specific exceptions
```

### Core Components

#### RedisCache

Low-level Redis client with connection management, error handling, and performance optimization.

#### CacheManager

High-level interface providing business logic, trading-specific operations, and policy management.

#### Cache Decorators

Automatic caching decorators for transparent performance optimization.

#### Serializers

Efficient serialization for complex trading objects including Decimal numbers and datetime objects.

## Configuration

### Environment Variables

```bash
# Redis Connection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password
REDIS_SSL=false
REDIS_MAX_CONNECTIONS=20

# Cache Policies
CACHE_DEFAULT_TTL=3600
CACHE_MARKET_DATA_TTL=60
CACHE_PORTFOLIO_TTL=300
CACHE_RISK_TTL=180
CACHE_SESSION_TTL=1800

# Cache Behavior
CACHE_KEY_PREFIX=trading
CACHE_COMPRESS_THRESHOLD=1024
CACHE_MAX_BATCH_SIZE=100

# Monitoring
CACHE_ENABLE_METRICS=true
CACHE_LOG_SLOW_OPS=true
CACHE_SLOW_OP_THRESHOLD=0.1
```

### Programmatic Configuration

```python
from src.infrastructure.cache import CacheConfig, CacheEnvironment

config = CacheConfig.from_env(environment="production")
config.validate()  # Ensures configuration is valid
```

## Usage Examples

### Basic Operations

```python
from src.infrastructure.cache import CacheManager

async def basic_usage():
    cache_manager = CacheManager()

    async with cache_manager:
        # Set value with TTL
        await cache_manager.set("key", {"value": 123}, ttl=300)

        # Get value
        result = await cache_manager.get("key", default={})

        # Check existence
        exists = await cache_manager._redis_cache.exists("key")

        # Delete value
        deleted = await cache_manager.delete("key")
```

### Decorator-Based Caching

```python
from src.infrastructure.cache import cache_result, cache_market_data

@cache_market_data(ttl=60)
async def fetch_stock_price(symbol: str):
    # Expensive API call
    return await external_api.get_price(symbol)

@cache_result(ttl=300, namespace="portfolio")
async def calculate_portfolio_value(portfolio_id: str):
    # Complex calculation
    return await portfolio_service.calculate_value(portfolio_id)

# Usage - automatically cached
price = await fetch_stock_price("AAPL")
value = await calculate_portfolio_value("portfolio_123")
```

### Trading-Specific Operations

```python
async def trading_operations():
    cache_manager = CacheManager()

    async with cache_manager:
        # Cache market data
        market_data = {
            "price": Decimal("150.00"),
            "volume": 1000000,
            "timestamp": datetime.now()
        }
        await cache_manager.cache_market_data("AAPL", market_data, "quote")

        # Cache portfolio calculation
        portfolio_result = {
            "total_value": Decimal("100000.00"),
            "total_pnl": Decimal("5000.00")
        }
        await cache_manager.cache_portfolio_calculation(
            "portfolio_123", "value", portfolio_result
        )

        # Cache risk metrics
        risk_metrics = {
            "var_95": Decimal("1000.00"),
            "beta": Decimal("1.2")
        }
        await cache_manager.cache_risk_calculation(
            "portfolio_123", "var", risk_metrics
        )
```

### Batch Operations

```python
async def batch_operations():
    cache_manager = CacheManager()

    async with cache_manager:
        # Set multiple keys
        data = {
            "stock_1": {"price": Decimal("100.00")},
            "stock_2": {"price": Decimal("200.00")},
            "stock_3": {"price": Decimal("300.00")}
        }
        await cache_manager.set_many(data, ttl=300)

        # Get multiple keys
        keys = ["stock_1", "stock_2", "stock_3"]
        results = await cache_manager.get_many(keys)

        # Clear namespace
        deleted_count = await cache_manager.clear_namespace("stocks")
```

## Performance Characteristics

### Latency Targets

- **Get Operations**: < 1ms average, < 5ms 99th percentile
- **Set Operations**: < 2ms average, < 10ms 99th percentile
- **Batch Operations**: < 0.1ms per key for batches > 10 keys

### Throughput Targets

- **Individual Operations**: > 1000 ops/second per connection
- **Batch Operations**: > 10,000 keys/second
- **Concurrent Connections**: Up to 50 concurrent connections

### Memory Efficiency

- **Compression**: Automatic compression for values > 1KB
- **Serialization**: Efficient binary serialization for complex objects
- **TTL Management**: Automatic expiration to prevent memory leaks

## Monitoring and Metrics

### Built-in Metrics

- **Hit/Miss Rates**: Cache effectiveness metrics
- **Latency Statistics**: Operation timing and percentiles
- **Error Rates**: Failed operations and error types
- **Connection Health**: Redis connection status and pool utilization
- **Memory Usage**: Cache memory consumption and growth patterns

### Accessing Metrics

```python
async def monitor_cache():
    cache_manager = CacheManager()

    async with cache_manager:
        stats = await cache_manager.get_stats()

        print(f"Hit Rate: {stats['cache_manager']['hit_rate']:.2%}")
        print(f"Avg Latency: {stats['cache_manager']['avg_latency_ms']:.2f}ms")
        print(f"Error Rate: {stats['cache_manager']['error_rate']:.2%}")
```

### Integration with Monitoring Systems

The cache layer integrates with the existing metrics collection system:

```python
from src.infrastructure.monitoring.metrics import MetricsCollector

metrics_collector = MetricsCollector()
cache_manager = CacheManager(metrics_collector=metrics_collector)

# Metrics are automatically reported to the monitoring system
```

## Error Handling and Resilience

### Circuit Breaker Integration

```python
from src.infrastructure.resilience.circuit_breaker import CircuitBreaker

circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout=60.0
)

cache_manager = CacheManager(circuit_breaker=circuit_breaker)

# Cache operations are protected by circuit breaker
```

### Graceful Degradation

```python
@cache_result(ttl=300)
async def resilient_function(arg):
    # Function continues to work even if cache is unavailable
    return expensive_computation(arg)

# If Redis is down:
# 1. Function executes normally
# 2. Results are not cached
# 3. No exceptions are raised
# 4. Application continues operating
```

### Error Types and Handling

- **RedisConnectionError**: Redis server unavailable
- **CacheTimeoutError**: Operation exceeded timeout
- **SerializationError**: Object cannot be serialized
- **CacheConfigurationError**: Invalid configuration

All errors include context information for debugging and monitoring.

## Testing

### Unit Tests

- Full test coverage for all cache operations
- Mock Redis client for fast, isolated testing
- Error scenario testing and edge cases

### Integration Tests

- Real Redis integration testing
- Performance benchmarking
- Concurrent operation testing

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-cache.txt

# Run unit tests
pytest tests/unit/infrastructure/cache/

# Run integration tests (requires Redis)
pytest tests/unit/infrastructure/cache/test_integration.py

# Run with coverage
pytest --cov=src/infrastructure/cache tests/unit/infrastructure/cache/
```

## Production Deployment

### Redis Configuration

```redis
# /etc/redis/redis.conf

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence (adjust based on durability requirements)
save 900 1
save 300 10
save 60 10000

# Network
tcp-keepalive 300
timeout 300

# Security
requirepass your_strong_password

# Performance
tcp-backlog 511
databases 16
```

### Environment Setup

```bash
# Production environment variables
export CACHE_ENVIRONMENT=production
export REDIS_HOST=redis-cluster.internal
export REDIS_PASSWORD=your_secure_password
export REDIS_MAX_CONNECTIONS=50
export CACHE_ENABLE_METRICS=true
```

### Monitoring Setup

1. **Redis Monitoring**: Monitor Redis server health, memory usage, and performance
2. **Application Metrics**: Track cache hit rates, latency, and error rates
3. **Alerting**: Set up alerts for cache performance degradation
4. **Capacity Planning**: Monitor growth trends and plan scaling

### Security Considerations

- **Password Protection**: Always use strong passwords in production
- **Network Security**: Use VPC/private networks for Redis communication
- **SSL/TLS**: Enable encryption for sensitive environments
- **Access Control**: Implement proper network ACLs
- **Data Encryption**: Consider encrypting sensitive cached data

## Troubleshooting

### Common Issues

#### High Cache Miss Rate

```python
# Check if TTL is too short
config = get_cache_config()
print(f"Market data TTL: {config.policy.market_data_ttl}s")

# Monitor key expiration patterns
stats = await cache_manager.get_stats()
print(f"Hit rate: {stats['cache_manager']['hit_rate']:.2%}")
```

#### High Latency

```python
# Check Redis server health
stats = await cache_manager.get_stats()
redis_stats = stats['redis']
print(f"Connected clients: {redis_stats['connected_clients']}")
print(f"Used memory: {redis_stats['used_memory_human']}")
```

#### Connection Issues

```python
# Test Redis connectivity
cache = RedisCache()
try:
    await cache.connect()
    await cache._client.ping()
    print("Redis connection successful")
except Exception as e:
    print(f"Redis connection failed: {e}")
finally:
    await cache.disconnect()
```

### Performance Tuning

1. **Connection Pool Sizing**: Adjust `max_connections` based on concurrency needs
2. **Batch Operations**: Use batch operations for multiple keys
3. **Compression**: Tune compression thresholds for your data patterns
4. **TTL Optimization**: Set appropriate TTLs based on data freshness requirements
5. **Memory Management**: Monitor and optimize Redis memory usage

### Debugging

Enable debug logging for detailed cache operation information:

```python
import logging

logging.getLogger('src.infrastructure.cache').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

- **Redis Cluster Support**: Horizontal scaling with Redis Cluster
- **Read Replicas**: Load balancing across Redis read replicas
- **Cache Warming**: Proactive cache population strategies
- **Smart Invalidation**: Dependency-based cache invalidation
- **Compression Algorithms**: Additional compression options for specific data types

### Integration Opportunities

- **Message Queues**: Cache integration with Redis Streams
- **Event Sourcing**: Cache layer for event replay optimization
- **Machine Learning**: Cache for model predictions and feature data
- **Real-time Analytics**: Cache for streaming analytics results

## Conclusion

The Redis caching layer provides a robust, high-performance foundation for the AI trading system. With comprehensive error handling, monitoring, and trading-specific optimizations, it enables the system to achieve sub-millisecond response times while maintaining data consistency and reliability.

For additional support or questions, refer to the test files for usage examples or consult the inline documentation in the source code.

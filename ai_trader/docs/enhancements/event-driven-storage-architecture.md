# Event-Driven Storage Architecture

## Executive Summary

This document outlines the evolution path from our current hybrid "Write to Both" storage implementation to a fully event-driven architecture. The migration is designed to be incremental, allowing each phase to deliver immediate value while building towards a more scalable and maintainable system.

## Current State (Phase 1 - Implemented)

### Overview
We've implemented a hybrid approach that combines direct writes to hot storage with event publishing for eventual consistency in cold storage.

### Components
- **DualStorageWriter**: Manages writes to both storage tiers
- **DataWrittenEvent**: Event type for storage write notifications
- **ColdStorageConsumer**: Processes events to write to cold storage
- **Repository Integration**: MarketDataRepository and NewsRepository support dual writes

### Architecture
```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Repository  │────▶│ DualStorageWriter│────▶│ Hot Storage │
└─────────────┘     └──────────────────┘     │ (PostgreSQL)│
                             │                └─────────────┘
                             │
                             ▼
                    ┌──────────────┐
                    │  Event Bus   │
                    └──────────────┘
                             │
                             ▼
                    ┌──────────────────┐     ┌─────────────┐
                    │ColdStorageConsumer│────▶│Cold Storage │
                    └──────────────────┘     │(Data Lake)  │
                                             └─────────────┘
```

### Benefits
- Immediate implementation of "Write to Both" principle
- Non-blocking cold storage writes
- Event-driven extensibility
- Graceful degradation if cold storage fails

## Current Implementation Status (December 2024)

### ✅ Phase 1 Complete

We have successfully implemented Phase 1 with the following achievements:

#### All 11 Repositories Integrated
- MarketDataRepository ✅
- NewsRepository ✅
- CompanyRepository ✅
- CryptocurrencyRepository ✅
- DividendsRepository ✅
- FinancialsRepository ✅
- GuidanceRepository ✅
- RatingsRepository ✅
- SentimentRepository ✅
- SocialSentimentRepository ✅
- FeatureRepository ✅

#### Infrastructure Components
- DualStorageWriter with circuit breaker protection ✅
- Event-driven cold storage consumer ✅
- Comprehensive metrics tracking ✅
- Full test coverage with real database connections ✅

#### Documentation
- Architecture documentation: `docs/architecture/dual_storage_implementation.md`
- Integration tests: `tests/integration/test_dual_storage_complete.py`
- Database audit: `docs/reports/database_execute_audit.md`

### Identified Improvements

From our database audit, we identified 8 files with 15 execute() calls that should be refactored to use repositories:
- Catalyst event generation
- Model deployment tracking
- Monitoring reports
- Data lifecycle management

These will be addressed in future iterations to ensure all business data flows through the dual storage system.

## Phase 2: Write-Ahead Log (WAL) Pattern

### Timeline: Q2 2025

### Overview
Introduce a Write-Ahead Log to ensure durability and enable replay capabilities.

### New Components
- **WAL Service**: Durable message queue (Kafka/Redis Streams)
- **WAL Writer**: Writes all data changes to WAL first
- **WAL Processor**: Processes WAL entries to storage tiers

### Architecture
```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│ Repository  │────▶│ WAL Writer  │────▶│     WAL      │
└─────────────┘     └─────────────┘     │(Kafka/Redis) │
                                         └──────────────┘
                                                 │
                    ┌────────────────────────────┴─────────────────┐
                    ▼                                              ▼
            ┌───────────────┐                              ┌──────────────┐
            │ Hot Processor │                              │Cold Processor│
            └───────────────┘                              └──────────────┘
                    │                                              │
                    ▼                                              ▼
            ┌─────────────┐                                ┌─────────────┐
            │ Hot Storage │                                │Cold Storage │
            └─────────────┘                                └─────────────┘
```

### Implementation Steps

#### 2.1 WAL Infrastructure
```python
# wal_service.py
class WALService:
    def __init__(self, backend: str = "kafka"):
        self.backend = self._initialize_backend(backend)
    
    async def append(self, entry: WALEntry) -> str:
        """Append entry to WAL and return sequence ID"""
        pass
    
    async def read(self, from_sequence: str) -> AsyncIterator[WALEntry]:
        """Read entries from given sequence"""
        pass

# wal_entry.py
@dataclass
class WALEntry:
    sequence_id: str
    timestamp: datetime
    operation: str  # insert, update, delete
    table_name: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
```

#### 2.2 Modified DualStorageWriter
```python
class DualStorageWriter:
    def __init__(self, wal_service: WALService, ...):
        self.wal = wal_service
    
    async def write(self, table_name: str, data: Dict) -> WriteResult:
        # Write to WAL first
        wal_entry = WALEntry(
            operation="upsert",
            table_name=table_name,
            data=data
        )
        sequence_id = await self.wal.append(wal_entry)
        
        # Continue with current implementation
        # but now we can replay from WAL if needed
```

#### 2.3 WAL Processors
```python
class HotStorageProcessor:
    async def process_wal_entries(self):
        """Process WAL entries to hot storage"""
        async for entry in self.wal.read(self.last_sequence):
            await self._write_to_hot(entry)
            await self._update_checkpoint(entry.sequence_id)

class ColdStorageProcessor:
    async def process_wal_entries(self):
        """Process WAL entries to cold storage with batching"""
        batch = []
        async for entry in self.wal.read(self.last_sequence):
            batch.append(entry)
            if len(batch) >= self.batch_size:
                await self._write_batch_to_cold(batch)
                await self._update_checkpoint(batch[-1].sequence_id)
```

### Benefits
- **Durability**: No data loss even if both storage tiers fail
- **Replay Capability**: Can replay from any point in time
- **Audit Trail**: Complete history of all changes
- **Decoupling**: Storage writes completely decoupled from ingestion

## Phase 3: Change Data Capture (CDC)

### Timeline: Q4 2025

### Overview
Implement database-level CDC to automatically capture all changes without application code modifications.

### Technology Stack
- **Debezium**: For PostgreSQL CDC
- **Kafka Connect**: For data routing
- **Schema Registry**: For schema evolution

### Architecture
```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│ Application │────▶│ PostgreSQL  │────▶│  Debezium    │
└─────────────┘     └─────────────┘     │  Connector   │
                                         └──────────────┘
                                                 │
                                         ┌───────▼────────┐
                                         │     Kafka      │
                                         │    Topics      │
                                         └────────────────┘
                                                 │
                    ┌────────────────────────────┼─────────────────┐
                    ▼                            ▼                 ▼
            ┌──────────────┐           ┌──────────────┐   ┌──────────────┐
            │ Cold Storage │           │  Analytics   │   │   Real-time  │
            │  Connector   │           │   Connector  │   │   Processor  │
            └──────────────┘           └──────────────┘   └──────────────┘
```

### Implementation Steps

#### 3.1 PostgreSQL Configuration
```sql
-- Enable logical replication
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_replication_slots = 10;
ALTER SYSTEM SET max_wal_senders = 10;

-- Create replication slot
SELECT * FROM pg_create_logical_replication_slot('debezium', 'pgoutput');
```

#### 3.2 Debezium Connector Configuration
```json
{
  "name": "ai-trader-postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "localhost",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "password",
    "database.dbname": "ai_trader",
    "database.server.name": "ai_trader",
    "table.include.list": "public.market_data,public.news_data,public.financials",
    "plugin.name": "pgoutput",
    "slot.name": "debezium"
  }
}
```

#### 3.3 Cold Storage Sink Connector
```json
{
  "name": "cold-storage-sink",
  "config": {
    "connector.class": "io.confluent.connect.s3.S3SinkConnector",
    "topics": "ai_trader.public.market_data,ai_trader.public.news_data",
    "s3.region": "us-west-2",
    "s3.bucket.name": "ai-trader-cold-storage",
    "s3.part.size": "5242880",
    "flush.size": "10000",
    "storage.class": "io.confluent.connect.s3.storage.S3Storage",
    "format.class": "io.confluent.connect.s3.format.parquet.ParquetFormat",
    "partitioner.class": "io.confluent.connect.storage.partitioner.TimeBasedPartitioner",
    "path.format": "'year'=YYYY/'month'=MM/'day'=dd",
    "timestamp.extractor": "Record"
  }
}
```

### Benefits
- **Zero Application Changes**: CDC works at database level
- **Guaranteed Consistency**: Captures every change
- **Multiple Consumers**: Easy to add new data consumers
- **Schema Evolution**: Handles schema changes gracefully

## Phase 4: Stream Processing Platform

### Timeline: 2026

### Overview
Build a complete stream processing platform for real-time analytics and ML feature computation.

### Components
- **Apache Flink**: Stream processing engine
- **Feature Store**: Real-time feature serving
- **Stream Analytics**: Real-time dashboards and alerts

### Architecture
```
┌─────────────────────────────────────────────────────┐
│                   Kafka Topics                      │
└─────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Flink Job 1  │   │ Flink Job 2  │   │ Flink Job 3  │
│ (Aggregation) │   │(Enrichment)  │   │ (ML Features)│
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│Time Series DB│   │ Feature Store │   │Analytics Lake│
└──────────────┘   └──────────────┘   └──────────────┘
```

### Example Stream Processing Jobs

#### 4.1 Real-time Aggregations
```python
# Flink SQL for 5-minute OHLCV
CREATE TABLE market_data_5min AS
SELECT 
    symbol,
    TUMBLE_START(event_time, INTERVAL '5' MINUTE) as window_start,
    MIN(price) as low,
    MAX(price) as high,
    FIRST_VALUE(price) as open,
    LAST_VALUE(price) as close,
    SUM(volume) as volume,
    COUNT(*) as trades
FROM market_data_stream
GROUP BY 
    symbol,
    TUMBLE(event_time, INTERVAL '5' MINUTE);
```

#### 4.2 Complex Event Processing
```python
# Detect unusual trading patterns
@flink.udf
def detect_anomaly(symbol, price, volume, avg_volume):
    if volume > avg_volume * 3 and price_change > 0.05:
        return {
            "type": "volume_spike",
            "severity": "high",
            "details": {...}
        }
    return None

# Apply to stream
anomalies = (
    market_stream
    .key_by(lambda x: x.symbol)
    .window(TumblingEventTimeWindows.of(Time.minutes(1)))
    .process(AnomalyDetector())
)
```

## Migration Strategy

### Principles
1. **No Breaking Changes**: Each phase maintains backward compatibility
2. **Incremental Value**: Each phase delivers immediate benefits
3. **Rollback Capability**: Can revert to previous phase if needed
4. **Monitoring First**: Comprehensive monitoring before migration

### Migration Checklist

#### Before Phase 2 (WAL)
- [ ] Dual storage writer stable for 30 days
- [ ] Cold storage consumer processing < 1 minute behind
- [ ] Zero data loss confirmed
- [ ] WAL infrastructure provisioned
- [ ] Replay procedures tested

#### Before Phase 3 (CDC)
- [ ] WAL system stable for 90 days
- [ ] Kafka cluster operational
- [ ] CDC connectors tested in staging
- [ ] Schema registry configured
- [ ] Monitoring dashboards ready

#### Before Phase 4 (Streaming)
- [ ] CDC pipeline stable for 90 days
- [ ] Flink cluster provisioned
- [ ] Stream processing jobs tested
- [ ] Feature store integrated
- [ ] ML models adapted for streaming

## Monitoring and Observability

### Key Metrics

#### Phase 1 (Current)
- Hot storage write latency (p50, p95, p99)
- Cold storage lag (seconds behind hot)
- Event publishing success rate
- Circuit breaker status

#### Phase 2 (WAL)
- WAL append latency
- WAL size and growth rate
- Processor lag by storage tier
- Replay operations count

#### Phase 3 (CDC)
- CDC lag (milliseconds)
- Connector status and errors
- Topic throughput (messages/sec)
- Schema evolution events

#### Phase 4 (Streaming)
- Job processing latency
- Checkpoint duration
- State size per job
- Output throughput

### Alerting Rules

```yaml
# Example Prometheus alerts
groups:
  - name: storage_architecture
    rules:
      - alert: ColdStorageLagHigh
        expr: cold_storage_lag_seconds > 300
        for: 5m
        annotations:
          summary: "Cold storage is more than 5 minutes behind"
          
      - alert: WALGrowthHigh
        expr: rate(wal_size_bytes[5m]) > 1e9
        for: 10m
        annotations:
          summary: "WAL growing faster than 1GB/5min"
          
      - alert: CDCConnectorDown
        expr: kafka_connect_connector_status != 1
        for: 1m
        annotations:
          summary: "CDC connector is not running"
```

## Cost Analysis

### Phase 1 (Current)
- Additional event bus overhead: ~5%
- Cold storage consumer compute: 1 instance
- Estimated monthly cost: +$200

### Phase 2 (WAL)
- Kafka/Redis cluster: 3 nodes
- Additional storage for WAL: 1TB
- Estimated monthly cost: +$800

### Phase 3 (CDC)
- Kafka expansion: 5 nodes
- Debezium compute: 2 instances
- Schema registry: 2 instances
- Estimated monthly cost: +$1,500

### Phase 4 (Streaming)
- Flink cluster: 10 nodes
- Feature store: 5 nodes
- Additional monitoring: 3 nodes
- Estimated monthly cost: +$5,000

## Risks and Mitigations

### Technical Risks

1. **Data Loss During Migration**
   - Mitigation: Dual-write during transition periods
   - Validation: Reconciliation jobs to verify data integrity

2. **Performance Degradation**
   - Mitigation: Gradual rollout with feature flags
   - Monitoring: Comprehensive performance baselines

3. **Schema Evolution Complexity**
   - Mitigation: Schema registry from Phase 3
   - Testing: Automated schema compatibility tests

### Operational Risks

1. **Increased Complexity**
   - Mitigation: Extensive documentation and runbooks
   - Training: Team training before each phase

2. **Vendor Lock-in**
   - Mitigation: Use open-source components
   - Abstraction: Interface-based design

## Additional Enhancement: Monitoring and Observability

### Overview
Enhance the current metrics collection with comprehensive monitoring and observability features.

### Components

#### 1. Enhanced Metrics Collection
```python
class DualStorageMetrics:
    # Timing metrics
    hot_write_latency_ms: Histogram
    cold_write_latency_ms: Histogram
    event_publish_latency_ms: Histogram
    
    # Throughput metrics
    writes_per_second: Gauge
    bytes_per_second: Gauge
    
    # Queue depth
    event_queue_depth: Gauge
    cold_storage_backlog: Gauge
    
    # Error categorization
    error_by_type: Counter  # timeout, connection, validation, etc.
    
    # Business metrics
    records_by_repository: Counter
    records_by_data_type: Counter
```

#### 2. Prometheus Integration
- Export all metrics in Prometheus format
- Create recording rules for common queries
- Set up alerting rules

#### 3. Grafana Dashboards
- **Overview Dashboard**: System health, write rates, error rates
- **Repository Dashboard**: Per-repository metrics, top writers
- **Performance Dashboard**: Latency percentiles, throughput trends
- **Circuit Breaker Dashboard**: State transitions, failure patterns

#### 4. Distributed Tracing
- Integrate OpenTelemetry for request tracing
- Track writes across hot/cold storage
- Correlate with event processing

#### 5. Alerting Strategy
- **Critical**: Circuit breaker open, hot storage failures
- **Warning**: Cold storage lag, high error rates
- **Info**: Performance degradation, queue depth

### Implementation Priority
1. Enhanced metrics collection (Phase 1.1)
2. Prometheus integration (Phase 1.2)
3. Grafana dashboards (Phase 1.3)
4. Distributed tracing (Phase 2)
5. Advanced alerting (Phase 2)

## Conclusion

This phased approach to event-driven storage architecture provides a clear path from our current implementation to a highly scalable, real-time data platform. Each phase builds on the previous one, delivering immediate value while maintaining system stability.

The key to success is:
1. Careful monitoring at each phase
2. Gradual rollout with rollback capabilities
3. Team training and documentation
4. Clear success criteria before proceeding

By following this roadmap, we can evolve our storage architecture to meet growing demands while maintaining the reliability our trading system requires.
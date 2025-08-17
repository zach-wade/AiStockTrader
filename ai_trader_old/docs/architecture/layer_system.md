# Layer System Architecture

## Overview

The layer system is a hierarchical classification system for symbols based on their trading importance and data requirements. It replaces the previous tier system and boolean layer qualifications with a single, clean integer-based layer assignment.

## Layer Definitions

### Layer 0: BASIC

- **Symbol Count**: ~10,000 symbols
- **Data Retention**: 30 days
- **Hot Storage**: 7 days
- **Intervals**: Daily only
- **Description**: Basic tradable symbols with minimal data requirements
- **Use Case**: Universe coverage, basic screening

### Layer 1: LIQUID

- **Symbol Count**: ~2,000 symbols
- **Data Retention**: 365 days (1 year)
- **Hot Storage**: 30 days
- **Intervals**: Daily, Hourly, 5-minute
- **Description**: Liquid symbols with regular trading activity
- **Use Case**: Active screening, moderate-frequency trading

### Layer 2: CATALYST

- **Symbol Count**: ~500 symbols
- **Data Retention**: 730 days (2 years)
- **Hot Storage**: 60 days
- **Intervals**: Daily, Hourly, 5-minute, 1-minute
- **Description**: Catalyst-driven symbols with events or momentum
- **Use Case**: Event-driven trading, high-frequency monitoring

### Layer 3: ACTIVE

- **Symbol Count**: ~50 symbols
- **Data Retention**: 1825 days (5 years)
- **Hot Storage**: 90 days
- **Intervals**: All (including tick data)
- **Description**: Actively traded symbols requiring maximum data
- **Use Case**: Active positions, algorithmic trading, deep analysis

## Database Schema

### Companies Table

```sql
-- New columns (primary system)
layer INTEGER (0-3)           -- Current layer assignment
layer_updated_at TIMESTAMP    -- When layer was last changed
layer_reason TEXT             -- Reason for current layer

-- Old columns (kept for migration period)
layer1_qualified BOOLEAN      -- Legacy Layer 1 qualification
layer2_qualified BOOLEAN      -- Legacy Layer 2 qualification
layer3_qualified BOOLEAN      -- Legacy Layer 3 qualification
```

### Layer Change History Table

```sql
CREATE TABLE layer_change_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    old_layer INTEGER,
    new_layer INTEGER,
    reason TEXT,
    changed_at TIMESTAMP,
    changed_by VARCHAR(100)
);
```

## Code Mapping

### DataLayer Enum

```python
from main.data_pipeline.core.enums import DataLayer

class DataLayer(IntEnum):
    BASIC = 0     # Maps to layer=0 in database
    LIQUID = 1    # Maps to layer=1 in database
    CATALYST = 2  # Maps to layer=2 in database
    ACTIVE = 3    # Maps to layer=3 in database
```

### Repository Methods

```python
# Update a symbol's layer
await company_repo.update_layer(
    symbol="AAPL",
    layer=2,  # DataLayer.CATALYST
    reason="High volume catalyst detected"
)

# Get symbols at a specific layer
symbols = await company_repo.get_symbols_by_layer(
    layer=2,  # CATALYST
    is_active=True
)

# Get symbols at or above a layer
symbols = await company_repo.get_symbols_above_layer(
    min_layer=1,  # LIQUID and above
    is_active=True
)
```

## Migration Strategy

### Phase 1: Dual-Write (Current)

- Add new `layer` column to companies table
- Migrate existing data from layer[1-3]_qualified columns
- CompanyRepository supports both old and new columns

### Phase 2: Dual-Read

- All writes go to new `layer` column
- Reads check both new and old columns for compatibility
- Scanner updates to use new system

### Phase 3: Single System

- All components use only the `layer` column
- Old columns remain for rollback capability

### Phase 4: Cleanup

- Drop old layer[1-3]_qualified columns
- Remove dual-read logic from code

## Event Integration

### Scanner → Layer Update → Event → Backfill

1. **Scanner Detects Qualification**

   ```python
   # Scanner detects symbol should be Layer 2
   await company_repo.update_layer("AAPL", 2, "Volume spike detected")
   ```

2. **Event Published**

   ```python
   # ScannerEventPublisher publishes event
   await publisher.publish_symbol_qualified(
       symbol="AAPL",
       layer=DataLayer.CATALYST,
       reason="Volume spike",
       metrics={...}
   )
   ```

3. **Backfill Triggered**

   ```python
   # Event coordinator receives event and triggers backfill
   # based on layer requirements
   ```

## Query Examples

### Get Layer 2 Symbols

```sql
-- New system
SELECT symbol FROM companies
WHERE layer = 2 AND is_active = true;

-- During migration (dual-read)
SELECT symbol FROM companies
WHERE (layer = 2 OR (layer IS NULL AND layer2_qualified = true))
AND is_active = true;
```

### Get High-Priority Symbols (Layer 2+)

```sql
SELECT symbol FROM companies
WHERE layer >= 2 AND is_active = true
ORDER BY layer DESC, symbol;
```

### Layer Distribution

```sql
SELECT
    layer,
    COUNT(*) as symbol_count,
    COUNT(*) FILTER (WHERE is_active = true) as active_count
FROM companies
GROUP BY layer
ORDER BY layer;
```

## Benefits

1. **Simplicity**: Single integer field instead of multiple booleans
2. **Hierarchy**: Natural ordering (3 > 2 > 1 > 0)
3. **Scalability**: Easy to add more layers if needed
4. **Performance**: Single indexed column for queries
5. **Clarity**: Symbol importance is immediately obvious
6. **Auditability**: Complete history of layer changes

## Migration Commands

```bash
# Run migration
python scripts/migrations/migrate_to_layer_system.py

# Verify migration
python scripts/migrations/migrate_to_layer_system.py --verify-only

# Rollback if needed
python scripts/migrations/rollback_layer_migration.py
```

## Monitoring

The layer system provides several monitoring capabilities:

- **Layer Summary View**: `SELECT * FROM layer_summary;`
- **Change History**: `SELECT * FROM layer_change_history WHERE symbol = 'AAPL';`
- **Distribution Metrics**: Track symbol counts per layer
- **Promotion/Demotion Events**: Audit trail of all changes

## Future Enhancements

1. **Automatic Demotion**: Scheduled jobs to demote inactive symbols
2. **Layer 4+**: Additional layers for ultra-high-frequency trading
3. **Dynamic Thresholds**: ML-based layer assignment
4. **Cost Optimization**: Automatic data purging based on layer

-- Migration: Add layer column to companies table
-- Purpose: Replace layer[1-3]_qualified boolean columns with single layer integer column
-- Author: System Migration
-- Date: 2025-08-06

-- Step 1: Add new columns for layer-based system
ALTER TABLE companies
ADD COLUMN IF NOT EXISTS layer INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS layer_updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN IF NOT EXISTS layer_reason TEXT;

-- Create index on layer column for efficient queries
CREATE INDEX IF NOT EXISTS idx_companies_layer ON companies(layer) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_companies_layer_updated ON companies(layer_updated_at);

-- Step 2: Migrate existing data from boolean columns to layer column
-- Priority: layer3 > layer2 > layer1 > default to 0
-- Only update if layer column exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'companies' AND column_name = 'layer') THEN
        UPDATE companies
        SET
            layer = CASE
                WHEN layer3_qualified = true THEN 3  -- ACTIVE layer
                WHEN layer2_qualified = true THEN 2  -- CATALYST layer
                WHEN layer1_qualified = true THEN 1  -- LIQUID layer
                ELSE 0  -- BASIC layer (default)
            END,
            layer_updated_at = CURRENT_TIMESTAMP,
            layer_reason = CASE
                WHEN layer3_qualified = true THEN 'Migrated from layer3_qualified'
                WHEN layer2_qualified = true THEN 'Migrated from layer2_qualified'
                WHEN layer1_qualified = true THEN 'Migrated from layer1_qualified'
                ELSE 'Default BASIC layer assignment'
            END
        WHERE layer IS NULL OR layer = 0;
    END IF;
END $$;

-- Step 3: Add constraints to ensure data integrity
ALTER TABLE companies
ADD CONSTRAINT chk_layer_valid CHECK (layer IN (0, 1, 2, 3));

-- Step 4: Create function to track layer changes
CREATE OR REPLACE FUNCTION track_layer_change() RETURNS TRIGGER AS $$
BEGIN
    IF OLD.layer IS DISTINCT FROM NEW.layer THEN
        NEW.layer_updated_at = CURRENT_TIMESTAMP;

        -- Insert into history table (if it exists)
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'layer_change_history') THEN
            INSERT INTO layer_change_history (
                symbol,
                old_layer,
                new_layer,
                reason,
                changed_at
            ) VALUES (
                NEW.symbol,
                OLD.layer,
                NEW.layer,
                NEW.layer_reason,
                CURRENT_TIMESTAMP
            );
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Step 5: Create trigger for automatic tracking
DROP TRIGGER IF EXISTS trg_companies_layer_change ON companies;
CREATE TRIGGER trg_companies_layer_change
    BEFORE UPDATE ON companies
    FOR EACH ROW
    EXECUTE FUNCTION track_layer_change();

-- Step 6: Create layer history table for audit trail
CREATE TABLE IF NOT EXISTS layer_change_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    old_layer INTEGER,
    new_layer INTEGER NOT NULL,
    reason TEXT,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    changed_by VARCHAR(100) DEFAULT current_user
);

CREATE INDEX IF NOT EXISTS idx_layer_history_symbol ON layer_change_history(symbol);
CREATE INDEX IF NOT EXISTS idx_layer_history_changed_at ON layer_change_history(changed_at);

-- Step 7: Create helper views for easy querying
CREATE OR REPLACE VIEW layer_summary AS
SELECT
    layer,
    CASE layer
        WHEN 0 THEN 'BASIC (Layer 0)'
        WHEN 1 THEN 'LIQUID (Layer 1)'
        WHEN 2 THEN 'CATALYST (Layer 2)'
        WHEN 3 THEN 'ACTIVE (Layer 3)'
    END as layer_name,
    COUNT(*) as symbol_count,
    COUNT(*) FILTER (WHERE is_active = true) as active_count
FROM companies
GROUP BY layer
ORDER BY layer;

-- Step 8: Verification queries (commented out, run manually)
-- SELECT * FROM layer_summary;
-- SELECT symbol, layer, layer_reason, layer_updated_at FROM companies WHERE layer > 0 ORDER BY layer DESC, symbol;
-- SELECT COUNT(*) as migrated_count FROM companies WHERE layer IS NOT NULL;

-- Note: Old columns (layer1_qualified, layer2_qualified, layer3_qualified) are NOT dropped yet
-- They will be dropped in a future migration after verifying the new system works correctly

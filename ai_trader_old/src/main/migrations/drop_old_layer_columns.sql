-- Migration script to drop old layer qualification columns
-- This should only be run AFTER verifying all code uses the new 'layer' field

-- First, verify that the new layer column exists and has data
DO $$
BEGIN
    -- Check if new layer column exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'companies' AND column_name = 'layer'
    ) THEN
        RAISE EXCEPTION 'New layer column does not exist. Migration cannot proceed.';
    END IF;

    -- Check if layer_transitions table exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'layer_transitions'
    ) THEN
        RAISE NOTICE 'layer_transitions table does not exist. Please run create_layer_transitions_table.sql first.';
    END IF;
END
$$;

-- Create backup of old layer qualification data before dropping
-- This preserves the historical data in case it's needed
CREATE TABLE IF NOT EXISTS companies_layer_backup AS
SELECT
    symbol,
    layer1_qualified,
    layer1_qualified_at,
    layer2_qualified,
    layer2_qualified_at,
    layer3_qualified,
    layer3_qualified_at,
    CURRENT_TIMESTAMP as backed_up_at
FROM companies
WHERE layer1_qualified = true
   OR layer2_qualified = true
   OR layer3_qualified = true;

-- Add comment to backup table
COMMENT ON TABLE companies_layer_backup IS 'Backup of old layer qualification columns before migration to single layer field';

-- Now drop the old columns
ALTER TABLE companies DROP COLUMN IF EXISTS layer1_qualified;
ALTER TABLE companies DROP COLUMN IF EXISTS layer1_qualified_at;
ALTER TABLE companies DROP COLUMN IF EXISTS layer2_qualified;
ALTER TABLE companies DROP COLUMN IF EXISTS layer2_qualified_at;
ALTER TABLE companies DROP COLUMN IF EXISTS layer3_qualified;
ALTER TABLE companies DROP COLUMN IF EXISTS layer3_qualified_at;

-- Verify the migration
DO $$
DECLARE
    layer_count INTEGER;
    transition_count INTEGER;
BEGIN
    -- Count companies with layer assignments
    SELECT COUNT(*) INTO layer_count
    FROM companies
    WHERE layer IS NOT NULL;

    -- Count recorded transitions
    SELECT COUNT(*) INTO transition_count
    FROM layer_transitions;

    RAISE NOTICE 'Migration complete:';
    RAISE NOTICE '  - Companies with layer assignments: %', layer_count;
    RAISE NOTICE '  - Layer transitions recorded: %', transition_count;
    RAISE NOTICE '  - Old columns dropped successfully';
    RAISE NOTICE '  - Backup data saved to companies_layer_backup table';
END
$$;

-- Create index on layer column if it doesn't exist
CREATE INDEX IF NOT EXISTS idx_companies_layer ON companies(layer);
CREATE INDEX IF NOT EXISTS idx_companies_layer_active ON companies(layer, is_active);

-- Grant appropriate permissions (adjust as needed for your setup)
-- GRANT SELECT ON companies_layer_backup TO readonly_user;

COMMENT ON COLUMN companies.layer IS 'Symbol layer (0=BASIC, 1=LIQUID, 2=CATALYST, 3=ACTIVE)';
COMMENT ON COLUMN companies.layer_updated_at IS 'Timestamp of last layer change';
COMMENT ON COLUMN companies.layer_reason IS 'Reason for current layer assignment';

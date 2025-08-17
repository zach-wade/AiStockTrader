-- Rollback: Remove layer column from companies table
-- Purpose: Rollback the layer-based system if issues arise
-- Author: System Migration
-- Date: 2025-08-06

-- WARNING: This will remove all layer column data!
-- Make sure to backup the data first if needed

-- Step 1: Drop the trigger
DROP TRIGGER IF EXISTS trg_companies_layer_change ON companies;

-- Step 2: Drop the function
DROP FUNCTION IF EXISTS track_layer_change();

-- Step 3: Drop the history table
DROP TABLE IF EXISTS layer_change_history;

-- Step 4: Drop the view
DROP VIEW IF EXISTS layer_summary;

-- Step 5: Drop indexes
DROP INDEX IF EXISTS idx_companies_layer;
DROP INDEX IF EXISTS idx_companies_layer_updated;

-- Step 6: Remove the constraint
ALTER TABLE companies DROP CONSTRAINT IF EXISTS chk_layer_valid;

-- Step 7: Drop the columns
ALTER TABLE companies
DROP COLUMN IF EXISTS layer,
DROP COLUMN IF EXISTS layer_updated_at,
DROP COLUMN IF EXISTS layer_reason;

-- Verification
-- SELECT column_name FROM information_schema.columns
-- WHERE table_name = 'companies' AND column_name LIKE 'layer%';

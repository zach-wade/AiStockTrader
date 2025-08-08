-- Check table structure
\d corporate_actions

-- Count records
SELECT COUNT(*) as total_records FROM corporate_actions;

-- Count by ticker and type
SELECT ticker, action_type, COUNT(*) as count
FROM corporate_actions
GROUP BY ticker, action_type
ORDER BY ticker, action_type;

-- Show AAPL records
SELECT ticker, action_type, ex_date, cash_amount, split_from, split_to, dividend_type
FROM corporate_actions
WHERE ticker = 'AAPL'
ORDER BY ex_date DESC
LIMIT 10;
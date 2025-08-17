-- Test queries to verify corporate actions data

-- 1. Count all corporate actions
SELECT COUNT(*) as total_actions FROM corporate_actions;

-- 2. Summary by ticker
SELECT
    ticker,
    COUNT(*) as total_actions,
    SUM(CASE WHEN action_type = 'dividend' THEN 1 ELSE 0 END) as dividends,
    SUM(CASE WHEN action_type = 'split' THEN 1 ELSE 0 END) as splits,
    MIN(ex_date) as earliest_date,
    MAX(ex_date) as latest_date
FROM corporate_actions
GROUP BY ticker
ORDER BY ticker;

-- 3. AAPL corporate actions in detail
SELECT
    action_type,
    ex_date,
    cash_amount,
    split_from || ':' || split_to as split_ratio,
    dividend_type,
    frequency
FROM corporate_actions
WHERE ticker = 'AAPL'
ORDER BY ex_date DESC;

-- 4. Verify the critical AAPL split
SELECT * FROM corporate_actions
WHERE ticker = 'AAPL'
  AND action_type = 'split'
  AND ex_date = '2020-08-31';

-- 5. Recent AAPL dividends
SELECT
    ex_date,
    cash_amount,
    dividend_type,
    pay_date
FROM corporate_actions
WHERE ticker = 'AAPL'
  AND action_type = 'dividend'
  AND ex_date >= '2024-01-01'
ORDER BY ex_date DESC;

-- 6. All splits in the database
SELECT
    ticker,
    ex_date,
    split_to || ':' || split_from as split_ratio
FROM corporate_actions
WHERE action_type = 'split'
ORDER BY ex_date DESC;

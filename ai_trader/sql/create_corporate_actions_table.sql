-- Drop existing table and recreate with proper schema based on Polygon API data
DROP TABLE IF EXISTS corporate_actions CASCADE;

-- Create table with optimal schema for Polygon data
CREATE TABLE corporate_actions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    action_type VARCHAR(20) NOT NULL CHECK (action_type IN ('dividend', 'split')),
    
    -- Common fields
    ex_date DATE NOT NULL,  -- ex_dividend_date for dividends, execution_date for splits
    
    -- Dividend-specific fields
    cash_amount NUMERIC(10, 4),
    currency VARCHAR(3) DEFAULT 'USD',
    dividend_type VARCHAR(10),  -- CD = Cash Dividend, SD = Special Dividend
    frequency INTEGER,  -- 1=annual, 4=quarterly, 12=monthly
    pay_date DATE,
    record_date DATE,
    declaration_date DATE,
    
    -- Split-specific fields
    split_from INTEGER,
    split_to INTEGER,
    
    -- Metadata
    polygon_id VARCHAR(100) UNIQUE,  -- Store the API's unique ID
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(ticker, action_type, ex_date)
);

-- Create indexes for performance
CREATE INDEX idx_corporate_actions_ticker_ex_date ON corporate_actions(ticker, ex_date);
CREATE INDEX idx_corporate_actions_ex_date ON corporate_actions(ex_date);
CREATE INDEX idx_corporate_actions_ticker ON corporate_actions(ticker);
CREATE INDEX idx_corporate_actions_action_type ON corporate_actions(action_type);

-- Insert all the corporate actions from the downloaded data
-- AAPL Dividends
INSERT INTO corporate_actions (ticker, action_type, ex_date, cash_amount, currency, dividend_type, frequency, pay_date, record_date, declaration_date, polygon_id) VALUES
('AAPL', 'dividend', '2020-08-07', 0.82, 'USD', 'CD', 4, '2020-08-13', '2020-08-10', '2020-07-30', 'Ef6ae7637aca0b62b5a1353bfe22c6853d4c6171a00aba9f06b6cec69f420952a'),
('AAPL', 'dividend', '2020-11-06', 0.205, 'USD', 'CD', 4, '2020-11-12', '2020-11-09', '2020-10-29', 'E8631adda3d94779a90d14f53d77efe3eec85d0fd3e6f9a0698407b787fe761fb'),
('AAPL', 'dividend', '2021-02-05', 0.205, 'USD', 'CD', 4, '2021-02-11', '2021-02-08', '2021-01-27', 'Eba2c89f79ce937dfe3d1b9b5e15ce4a21cd20d98c420a19d70c912354e039235'),
('AAPL', 'dividend', '2021-05-07', 0.22, 'USD', 'CD', 4, '2021-05-13', '2021-05-10', '2021-04-28', 'Edbd6674f2bacf55025a504fcabf9bc5cc25c174b9d69c1a6f1e07ee3ca37e843'),
('AAPL', 'dividend', '2021-08-06', 0.22, 'USD', 'CD', 4, '2021-08-12', '2021-08-09', NULL, 'E6436c5475706773f03490acf0b63fdb90b2c72bfeed329a6eb4afc080acd80ae'),
('AAPL', 'dividend', '2021-11-05', 0.22, 'USD', 'CD', 4, '2021-11-11', '2021-11-08', '2021-10-28', 'E8e3c4f794613e9205e2f178a36c53fcc57cdabb55e1988c87b33f9e52e221444'),
('AAPL', 'dividend', '2022-02-04', 0.22, 'USD', 'CD', 4, '2022-02-10', '2022-02-07', '2022-01-27', 'Eb672a10293aa431ae8d9d580589902263f16cdfdeaedf2a6dd36e9058e64e179'),
('AAPL', 'dividend', '2022-05-06', 0.23, 'USD', 'CD', 4, '2022-05-12', '2022-05-09', '2022-04-28', 'E6a8bb996a7b4db68fb8b9cac903ea5d69e6d5fcf4a8a3ff45e94c327b268d770'),
('AAPL', 'dividend', '2022-08-05', 0.23, 'USD', 'CD', 4, '2022-08-11', '2022-08-08', '2022-07-28', 'E68b4284b4e4874b1777b1434c602614d26e73394a5292aa4ca5596e3dd17c291'),
('AAPL', 'dividend', '2022-11-04', 0.23, 'USD', 'CD', 4, '2022-11-10', '2022-11-07', '2022-10-27', 'E010840de0c264bcfa41450d0370c1556954564f3af2238c2396fd9770160c645'),
('AAPL', 'dividend', '2023-02-10', 0.23, 'USD', 'CD', 4, '2023-02-16', '2023-02-13', '2023-02-02', 'E9d8d2564c86b61f9713e2fd11fd4f429f8aebad02002d7dd405deddce672d645'),
('AAPL', 'dividend', '2023-05-12', 0.24, 'USD', 'CD', 4, '2023-05-18', '2023-05-15', '2023-05-04', 'E40523b0f8cdfef5de834c3d3e27d3c40267dacba8d726d6938a0a322fdcfe880'),
('AAPL', 'dividend', '2023-08-11', 0.24, 'USD', 'CD', 4, '2023-08-17', '2023-08-14', '2023-08-03', 'Ed2b8a80dc190d1c53e410f566b3e72b06f3c2b640730b476bebe4fcb29ccaa6c'),
('AAPL', 'dividend', '2023-11-10', 0.24, 'USD', 'CD', 4, '2023-11-16', '2023-11-13', '2023-11-02', 'E4851913b4ca383b7bf0ed6a597e4fc12d8b88c92a152b9c918f4354c8572cd8d'),
('AAPL', 'dividend', '2024-02-09', 0.24, 'USD', 'CD', 4, '2024-02-15', '2024-02-12', '2024-02-01', 'E8e4edd3b27095b7e918c127648feb5c4c256b8650b7f5c36d8ac260d8ec08086'),
('AAPL', 'dividend', '2024-05-10', 0.25, 'USD', 'CD', 4, '2024-05-16', '2024-05-13', '2024-05-02', 'E495b1d61b65ddc1eb3abd60eaacf33c1313433df83f5fd63f1971612ff684f79'),
('AAPL', 'dividend', '2024-08-12', 0.25, 'USD', 'CD', 4, '2024-08-15', '2024-08-12', '2024-08-01', 'E47fd49ef418c51f7d2415b66030f735adaccf72d85f6b86eea12f96aa1d1c535'),
('AAPL', 'dividend', '2024-11-08', 0.25, 'USD', 'CD', 4, '2024-11-14', '2024-11-11', '2024-10-31', 'E416a068758f85277196150c3eb73a3331d04698856c141e883ad95710dd0b189'),
('AAPL', 'dividend', '2025-02-10', 0.25, 'USD', 'CD', 4, '2025-02-13', '2025-02-10', '2025-01-30', 'Ecc0fc5473a63ec503030d7aff356e1c4f467301d08791ac000e10ef831b58a71'),
('AAPL', 'dividend', '2025-05-12', 0.26, 'USD', 'CD', 4, '2025-05-15', '2025-05-12', '2025-05-01', 'E508c0a63699951e57b51c4cf746a7f3b4101c39232372228d68585046c0b4a67');

-- AAPL Split
INSERT INTO corporate_actions (ticker, action_type, ex_date, split_from, split_to, polygon_id) VALUES
('AAPL', 'split', '2020-08-31', 1, 4, 'E36416cce743c3964c5da63e1ef1626c0aece30fb47302eea5a49c0055c04e8d0');

-- MSFT Dividends
INSERT INTO corporate_actions (ticker, action_type, ex_date, cash_amount, currency, dividend_type, frequency, pay_date, record_date, declaration_date, polygon_id) VALUES
('MSFT', 'dividend', '2020-08-19', 0.51, 'USD', 'CD', 4, '2020-09-10', '2020-08-20', '2020-06-17', 'E9f61ba5f4d836ecbdda15c880f3320a57ea17e6a1a4281e6a145a74f01ca1b01'),
('MSFT', 'dividend', '2020-11-18', 0.56, 'USD', 'CD', 4, '2020-12-10', '2020-11-19', '2020-09-15', 'Ebfb3c62abe333c31e8e20f68a5897c76b754743fb7050a2df0216ab95be9af9f'),
('MSFT', 'dividend', '2021-02-17', 0.56, 'USD', 'CD', 4, '2021-03-11', '2021-02-18', '2020-12-02', 'E6293edf01cf837e057692aa0b430746612a57e13f6472572c7ee1593f4cd4897'),
('MSFT', 'dividend', '2021-05-19', 0.56, 'USD', 'CD', 4, '2021-06-10', '2021-05-20', '2021-03-16', 'E60403372829101540224a5bfb68cf07b341fd0db6797a015f44dad75c09660fb'),
('MSFT', 'dividend', '2021-08-18', 0.56, 'USD', 'CD', 4, '2021-09-09', '2021-08-19', '2021-06-16', 'E094e38e57fdf26bc5a12edb3f1bcefea1ddff7edda3df45d33f078c69583e79d'),
('MSFT', 'dividend', '2021-11-17', 0.62, 'USD', 'CD', 4, '2021-12-09', '2021-11-18', '2021-09-14', 'E7068b9714a4aa852526141ebafe615f2f39b3f040b8e87ae7ee5cdc781e8257b'),
('MSFT', 'dividend', '2022-02-16', 0.62, 'USD', 'CD', 4, '2022-03-10', '2022-02-17', '2021-12-07', 'E9d82c40df315e71235f45b50a870347e05104d382899961bf532606bf2f87f59'),
('MSFT', 'dividend', '2022-05-18', 0.62, 'USD', 'CD', 4, '2022-06-09', '2022-05-19', '2022-03-14', 'Efbdc6ed0934de4656330289f9f9e44dec391bcb8cf8b71c596539456b7b8ac2d'),
('MSFT', 'dividend', '2022-08-17', 0.62, 'USD', 'CD', 4, '2022-09-08', '2022-08-18', '2022-06-14', 'Ec61c7a412fdffd60f646202361b869ea302be23957a615b790316895416fc602'),
('MSFT', 'dividend', '2022-11-16', 0.68, 'USD', 'CD', 4, '2022-12-08', '2022-11-17', '2022-09-20', 'E1e9f39365baa699a74a56fd1084be8714082e90d5fe4891dd5fe28fc965c038c'),
('MSFT', 'dividend', '2023-02-15', 0.68, 'USD', 'CD', 4, '2023-03-09', '2023-02-16', '2022-11-29', 'E7e3329994f4ae9ba9c1062b3924ad38ef3390559c1bb2871268f0531397eb180'),
('MSFT', 'dividend', '2023-05-17', 0.68, 'USD', 'CD', 4, '2023-06-08', '2023-05-18', '2023-03-14', 'E5ab30221f595c9fdb5aa5ccbdc6d474738404955a86ed4f0960a04feb5da9222'),
('MSFT', 'dividend', '2023-08-16', 0.68, 'USD', 'CD', 4, '2023-09-14', '2023-08-17', '2023-06-13', 'E0d18b51ab7eaa7399bf199f9e9288f0d5bdad38aaf10a146f378682a993c7cd1'),
('MSFT', 'dividend', '2023-11-15', 0.75, 'USD', 'CD', 4, '2023-12-14', '2023-11-16', '2023-09-19', 'E5f2084116a8656dbf17afede4f20f089c59a72394ea0aaebabf5852bc127cef5'),
('MSFT', 'dividend', '2024-02-14', 0.75, 'USD', 'CD', 4, '2024-03-14', '2024-02-15', '2023-11-28', 'Efc39ba5de8c72013c7715c6c6f1b9b96ee04b5167397d227b9d7b25cef52f3c3'),
('MSFT', 'dividend', '2024-05-15', 0.75, 'USD', 'CD', 4, '2024-06-13', '2024-05-16', '2024-03-12', 'E4d5ec44e3af6d9c003c795a6a865945539467c8337621b958aabee1727b94d22'),
('MSFT', 'dividend', '2024-08-15', 0.75, 'USD', 'CD', 4, '2024-09-12', '2024-08-15', '2024-06-12', 'E86c03763df7f6fd3ff98aaa5f589bec9656ec368f02c66f9d1fe23b7f852b495'),
('MSFT', 'dividend', '2024-11-21', 0.83, 'USD', 'CD', 4, '2024-12-12', '2024-11-21', '2024-09-16', 'Ef6b22a952d02416181303e714b00cb2a525188b4c145e99c594ac18fd66c651c'),
('MSFT', 'dividend', '2025-02-20', 0.83, 'USD', 'CD', 4, '2025-03-13', '2025-02-20', '2024-12-03', 'Ecef5e6c1102feab369ba58fd3711d50321893d7fd7aabbc10112c621327476f9'),
('MSFT', 'dividend', '2025-05-15', 0.83, 'USD', 'CD', 4, '2025-06-12', '2025-05-15', '2025-03-11', 'Ec79a97079e6a62876fdb04c4c79b9bb34931fe1b368849751b456de1dae967cb');

-- GOOGL Dividends
INSERT INTO corporate_actions (ticker, action_type, ex_date, cash_amount, currency, dividend_type, frequency, pay_date, record_date, declaration_date, polygon_id) VALUES
('GOOGL', 'dividend', '2024-06-10', 0.2, 'USD', 'CD', 4, '2024-06-17', '2024-06-10', '2024-04-25', 'E22b716d8cc8fbd28e9172343f99ae9619d0ded112cd2d8c39c990de4df8ccf7b'),
('GOOGL', 'dividend', '2024-09-09', 0.2, 'USD', 'CD', 4, '2024-09-16', '2024-09-09', '2024-07-23', 'E875229bc5dc90c8b1df144a7ff7d0f726e1f3b0996ab365d8300969139e8f44d'),
('GOOGL', 'dividend', '2024-12-09', 0.2, 'USD', 'CD', 4, '2024-12-16', '2024-12-09', '2024-10-28', 'Ead65a9fcc755e36a2781ae7f72ddd735db8f828bb054ce31251ac4498f86c547'),
('GOOGL', 'dividend', '2025-03-10', 0.2, 'USD', 'CD', 4, '2025-03-17', '2025-03-10', '2025-02-04', 'E1cdc29fc580666cc27dbd3bfa5a79b65dca59bfcba64b62af8fa45833a79a652'),
('GOOGL', 'dividend', '2025-06-09', 0.21, 'USD', 'CD', 4, '2025-06-16', '2025-06-09', '2025-04-23', 'Eb0768616defecdc7ed9f208ce457ee0ab571656a30a1e197abdd334dfd3a5f42');

-- GOOGL Split
INSERT INTO corporate_actions (ticker, action_type, ex_date, split_from, split_to, polygon_id) VALUES
('GOOGL', 'split', '2022-07-18', 1, 20, 'E75f0e6c8145d353f2226725ab925f972bcd3f5d050116c4318aba162466c269c');

-- META Dividends
INSERT INTO corporate_actions (ticker, action_type, ex_date, cash_amount, currency, dividend_type, frequency, pay_date, record_date, declaration_date, polygon_id) VALUES
('META', 'dividend', '2024-02-21', 0.5, 'USD', 'CD', 4, '2024-03-26', '2024-02-22', '2024-02-01', 'Ef8795a4515fee5e57f05030294218f0aac8514e8f060886876b30f6d92337cd1'),
('META', 'dividend', '2024-06-14', 0.5, 'USD', 'CD', 4, '2024-06-26', '2024-06-14', '2024-05-30', 'Edbd30f584bd978a9c21fa5a5f829a94f2c66b401057211d3bbf2f8b2963750b8'),
('META', 'dividend', '2024-09-16', 0.5, 'USD', 'CD', 4, '2024-09-26', '2024-09-16', '2024-09-05', 'Ebb8825e317cd430d0ff8210f0b8cd537870a77a983de9d499bf86ed2a7b14c4e'),
('META', 'dividend', '2024-12-16', 0.5, 'USD', 'CD', 4, '2024-12-27', '2024-12-16', '2024-12-05', 'Ecec59fc797c01fe342c9693ce95f194450e2ca76f632da986c5e24e4836c64cf'),
('META', 'dividend', '2025-03-14', 0.525, 'USD', 'CD', 4, '2025-03-26', '2025-03-14', '2025-02-13', 'E27c60f82e95faa5cf7f8c34f621350af97b3081412a630dfba702f9679f7ec36'),
('META', 'dividend', '2025-06-16', 0.525, 'USD', 'CD', 4, '2025-06-26', '2025-06-16', '2025-05-29', 'Ea926aaf34c234b1c844cac965e0f8a3991e2bf42f88b3f0148576cc215a31a14');

-- AMZN Split
INSERT INTO corporate_actions (ticker, action_type, ex_date, split_from, split_to, polygon_id) VALUES
('AMZN', 'split', '2022-06-06', 1, 20, 'Ef72af690fef1f9db3fd5382ca3c92c8885ea75761e9cdf54825fc7139bd88c6b');

-- Verify the data
SELECT ticker, action_type, COUNT(*) as count
FROM corporate_actions
GROUP BY ticker, action_type
ORDER BY ticker, action_type;

-- Show AAPL details
SELECT action_type, ex_date, cash_amount, split_from, split_to, dividend_type, frequency
FROM corporate_actions
WHERE ticker = 'AAPL'
ORDER BY ex_date DESC;
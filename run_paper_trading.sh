#!/bin/bash
# Load environment variables and run paper trading with real Alpaca data

# Load .env file if it exists
if [ -f .env ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs)
fi

# Run the paper trading script
echo "🚀 Starting paper trading with real Alpaca market data..."
python paper_trading_alpaca.py

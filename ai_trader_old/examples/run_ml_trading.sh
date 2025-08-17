#!/bin/bash
# Example: Run ML Trading with AI Trader

echo "=== AI Trader ML Trading Example ==="
echo ""
echo "This script demonstrates how to run ML-powered trading through ai_trader.py"
echo ""

# Change to project directory
cd "$(dirname "$0")/.."

# First, ensure AAPL model is deployed
echo "1. Checking if AAPL model is deployed..."
python scripts/deploy_ml_model.py --list | grep -q "aapl_xgboost"
if [ $? -ne 0 ]; then
    echo "   Deploying AAPL model..."
    python scripts/deploy_ml_model.py --deploy aapl_xgboost
else
    echo "   ✅ AAPL model already deployed"
fi

echo ""
echo "2. Starting ML trading in paper mode..."
echo "   Command: python ai_trader.py trade --mode paper --symbols AAPL --enable-ml"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run ML trading
python ai_trader.py trade --mode paper --symbols AAPL --enable-ml

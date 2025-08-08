#!/bin/bash
# AI Trader - Paper Trading Startup Script
# This script starts the AI Trader system in paper trading mode with monitoring dashboards

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}AI Trader - Paper Trading Mode${NC}"
echo -e "${GREEN}======================================${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo -e "${YELLOW}Please create a .env file with your API keys and database credentials${NC}"
    echo -e "${YELLOW}Example:${NC}"
    echo "  ALPACA_API_KEY=your_key_here"
    echo "  ALPACA_SECRET_KEY=your_secret_here"
    echo "  POLYGON_API_KEY=your_key_here"
    echo "  DB_HOST=localhost"
    echo "  DB_PORT=5432"
    echo "  DB_NAME=ai_trader"
    echo "  DB_USER=your_user"
    echo "  DB_PASSWORD=your_password"
    exit 1
fi

# Default symbols if not provided
DEFAULT_SYMBOLS="AAPL,MSFT,GOOGL,AMZN,TSLA"

# Parse command line arguments
SYMBOLS=${1:-$DEFAULT_SYMBOLS}
ENABLE_ML=${2:-""}

echo -e "${YELLOW}Configuration:${NC}"
echo "  Mode: Paper Trading"
echo "  Symbols: $SYMBOLS"
echo "  ML Trading: ${ENABLE_ML:-Disabled}"
echo "  Dashboard: http://localhost:8080"
echo "  WebSocket: ws://localhost:8081"
echo ""

# Build the command
CMD="python3 ai_trader.py trade --mode paper --symbols $SYMBOLS"

# Add ML flag if requested
if [ "$ENABLE_ML" = "--enable-ml" ]; then
    CMD="$CMD --enable-ml"
fi

# Add monitoring and streaming (enabled by default, no flags needed)
# CMD="$CMD"  # Monitoring and streaming are enabled by default

echo -e "${YELLOW}Starting AI Trader...${NC}"
echo "Command: $CMD"
echo ""

# Check if database is accessible
echo -e "${YELLOW}Checking database connection...${NC}"
python3 -c "
import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()

try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'ai_trader'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    conn.close()
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Database connection failed. Please check your database settings in .env${NC}"
    exit 1
fi

# Check if required Python packages are installed
echo -e "${YELLOW}Checking required packages...${NC}"
python3 -c "
required_packages = [
    'click',
    'asyncio',
    'pandas',
    'numpy',
    'sqlalchemy',
    'alpaca',
    'polygon',
    'yfinance',
    'omegaconf',
    'asyncpg'
]

missing = []
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        missing.append(package)

if missing:
    print(f'❌ Missing packages: {missing}')
    print('Please install with: pip install -r requirements.txt')
    exit(1)
else:
    print('✅ All required packages installed')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Missing required packages. Please install dependencies.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Starting AI Trader Paper Trading${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Press Ctrl+C to stop the system"
echo ""

# Run the trading system
exec $CMD
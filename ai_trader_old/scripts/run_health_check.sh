#!/bin/bash
# AI Trading System - Health Check with .env Loading
# This script automatically loads your .env file before running the health check

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a  # Automatically export all variables
    source .env
    set +a  # Stop auto-exporting
    echo "✅ Environment variables loaded successfully"
else
    echo "⚠️  Warning: .env file not found in project root"
fi

# Run the health check
echo ""
echo "🔍 Running AI Trading System Health Check..."
echo ""

./deployment/scripts/health_check.sh

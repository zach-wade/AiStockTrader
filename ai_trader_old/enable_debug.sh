#!/bin/bash
# Enable debug logging for backfill

export BACKFILL_DEBUG=true
export LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1

echo "🔧 Debug logging enabled!"
echo "Run backfill with: python ai_trader.py backfill --symbols TSLA --config debug_logging_config.yaml"

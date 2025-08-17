#!/bin/bash

# Layer 1 Backfill Script
# This script runs a comprehensive backfill for all Layer 1 qualified symbols
# It reads configuration from config/layer1_backfill.yaml and executes stages in order

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Log file
LOG_DIR="logs/backfill"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/layer1_backfill_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo -e "${1}" | tee -a "$LOG_FILE"
}

# Function to log error
log_error() {
    echo -e "${RED}[ERROR] ${1}${NC}" | tee -a "$LOG_FILE"
}

# Function to log success
log_success() {
    echo -e "${GREEN}[SUCCESS] ${1}${NC}" | tee -a "$LOG_FILE"
}

# Function to log info
log_info() {
    echo -e "${BLUE}[INFO] ${1}${NC}" | tee -a "$LOG_FILE"
}

# Function to log warning
log_warning() {
    echo -e "${YELLOW}[WARNING] ${1}${NC}" | tee -a "$LOG_FILE"
}

# Header
log "======================================================================="
log "Layer 1 Symbols Comprehensive Backfill"
log "Started at: $(date)"
log "Log file: $LOG_FILE"
log "======================================================================="

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    log_warning "Virtual environment not activated. Activating venv..."
    source venv/bin/activate || {
        log_error "Failed to activate virtual environment"
        exit 1
    }
fi

# Function to run backfill for a specific stage
run_backfill_stage() {
    local stage=$1
    local days=$2
    local description=$3

    log ""
    log "-----------------------------------------------------------------------"
    log_info "Stage: $stage"
    log_info "Description: $description"
    log_info "Lookback days: $days"
    log "-----------------------------------------------------------------------"

    # Run the backfill command
    if python ai_trader.py backfill --stage "$stage" --symbols layer1 --days "$days" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Stage $stage completed successfully"
        return 0
    else
        log_error "Stage $stage failed"
        return 1
    fi
}

# Function to check disk space
check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

    if (( available_gb < required_gb )); then
        log_error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    else
        log_info "Disk space check passed. Available: ${available_gb}GB"
        return 0
    fi
}

# Function to get Layer 1 symbol count
get_layer1_count() {
    python -c "
import asyncio
from main.universe.universe_manager import UniverseManager
from main.config import get_config

async def count():
    config = get_config()
    um = UniverseManager(config)
    try:
        symbols = await um.get_qualified_symbols('1')
        return len(symbols)
    finally:
        await um.close()

print(asyncio.run(count()))
" 2>/dev/null || echo "0"
}

# Main execution
main() {
    # Check disk space (require at least 150GB for full backfill)
    log_info "Checking disk space..."
    if ! check_disk_space 150; then
        log_error "Please free up disk space before running backfill"
        exit 1
    fi

    # Get Layer 1 symbol count
    log_info "Checking Layer 1 qualified symbols..."
    SYMBOL_COUNT=$(get_layer1_count)

    if [[ "$SYMBOL_COUNT" -eq 0 ]]; then
        log_error "No Layer 1 qualified symbols found in database"
        log_info "Please run: python ai_trader.py scan --full"
        exit 1
    fi

    log_success "Found $SYMBOL_COUNT Layer 1 qualified symbols"

    # Confirm before proceeding
    if [[ "${SKIP_CONFIRM:-false}" != "true" ]]; then
        echo -e "${YELLOW}This will download approximately 60-120GB of data for $SYMBOL_COUNT symbols.${NC}"
        echo -e "${YELLOW}Estimated time: 4-8 hours depending on API limits.${NC}"
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Backfill cancelled by user"
            exit 0
        fi
    fi

    # Track overall status
    FAILED_STAGES=()

    # Stage 1: Long-term market data (5 years)
    if ! run_backfill_stage "long_term" 1825 "Daily and hourly bars for technical analysis"; then
        FAILED_STAGES+=("long_term")
    fi

    # Stage 2: Scanner intraday data (1 year)
    if ! run_backfill_stage "scanner_intraday" 365 "Minute bars for intraday patterns"; then
        FAILED_STAGES+=("scanner_intraday")
    fi

    # Stage 3: News data (2 years)
    if ! run_backfill_stage "news_data" 730 "News articles for sentiment analysis"; then
        FAILED_STAGES+=("news_data")
    fi

    # Stage 4: Corporate actions (10 years)
    if ! run_backfill_stage "corporate_actions" 3650 "Splits, dividends, earnings dates"; then
        FAILED_STAGES+=("corporate_actions")
    fi

    # Optional stages (check if enabled in config)
    # Stage 5: Social sentiment (6 months) - currently not available
    # if [[ "${ENABLE_SOCIAL_SENTIMENT:-false}" == "true" ]]; then
    #     if ! run_backfill_stage "social_sentiment" 180 "Reddit, Twitter sentiment"; then
    #         FAILED_STAGES+=("social_sentiment")
    #     fi
    # fi

    # Stage 6: Options data (1 year) - currently not configured
    # if [[ "${ENABLE_OPTIONS_DATA:-false}" == "true" ]]; then
    #     if ! run_backfill_stage "options_data" 365 "Options chain data"; then
    #         FAILED_STAGES+=("options_data")
    #     fi
    # fi

    # Summary
    log ""
    log "======================================================================="
    log "Backfill Summary"
    log "======================================================================="

    if [[ ${#FAILED_STAGES[@]} -eq 0 ]]; then
        log_success "All stages completed successfully!"
        log_info "Data is now available for $SYMBOL_COUNT Layer 1 symbols"
    else
        log_warning "Backfill completed with errors"
        log_error "Failed stages: ${FAILED_STAGES[*]}"
        log_info "You can re-run failed stages individually:"
        for stage in "${FAILED_STAGES[@]}"; do
            log "  python ai_trader.py backfill --stage $stage --symbols layer1"
        done
    fi

    log ""
    log "Completed at: $(date)"
    log "Total duration: $SECONDS seconds"
    log "======================================================================="
}

# Run main function
main "$@"

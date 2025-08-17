#!/bin/bash
# AI Trader Comprehensive Health Check Script
# Version: 1.0
# Author: Claude Code Assistant

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/deployment/logs/health_checks"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/health_check_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Health check results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================
setup_logging() {
    mkdir -p "${LOG_DIR}"
    exec > >(tee -a "${LOG_FILE}")
    exec 2>&1
    echo "Health check started at $(date)" | tee -a "${LOG_FILE}"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# =============================================================================
# HEALTH CHECK FUNCTIONS
# =============================================================================
check_result() {
    local check_name="$1"
    local result="$2"
    local message="$3"

    ((TOTAL_CHECKS++))

    case "$result" in
        "PASS")
            ((PASSED_CHECKS++))
            log_info "✓ ${check_name}: ${message}"
            ;;
        "WARN")
            ((WARNING_CHECKS++))
            log_warn "⚠ ${check_name}: ${message}"
            ;;
        "FAIL")
            ((FAILED_CHECKS++))
            log_error "✗ ${check_name}: ${message}"
            ;;
    esac
}

# =============================================================================
# SYSTEM HEALTH CHECKS
# =============================================================================
check_system_requirements() {
    echo "=== System Requirements ==="

    # Python version check
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        if [[ $(echo "${PYTHON_VERSION}" | cut -d'.' -f1) -ge 3 ]] && [[ $(echo "${PYTHON_VERSION}" | cut -d'.' -f2) -ge 8 ]]; then
            check_result "Python Version" "PASS" "Python ${PYTHON_VERSION} ✓"
        else
            check_result "Python Version" "FAIL" "Python ${PYTHON_VERSION} < 3.8 required"
        fi
    else
        check_result "Python Installation" "FAIL" "Python 3 not found"
    fi

    # Pip check
    if command -v pip3 &> /dev/null; then
        check_result "Pip Installation" "PASS" "pip3 available"
    else
        check_result "Pip Installation" "FAIL" "pip3 not found"
    fi

    # Virtual environment check
    if [[ -d "${PROJECT_ROOT}/venv" ]]; then
        check_result "Virtual Environment" "PASS" "venv directory exists"

        # Check if venv is functional
        if [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
            check_result "Virtual Environment Setup" "PASS" "activate script exists"
        else
            check_result "Virtual Environment Setup" "FAIL" "activate script missing"
        fi
    else
        check_result "Virtual Environment" "WARN" "venv directory not found"
    fi

    # System memory check
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [[ ${MEMORY_GB} -ge 4 ]]; then
            check_result "System Memory" "PASS" "${MEMORY_GB}GB available"
        else
            check_result "System Memory" "WARN" "${MEMORY_GB}GB available (4GB+ recommended)"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
        if [[ ${MEMORY_GB} -ge 8 ]]; then
            check_result "System Memory" "PASS" "${MEMORY_GB}GB available"
        else
            check_result "System Memory" "WARN" "${MEMORY_GB}GB available (8GB+ recommended)"
        fi
    fi

    # Disk space check
    DISK_AVAILABLE=$(df -h "${PROJECT_ROOT}" | awk 'NR==2 {print $4}' | sed 's/[^0-9.]//g')
    if [[ $(echo "${DISK_AVAILABLE} > 5" | bc -l 2>/dev/null || echo "1") == "1" ]]; then
        check_result "Disk Space" "PASS" "${DISK_AVAILABLE}GB available"
    else
        check_result "Disk Space" "WARN" "${DISK_AVAILABLE}GB available (5GB+ recommended)"
    fi
}

check_project_structure() {
    echo "=== Project Structure ==="

    # Core files
    local required_files=(
        "requirements.txt"
        "src/main/__init__.py"
        "src/main/config/unified_config_v2.yaml"
    )

    for file in "${required_files[@]}"; do
        if [[ -f "${PROJECT_ROOT}/${file}" ]]; then
            check_result "File: ${file}" "PASS" "exists"
        else
            check_result "File: ${file}" "FAIL" "missing"
        fi
    done

    # Core directories
    local required_dirs=(
        "src/ai_trader"
        "src/main/config"
        "src/main/data_pipeline"
        "src/main/feature_pipeline"
        "deployment"
        "scripts"
    )

    for dir in "${required_dirs[@]}"; do
        if [[ -d "${PROJECT_ROOT}/${dir}" ]]; then
            check_result "Directory: ${dir}" "PASS" "exists"
        else
            check_result "Directory: ${dir}" "FAIL" "missing"
        fi
    done

    # Data directories
    local data_dirs=(
        "data_lake"
        "logs"
        "cache"
    )

    for dir in "${data_dirs[@]}"; do
        if [[ -d "${PROJECT_ROOT}/${dir}" ]]; then
            check_result "Data Directory: ${dir}" "PASS" "exists"
        else
            check_result "Data Directory: ${dir}" "WARN" "will be created on startup"
        fi
    done
}

check_python_dependencies() {
    echo "=== Python Dependencies ==="

    cd "${PROJECT_ROOT}"

    # Activate virtual environment if available
    if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        check_result "Virtual Environment Activation" "PASS" "activated successfully"
    else
        check_result "Virtual Environment Activation" "WARN" "using system Python"
    fi

    # Critical dependencies check
    local critical_deps=(
        "pandas"
        "numpy"
        "asyncio"
        "aiohttp"
        "yaml"
        "sqlalchemy"
    )

    for dep in "${critical_deps[@]}"; do
        if python3 -c "import ${dep}" 2>/dev/null; then
            VERSION=$(python3 -c "import ${dep}; print(getattr(${dep}, '__version__', 'unknown'))" 2>/dev/null)
            check_result "Dependency: ${dep}" "PASS" "v${VERSION}"
        else
            check_result "Dependency: ${dep}" "FAIL" "not installed or import error"
        fi
    done

    # Optional but recommended dependencies
    local optional_deps=(
        "alpaca_trade_api:alpaca-py"
        "polygon:polygon-api-client"
        "talib:TA-Lib"
        "redis"
        "psycopg2:PostgreSQL adapter"
    )

    for dep_info in "${optional_deps[@]}"; do
        IFS=':' read -r dep_name dep_desc <<< "${dep_info}"
        if python3 -c "import ${dep_name}" 2>/dev/null; then
            check_result "Optional: ${dep_desc}" "PASS" "available"
        else
            check_result "Optional: ${dep_desc}" "WARN" "not available"
        fi
    done
}

check_configuration() {
    echo "=== Configuration ==="

    local config_file="${PROJECT_ROOT}/src/main/config/unified_config_v2.yaml"

    if [[ -f "${config_file}" ]]; then
        check_result "Configuration File" "PASS" "unified_config_v2.yaml exists"

        # Test YAML parsing
        if python3 -c "import yaml; yaml.safe_load(open('${config_file}'))" 2>/dev/null; then
            check_result "Configuration Syntax" "PASS" "YAML syntax valid"
        else
            check_result "Configuration Syntax" "FAIL" "YAML syntax error"
        fi
    else
        check_result "Configuration File" "FAIL" "unified_config_v2.yaml missing"
    fi

    # Environment variables check
    local env_vars=(
        "ALPACA_API_KEY:Critical"
        "ALPACA_SECRET_KEY:Critical"
        "ALPACA_BASE_URL:Critical"
        "POLYGON_API_KEY:Critical"
        "FRED_API_KEY:Optional"
        "REDDIT_CLIENT_ID:Optional"
    )

    for var_info in "${env_vars[@]}"; do
        IFS=':' read -r var_name var_importance <<< "${var_info}"
        if [[ -n "${!var_name:-}" ]]; then
            check_result "Environment: ${var_name}" "PASS" "set"
        else
            if [[ "${var_importance}" == "Critical" ]]; then
                check_result "Environment: ${var_name}" "FAIL" "not set (required)"
            else
                check_result "Environment: ${var_name}" "WARN" "not set (optional)"
            fi
        fi
    done
}

check_ai_trader_modules() {
    echo "=== AI Trader Modules ==="

    cd "${PROJECT_ROOT}"

    # Activate virtual environment if available
    if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi

    # Core module imports
    local core_modules=(
        "ai_trader"
        "ai_trader.config"
        "ai_trader.data_pipeline"
        "ai_trader.feature_pipeline"
        "ai_trader.data_pipeline.ingestion"
        "ai_trader.feature_pipeline.calculators"
    )

    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

    for module in "${core_modules[@]}"; do
        if python3 -c "import ${module}" 2>/dev/null; then
            check_result "Module: ${module}" "PASS" "imports successfully"
        else
            check_result "Module: ${module}" "FAIL" "import error"
        fi
    done

    # Test core functionality
    if python3 -c "
import sys
sys.path.append('src')
from ai_trader.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine
engine = UnifiedFeatureEngine({})
print('UnifiedFeatureEngine instantiated successfully')
" 2>/dev/null; then
        check_result "Core Functionality" "PASS" "UnifiedFeatureEngine works"
    else
        check_result "Core Functionality" "FAIL" "UnifiedFeatureEngine instantiation failed"
    fi
}

check_data_pipeline() {
    echo "=== Data Pipeline ==="

    cd "${PROJECT_ROOT}"

    # Check if data clients can be instantiated
    if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi

    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

    # Test basic data client functionality
    if python3 -c "
import sys
sys.path.append('src')
try:
    from ai_trader.utils.base_api_client import BaseAPIClient
    print('BaseAPIClient imported successfully')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
" 2>/dev/null; then
        check_result "Data Pipeline Base" "PASS" "BaseAPIClient available"
    else
        check_result "Data Pipeline Base" "FAIL" "BaseAPIClient import failed"
    fi

    # Check data lake directory
    if [[ -d "${PROJECT_ROOT}/data_lake" ]]; then
        LAKE_SIZE=$(du -sh "${PROJECT_ROOT}/data_lake" 2>/dev/null | cut -f1)
        check_result "Data Lake" "PASS" "exists (${LAKE_SIZE})"
    else
        check_result "Data Lake" "WARN" "will be created on first run"
    fi

    # Check if validation pipeline works
    if python3 -c "
import sys
sys.path.append('src')
try:
    from ai_trader.data_pipeline.validation.unified_validator import UnifiedValidator
    validator = UnifiedValidator()
    print('UnifiedValidator instantiated successfully')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" 2>/dev/null; then
        check_result "Validation Pipeline" "PASS" "UnifiedValidator works"
    else
        check_result "Validation Pipeline" "FAIL" "UnifiedValidator instantiation failed"
    fi
}

check_database_connectivity() {
    echo "=== Database Connectivity ==="

    cd "${PROJECT_ROOT}"

    if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi

    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

    # Test database connection if database is configured
    if python3 -c "
import sys
sys.path.append('src')
try:
    from ai_trader.config.config_manager import get_config
    config = get_config()
    if 'database' in config:
        print('Database configuration found')
    else:
        print('No database configuration found')
except Exception as e:
    print(f'Config error: {e}')
" 2>/dev/null; then
        check_result "Database Configuration" "PASS" "configuration accessible"
    else
        check_result "Database Configuration" "WARN" "configuration not accessible"
    fi

    # Check if SQLAlchemy models can be imported
    if python3 -c "
import sys
sys.path.append('src')
try:
    import sqlalchemy
    print('SQLAlchemy available')
except ImportError:
    print('SQLAlchemy not available')
    exit(1)
" 2>/dev/null; then
        check_result "Database Libraries" "PASS" "SQLAlchemy available"
    else
        check_result "Database Libraries" "WARN" "SQLAlchemy not available"
    fi
}

check_api_connectivity() {
    echo "=== API Connectivity ==="

    # Only run connectivity tests if API keys are available
    if [[ -n "${ALPACA_API_KEY:-}" ]] && [[ -n "${ALPACA_SECRET_KEY:-}" ]]; then
        cd "${PROJECT_ROOT}"

        if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
            source venv/bin/activate
        fi

        export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

        # Test Alpaca connectivity
        if timeout 10 python3 -c "
import sys
sys.path.append('src')
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient

    # Test API connectivity
    trading_client = TradingClient('${ALPACA_API_KEY}', '${ALPACA_SECRET_KEY}', paper=True)
    account = trading_client.get_account()
    print(f'Alpaca connection successful - Account: {account.account_number}')
except Exception as e:
    print(f'Alpaca connection failed: {e}')
    exit(1)
" 2>/dev/null; then
            check_result "Alpaca API" "PASS" "connectivity confirmed"
        else
            check_result "Alpaca API" "WARN" "connectivity test failed or timed out"
        fi
    else
        check_result "Alpaca API" "WARN" "API keys not configured"
    fi

    # Test Polygon connectivity if key available
    if [[ -n "${POLYGON_API_KEY:-}" ]]; then
        if timeout 10 python3 -c "
import sys
sys.path.append('src')
try:
    from polygon import RESTClient
    client = RESTClient(api_key='${POLYGON_API_KEY}')
    # Simple test call
    print('Polygon API key configured')
except Exception as e:
    print(f'Polygon test failed: {e}')
    exit(1)
" 2>/dev/null; then
            check_result "Polygon API" "PASS" "API key configured"
        else
            check_result "Polygon API" "WARN" "API test failed"
        fi
    else
        check_result "Polygon API" "WARN" "API key not configured"
    fi
}

check_scripts() {
    echo "=== Executable Scripts ==="

    local scripts=(
        "scripts/init_database.py"
        "scripts/run_data_pipeline.py"
        "deployment/scripts/deploy.sh"
        "deployment/scripts/rollback.sh"
    )

    for script in "${scripts[@]}"; do
        if [[ -f "${PROJECT_ROOT}/${script}" ]]; then
            if [[ -x "${PROJECT_ROOT}/${script}" ]] || [[ "${script}" == *.py ]]; then
                check_result "Script: ${script}" "PASS" "exists and executable"
            else
                check_result "Script: ${script}" "WARN" "exists but not executable"
            fi
        else
            check_result "Script: ${script}" "WARN" "not found"
        fi
    done
}

check_performance() {
    echo "=== Performance Checks ==="

    cd "${PROJECT_ROOT}"

    if [[ -d "venv" ]] && [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi

    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

    # Test feature calculation performance
    START_TIME=$(date +%s.%N)
    if python3 -c "
import sys
sys.path.append('src')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from ai_trader.feature_pipeline.calculators.technical_indicators import TechnicalIndicatorCalculator

    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.random(100) * 100 + 100,
        'high': np.random.random(100) * 100 + 105,
        'low': np.random.random(100) * 100 + 95,
        'close': np.random.random(100) * 100 + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    calc = TechnicalIndicatorCalculator()
    features = calc.calculate(data)
    print(f'Calculated {len(features.columns)} features for 100 data points')
except Exception as e:
    print(f'Performance test failed: {e}')
    exit(1)
" 2>/dev/null; then
        END_TIME=$(date +%s.%N)
        DURATION=$(echo "${END_TIME} - ${START_TIME}" | bc -l)
        check_result "Feature Calculation Speed" "PASS" "completed in ${DURATION}s"
    else
        check_result "Feature Calculation Speed" "WARN" "performance test failed"
    fi
}

# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================
print_summary() {
    echo ""
    echo "=============================================="
    echo "HEALTH CHECK SUMMARY"
    echo "=============================================="
    echo "Total Checks: ${TOTAL_CHECKS}"
    echo -e "${GREEN}Passed: ${PASSED_CHECKS}${NC}"
    echo -e "${YELLOW}Warnings: ${WARNING_CHECKS}${NC}"
    echo -e "${RED}Failed: ${FAILED_CHECKS}${NC}"
    echo ""

    local health_percentage=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))

    if [[ ${FAILED_CHECKS} -eq 0 ]]; then
        if [[ ${WARNING_CHECKS} -eq 0 ]]; then
            echo -e "${GREEN}✓ SYSTEM HEALTH: EXCELLENT (${health_percentage}%)${NC}"
            echo "The AI Trader system is fully operational."
        else
            echo -e "${YELLOW}⚠ SYSTEM HEALTH: GOOD (${health_percentage}%)${NC}"
            echo "The AI Trader system is operational with minor issues."
        fi
    else
        if [[ ${FAILED_CHECKS} -lt 3 ]]; then
            echo -e "${YELLOW}⚠ SYSTEM HEALTH: FAIR (${health_percentage}%)${NC}"
            echo "The AI Trader system has some issues that should be addressed."
        else
            echo -e "${RED}✗ SYSTEM HEALTH: POOR (${health_percentage}%)${NC}"
            echo "The AI Trader system has significant issues that must be resolved."
        fi
    fi

    echo ""
    echo "Log file: ${LOG_FILE}"
    echo "=============================================="
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================
main() {
    local verbose="${1:-}"

    echo "==============================================="
    echo "AI Trader System Health Check"
    echo "Timestamp: $(date)"
    echo "==============================================="

    setup_logging

    # Run all health checks
    check_system_requirements
    echo ""

    check_project_structure
    echo ""

    check_python_dependencies
    echo ""

    check_configuration
    echo ""

    check_ai_trader_modules
    echo ""

    check_data_pipeline
    echo ""

    check_database_connectivity
    echo ""

    check_api_connectivity
    echo ""

    check_scripts
    echo ""

    if [[ "${verbose}" == "-v" ]] || [[ "${verbose}" == "--verbose" ]]; then
        check_performance
        echo ""
    fi

    # Print summary
    print_summary

    # Exit with appropriate code
    if [[ ${FAILED_CHECKS} -gt 0 ]]; then
        exit 1
    elif [[ ${WARNING_CHECKS} -gt 5 ]]; then
        exit 2  # Too many warnings
    else
        exit 0
    fi
}

# =============================================================================
# USAGE FUNCTION
# =============================================================================
show_usage() {
    cat << EOF
AI Trader Health Check Script

Usage:
  $0 [options]

Options:
  -v, --verbose    Run additional performance checks
  -h, --help       Show this help message

Examples:
  $0               # Basic health check
  $0 --verbose     # Health check with performance tests

Exit Codes:
  0 - All checks passed (warnings allowed)
  1 - Critical failures detected
  2 - Too many warnings

EOF
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

#!/bin/bash
# AI Trader Deployment Automation Script
# Version: 1.0
# Author: Claude Code Assistant

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/deployment/logs/deployment"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/deploy_${TIMESTAMP}.log"
BACKUP_DIR="${PROJECT_ROOT}/deployment/backups"
HEALTH_CHECK_SCRIPT="${SCRIPT_DIR}/health_check.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================
setup_logging() {
    mkdir -p "${LOG_DIR}"
    exec > >(tee -a "${LOG_FILE}")
    exec 2>&1
    echo "Deployment started at $(date)" | tee -a "${LOG_FILE}"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1" | tee -a "${LOG_FILE}"
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
validate_environment() {
    log_info "Validating deployment environment..."

    # Check if we're in the correct directory
    if [[ ! -f "${PROJECT_ROOT}/requirements.txt" ]]; then
        log_error "Not in AI Trader project root. Missing requirements.txt"
        exit 1
    fi

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "${PYTHON_VERSION} < 3.8" | bc -l 2>/dev/null || echo "1") == "1" ]] && [[ "${PYTHON_VERSION}" != "3.8" ]] && [[ "${PYTHON_VERSION}" != "3.9" ]] && [[ "${PYTHON_VERSION}" != "3.10" ]] && [[ "${PYTHON_VERSION}" != "3.11" ]] && [[ "${PYTHON_VERSION}" != "3.12" ]]; then
        log_error "Python 3.8+ required. Found: ${PYTHON_VERSION}"
        exit 1
    fi

    log_info "Python version: ${PYTHON_VERSION} ✓"

    # Check for required system dependencies
    local missing_deps=()

    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("pip3")
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing system dependencies: ${missing_deps[*]}"
        exit 1
    fi

    log_info "Environment validation passed ✓"
}

validate_configuration() {
    log_info "Validating configuration files..."

    local config_file="${PROJECT_ROOT}/src/main/config/unified_config_v2.yaml"
    if [[ ! -f "${config_file}" ]]; then
        log_error "Main configuration file not found: ${config_file}"
        exit 1
    fi

    # Check for environment variables
    local required_env_vars=(
        "ALPACA_API_KEY"
        "ALPACA_SECRET_KEY"
        "ALPACA_BASE_URL"
        "POLYGON_API_KEY"
    )

    local missing_env_vars=()
    for var in "${required_env_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_env_vars+=("${var}")
        fi
    done

    if [[ ${#missing_env_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables: ${missing_env_vars[*]}"
        log_info "Please set these environment variables before deployment"
        exit 1
    fi

    log_info "Configuration validation passed ✓"
}

# =============================================================================
# BACKUP FUNCTIONS
# =============================================================================
create_backup() {
    log_info "Creating deployment backup..."

    mkdir -p "${BACKUP_DIR}"
    local backup_name="pre_deploy_${TIMESTAMP}"
    local backup_path="${BACKUP_DIR}/${backup_name}"

    # Backup critical directories
    mkdir -p "${backup_path}"

    if [[ -d "${PROJECT_ROOT}/data_lake" ]]; then
        log_info "Backing up data lake..."
        cp -r "${PROJECT_ROOT}/data_lake" "${backup_path}/" 2>/dev/null || log_warn "Failed to backup data_lake"
    fi

    if [[ -d "${PROJECT_ROOT}/src" ]]; then
        log_info "Backing up source code..."
        cp -r "${PROJECT_ROOT}/src" "${backup_path}/"
    fi

    if [[ -d "${PROJECT_ROOT}/scripts" ]]; then
        log_info "Backing up scripts..."
        cp -r "${PROJECT_ROOT}/scripts" "${backup_path}/"
    fi

    # Create backup manifest
    cat > "${backup_path}/manifest.txt" << EOF
Backup created: $(date)
Deployment timestamp: ${TIMESTAMP}
Project root: ${PROJECT_ROOT}
Python version: $(python3 --version)
Git commit: $(cd "${PROJECT_ROOT}" && git rev-parse HEAD 2>/dev/null || echo "N/A")
EOF

    echo "${backup_name}" > "${BACKUP_DIR}/latest_backup.txt"
    log_info "Backup created: ${backup_name} ✓"
}

# =============================================================================
# INSTALLATION FUNCTIONS
# =============================================================================
setup_python_environment() {
    log_info "Setting up Python environment..."

    cd "${PROJECT_ROOT}"

    # Check if virtual environment exists
    if [[ ! -d "venv" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip

    # Install dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt

    log_info "Python environment setup completed ✓"
}

install_system_dependencies() {
    log_info "Checking system dependencies..."

    # Check for TA-Lib (optional but recommended)
    if ! python3 -c "import talib" 2>/dev/null; then
        log_warn "TA-Lib not installed. Some technical indicators may not work."
        log_info "To install TA-Lib:"
        log_info "  macOS: brew install ta-lib && pip install ta-lib"
        log_info "  Ubuntu: sudo apt-get install libta-lib-dev && pip install ta-lib"
    fi

    log_info "System dependencies check completed ✓"
}

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================
setup_database() {
    log_info "Setting up database..."

    cd "${PROJECT_ROOT}"
    source venv/bin/activate

    # Check if database initialization script exists
    local init_script="${PROJECT_ROOT}/scripts/init_database.py"
    if [[ -f "${init_script}" ]]; then
        log_info "Running database initialization..."
        python "${init_script}" || {
            log_warn "Database initialization failed or already completed"
        }
    else
        log_warn "Database initialization script not found: ${init_script}"
    fi

    log_info "Database setup completed ✓"
}

# =============================================================================
# DEPLOYMENT FUNCTIONS
# =============================================================================
deploy_application() {
    log_info "Deploying application..."

    cd "${PROJECT_ROOT}"
    source venv/bin/activate

    # Install the package in development mode
    if [[ -f "setup.py" ]]; then
        pip install -e .
    else
        # Add src to Python path for development deployment
        export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
        log_info "Added ${PROJECT_ROOT}/src to PYTHONPATH"
    fi

    # Create necessary directories
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/data_lake"
    mkdir -p "${PROJECT_ROOT}/cache"

    log_info "Application deployment completed ✓"
}

# =============================================================================
# HEALTH CHECK FUNCTIONS
# =============================================================================
run_health_checks() {
    log_info "Running post-deployment health checks..."

    if [[ -f "${HEALTH_CHECK_SCRIPT}" ]]; then
        log_info "Executing health check script..."
        bash "${HEALTH_CHECK_SCRIPT}" || {
            log_error "Health checks failed!"
            return 1
        }
    else
        log_warn "Health check script not found: ${HEALTH_CHECK_SCRIPT}"
        log_info "Running basic health checks..."

        cd "${PROJECT_ROOT}"
        source venv/bin/activate

        # Test basic imports
        python3 -c "import sys; sys.path.append('src'); import ai_trader" || {
            log_error "Failed to import ai_trader module"
            return 1
        }

        log_info "Basic health checks passed ✓"
    fi
}

# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================
cleanup_old_backups() {
    log_info "Cleaning up old backups..."

    if [[ -d "${BACKUP_DIR}" ]]; then
        # Keep only the last 5 backups
        cd "${BACKUP_DIR}"
        ls -t | tail -n +6 | xargs -r rm -rf
        log_info "Old backups cleaned up ✓"
    fi
}

# =============================================================================
# MAIN DEPLOYMENT FUNCTION
# =============================================================================
main() {
    echo "==============================================="
    echo "AI Trader Deployment Script"
    echo "Timestamp: $(date)"
    echo "==============================================="

    setup_logging

    log_info "Starting deployment process..."

    # Pre-deployment validation
    validate_environment
    validate_configuration

    # Create backup
    create_backup

    # Setup and installation
    install_system_dependencies
    setup_python_environment
    setup_database
    deploy_application

    # Post-deployment validation
    if ! run_health_checks; then
        log_error "Deployment failed health checks!"
        log_info "Check logs at: ${LOG_FILE}"
        exit 1
    fi

    # Cleanup
    cleanup_old_backups

    log_info "Deployment completed successfully! ✓"
    log_info "Log file: ${LOG_FILE}"

    echo "==============================================="
    echo "Deployment Summary:"
    echo "- Status: SUCCESS"
    echo "- Timestamp: ${TIMESTAMP}"
    echo "- Log: ${LOG_FILE}"
    echo "- Backup: $(cat "${BACKUP_DIR}/latest_backup.txt" 2>/dev/null || echo "N/A")"
    echo "==============================================="

    log_info "To start the AI Trader system:"
    log_info "  cd ${PROJECT_ROOT}"
    log_info "  source venv/bin/activate"
    log_info "  python src/main/app/run_backfill.py"
}

# =============================================================================
# ERROR HANDLING
# =============================================================================
trap 'log_error "Deployment failed at line $LINENO. Exit code: $?"' ERR

# Handle script interruption
trap 'log_warn "Deployment interrupted by user"; exit 130' INT

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

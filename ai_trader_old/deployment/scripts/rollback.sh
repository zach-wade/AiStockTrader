#!/bin/bash
# AI Trader Rollback Automation Script
# Version: 1.0
# Author: Claude Code Assistant

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/deployment/logs/rollback"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/rollback_${TIMESTAMP}.log"
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
    echo "Rollback started at $(date)" | tee -a "${LOG_FILE}"
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
validate_rollback_environment() {
    log_info "Validating rollback environment..."

    # Check if we're in the correct directory
    if [[ ! -d "${BACKUP_DIR}" ]]; then
        log_error "Backup directory not found: ${BACKUP_DIR}"
        exit 1
    fi

    # Check if there are any backups available
    if [[ ! "$(ls -A "${BACKUP_DIR}" 2>/dev/null)" ]]; then
        log_error "No backups found in ${BACKUP_DIR}"
        exit 1
    fi

    log_info "Rollback environment validation passed ✓"
}

# =============================================================================
# BACKUP DISCOVERY FUNCTIONS
# =============================================================================
list_available_backups() {
    log_info "Available backups:"
    local backup_count=0

    if [[ -d "${BACKUP_DIR}" ]]; then
        for backup in "${BACKUP_DIR}"/pre_deploy_*; do
            if [[ -d "${backup}" ]]; then
                ((backup_count++))
                local backup_name=$(basename "${backup}")
                local backup_timestamp=${backup_name#pre_deploy_}
                local backup_date=$(echo "${backup_timestamp}" | sed 's/_/ /' | sed 's/\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\) \([0-9]\{2\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1-\2-\3 \4:\5:\6/')

                echo "  ${backup_count}. ${backup_name} (${backup_date})"

                # Show manifest if available
                if [[ -f "${backup}/manifest.txt" ]]; then
                    echo "     $(grep "Git commit:" "${backup}/manifest.txt" 2>/dev/null || echo "     No git info")"
                fi
            fi
        done
    fi

    if [[ ${backup_count} -eq 0 ]]; then
        log_error "No valid backups found"
        exit 1
    fi

    return ${backup_count}
}

get_latest_backup() {
    local latest_backup=""

    if [[ -f "${BACKUP_DIR}/latest_backup.txt" ]]; then
        latest_backup=$(cat "${BACKUP_DIR}/latest_backup.txt")
        if [[ -d "${BACKUP_DIR}/${latest_backup}" ]]; then
            echo "${latest_backup}"
            return 0
        fi
    fi

    # Fallback: find the most recent backup by directory timestamp
    latest_backup=$(ls -t "${BACKUP_DIR}"/pre_deploy_* 2>/dev/null | head -n1 | xargs basename)
    if [[ -n "${latest_backup}" ]]; then
        echo "${latest_backup}"
        return 0
    fi

    return 1
}

select_backup() {
    local target_backup="${1:-}"

    if [[ -n "${target_backup}" ]]; then
        # Specific backup provided
        if [[ ! -d "${BACKUP_DIR}/${target_backup}" ]]; then
            log_error "Specified backup not found: ${target_backup}"
            exit 1
        fi
        echo "${target_backup}"
    else
        # No specific backup provided, use latest
        if ! target_backup=$(get_latest_backup); then
            log_error "Could not determine latest backup"
            exit 1
        fi
        echo "${target_backup}"
    fi
}

# =============================================================================
# SERVICE MANAGEMENT FUNCTIONS
# =============================================================================
stop_services() {
    log_info "Stopping AI Trader services..."

    # Stop any running processes
    local pids=$(pgrep -f "ai_trader" 2>/dev/null || true)
    if [[ -n "${pids}" ]]; then
        log_info "Stopping running AI Trader processes: ${pids}"
        echo "${pids}" | xargs -r kill -TERM
        sleep 5

        # Force kill if still running
        pids=$(pgrep -f "ai_trader" 2>/dev/null || true)
        if [[ -n "${pids}" ]]; then
            log_warn "Force killing processes: ${pids}"
            echo "${pids}" | xargs -r kill -KILL
        fi
    fi

    log_info "Services stopped ✓"
}

# =============================================================================
# ROLLBACK FUNCTIONS
# =============================================================================
create_pre_rollback_backup() {
    log_info "Creating pre-rollback backup..."

    local current_backup_name="pre_rollback_${TIMESTAMP}"
    local current_backup_path="${BACKUP_DIR}/${current_backup_name}"

    mkdir -p "${current_backup_path}"

    # Backup current state
    if [[ -d "${PROJECT_ROOT}/src" ]]; then
        cp -r "${PROJECT_ROOT}/src" "${current_backup_path}/"
    fi

    if [[ -d "${PROJECT_ROOT}/scripts" ]]; then
        cp -r "${PROJECT_ROOT}/scripts" "${current_backup_path}/"
    fi

    # Create manifest
    cat > "${current_backup_path}/manifest.txt" << EOF
Pre-rollback backup created: $(date)
Rollback timestamp: ${TIMESTAMP}
Project root: ${PROJECT_ROOT}
Python version: $(python3 --version 2>/dev/null || echo "N/A")
Git commit: $(cd "${PROJECT_ROOT}" && git rev-parse HEAD 2>/dev/null || echo "N/A")
EOF

    log_info "Pre-rollback backup created: ${current_backup_name} ✓"
}

restore_from_backup() {
    local backup_name="${1}"
    local backup_path="${BACKUP_DIR}/${backup_name}"

    log_info "Restoring from backup: ${backup_name}..."

    if [[ ! -d "${backup_path}" ]]; then
        log_error "Backup path not found: ${backup_path}"
        exit 1
    fi

    # Show what we're restoring
    if [[ -f "${backup_path}/manifest.txt" ]]; then
        log_info "Backup manifest:"
        cat "${backup_path}/manifest.txt" | sed 's/^/  /'
    fi

    # Stop services before restoration
    stop_services

    # Restore source code
    if [[ -d "${backup_path}/src" ]]; then
        log_info "Restoring source code..."
        rm -rf "${PROJECT_ROOT}/src"
        cp -r "${backup_path}/src" "${PROJECT_ROOT}/"
    fi

    # Restore scripts
    if [[ -d "${backup_path}/scripts" ]]; then
        log_info "Restoring scripts..."
        rm -rf "${PROJECT_ROOT}/scripts"
        cp -r "${backup_path}/scripts" "${PROJECT_ROOT}/"
    fi

    # Restore data lake if it was backed up (optional)
    if [[ -d "${backup_path}/data_lake" ]]; then
        read -p "Restore data lake? This will overwrite current data (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Restoring data lake..."
            rm -rf "${PROJECT_ROOT}/data_lake"
            cp -r "${backup_path}/data_lake" "${PROJECT_ROOT}/"
        else
            log_info "Skipping data lake restoration"
        fi
    fi

    log_info "Restoration completed ✓"
}

setup_python_environment_rollback() {
    log_info "Setting up Python environment after rollback..."

    cd "${PROJECT_ROOT}"

    # Activate virtual environment if it exists
    if [[ -d "venv" ]]; then
        source venv/bin/activate

        # Reinstall dependencies to match restored code
        if [[ -f "requirements.txt" ]]; then
            log_info "Reinstalling dependencies..."
            pip install -r requirements.txt
        fi
    else
        log_warn "Virtual environment not found. You may need to recreate it."
    fi

    log_info "Python environment setup completed ✓"
}

# =============================================================================
# HEALTH CHECK FUNCTIONS
# =============================================================================
run_rollback_health_checks() {
    log_info "Running post-rollback health checks..."

    if [[ -f "${HEALTH_CHECK_SCRIPT}" ]]; then
        log_info "Executing health check script..."
        bash "${HEALTH_CHECK_SCRIPT}" || {
            log_error "Health checks failed after rollback!"
            return 1
        }
    else
        log_warn "Health check script not found: ${HEALTH_CHECK_SCRIPT}"
        log_info "Running basic health checks..."

        cd "${PROJECT_ROOT}"
        if [[ -d "venv" ]]; then
            source venv/bin/activate
        fi

        # Test basic imports
        python3 -c "import sys; sys.path.append('src'); import ai_trader" || {
            log_error "Failed to import ai_trader module after rollback"
            return 1
        }

        log_info "Basic health checks passed ✓"
    fi
}

# =============================================================================
# INTERACTIVE FUNCTIONS
# =============================================================================
confirm_rollback() {
    local backup_name="${1}"

    echo "==============================================="
    echo "ROLLBACK CONFIRMATION"
    echo "==============================================="
    echo "Target backup: ${backup_name}"
    echo "Current time: $(date)"
    echo "Project root: ${PROJECT_ROOT}"
    echo
    echo "This will:"
    echo "1. Stop all AI Trader services"
    echo "2. Create a pre-rollback backup"
    echo "3. Restore code from the specified backup"
    echo "4. Reinstall Python dependencies"
    echo "5. Run health checks"
    echo
    echo "WARNING: This operation cannot be easily undone!"
    echo

    read -p "Are you sure you want to proceed? (yes/no): " -r
    if [[ ! $REPLY == "yes" ]]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
}

# =============================================================================
# MAIN ROLLBACK FUNCTION
# =============================================================================
main() {
    echo "==============================================="
    echo "AI Trader Rollback Script"
    echo "Timestamp: $(date)"
    echo "==============================================="

    setup_logging

    local target_backup="${1:-}"
    local force_flag="${2:-}"

    # Show usage if help requested
    if [[ "${target_backup}" == "-h" ]] || [[ "${target_backup}" == "--help" ]]; then
        show_usage
        exit 0
    fi

    # List backups if requested
    if [[ "${target_backup}" == "--list" ]] || [[ "${target_backup}" == "-l" ]]; then
        list_available_backups
        exit 0
    fi

    log_info "Starting rollback process..."

    # Validation
    validate_rollback_environment

    # Select backup
    target_backup=$(select_backup "${target_backup}")
    log_info "Selected backup: ${target_backup}"

    # Confirmation (skip if force flag provided)
    if [[ "${force_flag}" != "--force" ]] && [[ "${force_flag}" != "-f" ]]; then
        confirm_rollback "${target_backup}"
    fi

    # Create pre-rollback backup
    create_pre_rollback_backup

    # Perform rollback
    restore_from_backup "${target_backup}"
    setup_python_environment_rollback

    # Post-rollback validation
    if ! run_rollback_health_checks; then
        log_error "Rollback completed but health checks failed!"
        log_info "Check logs at: ${LOG_FILE}"
        exit 1
    fi

    log_info "Rollback completed successfully! ✓"
    log_info "Log file: ${LOG_FILE}"

    echo "==============================================="
    echo "Rollback Summary:"
    echo "- Status: SUCCESS"
    echo "- Restored from: ${target_backup}"
    echo "- Timestamp: ${TIMESTAMP}"
    echo "- Log: ${LOG_FILE}"
    echo "==============================================="

    log_info "To start the AI Trader system:"
    log_info "  cd ${PROJECT_ROOT}"
    log_info "  source venv/bin/activate"
    log_info "  python src/main/app/run_backfill.py"
}

show_usage() {
    cat << EOF
AI Trader Rollback Script

Usage:
  $0 [backup_name] [options]

Arguments:
  backup_name    Name of backup to restore from (optional, uses latest if not specified)

Options:
  -l, --list     List available backups
  -f, --force    Skip confirmation prompt
  -h, --help     Show this help message

Examples:
  $0                                    # Rollback to latest backup
  $0 pre_deploy_20240101_120000         # Rollback to specific backup
  $0 --list                            # List available backups
  $0 pre_deploy_20240101_120000 --force # Force rollback without confirmation

EOF
}

# =============================================================================
# ERROR HANDLING
# =============================================================================
trap 'log_error "Rollback failed at line $LINENO. Exit code: $?"' ERR

# Handle script interruption
trap 'log_warn "Rollback interrupted by user"; exit 130' INT

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

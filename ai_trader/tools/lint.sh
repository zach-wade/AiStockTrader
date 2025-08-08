#!/bin/bash
# Linting script for AI Trader codebase

set -e  # Exit on error

echo "🔍 Running AI Trader Code Quality Checks..."
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if tools are installed
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed. Please install it first.${NC}"
        exit 1
    fi
}

# Run black formatter check
run_black() {
    echo -e "\n${YELLOW}Running Black formatter check...${NC}"
    if black --check src/ tests/ 2>/dev/null; then
        echo -e "${GREEN}✅ Black: All files properly formatted${NC}"
    else
        echo -e "${RED}❌ Black: Some files need formatting${NC}"
        echo "Run 'black src/ tests/' to fix"
        return 1
    fi
}

# Run flake8 linter
run_flake8() {
    echo -e "\n${YELLOW}Running Flake8 linter...${NC}"
    if flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__,*.pyc --ignore=E203,W503 2>/dev/null; then
        echo -e "${GREEN}✅ Flake8: No linting errors found${NC}"
    else
        echo -e "${RED}❌ Flake8: Linting errors found${NC}"
        return 1
    fi
}

# Run isort import checker
run_isort() {
    echo -e "\n${YELLOW}Running isort import checker...${NC}"
    if isort --check-only --diff src/ tests/ 2>/dev/null; then
        echo -e "${GREEN}✅ isort: Imports properly sorted${NC}"
    else
        echo -e "${RED}❌ isort: Imports need sorting${NC}"
        echo "Run 'isort src/ tests/' to fix"
        return 1
    fi
}

# Run mypy type checker
run_mypy() {
    echo -e "\n${YELLOW}Running mypy type checker...${NC}"
    if mypy src/ --ignore-missing-imports --no-strict-optional 2>/dev/null; then
        echo -e "${GREEN}✅ mypy: Type checking passed${NC}"
    else
        echo -e "${YELLOW}⚠️  mypy: Type checking warnings${NC}"
        # Don't fail on mypy warnings
    fi
}

# Run pylint
run_pylint() {
    echo -e "\n${YELLOW}Running pylint...${NC}"
    # Run with a reasonable config, don't fail on warnings
    pylint src/ --rcfile=.pylintrc --exit-zero 2>/dev/null || true
    echo -e "${GREEN}✅ pylint: Analysis complete${NC}"
}

# Check for security issues with bandit
run_bandit() {
    echo -e "\n${YELLOW}Running bandit security checker...${NC}"
    if bandit -r src/ -ll -i 2>/dev/null; then
        echo -e "${GREEN}✅ bandit: No security issues found${NC}"
    else
        echo -e "${YELLOW}⚠️  bandit: Security warnings found${NC}"
    fi
}

# Main execution
main() {
    local failed=0
    
    # Check if we're in the right directory
    if [ ! -d "src/main" ]; then
        echo -e "${RED}❌ Error: Must run from AI Trader root directory${NC}"
        exit 1
    fi
    
    # Check tools
    echo "Checking required tools..."
    check_tool black
    check_tool flake8
    check_tool isort
    
    # Optional tools (don't fail if not installed)
    if command -v mypy &> /dev/null; then
        HAS_MYPY=1
    fi
    if command -v pylint &> /dev/null; then
        HAS_PYLINT=1
    fi
    if command -v bandit &> /dev/null; then
        HAS_BANDIT=1
    fi
    
    # Run checks
    run_black || ((failed++))
    run_flake8 || ((failed++))
    run_isort || ((failed++))
    
    if [ ! -z "$HAS_MYPY" ]; then
        run_mypy
    fi
    
    if [ ! -z "$HAS_PYLINT" ]; then
        run_pylint
    fi
    
    if [ ! -z "$HAS_BANDIT" ]; then
        run_bandit
    fi
    
    # Summary
    echo -e "\n=========================================="
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}✅ All required checks passed!${NC}"
        exit 0
    else
        echo -e "${RED}❌ $failed check(s) failed${NC}"
        echo -e "Fix the issues above and run again"
        exit 1
    fi
}

# Run main function
main "$@"
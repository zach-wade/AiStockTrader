#!/bin/bash
# Setup and run tests with coverage for AI Trader

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}AI Trader Test Setup and Run Script${NC}"
echo "===================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
echo -e "${YELLOW}Set PYTHONPATH to: $PYTHONPATH${NC}"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
    echo "Looking for venv..."
    if [ -d "venv" ]; then
        echo "Activating venv..."
        source venv/bin/activate
    else
        echo -e "${RED}No venv found. Please create and activate a virtual environment.${NC}"
        exit 1
    fi
fi

# Install required packages
echo -e "\n${YELLOW}Installing required packages...${NC}"
pip install -q pytest pytest-asyncio pytest-cov coverage[toml]

# Check installation
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest installation failed${NC}"
    exit 1
fi

if ! python -c "import pytest_cov" &> /dev/null; then
    echo -e "${RED}pytest-cov installation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All required packages installed${NC}"

# Run tests based on argument
if [ "$1" == "coverage" ] || [ -z "$1" ]; then
    echo -e "\n${YELLOW}Running tests with coverage...${NC}"
    pytest --cov=src/main --cov-report=xml --cov-report=html --cov-report=term-missing tests/
elif [ "$1" == "unit" ]; then
    echo -e "\n${YELLOW}Running unit tests...${NC}"
    pytest tests/unit/ -v
elif [ "$1" == "integration" ]; then
    echo -e "\n${YELLOW}Running integration tests...${NC}"
    pytest tests/integration/ -v
elif [ "$1" == "report" ]; then
    echo -e "\n${YELLOW}Generating coverage report...${NC}"
    python tests/coverage_report.py
elif [ "$1" == "help" ]; then
    echo "Usage: $0 [coverage|unit|integration|report|help]"
    echo "  coverage    - Run all tests with coverage (default)"
    echo "  unit        - Run only unit tests"
    echo "  integration - Run only integration tests"
    echo "  report      - Generate coverage report from existing data"
    echo "  help        - Show this help message"
else
    echo -e "${RED}Unknown option: $1${NC}"
    echo "Use '$0 help' for usage information"
    exit 1
fi

# Check if coverage report was generated
if [ "$1" == "coverage" ] || [ -z "$1" ]; then
    if [ -f "coverage.xml" ]; then
        echo -e "\n${GREEN}✓ Coverage report generated${NC}"
        echo -e "View HTML report: open htmlcov/index.html"

        # Generate summary report
        echo -e "\n${YELLOW}Generating coverage summary...${NC}"
        python tests/coverage_report.py
    else
        echo -e "\n${RED}✗ Coverage report not generated${NC}"
    fi
fi

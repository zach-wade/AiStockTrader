#!/bin/bash
# Test runner script for AI Trader

set -e  # Exit on error

echo "🧪 Running AI Trader Tests..."
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
COVERAGE=0
VERBOSE=0
MARKERS=""
TEST_PATH="tests/"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=1
            shift
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --unit)
            MARKERS="unit"
            shift
            ;;
        --integration)
            MARKERS="integration"
            shift
            ;;
        --slow)
            MARKERS="slow"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options] [test_path]"
            echo ""
            echo "Options:"
            echo "  -c, --coverage     Run with coverage report"
            echo "  -v, --verbose      Verbose output"
            echo "  --unit             Run only unit tests"
            echo "  --integration      Run only integration tests"
            echo "  --slow             Include slow tests"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all tests"
            echo "  $0 --coverage                # Run with coverage"
            echo "  $0 --unit                    # Run only unit tests"
            echo "  $0 tests/unit/test_config.py # Run specific test file"
            exit 0
            ;;
        *)
            TEST_PATH="$1"
            shift
            ;;
    esac
done

# Check if we're in the right directory
if [ ! -d "src/main" ]; then
    echo -e "${RED}❌ Error: Must run from AI Trader root directory${NC}"
    exit 1
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest is not installed. Please install it first.${NC}"
    echo "Run: pip install pytest pytest-cov pytest-asyncio"
    exit 1
fi

# Build pytest command
PYTEST_CMD="pytest"

# Add test path
PYTEST_CMD="$PYTEST_CMD $TEST_PATH"

# Add verbosity
if [ $VERBOSE -eq 1 ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add markers
if [ ! -z "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD -m $MARKERS"
fi

# Add coverage
if [ $COVERAGE -eq 1 ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term"
fi

# Add color
PYTEST_CMD="$PYTEST_CMD --color=yes"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run tests
echo -e "${BLUE}Running command: $PYTEST_CMD${NC}"
echo ""

if $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}✅ All tests passed!${NC}"

    if [ $COVERAGE -eq 1 ]; then
        echo ""
        echo -e "${YELLOW}📊 Coverage report generated in htmlcov/index.html${NC}"
    fi

    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi

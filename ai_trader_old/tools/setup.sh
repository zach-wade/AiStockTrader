#!/bin/bash
# Setup script for AI Trader development environment

set -e  # Exit on error

echo "🚀 AI Trader Development Setup"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
check_python() {
    echo -e "${BLUE}Checking Python version...${NC}"

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        echo "Found Python $PYTHON_VERSION"

        # Check if Python >= 3.8
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
            echo -e "${GREEN}✅ Python version is compatible${NC}"
        else
            echo -e "${RED}❌ Python 3.8 or higher is required${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Python 3 is not installed${NC}"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    echo -e "\n${BLUE}Setting up virtual environment...${NC}"

    VENV_PATH="../venv"

    if [ -d "$VENV_PATH" ]; then
        echo -e "${YELLOW}Virtual environment already exists at $VENV_PATH${NC}"
        read -p "Do you want to recreate it? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Keeping existing virtual environment"
            return
        fi
        rm -rf "$VENV_PATH"
    fi

    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✅ Virtual environment created at $VENV_PATH${NC}"
}

# Install dependencies
install_dependencies() {
    echo -e "\n${BLUE}Installing dependencies...${NC}"

    # Activate virtual environment
    source ../venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        echo -e "${GREEN}✅ Core dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠️  requirements.txt not found${NC}"
    fi

    # Install development dependencies
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        echo -e "${GREEN}✅ Development dependencies installed${NC}"
    else
        echo -e "${YELLOW}Creating requirements-dev.txt...${NC}"
        cat > requirements-dev.txt << EOF
# Development dependencies
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.20.0
pytest-mock>=3.10.0
black>=23.0.0
flake8>=6.0.0
isort>=5.11.0
mypy>=1.0.0
pylint>=2.16.0
bandit>=1.7.0
pre-commit>=3.0.0
tox>=4.0.0
EOF
        pip install -r requirements-dev.txt
        echo -e "${GREEN}✅ Development dependencies installed${NC}"
    fi
}

# Setup pre-commit hooks
setup_precommit() {
    echo -e "\n${BLUE}Setting up pre-commit hooks...${NC}"

    if [ -f ".pre-commit-config.yaml" ]; then
        pre-commit install
        echo -e "${GREEN}✅ Pre-commit hooks installed${NC}"
    else
        echo -e "${YELLOW}⚠️  .pre-commit-config.yaml not found, skipping pre-commit setup${NC}"
    fi
}

# Create necessary directories
create_directories() {
    echo -e "\n${BLUE}Creating necessary directories...${NC}"

    directories=(
        "data/dev/lake"
        "data/dev/models"
        "data/dev/cache"
        "data/dev/backtest"
        "logs/dev"
        "profiles/dev"
    )

    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        echo "Created: $dir"
    done

    echo -e "${GREEN}✅ Directories created${NC}"
}

# Setup environment file
setup_env_file() {
    echo -e "\n${BLUE}Setting up environment file...${NC}"

    if [ -f ".env" ]; then
        echo -e "${YELLOW}⚠️  .env file already exists${NC}"
    else
        cat > .env.example << EOF
# AI Trader Environment Variables

# Environment
AI_TRADER_ENV=dev

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_trader_dev
DB_USER=ai_trader_dev
DB_PASSWORD=dev_password

# API Keys (obtain from providers)
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
POLYGON_API_KEY=

# Optional API Keys
ALPHA_VANTAGE_KEY=
FRED_API_KEY=
BENZINGA_API_KEY=
FINNHUB_API_KEY=
NEWS_API_KEY=

# Notifications (optional)
SLACK_WEBHOOK_DEV=
SMTP_HOST=
SMTP_USERNAME=
SMTP_PASSWORD=
EOF

        cp .env.example .env
        echo -e "${GREEN}✅ Created .env file (update with your credentials)${NC}"
        echo -e "${YELLOW}⚠️  Remember to update .env with your actual API keys${NC}"
    fi
}

# Run initial tests
run_initial_tests() {
    echo -e "\n${BLUE}Running initial tests...${NC}"

    # Try to import main modules
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from main.utils import core
    from main.config import config_manager
    print('✅ Core imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"
}

# Main setup flow
main() {
    echo "Starting AI Trader development setup..."
    echo ""

    # Check if we're in the right directory
    if [ ! -d "src/main" ]; then
        echo -e "${RED}❌ Error: Must run from AI Trader root directory${NC}"
        exit 1
    fi

    # Run setup steps
    check_python
    create_venv
    install_dependencies
    setup_precommit
    create_directories
    setup_env_file
    run_initial_tests

    # Summary
    echo ""
    echo "=============================="
    echo -e "${GREEN}✅ Setup completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Update .env file with your API credentials"
    echo "2. Activate virtual environment: source ../venv/bin/activate"
    echo "3. Run tests: ./tools/run_tests.sh"
    echo "4. Start development!"
    echo ""
    echo "Useful commands:"
    echo "  ./tools/validate_config.py --env dev  # Validate configuration"
    echo "  ./tools/lint.sh                       # Run code quality checks"
    echo "  ./tools/run_tests.sh --coverage       # Run tests with coverage"
}

# Run main function
main "$@"

# CLAUDE-SETUP.md - Initial Setup & Configuration Guide

This document provides comprehensive setup instructions for the AI Trading System.

---

## 📦 Repository Information

### Repository Access
```bash
# Clone repository (when available)
git clone https://github.com/[username]/ai-trader.git
cd ai-trader

# Preferred access method: HTTPS (for CI/CD)
git remote set-url origin https://github.com/[username]/ai-trader.git

# Alternative: SSH (for development)
git remote set-url origin git@github.com:[username]/ai-trader.git
```

### Branch Strategy
```
main            # Production-ready code
├── develop     # Integration branch
├── feature/*   # Feature branches
├── bugfix/*    # Bug fix branches
└── release/*   # Release preparation
```

### Git Configuration
```bash
# Set up git identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Configure line endings
git config --global core.autocrlf input  # Mac/Linux
git config --global core.autocrlf true   # Windows

# Set default branch
git config --global init.defaultBranch main

# Enable color output
git config --global color.ui auto
```

---

## 🚀 Quick Start Setup

### Prerequisites Installation

#### macOS
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Install PostgreSQL
brew install postgresql@15
brew services start postgresql@15

# Install Redis
brew install redis
brew services start redis

# Install development tools
brew install git wget curl jq
```

#### Ubuntu/Debian
```bash
# Update packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip

# Install PostgreSQL
sudo apt install postgresql-15 postgresql-contrib

# Install Redis
sudo apt install redis-server

# Install development tools
sudo apt install git curl wget jq build-essential
```

#### Windows (WSL2)
```bash
# Install WSL2
wsl --install

# Follow Ubuntu instructions above within WSL2
```

---

## 🐍 Python Environment Setup

### Virtual Environment
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Environment Variables
```bash
# Create .env file
cat > .env << EOF
# API Keys (REQUIRED)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_API_SECRET=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
POLYGON_API_KEY=your_polygon_api_key_here

# Database Configuration
DATABASE_URL=postgresql://ai_trader:password@localhost:5432/ai_trader
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_trader
DB_USER=ai_trader
DB_PASSWORD=secure_password_here

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DATA_LAKE_PATH=/path/to/data_lake

# Optional Settings
ENABLE_SLACK_ALERTS=false
SLACK_WEBHOOK_URL=
ENABLE_EMAIL_ALERTS=false
EMAIL_SMTP_SERVER=
EMAIL_FROM_ADDRESS=
EMAIL_TO_ADDRESS=
EOF

# Load environment variables
source .env  # For current session
```

---

## 🗄️ Database Initialization

### PostgreSQL Setup
```bash
# Create database user
sudo -u postgres createuser -P ai_trader
# Enter password when prompted

# Create database
sudo -u postgres createdb -O ai_trader ai_trader

# Grant permissions
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ai_trader TO ai_trader;"

# Test connection
psql -h localhost -U ai_trader -d ai_trader -c "SELECT version();"
```

### Initialize Schema
```bash
# Run initialization script
python scripts/init_database.py

# Or manually with SQL
psql -h localhost -U ai_trader -d ai_trader < deployment/sql/init.sql

# Create partitions
python scripts/create_partitions.py --months-ahead 3

# Verify tables
psql -h localhost -U ai_trader -d ai_trader -c "\dt"
```

### Sample Data (Optional)
```bash
# Load sample data for testing
python scripts/load_sample_data.py

# Generate mock historical data
python scripts/generate_mock_data.py --symbols AAPL,MSFT,GOOGL --days 30
```

---

## 🔑 API Configuration

### Alpaca Setup
1. Create account at https://alpaca.markets
2. Generate API keys from dashboard
3. Choose endpoint:
   - Paper Trading: `https://paper-api.alpaca.markets`
   - Live Trading: `https://api.alpaca.markets`
4. Test connection:
```bash
python scripts/test_alpaca_connection.py
```

### Polygon.io Setup
1. Create account at https://polygon.io
2. Subscribe to appropriate plan (Stocks Starter minimum)
3. Get API key from dashboard
4. Test connection:
```bash
python scripts/test_polygon_connection.py
```

### API Key Security
```bash
# Never commit API keys
echo ".env" >> .gitignore

# Use environment variables
export ALPACA_API_KEY="your_key"  # Temporary
source .env                       # From file

# Encrypt sensitive files (optional)
gpg --symmetric .env
```

---

## 🐳 Docker Setup (Alternative)

### Docker Installation
```bash
# macOS
brew install docker docker-compose

# Ubuntu
sudo apt install docker.io docker-compose

# Start Docker daemon
sudo systemctl start docker  # Linux
open -a Docker              # macOS
```

### Docker Compose Setup
```bash
# Build containers
docker-compose build

# Start services
docker-compose up -d

# Initialize database
docker-compose exec app python scripts/init_database.py

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Docker Environment
```bash
# Create docker.env file
cat > docker.env << EOF
POSTGRES_DB=ai_trader
POSTGRES_USER=ai_trader
POSTGRES_PASSWORD=secure_password
REDIS_PASSWORD=redis_password
EOF

# Reference in docker-compose.yml
env_file:
  - docker.env
```

---

## 🛠️ Development Tools Setup

### IDE Configuration

#### VS Code
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true,
  "editor.rulers": [120],
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm
```
1. File → Settings → Project → Python Interpreter
2. Select: ./venv/bin/python
3. Enable: Format on save (Black)
4. Enable: Optimize imports on save
5. Set line length: 120
```

### Linting & Formatting
```bash
# Install tools
pip install black ruff mypy isort

# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/
mypy src/

# Pre-commit hooks
pip install pre-commit
pre-commit install
```

### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.261
    hooks:
      - id: ruff
```

---

## 🧪 Testing Setup

### Test Database
```bash
# Create test database
sudo -u postgres createdb ai_trader_test

# Configure test environment
cat > .env.test << EOF
DATABASE_URL=postgresql://ai_trader:password@localhost:5432/ai_trader_test
ENVIRONMENT=testing
LOG_LEVEL=DEBUG
EOF
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/main --cov-report=html

# Run specific test
pytest tests/unit/test_market_data.py::TestMarketData::test_fetch_data

# Run integration tests
pytest tests/integration/ -v

# Run with parallel execution
pytest -n auto
```

### Test Data Generation
```bash
# Generate test fixtures
python scripts/generate_test_fixtures.py

# Create mock market data
python scripts/create_mock_data.py --output tests/fixtures/
```

---

## 📊 Initial Data Setup

### Historical Data Backfill
```bash
# 1. Start with a small test
python ai_trader.py backfill --symbols AAPL --days 7 --stage market_data

# 2. Verify data loaded correctly
psql -h localhost -U ai_trader -d ai_trader \
  -c "SELECT COUNT(*) FROM market_data_1h WHERE symbol='AAPL';"

# 3. Backfill Layer 1 symbols (liquid)
python ai_trader.py backfill --layer layer1 --days 30

# 4. Calculate initial features
python ai_trader.py features --layer layer1 --lookback 20
```

### Universe Initialization
```bash
# Load initial universe
python ai_trader.py universe init

# Run initial scan
python ai_trader.py universe scan --layer layer0

# Verify symbols loaded
psql -c "SELECT layer, COUNT(*) FROM companies GROUP BY layer;"
```

---

## 🔍 Validation & Testing

### System Validation
```bash
# Validate all components
python ai_trader.py validate --all

# Test individual components
python ai_trader.py validate --component database
python ai_trader.py validate --component redis
python ai_trader.py validate --component apis
python ai_trader.py validate --component trading
```

### Connection Tests
```bash
# Test database
python scripts/test_database.py

# Test Redis
python scripts/test_redis.py

# Test APIs
python scripts/test_apis.py

# Test broker connection
python scripts/test_broker.py
```

### Performance Baseline
```bash
# Run performance tests
python scripts/benchmark.py

# Expected results:
# - Database writes: >10K records/sec
# - Feature calculation: >1M features/sec
# - API response time: <100ms
```

---

## 🚦 First Run Checklist

### Pre-flight Checks
```bash
# ✅ Environment variables set
env | grep -E "ALPACA|POLYGON|DATABASE"

# ✅ Database running
pg_isready -h localhost

# ✅ Redis running
redis-cli ping

# ✅ Python environment active
which python  # Should show venv path

# ✅ Dependencies installed
pip list | grep -E "pandas|asyncpg|alpaca"

# ✅ Configuration valid
python scripts/validate_config.py
```

### Initial Test Trade (Paper Mode)
```bash
# 1. Start in paper mode
python ai_trader.py trade --mode paper --symbols AAPL --dry-run

# 2. Monitor logs
tail -f logs/ai_trader.log

# 3. Check for errors
grep ERROR logs/ai_trader.log

# 4. Verify no real trades executed
python scripts/check_paper_trades.py
```

---

## 🔧 Troubleshooting Setup Issues

### Common Setup Problems

#### Python Version Issues
```bash
# Check Python version
python --version  # Should be 3.8+

# If wrong version, use explicit path
/usr/bin/python3.11 -m venv venv

# Or use pyenv
pyenv install 3.11.0
pyenv local 3.11.0
```

#### Permission Errors
```bash
# Fix ownership
sudo chown -R $USER:$USER .

# Fix permissions
chmod -R 755 scripts/
chmod 600 .env
```

#### Missing Dependencies
```bash
# Install system dependencies
# macOS
brew install postgresql libpq

# Ubuntu
sudo apt install python3-dev libpq-dev

# Reinstall Python packages
pip install --force-reinstall -r requirements.txt
```

#### Database Connection Failed
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql  # Linux
brew services list | grep postgres # macOS

# Check connection parameters
psql -h localhost -U ai_trader -d postgres -c "\l"

# Reset password if needed
sudo -u postgres psql -c "ALTER USER ai_trader PASSWORD 'new_password';"
```

---

## 📚 Additional Resources

### Documentation
- Project README: `README.md`
- Architecture: `docs/architecture/`
- API Documentation: `docs/api/`
- Database Schema: `docs/database/`

### Support Commands
```bash
# Get help
python ai_trader.py --help

# Check version
python ai_trader.py --version

# System info
python scripts/system_info.py
```

### Community & Support
- GitHub Issues: [Report bugs and request features]
- Documentation: [Comprehensive guides]
- Discord/Slack: [Community support]

---

## ✅ Setup Complete!

Once setup is complete, you should be able to:

1. **Run the application**: `python ai_trader.py status`
2. **Access the database**: `psql -h localhost -U ai_trader -d ai_trader`
3. **View logs**: `tail -f logs/ai_trader.log`
4. **Start paper trading**: `python ai_trader.py trade --mode paper`

---

*Last Updated: 2025-08-08*
*Version: 1.0*
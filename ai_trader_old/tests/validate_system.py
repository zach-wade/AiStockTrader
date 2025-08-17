#!/usr/bin/env python3
"""
System Validation Script for AI Trading System
Checks if all components are properly configured and ready for production.

Run: python validate_system.py
"""

# Standard library imports
from datetime import datetime
import os
from pathlib import Path
import sys

# Third-party imports
import psycopg2
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
# Third-party imports
from test_setup import setup_test_path

setup_test_path()

# Track validation results
validation_results = {"passed": [], "warnings": [], "failures": []}


def check_pass(component, message):
    """Mark a check as passed."""
    validation_results["passed"].append(f"{component}: {message}")
    print(f"✅ {component}: {message}")


def check_warn(component, message):
    """Mark a check as warning."""
    validation_results["warnings"].append(f"{component}: {message}")
    print(f"⚠️  {component}: {message}")


def check_fail(component, message):
    """Mark a check as failed."""
    validation_results["failures"].append(f"{component}: {message}")
    print(f"❌ {component}: {message}")


def validate_config():
    """Validate configuration files."""
    print("\n🔍 Validating Configuration...")

    config_path = "config/unified_config.yaml"

    # Check if config exists
    if not os.path.exists(config_path):
        check_fail("Config", f"Configuration file not found: {config_path}")
        return

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ["database", "api_keys", "universe", "risk", "trading"]
        for section in required_sections:
            if section in config:
                check_pass("Config", f"Section '{section}' found")
            else:
                check_fail("Config", f"Missing required section: {section}")

        # Check API keys
        if "api_keys" in config:
            for service in ["alpaca", "polygon", "benzinga"]:
                if service in config["api_keys"] and config["api_keys"][service].get("key"):
                    check_pass("API Keys", f"{service.capitalize()} key configured")
                else:
                    check_warn("API Keys", f"{service.capitalize()} key not configured")

    except Exception as e:
        check_fail("Config", f"Error loading config: {e}")


def validate_database():
    """Validate database connection and schema."""
    print("\n🔍 Validating Database...")

    try:
        # Try to load config for database settings
        with open("config/unified_config.yaml") as f:
            config = yaml.safe_load(f)

        db_config = config.get("database", {})

        # Check if using environment variables
        host = os.getenv("DB_HOST", db_config.get("host", "localhost"))
        port = os.getenv("DB_PORT", db_config.get("port", 5432))
        name = os.getenv("DB_NAME", db_config.get("name", "ai_trader"))
        user = os.getenv("DB_USER", db_config.get("user"))
        password = os.getenv("DB_PASSWORD", db_config.get("password"))

        if not all([user, password]):
            check_warn("Database", "Database credentials not fully configured")
            return

        # Try to connect
        conn_string = f"host={host} port={port} dbname={name} user={user} password={password}"
        conn = psycopg2.connect(conn_string)
        conn.close()

        check_pass("Database", "Connection successful")

        # Check for required tables
        required_tables = ["market_data", "news_articles", "features", "trades"]

        conn = psycopg2.connect(conn_string)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """
        )

        existing_tables = [row[0] for row in cur.fetchall()]

        for table in required_tables:
            if table in existing_tables:
                check_pass("Database", f"Table '{table}' exists")
            else:
                check_fail("Database", f"Missing table: {table}")

        cur.close()
        conn.close()

    except psycopg2.OperationalError as e:
        check_fail("Database", f"Cannot connect to database: {e}")
    except Exception as e:
        check_fail("Database", f"Database validation error: {e}")


def validate_file_structure():
    """Validate required directories and files exist."""
    print("\n🔍 Validating File Structure...")

    required_dirs = [
        "config",
        "data_pipeline",
        "feature_pipeline",
        "models",
        "trading_engine",
        "risk_management",
        "monitoring",
        "orchestration",
        "logs",
        "data",
    ]

    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            check_pass("Directory", f"{dir_name}/ exists")
        else:
            check_fail("Directory", f"{dir_name}/ missing")

    # Check for key files
    key_files = [
        "orchestration/main_orchestrator.py",
        "config/config_manager.py",
        "data_pipeline/orchestrator.py",
        "models/strategies/ensemble.py",
    ]

    for file_path in key_files:
        if os.path.isfile(file_path):
            check_pass("File", f"{file_path} exists")
        else:
            check_fail("File", f"{file_path} missing")


def validate_dependencies():
    """Validate Python dependencies."""
    print("\n🔍 Validating Dependencies...")

    required_packages = [
        "pandas",
        "numpy",
        "sqlalchemy",
        "asyncio",
        "aiohttp",
        "psycopg2",
        "pyyaml",
        "scikit-learn",
    ]

    for package in required_packages:
        try:
            __import__(package)
            check_pass("Package", f"{package} installed")
        except ImportError:
            check_fail("Package", f"{package} not installed")


def validate_models():
    """Validate ML models and strategies."""
    print("\n🔍 Validating Models...")

    # Check if model directories exist
    model_dirs = ["models/strategies", "models/ml", "models/saved"]

    for dir_path in model_dirs:
        if os.path.isdir(dir_path):
            check_pass("Models", f"{dir_path}/ exists")
        else:
            check_warn("Models", f"{dir_path}/ missing")

    # Check for strategy files
    strategy_files = [
        "models/strategies/mean_reversion.py",
        "models/strategies/ml_momentum.py",
        "models/strategies/ensemble.py",
    ]

    for file_path in strategy_files:
        if os.path.isfile(file_path):
            check_pass("Strategy", f"{os.path.basename(file_path)} found")
        else:
            check_fail("Strategy", f"{os.path.basename(file_path)} missing")


def validate_permissions():
    """Validate file permissions."""
    print("\n🔍 Validating Permissions...")

    # Check write permissions for key directories
    writable_dirs = ["logs", "data", "models/saved"]

    for dir_path in writable_dirs:
        if os.path.isdir(dir_path):
            test_file = os.path.join(dir_path, ".write_test")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                check_pass("Permissions", f"{dir_path}/ is writable")
            except Exception:
                check_fail("Permissions", f"{dir_path}/ not writable")
        else:
            # Try to create the directory
            try:
                os.makedirs(dir_path, exist_ok=True)
                check_pass("Permissions", f"Created {dir_path}/")
            except Exception:
                check_fail("Permissions", f"Cannot create {dir_path}/")


def generate_report():
    """Generate validation report."""
    print("\n" + "=" * 60)
    print("📊 SYSTEM VALIDATION REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ Passed: {len(validation_results['passed'])}")
    print(f"⚠️  Warnings: {len(validation_results['warnings'])}")
    print(f"❌ Failed: {len(validation_results['failures'])}")

    if validation_results["failures"]:
        print("\n🔴 SYSTEM NOT READY FOR PRODUCTION")
        print("\nCritical Issues:")
        for failure in validation_results["failures"]:
            print(f"  - {failure}")
    elif validation_results["warnings"]:
        print("\n🟡 SYSTEM READY WITH WARNINGS")
        print("\nWarnings to address:")
        for warning in validation_results["warnings"]:
            print(f"  - {warning}")
    else:
        print("\n🟢 SYSTEM READY FOR PRODUCTION")

    # Save report
    report_path = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as f:
        f.write("System Validation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")

        f.write(f"Passed ({len(validation_results['passed'])}):\n")
        for item in validation_results["passed"]:
            f.write(f"  ✓ {item}\n")

        f.write(f"\nWarnings ({len(validation_results['warnings'])}):\n")
        for item in validation_results["warnings"]:
            f.write(f"  ⚠ {item}\n")

        f.write(f"\nFailed ({len(validation_results['failures'])}):\n")
        for item in validation_results["failures"]:
            f.write(f"  ✗ {item}\n")

    print(f"\nReport saved to: {report_path}")
    print("=" * 60)


def main():
    """Run all validation checks."""
    print("🚀 AI Trading System Validation")
    print("Checking if system is ready for production...")

    # Run all validations
    validate_config()
    validate_database()
    validate_file_structure()
    validate_dependencies()
    validate_models()
    validate_permissions()

    # Generate report
    generate_report()

    # Return exit code
    return 0 if not validation_results["failures"] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

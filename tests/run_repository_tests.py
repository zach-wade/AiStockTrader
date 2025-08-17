#!/usr/bin/env python3
"""
Repository Test Runner

Comprehensive test runner for the repository layer tests.
Supports different test categories and provides detailed reporting.
"""

# Standard library imports
import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    # Third-party imports
    import pytest
except ImportError:
    print("pytest is required to run tests. Install with: pip install pytest pytest-asyncio")
    sys.exit(1)


class RepositoryTestRunner:
    """Test runner for repository layer tests."""

    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.root_dir = self.test_dir.parent

    def get_test_categories(self) -> dict:
        """Get available test categories."""
        return {
            "unit": {
                "description": "Unit tests for repository layer components",
                "paths": [
                    "tests/unit/application/interfaces/",
                    "tests/unit/infrastructure/database/",
                    "tests/unit/infrastructure/repositories/",
                ],
                "markers": ["unit"],
            },
            "integration": {
                "description": "Integration tests with real database",
                "paths": ["tests/integration/repositories/"],
                "markers": ["integration"],
                "env_vars": {"RUN_INTEGRATION_TESTS": "true"},
            },
            "interfaces": {
                "description": "Repository interface contract tests",
                "paths": ["tests/unit/application/interfaces/"],
                "markers": ["unit"],
            },
            "database": {
                "description": "Database infrastructure tests",
                "paths": ["tests/unit/infrastructure/database/"],
                "markers": ["unit"],
            },
            "repositories": {
                "description": "Repository implementation tests",
                "paths": ["tests/unit/infrastructure/repositories/"],
                "markers": ["unit"],
            },
            "transactions": {
                "description": "Transaction behavior tests",
                "paths": ["tests/integration/repositories/test_transaction_integration.py"],
                "markers": ["integration"],
                "env_vars": {"RUN_INTEGRATION_TESTS": "true"},
            },
        }

    def set_environment_variables(self, env_vars: dict | None = None):
        """Set environment variables for tests."""
        if env_vars:
            for key, value in env_vars.items():
                os.environ[key] = value

        # Set default test environment variables
        os.environ.setdefault("TESTING", "true")
        os.environ.setdefault("LOG_LEVEL", "DEBUG")

        # Database configuration for integration tests
        if os.environ.get("RUN_INTEGRATION_TESTS") == "true":
            os.environ.setdefault("TEST_DATABASE_HOST", "localhost")
            os.environ.setdefault("TEST_DATABASE_PORT", "5432")
            os.environ.setdefault("TEST_DATABASE_NAME", "ai_trader_test")
            os.environ.setdefault("TEST_DATABASE_USER", "zachwade")
            os.environ.setdefault("TEST_DATABASE_PASSWORD", "")

    def build_pytest_args(
        self,
        category: str | None = None,
        verbose: bool = False,
        coverage: bool = False,
        fail_fast: bool = False,
        test_filter: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build pytest command arguments."""
        args = []

        # Add test paths
        if category:
            categories = self.get_test_categories()
            if category in categories:
                for path in categories[category]["paths"]:
                    full_path = self.root_dir / path
                    if full_path.exists():
                        args.append(str(full_path))

                # Add markers if specified
                markers = categories[category].get("markers", [])
                if markers:
                    args.extend(["-m", " and ".join(markers)])
        else:
            # Run all repository tests
            args.append(str(self.test_dir))

        # Verbosity
        if verbose:
            args.append("-v")
        else:
            args.append("-q")

        # Coverage
        if coverage:
            args.extend(
                [
                    "--cov=src",
                    "--cov-report=html:htmlcov",
                    "--cov-report=xml:coverage.xml",
                    "--cov-report=term-missing",
                ]
            )

        # Fail fast
        if fail_fast:
            args.append("-x")

        # Test filter
        if test_filter:
            args.extend(["-k", test_filter])

        # Async support
        args.append("--asyncio-mode=auto")

        # Add extra args
        if extra_args:
            args.extend(extra_args)

        return args

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        required_packages = ["pytest", "pytest-asyncio"]
        optional_packages = ["pytest-cov"]

        missing_required = []
        missing_optional = []

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_required.append(package)

        for package in optional_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_optional.append(package)

        if missing_required:
            print(f"Missing required packages: {', '.join(missing_required)}")
            print("Install with: pip install " + " ".join(missing_required))
            return False

        if missing_optional:
            print(f"Missing optional packages: {', '.join(missing_optional)}")
            print("Install with: pip install " + " ".join(missing_optional))

        return True

    def check_database_connection(self) -> bool:
        """Check if database is available for integration tests."""
        if os.environ.get("RUN_INTEGRATION_TESTS") != "true":
            return True

        try:
            # Third-party imports
            import psycopg

            # Local imports
            from src.infrastructure.database.connection import DatabaseConfig

            config = DatabaseConfig(
                host=os.environ.get("TEST_DATABASE_HOST", "localhost"),
                port=int(os.environ.get("TEST_DATABASE_PORT", "5432")),
                database=os.environ.get("TEST_DATABASE_NAME", "ai_trader_test"),
                user=os.environ.get("TEST_DATABASE_USER", "zachwade"),
                password=os.environ.get("TEST_DATABASE_PASSWORD", ""),
            )

            # Quick connection test
            conn = psycopg.connect(config.connection_string)
            conn.close()
            return True

        except Exception as e:
            print(f"Database connection failed: {e}")
            print("Integration tests will be skipped")
            return False

    def run_tests(
        self,
        category: str | None = None,
        verbose: bool = False,
        coverage: bool = False,
        fail_fast: bool = False,
        test_filter: str | None = None,
        extra_args: list[str] | None = None,
    ) -> int:
        """Run repository tests."""
        print("🧪 Repository Layer Test Runner")
        print("=" * 50)

        # Check dependencies
        if not self.check_dependencies():
            return 1

        # Set environment variables
        categories = self.get_test_categories()
        if category and category in categories:
            env_vars = categories[category].get("env_vars", {})
            self.set_environment_variables(env_vars)
        else:
            self.set_environment_variables()

        # Check database connection for integration tests
        if not self.check_database_connection() and category in ["integration", "transactions"]:
            print("❌ Cannot run integration tests without database connection")
            return 1

        # Build pytest arguments
        pytest_args = self.build_pytest_args(
            category=category,
            verbose=verbose,
            coverage=coverage,
            fail_fast=fail_fast,
            test_filter=test_filter,
            extra_args=extra_args,
        )

        # Display test configuration
        print(f"Category: {category or 'all'}")
        print(f"Arguments: {' '.join(pytest_args)}")
        print()

        # Run tests
        exit_code = pytest.main(pytest_args)

        # Display results
        if exit_code == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code {exit_code}")

        return exit_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run repository layer tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  unit         - All unit tests for repository layer
  integration  - Integration tests with real database
  interfaces   - Repository interface contract tests
  database     - Database infrastructure tests
  repositories - Repository implementation tests
  transactions - Transaction behavior tests

Examples:
  python run_repository_tests.py                    # Run all tests
  python run_repository_tests.py --category unit    # Run unit tests only
  python run_repository_tests.py --category integration --verbose
  python run_repository_tests.py --filter "test_order" --coverage
        """,
    )

    parser.add_argument(
        "--category",
        "-c",
        choices=["unit", "integration", "interfaces", "database", "repositories", "transactions"],
        help="Test category to run",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")

    parser.add_argument("--fail-fast", "-x", action="store_true", help="Stop on first failure")

    parser.add_argument("--filter", "-k", help="Filter tests by name pattern")

    parser.add_argument(
        "--list-categories", action="store_true", help="List available test categories"
    )

    parser.add_argument("pytest_args", nargs="*", help="Additional arguments to pass to pytest")

    args = parser.parse_args()

    runner = RepositoryTestRunner()

    # List categories if requested
    if args.list_categories:
        categories = runner.get_test_categories()
        print("Available test categories:")
        print()
        for name, info in categories.items():
            print(f"  {name:12} - {info['description']}")
        return 0

    # Run tests
    exit_code = runner.run_tests(
        category=args.category,
        verbose=args.verbose,
        coverage=args.coverage,
        fail_fast=args.fail_fast,
        test_filter=args.filter,
        extra_args=args.pytest_args,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

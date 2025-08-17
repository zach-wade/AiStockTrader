#!/usr/bin/env python3
"""
Quick test runner to verify test setup and run basic tests.
"""

# Standard library imports
import os
from pathlib import Path
import subprocess
import sys

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        # Third-party imports
        import pytest
    except ImportError:
        missing.append("pytest")

    try:
        # Third-party imports
        import pytest_cov
    except ImportError:
        missing.append("pytest-cov")

    try:
        # Third-party imports
        import coverage
    except ImportError:
        missing.append("coverage")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    return True


def run_simple_test():
    """Run a simple test to verify setup."""
    print("Running simple infrastructure test...")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(project_root / "tests" / "integration" / "test_infrastructure.py"),
        "-v",
        "--tb=short",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path)

    result = subprocess.run(cmd, env=env, check=False)
    return result.returncode == 0


def run_coverage_test():
    """Run tests with coverage."""
    print("\nRunning tests with coverage...")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        f"--cov={src_path / 'main'}",
        "--cov-report=xml",
        "--cov-report=term",
        str(project_root / "tests"),
        "-v",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path)

    result = subprocess.run(cmd, env=env, check=False)
    return result.returncode == 0


def main():
    """Main entry point."""
    print("AI Trader Test Quick Check")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return 1

    print("✅ All dependencies installed")

    # Run simple test
    if not run_simple_test():
        print("\n❌ Basic test failed - check your setup")
        return 1

    print("\n✅ Basic test passed")

    # Ask if user wants to run full coverage
    response = input("\nRun full test suite with coverage? (y/n): ").lower()

    if response == "y":
        if run_coverage_test():
            print("\n✅ Coverage test completed")
            print(f"View coverage report: {project_root}/htmlcov/index.html")

            # Generate report
            print("\nGenerating coverage summary...")
            subprocess.run(
                [sys.executable, str(project_root / "tests" / "coverage_report.py")], check=False
            )
        else:
            print("\n❌ Coverage test failed")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

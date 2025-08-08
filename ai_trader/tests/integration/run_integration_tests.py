#!/usr/bin/env python3
"""
Integration Test Runner

Runs integration tests with proper configuration and reporting.
"""

import os
import sys
from pathlib import Path
from pathlib import Path
import argparse
import subprocess
from pathlib import Path


def run_integration_tests(
    verbose: bool = False,
    specific_test: str = None,
    markers: str = None,
    coverage: bool = False
):
    """Run integration tests with pytest."""
    # Build pytest command
    cmd = ["pytest", "tests/integration"]
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add specific test if provided
    if specific_test:
        cmd.append(specific_test)
    
    # Add markers
    if markers:
        cmd.extend(["-m", markers])
    else:
        # Default to integration tests only
        cmd.extend(["-m", "integration"])
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            "--cov=src/main",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add other useful flags
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Ensure all markers are registered
        "-p", "no:warnings",  # Disable warnings for cleaner output
    ])
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent / "src")
    env["AI_TRADER_ENV"] = "test"
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    
    # Run tests
    result = subprocess.run(cmd, env=env)
    
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run AI Trader integration tests"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-t", "--test",
        help="Run specific test file or test function"
    )
    
    parser.add_argument(
        "-m", "--markers",
        help="Run tests with specific markers (e.g., 'slow', 'requires_db')"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including slow ones"
    )
    
    args = parser.parse_args()
    
    # Override markers if --all is specified
    if args.all:
        args.markers = "integration or slow"
    
    # Run tests
    exit_code = run_integration_tests(
        verbose=args.verbose,
        specific_test=args.test,
        markers=args.markers,
        coverage=args.coverage
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Run all data pipeline tests with coverage reporting.

This script runs both unit and integration tests for the data pipeline
module and generates a coverage report.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


class DataPipelineTestRunner:
    """Test runner for data pipeline tests."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src" / "main" / "data_pipeline"
        
        # Ensure we're in the right directory
        os.chdir(self.project_root)
        
        # Add src to Python path
        sys.path.insert(0, str(self.project_root / "src"))
    
    def run_unit_tests(self, verbose=False):
        """Run data pipeline unit tests."""
        print("\n" + "="*60)
        print("Running Data Pipeline Unit Tests")
        print("="*60)
        
        cmd = [
            "pytest",
            str(self.tests_dir / "unit" / "data_pipeline"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--cov=main.data_pipeline",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_html/data_pipeline",
            "-x"  # Stop on first failure
        ]
        
        result = subprocess.run(cmd)
        return result.returncode
    
    def run_integration_tests(self, verbose=False):
        """Run data pipeline integration tests."""
        print("\n" + "="*60)
        print("Running Data Pipeline Integration Tests")
        print("="*60)
        
        cmd = [
            "pytest",
            str(self.tests_dir / "integration" / "data_pipeline"),
            "-v" if verbose else "-q",
            "--tb=short",
            "--cov=main.data_pipeline",
            "--cov-append",  # Append to existing coverage
            "--cov-report=term-missing",
            "--cov-report=html:coverage_html/data_pipeline",
            "-x"
        ]
        
        result = subprocess.run(cmd)
        return result.returncode
    
    def run_specific_test(self, test_path, verbose=False):
        """Run a specific test file or test case."""
        print(f"\n{'='*60}")
        print(f"Running: {test_path}")
        print("="*60)
        
        cmd = [
            "pytest",
            test_path,
            "-v" if verbose else "-q",
            "--tb=short",
            "-x"
        ]
        
        result = subprocess.run(cmd)
        return result.returncode
    
    def generate_coverage_report(self):
        """Generate detailed coverage report."""
        print("\n" + "="*60)
        print("Generating Coverage Report")
        print("="*60)
        
        cmd = [
            "coverage",
            "report",
            "--include=src/main/data_pipeline/*",
            "--show-missing"
        ]
        
        subprocess.run(cmd)
        
        # Also generate XML for CI/CD
        subprocess.run([
            "coverage",
            "xml",
            "-o",
            "coverage.xml",
            "--include=src/main/data_pipeline/*"
        ])
        
        print(f"\nHTML coverage report available at: {self.project_root}/coverage_html/data_pipeline/index.html")
    
    def list_tests(self):
        """List all available data pipeline tests."""
        print("\n" + "="*60)
        print("Available Data Pipeline Tests")
        print("="*60)
        
        # Unit tests
        print("\nUnit Tests:")
        unit_dir = self.tests_dir / "unit" / "data_pipeline"
        if unit_dir.exists():
            for test_file in unit_dir.rglob("test_*.py"):
                print(f"  - {test_file.relative_to(self.tests_dir)}")
        
        # Integration tests
        print("\nIntegration Tests:")
        integration_dir = self.tests_dir / "integration" / "data_pipeline"
        if integration_dir.exists():
            for test_file in integration_dir.rglob("test_*.py"):
                print(f"  - {test_file.relative_to(self.tests_dir)}")
    
    def run_all(self, verbose=False):
        """Run all data pipeline tests."""
        print(f"\nStarting Data Pipeline Test Suite - {datetime.now()}")
        
        # Run unit tests
        unit_result = self.run_unit_tests(verbose)
        if unit_result != 0:
            print("\n❌ Unit tests failed!")
            return unit_result
        
        # Run integration tests
        integration_result = self.run_integration_tests(verbose)
        if integration_result != 0:
            print("\n❌ Integration tests failed!")
            return integration_result
        
        # Generate coverage report
        self.generate_coverage_report()
        
        print("\n✅ All data pipeline tests passed!")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run data pipeline tests")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "-u", "--unit-only",
        action="store_true",
        help="Run only unit tests"
    )
    parser.add_argument(
        "-i", "--integration-only",
        action="store_true",
        help="Run only integration tests"
    )
    parser.add_argument(
        "-t", "--test",
        type=str,
        help="Run specific test file"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available tests"
    )
    
    args = parser.parse_args()
    runner = DataPipelineTestRunner()
    
    if args.list:
        runner.list_tests()
        return 0
    
    if args.test:
        return runner.run_specific_test(args.test, args.verbose)
    
    if args.unit_only:
        result = runner.run_unit_tests(args.verbose)
        runner.generate_coverage_report()
        return result
    
    if args.integration_only:
        result = runner.run_integration_tests(args.verbose)
        runner.generate_coverage_report()
        return result
    
    # Run all tests
    return runner.run_all(args.verbose)


if __name__ == "__main__":
    sys.exit(main())
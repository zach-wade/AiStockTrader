#!/usr/bin/env python3
"""
Setup test environment for AI Trader.

This script ensures all test dependencies are installed and the environment
is properly configured for running tests.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Tuple, Optional


class TestEnvironmentSetup:
    """Manages test environment setup and validation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.tests_dir = project_root / "tests"
        self.requirements_file = project_root / "requirements.txt"
        
    def setup_environment(self) -> bool:
        """Setup complete test environment."""
        print("🚀 Setting up AI Trader test environment...")
        print("=" * 60)
        
        success = True
        
        # Check Python version
        if not self._check_python_version():
            success = False
        
        # Setup Python path
        self._setup_python_path()
        
        # Install dependencies
        if not self._install_test_dependencies():
            success = False
        
        # Create necessary directories
        self._create_directories()
        
        # Validate test configuration
        if not self._validate_test_config():
            success = False
        
        # Check optional dependencies
        self._check_optional_dependencies()
        
        # Setup environment variables
        self._setup_environment_variables()
        
        if success:
            print("\n✅ Test environment setup completed successfully!")
            self._print_quick_start_guide()
        else:
            print("\n❌ Test environment setup failed!")
            
        return success
    
    def _check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        print("\n📋 Checking Python version...")
        
        version = sys.version_info
        min_version = (3, 8)
        
        if version >= min_version:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
            return True
        else:
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires >= {min_version[0]}.{min_version[1]})")
            return False
    
    def _setup_python_path(self):
        """Setup Python path for imports."""
        print("\n📂 Setting up Python path...")
        
        src_path = str(self.src_dir)
        
        # Add to PYTHONPATH environment variable
        current_path = os.environ.get("PYTHONPATH", "")
        if src_path not in current_path:
            if current_path:
                os.environ["PYTHONPATH"] = f"{src_path}:{current_path}"
            else:
                os.environ["PYTHONPATH"] = src_path
        
        # Add to sys.path for current session
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        print(f"✅ Added {src_path} to Python path")
    
    def _install_test_dependencies(self) -> bool:
        """Install required test dependencies."""
        print("\n📦 Installing test dependencies...")
        
        # Core test dependencies
        test_packages = [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0", 
            "pytest-cov>=4.1.0",
            "coverage[toml]>=7.3.0",
            "pytest-xdist>=3.3.0",  # For parallel testing
            "pytest-timeout>=2.1.0",  # For test timeouts
        ]
        
        success = True
        
        for package in test_packages:
            if not self._install_package(package):
                success = False
        
        # Install project requirements if available
        if self.requirements_file.exists():
            print(f"\n📋 Installing project requirements from {self.requirements_file}")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Project requirements installed")
            else:
                print(f"⚠️  Warning: Could not install all project requirements")
                print(f"Error: {result.stderr}")
        
        return success
    
    def _install_package(self, package: str) -> bool:
        """Install a single package."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ Installed {package}")
                return True
            else:
                print(f"❌ Failed to install {package}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout installing {package}")
            return False
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
            return False
    
    def _create_directories(self):
        """Create necessary test directories."""
        print("\n📁 Creating test directories...")
        
        directories = [
            self.tests_dir / "logs",
            self.tests_dir / "reports",
            self.tests_dir / "tmp",
            self.project_root / "htmlcov"  # Coverage reports
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {directory}")
    
    def _validate_test_config(self) -> bool:
        """Validate test configuration files."""
        print("\n⚙️  Validating test configuration...")
        
        required_files = [
            self.tests_dir / "pytest.ini",
            self.tests_dir / "conftest.py"
        ]
        
        success = True
        
        for file_path in required_files:
            if file_path.exists():
                print(f"✅ Found {file_path.name}")
            else:
                print(f"❌ Missing {file_path.name}")
                success = False
        
        # Test pytest configuration
        try:
            result = subprocess.run([
                "pytest", "--collect-only", "-q"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=30)
            
            if result.returncode == 0:
                print("✅ Pytest configuration valid")
            else:
                print(f"⚠️  Pytest configuration warning: {result.stderr}")
                
        except Exception as e:
            print(f"⚠️  Could not validate pytest config: {e}")
        
        return success
    
    def _check_optional_dependencies(self):
        """Check for optional but useful dependencies."""
        print("\n🔍 Checking optional dependencies...")
        
        optional_packages = {
            "psutil": "System monitoring for performance tests",
            "pytest-html": "HTML test reports",
            "pytest-benchmark": "Performance benchmarking",
            "pytest-mock": "Enhanced mocking capabilities"
        }
        
        for package, description in optional_packages.items():
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {package} - {description}")
            except ImportError:
                print(f"⚪ {package} - {description} (optional, not installed)")
    
    def _setup_environment_variables(self):
        """Setup test environment variables."""
        print("\n🌍 Setting up environment variables...")
        
        test_env_vars = {
            "AI_TRADER_ENV": "test",
            "TESTING": "1",
            "PYTHONPATH": str(self.src_dir),
        }
        
        for var, value in test_env_vars.items():
            os.environ[var] = value
            print(f"✅ Set {var}={value}")
    
    def _print_quick_start_guide(self):
        """Print quick start guide for running tests."""
        print("\n" + "=" * 60)
        print("🎯 QUICK START GUIDE")
        print("=" * 60)
        
        commands = [
            ("Run all tests with coverage", "python tests/run_all_tests.py"),
            ("Run unit tests only", "python tests/run_all_tests.py -t unit"),
            ("Run integration tests only", "python tests/run_all_tests.py -t integration"),
            ("Run events module tests", "pytest tests/unit/events/ tests/integration/events/ -v"),
            ("Run specific test file", "pytest tests/unit/events/test_event_bus.py -v"),
            ("Run with markers", "pytest -m 'unit and events' -v"),
            ("Generate coverage report", "pytest --cov=src/main --cov-report=html"),
            ("Run tests in parallel", "pytest -n auto"),
        ]
        
        for description, command in commands:
            print(f"\n📌 {description}:")
            print(f"   {command}")
        
        print(f"\n📊 Coverage reports will be available in: {self.project_root}/htmlcov/index.html")
        print(f"📋 Test logs will be saved in: {self.tests_dir}/logs/")
        print("\n🔗 More options: python tests/run_all_tests.py --help")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup AI Trader test environment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check environment, don't install packages"
    )
    
    parser.add_argument(
        "--minimal",
        action="store_true", 
        help="Install only core test dependencies"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    setup = TestEnvironmentSetup(project_root)
    
    try:
        if args.check_only:
            # Only validation, no installation
            print("🔍 Checking test environment (no installation)...")
            success = (setup._check_python_version() and 
                      setup._validate_test_config())
        else:
            # Full setup
            success = setup.setup_environment()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
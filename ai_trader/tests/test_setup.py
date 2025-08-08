"""Standardized test setup utilities for consistent import handling."""

import sys
from pathlib import Path
from pathlib import Path
import os
from pathlib import Path


def setup_test_path():
    """Add the project root to sys.path for test imports.
    
    This function should be called at the beginning of test files
    to ensure consistent import behavior across all tests.
    """
    # Get the absolute path to the tests directory
    tests_dir = Path(__file__).parent.absolute()
    
    # Get the project root (parent of tests directory)
    project_root = tests_dir.parent
    
    # Add project root to sys.path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root


def get_project_root():
    """Get the absolute path to the project root directory."""
    return Path(__file__).parent.parent.absolute()


def get_test_config_path():
    """Get the path to the test configuration file."""
    return get_project_root() / "tests" / "integration" / "test_config.yaml"


# Automatically setup path when this module is imported
PROJECT_ROOT = setup_test_path()
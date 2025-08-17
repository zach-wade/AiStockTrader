#!/bin/bash

# AI Trading System - Code Inventory Script
# Generates detailed statistics for each module

echo "AI Trading System - Code Inventory"
echo "==================================="
echo ""

BASE_DIR="src/main"

# Function to analyze a module
analyze_module() {
    local module=$1
    local path="$BASE_DIR/$module"

    if [ -d "$path" ]; then
        echo "Module: $module"
        echo "Path: $path"

        # Count Python files
        py_files=$(find "$path" -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "Python files: $py_files"

        # Count lines of code (excluding blank lines and comments)
        if [ "$py_files" -gt 0 ]; then
            total_lines=$(find "$path" -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
            echo "Total lines: $total_lines"

            # Count test files
            test_files=$(find "$path" -name "test_*.py" -o -name "*_test.py" 2>/dev/null | wc -l | tr -d ' ')
            echo "Test files: $test_files"
        else
            echo "Total lines: 0"
            echo "Test files: 0"
        fi

        # Check for __init__.py
        if [ -f "$path/__init__.py" ]; then
            echo "Has __init__.py: Yes"
        else
            echo "Has __init__.py: No"
        fi

        # Count subdirectories
        subdirs=$(find "$path" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
        echo "Subdirectories: $subdirs"

        echo "---"
        echo ""
    fi
}

# Analyze each module
modules=(
    "app"
    "backtesting"
    "config"
    "core"
    "data_pipeline"
    "events"
    "feature_pipeline"
    "features"
    "interfaces"
    "jobs"
    "migrations"
    "models"
    "monitoring"
    "orchestration"
    "risk_management"
    "scanners"
    "services"
    "trading_engine"
    "universe"
    "utils"
)

for module in "${modules[@]}"; do
    analyze_module "$module"
done

# Overall statistics
echo "==================================="
echo "Overall Statistics"
echo "==================================="
total_py=$(find "$BASE_DIR" -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
total_lines=$(find "$BASE_DIR" -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
total_tests=$(find "$BASE_DIR" -name "test_*.py" -o -name "*_test.py" 2>/dev/null | wc -l | tr -d ' ')

echo "Total Python files: $total_py"
echo "Total lines of code: $total_lines"
echo "Total test files: $total_tests"

# Check for common issues
echo ""
echo "==================================="
echo "Common Issues Check"
echo "==================================="

# Find files with more than 500 lines
echo "Large files (>500 lines):"
find "$BASE_DIR" -name "*.py" -type f -exec wc -l {} + 2>/dev/null | awk '$1 > 500 {print $2 " (" $1 " lines)"}' | head -10

echo ""
echo "Files without docstrings (sampling):"
find "$BASE_DIR" -name "*.py" -type f -exec grep -L '"""' {} \; 2>/dev/null | head -5

echo ""
echo "Potential circular imports (files importing from parent):"
find "$BASE_DIR" -name "*.py" -type f -exec grep -l "from \.\." {} \; 2>/dev/null | head -5

echo ""
echo "==================================="
echo "Inventory complete!"

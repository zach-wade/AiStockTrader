#!/bin/bash

# AI Trading System - Complete Code Inventory Script V2
# Generates detailed statistics for the entire project structure

echo "AI Trading System - Complete Code Inventory V2"
echo "==============================================="
echo ""

# Function to analyze a directory
analyze_directory() {
    local dir=$1
    local name=$2

    if [ -d "$dir" ]; then
        echo "### $name"
        echo "Path: $dir"

        # Count Python files
        py_files=$(find "$dir" -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "Python files: $py_files"

        # Count lines of code
        if [ "$py_files" -gt 0 ]; then
            total_lines=$(find "$dir" -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
            echo "Total lines: $total_lines"

            # Count test files
            test_files=$(find "$dir" \( -name "test_*.py" -o -name "*_test.py" \) 2>/dev/null | wc -l | tr -d ' ')
            echo "Test files: $test_files"
        else
            echo "Total lines: 0"
            echo "Test files: 0"
        fi

        # Count subdirectories
        subdirs=$(find "$dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
        echo "Subdirectories: $subdirs"

        echo ""
    fi
}

echo "==================================="
echo "PROJECT STRUCTURE OVERVIEW"
echo "==================================="
echo ""

# Main directories
echo "## Main Code (src/main/)"
echo "------------------------"
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

total_main_files=0
total_main_lines=0

for module in "${modules[@]}"; do
    if [ -d "src/main/$module" ]; then
        files=$(find "src/main/$module" -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
        lines=$(find "src/main/$module" -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
        echo "$module: $files files, $lines lines"
        total_main_files=$((total_main_files + files))
        total_main_lines=$((total_main_lines + lines))
    fi
done

echo ""
echo "Main Code Total: $total_main_files files, $total_main_lines lines"
echo ""

echo "## Test Suite (tests/)"
echo "----------------------"
# Analyze test directories
test_dirs=(
    "fixtures"
    "integration"
    "monitoring"
    "performance"
    "unit"
)

total_test_files=0
total_test_lines=0

for dir in "${test_dirs[@]}"; do
    if [ -d "tests/$dir" ]; then
        files=$(find "tests/$dir" -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
        lines=$(find "tests/$dir" -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
        echo "$dir: $files files, $lines lines"
        total_test_files=$((total_test_files + files))
        total_test_lines=$((total_test_lines + lines))
    fi
done

# Count root test files
root_test_files=$(find "tests" -maxdepth 1 -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
root_test_lines=$(find "tests" -maxdepth 1 -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
echo "root tests: $root_test_files files, $root_test_lines lines"
total_test_files=$((total_test_files + root_test_files))
total_test_lines=$((total_test_lines + root_test_lines))

echo ""
echo "Test Suite Total: $total_test_files files, $total_test_lines lines"
echo ""

echo "## Supporting Directories"
echo "------------------------"
# Other important directories
other_dirs=(
    "docs:Documentation"
    "scripts:Utility Scripts"
    "examples:Example Code"
    "config:Configuration Files"
    "data_pipeline:Data Pipeline Scripts"
)

for entry in "${other_dirs[@]}"; do
    dir=$(echo $entry | cut -d: -f1)
    desc=$(echo $entry | cut -d: -f2)
    if [ -d "$dir" ]; then
        py_files=$(find "$dir" -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [ "$py_files" -gt 0 ]; then
            lines=$(find "$dir" -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
            echo "$dir ($desc): $py_files Python files, $lines lines"
        fi

        # Count other file types
        yaml_files=$(find "$dir" -name "*.yaml" -o -name "*.yml" 2>/dev/null | wc -l | tr -d ' ')
        md_files=$(find "$dir" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$yaml_files" -gt 0 ] || [ "$md_files" -gt 0 ]; then
            echo "  - YAML files: $yaml_files, Markdown files: $md_files"
        fi
    fi
done

echo ""
echo "==================================="
echo "OVERALL PROJECT STATISTICS"
echo "==================================="

# Calculate totals
all_py_files=$(find . -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
all_py_lines=$(find . -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
all_test_files=$(find . \( -name "test_*.py" -o -name "*_test.py" \) 2>/dev/null | wc -l | tr -d ' ')

echo "Total Python files: $all_py_files"
echo "Total Python lines: $all_py_lines"
echo "Total test files: $all_test_files"
echo ""
echo "Main code: $total_main_files files ($total_main_lines lines)"
echo "Test code: $total_test_files files ($total_test_lines lines)"

# Calculate test coverage ratio
if [ $total_main_lines -gt 0 ]; then
    coverage_ratio=$(echo "scale=2; $total_test_lines * 100 / $total_main_lines" | bc)
    echo "Test-to-code ratio: ${coverage_ratio}%"
fi

echo ""
echo "==================================="
echo "CODE QUALITY METRICS"
echo "==================================="

# Large files
echo "## Large Files (>500 lines)"
echo "----------------------------"
find src/main -name "*.py" -type f -exec wc -l {} + 2>/dev/null | awk '$1 > 500 {print $2 " (" $1 " lines)"}' | head -10

echo ""
echo "## Very Large Files (>1000 lines)"
echo "----------------------------------"
find src/main -name "*.py" -type f -exec wc -l {} + 2>/dev/null | awk '$1 > 1000 {print $2 " (" $1 " lines)"}'

echo ""
echo "## Potential Issues"
echo "-------------------"

# Files without docstrings
no_docstring=$(find src/main -name "*.py" -type f -exec grep -L '"""' {} \; 2>/dev/null | wc -l | tr -d ' ')
echo "Files without docstrings: $no_docstring"

# Circular imports
circular=$(find src/main -name "*.py" -type f -exec grep -l "from \.\." {} \; 2>/dev/null | wc -l | tr -d ' ')
echo "Files with potential circular imports: $circular"

# TODO comments
todos=$(grep -r "TODO" src/main --include="*.py" 2>/dev/null | wc -l | tr -d ' ')
echo "TODO comments: $todos"

# FIXME comments
fixmes=$(grep -r "FIXME" src/main --include="*.py" 2>/dev/null | wc -l | tr -d ' ')
echo "FIXME comments: $fixmes"

echo ""
echo "==================================="
echo "Inventory complete!"
echo "Generated: $(date)"
echo "===================================="

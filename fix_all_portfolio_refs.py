#!/usr/bin/env python3
"""Fix all references to portfolio methods in tests."""

import re
from pathlib import Path


def fix_test_file(file_path: Path) -> None:
    """Fix a test file to use new service architecture."""
    
    content = file_path.read_text()
    
    # Add service initialization for each test method
    lines = content.split('\n')
    new_lines = []
    in_test_method = False
    service_added = False
    
    for i, line in enumerate(lines):
        # Check if we're starting a test method
        if re.match(r'    def test_\w+\(self', line):
            in_test_method = True
            service_added = False
            new_lines.append(line)
        # Add service initialization after docstring if needed
        elif in_test_method and not service_added and (line.strip().startswith('"""') and i > 0 and lines[i-1].strip().endswith('"""')):
            new_lines.append(line)
            # Check if next lines use portfolio operations
            upcoming_content = '\n'.join(lines[i+1:i+20])
            if 'service.open_position' in upcoming_content or 'service.close_position' in upcoming_content:
                if 'service = PortfolioService()' not in upcoming_content:
                    new_lines.append('        service = PortfolioService()')
            service_added = True
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    # Fix method calls that still reference PortfolioMetricsCalculator
    content = re.sub(
        r'PortfolioMetricsCalculator\.(\w+)',
        r'PortfolioCalculator.\1',
        content
    )
    
    # Save the file
    file_path.write_text(content)
    print(f"Fixed: {file_path}")


def main():
    """Fix all test files."""
    test_files = Path("/Users/zachwade/StockMonitoring/tests").rglob("*.py")
    
    for test_file in test_files:
        if "portfolio" in test_file.name.lower() or "Portfolio" in test_file.read_text():
            try:
                fix_test_file(test_file)
            except Exception as e:
                print(f"Error fixing {test_file}: {e}")
    
    print("\nAll test files updated!")


if __name__ == "__main__":
    main()
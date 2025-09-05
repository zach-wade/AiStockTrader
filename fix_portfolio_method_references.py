#!/usr/bin/env python3
"""Fix all references to old Portfolio methods that were removed."""

import os
import re
from pathlib import Path

# Define the replacements
REPLACEMENTS = [
    # Replace portfolio.get_total_value() with PortfolioCalculator.get_total_value(portfolio)
    (
        r'(\s*)(.+?)\.get_total_value\(\)',
        lambda m: f"{m.group(1)}from src.domain.services.portfolio_calculator import PortfolioCalculator\n{m.group(1)}PortfolioCalculator.get_total_value({m.group(2)})"
    ),
    # Replace portfolio.get_unrealized_pnl() with PortfolioCalculator.get_unrealized_pnl(portfolio)
    (
        r'(\s*)(.+?)\.get_unrealized_pnl\(\)',
        lambda m: f"{m.group(1)}from src.domain.services.portfolio_calculator import PortfolioCalculator\n{m.group(1)}PortfolioCalculator.get_unrealized_pnl({m.group(2)})"
    ),
    # Replace portfolio.get_return_percentage() with PortfolioCalculator.get_return_percentage(portfolio)
    (
        r'(\s*)(.+?)\.get_return_percentage\(\)',
        lambda m: f"{m.group(1)}from src.domain.services.portfolio_calculator import PortfolioCalculator\n{m.group(1)}PortfolioCalculator.get_return_percentage({m.group(2)})"
    ),
    # Replace portfolio.to_dict() with PortfolioCalculator.to_dict(portfolio)
    (
        r'(\s*)(.+?)\.to_dict\(\)',
        lambda m: f"{m.group(1)}from src.domain.services.portfolio_calculator import PortfolioCalculator\n{m.group(1)}PortfolioCalculator.to_dict({m.group(2)})"
    ),
]

def fix_file(filepath):
    """Fix a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Apply replacements
    for pattern, replacement in REPLACEMENTS:
        if callable(replacement):
            content = re.sub(pattern, replacement, content)
        else:
            content = re.sub(pattern, replacement, content)
    
    # Only write if changed
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
        return True
    return False

def main():
    """Main function to fix all Python files."""
    src_dir = Path('/Users/zachwade/StockMonitoring/src')
    fixed_count = 0
    
    for filepath in src_dir.rglob('*.py'):
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == '__main__':
    main()
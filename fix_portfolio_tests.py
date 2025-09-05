#!/usr/bin/env python3
"""Script to update portfolio tests to use new service architecture."""

import re
from pathlib import Path


def update_portfolio_tests(file_path: Path) -> None:
    """Update a test file to use the new service architecture."""
    
    content = file_path.read_text()
    
    # Add service initialization if not present
    if "service = PortfolioService()" not in content:
        # Add at the beginning of test methods that use portfolio operations
        patterns = [
            (r"(\s+def test_\w+\(self[^)]*\):.*?\n)(.*?portfolio\.open_position)",
             r"\1        service = PortfolioService()\n\2"),
            (r"(\s+def test_\w+\(self[^)]*\):.*?\n)(.*?portfolio\.close_position)",
             r"\1        service = PortfolioService()\n\2"),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Replace portfolio method calls with service calls
    replacements = [
        # Simple replacements
        (r"portfolio\.open_position\(request\)",
         "service.open_position(portfolio, request)"),
        (r"portfolio\.open_position\((request\d+)\)",
         r"service.open_position(portfolio, \1)"),
        (r"portfolio\.open_position\(PositionRequest\(",
         "service.open_position(portfolio, PositionRequest("),
        
        # Close position with various arguments
        (r"portfolio\.close_position\(\n",
         "service.close_position(portfolio,\n"),
        (r"portfolio\.close_position\(([^)]+)\)",
         r"service.close_position(portfolio, \1)"),
        
        # Can open position
        (r"portfolio\.can_open_position\(([^)]+)\)",
         r"PortfolioValidator.can_open_position(portfolio, \1)"),
        
        # Metric methods
        (r"portfolio\.get_total_value\(\)",
         "PortfolioCalculator.get_total_value(portfolio)"),
        (r"portfolio\.get_positions_value\(\)",
         "PortfolioCalculator.get_positions_value(portfolio)"),
        (r"portfolio\.get_unrealized_pnl\(\)",
         "PortfolioCalculator.get_unrealized_pnl(portfolio)"),
        (r"portfolio\.get_total_pnl\(\)",
         "PortfolioCalculator.get_total_pnl(portfolio)"),
        (r"portfolio\.get_return_percentage\(\)",
         "PortfolioCalculator.get_return_percentage(portfolio)"),
        (r"portfolio\.get_win_rate\(\)",
         "PortfolioCalculator.get_win_rate(portfolio)"),
        (r"portfolio\.get_profit_factor\(\)",
         "PortfolioCalculator.get_profit_factor(portfolio)"),
        (r"portfolio\.get_sharpe_ratio\(([^)]*)\)",
         r"PortfolioCalculator.get_sharpe_ratio(portfolio, \1)"),
        (r"portfolio\.get_max_drawdown\(([^)]*)\)",
         r"PortfolioCalculator.get_max_drawdown(portfolio, \1)"),
        
        # Serialization
        (r"portfolio\.to_dict\(\)",
         "PortfolioCalculator.portfolio_to_dict(portfolio)"),
        (r"str\(portfolio\)",
         "PortfolioCalculator.portfolio_to_string(portfolio)"),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Save the updated file
    file_path.write_text(content)
    print(f"Updated: {file_path}")


def main():
    """Update all portfolio test files."""
    test_files = [
        Path("/Users/zachwade/StockMonitoring/tests/unit/domain/entities/test_portfolio.py"),
    ]
    
    for test_file in test_files:
        if test_file.exists():
            update_portfolio_tests(test_file)
    
    print("\nTest files updated successfully!")


if __name__ == "__main__":
    main()
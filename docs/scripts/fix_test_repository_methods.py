#!/usr/bin/env python3
"""Fix repository method calls in test files to match interface definitions."""

import re
from pathlib import Path

# Mapping of old method names to new ones in tests
REPLACEMENTS = [
    # Order repository
    (r"orders\.get_by_id", "orders.get_order_by_id"),
    (r"order_repo\.get_by_id", "order_repo.get_order_by_id"),
    (r"orders\.save", "orders.save_order"),
    (r"order_repo\.save", "order_repo.save_order"),
    (r"orders\.update", "orders.update_order"),
    (r"order_repo\.update", "order_repo.update_order"),
    # Portfolio repository
    (r"portfolios\.get_by_id", "portfolios.get_portfolio_by_id"),
    (r"portfolio_repo\.get_by_id", "portfolio_repo.get_portfolio_by_id"),
    (r"portfolios\.save", "portfolios.save_portfolio"),
    (r"portfolio_repo\.save", "portfolio_repo.save_portfolio"),
    (r"portfolios\.update", "portfolios.update_portfolio"),
    (r"portfolio_repo\.update", "portfolio_repo.update_portfolio"),
    # Position repository
    (r"positions\.get_by_id", "positions.get_position_by_id"),
    (r"position_repo\.get_by_id", "position_repo.get_position_by_id"),
    (r"positions\.save", "positions.persist_position"),
    (r"position_repo\.save", "position_repo.persist_position"),
    (r"positions\.update", "positions.update_position"),
    (r"position_repo\.update", "position_repo.update_position"),
]


def fix_file(file_path):
    """Fix repository method calls in a file."""
    content = file_path.read_text()
    original = content

    for old_pattern, new_text in REPLACEMENTS:
        content = re.sub(old_pattern, new_text, content)

    if content != original:
        file_path.write_text(content)
        print(f"Fixed: {file_path}")
        return True
    return False


def main():
    """Fix all test files."""
    test_dirs = [
        Path("tests/unit/application/use_cases"),
        Path("tests/unit/application/coordinators"),
    ]

    fixed_count = 0

    for test_dir in test_dirs:
        if test_dir.exists():
            for file_path in test_dir.glob("test_*.py"):
                if fix_file(file_path):
                    fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()

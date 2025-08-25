#!/usr/bin/env python3
"""Fix repository method calls to match interface definitions."""

import re
from pathlib import Path

# Mapping of old method names to new ones
REPLACEMENTS = [
    # Order repository
    (r"order_repo\.get_by_id\(", "order_repo.get_order_by_id("),
    (r"order_repo\.save\(", "order_repo.save_order("),
    (r"order_repo\.update\(", "order_repo.update_order("),
    # Portfolio repository
    (r"portfolio_repo\.get_by_id\(", "portfolio_repo.get_portfolio_by_id("),
    (r"portfolio_repo\.save\(", "portfolio_repo.save_portfolio("),
    (r"portfolio_repo\.update\(", "portfolio_repo.update_portfolio("),
    # Position repository
    (r"position_repo\.get_by_id\(", "position_repo.get_position_by_id("),
    (r"position_repo\.save\(", "position_repo.persist_position("),
    (r"position_repo\.update\(", "position_repo.update_position("),
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
    """Fix all use case files."""
    use_cases_dir = Path("src/application/use_cases")
    fixed_count = 0

    for file_path in use_cases_dir.glob("*.py"):
        if fix_file(file_path):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()

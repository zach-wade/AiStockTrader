"""
Basic Configuration Example

This example shows how to create and use a basic configuration
for the AI Trader system.
"""

import asyncio
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from main.config.config_manager import get_config
from main.config.config_validator import ConfigValidator
from main.utils.logging import setup_logging


async def main():
    """Basic configuration example."""
    
    # Load configuration for development environment
    config = get_config(config_name='prod', environment='dev')
    
    print("=== Basic Configuration Example ===\n")
    
    # Access configuration values
    print(f"Environment: {config.environment}")
    print(f"Debug Mode: {config.feature_flags.debug_mode}")
    print(f"Database Host: {config.database.host}")
    print(f"API Base URL: {config.api_endpoints.alpaca.base_url}")
    
    # Setup logging with configuration
    setup_logging(config)
    
    # Validate configuration
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate_configuration(
        config, 
        environment='dev'
    )
    
    print(f"\nConfiguration Valid: {is_valid}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Access nested configuration
    print("\n=== Scanner Configuration ===")
    print(f"Max Workers: {config.scanners.max_workers}")
    print(f"Timeout: {config.scanners.timeout_seconds}s")
    print(f"Max Symbols: {config.scanners.max_symbols_per_scan}")
    
    # Feature flags
    print("\n=== Feature Flags ===")
    if config.feature_flags.enable_paper_trading:
        print("Paper trading is ENABLED")
    else:
        print("Paper trading is DISABLED")
    
    if config.feature_flags.enable_backtesting:
        print("Backtesting is ENABLED")
    else:
        print("Backtesting is DISABLED")


if __name__ == "__main__":
    asyncio.run(main())
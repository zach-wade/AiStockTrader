"""
Environment Override Example

This example demonstrates how to override configuration values
for different environments and use environment variables.
"""

import asyncio
import os
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from main.config.config_manager import ConfigManager, get_config


async def main():
    """Environment override configuration example."""
    
    print("=== Environment Override Example ===\n")
    
    # Create config manager instance
    config_manager = ConfigManager()
    
    # Load configurations for different environments
    environments = ['dev', 'staging', 'prod', 'paper']
    
    for env in environments:
        print(f"\n--- {env.upper()} Environment ---")
        config = config_manager.load_environment_config(env)
        
        # Show environment-specific values
        print(f"Database: {config.database.host}:{config.database.port}")
        print(f"Cache TTL: {config.cache.ttl_seconds}s")
        print(f"Max Workers: {config.scanners.max_workers}")
        print(f"Log Level: {config.logging.level}")
        print(f"Debug Mode: {config.feature_flags.debug_mode}")
    
    # Demonstrate environment variable override
    print("\n\n=== Environment Variable Override ===")
    
    # Set environment variables
    os.environ['DB_HOST'] = 'custom-db-host.example.com'
    os.environ['DB_PORT'] = '5433'
    os.environ['ALPACA_API_KEY'] = 'test-api-key'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # Reload configuration with env vars
    config = get_config(config_name='prod', environment='dev')
    
    print(f"DB Host (from env): {config.database.host}")
    print(f"DB Port (from env): {config.database.port}")
    print(f"Log Level (from env): {config.logging.level}")
    
    # Show how to access API keys safely
    print("\n=== API Key Access ===")
    api_key = os.getenv('ALPACA_API_KEY', config.api_keys.alpaca.api_key)
    if api_key:
        print(f"Alpaca API Key: {api_key[:10]}... (truncated)")
    else:
        print("Alpaca API Key: Not configured")
    
    # Custom configuration override
    print("\n=== Custom Configuration ===")
    
    # Create custom config dict
    custom_overrides = {
        'scanners': {
            'max_workers': 20,
            'timeout_seconds': 120
        },
        'feature_flags': {
            'enable_experimental_features': True
        }
    }
    
    # Apply custom overrides
    from omegaconf import OmegaConf
    base_config = config_manager.load_environment_config('dev')
    custom_config = OmegaConf.merge(base_config, custom_overrides)
    
    print(f"Custom Max Workers: {custom_config.scanners.max_workers}")
    print(f"Custom Timeout: {custom_config.scanners.timeout_seconds}s")
    print(f"Experimental Features: {custom_config.feature_flags.enable_experimental_features}")


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Debug Configuration Loading

Tests what the configuration manager is actually returning.
"""

# Standard library imports
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def debug_config_structure():
    """Debug what configuration is actually being loaded."""

    print("🔍 Debugging Configuration Structure")
    print("=" * 50)

    try:
        # Local imports
        from main.config.config_manager import get_config

        # Load configuration
        config = get_config()

        print("✅ Configuration loaded successfully")
        print(f"📊 Config type: {type(config)}")

        # Check top-level keys
        if hasattr(config, "keys"):
            top_keys = list(config.keys())
            print(f"🔑 Top-level keys: {sorted(top_keys)}")

            # Check if 'data' key exists
            if "data" in config:
                data_config = config["data"]
                print(
                    f"📊 Data config keys: {list(data_config.keys()) if hasattr(data_config, 'keys') else 'Not a dict'}"
                )

                if hasattr(data_config, "keys") and "backfill" in data_config:
                    backfill_config = data_config["backfill"]
                    print(
                        f"🔄 Backfill config keys: {list(backfill_config.keys()) if hasattr(backfill_config, 'keys') else 'Not a dict'}"
                    )

                    if hasattr(backfill_config, "keys") and "source_priorities" in backfill_config:
                        priorities = backfill_config["source_priorities"]
                        print(f"🎯 Source priorities: {priorities}")
                    else:
                        print("❌ No source_priorities in backfill config")
                else:
                    print("❌ No backfill in data config")
            else:
                print("❌ No data key in config")

        else:
            print("❌ Config doesn't have keys method")

        # Try direct access
        try:
            priorities = config.get("data.backfill.source_priorities", "NOT_FOUND")
            print(f"🎯 Direct access result: {priorities}")
        except Exception as e:
            print(f"❌ Direct access failed: {e}")

        # Print some sample config content
        print("\n📄 Sample config content:")
        if hasattr(config, "keys"):
            for key in sorted(list(config.keys())[:5]):  # First 5 keys
                value = config[key]
                if isinstance(value, dict):
                    print(f"  {key}: {list(value.keys())[:3]}...")  # First 3 sub-keys
                else:
                    print(f"  {key}: {str(value)[:50]}...")  # First 50 chars

        return True

    except Exception as e:
        print(f"❌ Configuration debug failed: {e}")
        # Standard library imports
        import traceback

        traceback.print_exc()
        return False


def check_config_file():
    """Check if the configuration file exists."""

    print("\n📁 Checking Configuration File")
    print("-" * 40)

    try:

        # Check config directory
        config_dir = Path(__file__).parent / "ai_trader" / "config"
        print(f"📂 Config directory: {config_dir}")
        print(f"📂 Config directory exists: {config_dir.exists()}")

        if config_dir.exists():
            config_files = list(config_dir.glob("*.yaml"))
            print(f"📄 Config files: {[f.name for f in config_files]}")

            # Check for unified_config.yaml
            unified_config = config_dir / "unified_config.yaml"
            print(f"📄 unified_config.yaml exists: {unified_config.exists()}")

            if unified_config.exists():
                print(f"📄 unified_config.yaml size: {unified_config.stat().st_size} bytes")

        return True

    except Exception as e:
        print(f"❌ Config file check failed: {e}")
        return False


def main():
    """Main debug runner."""
    print("🐛 Configuration Debug Tool")
    print("=" * 60)

    # Run debug tests
    config_debug = debug_config_structure()
    file_check = check_config_file()

    print("\n" + "=" * 60)
    print("🏁 Debug Results")
    print("=" * 60)
    print(f"Config Structure: {'✅ PASS' if config_debug else '❌ FAIL'}")
    print(f"File Check: {'✅ PASS' if file_check else '❌ FAIL'}")

    overall_success = config_debug and file_check
    print(f"Overall: {'✅ ALL GOOD' if overall_success else '❌ ISSUES FOUND'}")

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test TSLA through the complete scanner pipeline with detailed validation at each layer.
"""
# Standard library imports
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Standard library imports
import logging

# Third-party imports
from omegaconf import OmegaConf

# Local imports
from main.config import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.scanners.scanner_pipeline import ScannerPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def ensure_tsla_in_companies(db_adapter):
    """Ensure TSLA exists in companies table."""

    # Check if TSLA exists
    check_query = "SELECT symbol FROM companies WHERE symbol = 'TSLA'"
    result = await db_adapter.fetch_one(check_query)

    if not result:
        print("Adding TSLA to companies table...")
        insert_query = """
            INSERT INTO companies (symbol, name, exchange, sector, industry, is_active, created_at)
            VALUES ('TSLA', 'Tesla Inc', 'NASDAQ', 'Consumer Cyclical', 'Auto Manufacturers', true, NOW())
            ON CONFLICT (symbol) DO NOTHING
        """
        await db_adapter.execute(insert_query)
        print("✓ TSLA added to companies table")
    else:
        print("✓ TSLA already exists in companies table")


async def validate_layer_results(db_adapter, layer_name, layer_num):
    """Validate results after each layer."""

    print(f"\nValidating {layer_name} results...")

    # Check qualification status
    qual_query = """
        SELECT layer0_qualified, layer1_qualified, layer2_qualified, layer3_qualified
        FROM companies WHERE symbol = 'TSLA'
    """

    result = await db_adapter.fetch_one(qual_query)
    if result:
        print("  Qualification Status:")
        print(f"    Layer 0: {'✓' if result['layer0_qualified'] else '✗'}")
        print(f"    Layer 1: {'✓' if result['layer1_qualified'] else '✗'}")
        print(f"    Layer 2: {'✓' if result['layer2_qualified'] else '✗'}")
        print(f"    Layer 3: {'✓' if result['layer3_qualified'] else '✗'}")

    # Check for alerts
    if int(layer_num) >= 2:
        alerts_query = """
            SELECT alert_type, score, message, created_at
            FROM scanner_alerts
            WHERE symbol = 'TSLA'
            ORDER BY created_at DESC
            LIMIT 5
        """

        alerts = await db_adapter.fetch_all(alerts_query)
        if alerts:
            print("  Recent Alerts:")
            for alert in alerts:
                print(
                    f"    {alert['created_at']}: {alert['alert_type']} (score: {alert['score']:.2f})"
                )
                print(f"      {alert['message']}")
        else:
            print("  No alerts generated")


async def test_tsla_scanner_pipeline():
    """Test TSLA through scanner pipeline with validation."""

    # Get config
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    # Convert to dict and modify for TSLA-only test
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Override scanner settings for focused test
    if "scanner" not in config_dict:
        config_dict["scanner"] = {}

    # Force TSLA in Layer 0
    config_dict["scanner"]["layer0"] = {
        "test_mode": True,
        "test_symbols": ["TSLA"],
        "max_symbols": 1,
    }

    # Ensure all layers run
    config_dict["scanner"]["enable_all_layers"] = True

    # Convert back to OmegaConf
    config = OmegaConf.create(config_dict)

    # Initialize database
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)

    results = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "TSLA",
        "layers": {},
        "success": False,
    }

    try:
        print("=" * 80)
        print("TSLA SCANNER PIPELINE TEST")
        print("=" * 80)
        print(f"Timestamp: {results['timestamp']}")

        # Ensure TSLA is in companies table
        await ensure_tsla_in_companies(db_adapter)

        # Create and run scanner pipeline
        print("\nStarting scanner pipeline...")
        pipeline = ScannerPipeline(config, test_mode=True)

        # Run the pipeline
        pipeline_result = await pipeline.run()

        print("\n" + "=" * 80)
        print("PIPELINE RESULTS")
        print("=" * 80)

        print(f"Success: {pipeline_result.success}")
        print(f"Duration: {pipeline_result.total_duration:.2f} seconds")
        print(f"Final Symbols: {pipeline_result.final_symbols}")

        # Analyze each layer
        for layer_result in pipeline_result.layer_results:
            layer_name = layer_result.layer_name
            layer_num = layer_result.layer_number

            print(f"\n{layer_name} (Layer {layer_num}):")
            print(f"  Input: {layer_result.input_count} symbols")
            print(f"  Output: {layer_result.output_count} symbols")
            print(f"  Duration: {layer_result.execution_time:.2f}s")

            if layer_result.errors:
                print(f"  Errors: {layer_result.errors}")

            # Store layer results
            results["layers"][layer_num] = {
                "name": layer_name,
                "input_count": layer_result.input_count,
                "output_count": layer_result.output_count,
                "duration": layer_result.execution_time,
                "errors": layer_result.errors,
                "symbols": layer_result.symbols if hasattr(layer_result, "symbols") else [],
            }

            # Validate layer results
            await validate_layer_results(db_adapter, layer_name, layer_num)

            # Check if TSLA made it through this layer
            if hasattr(layer_result, "symbols") and "TSLA" in layer_result.symbols:
                print(f"  ✓ TSLA passed {layer_name}")
            elif layer_result.output_count == 0:
                print(f"  ⚠️  No symbols passed {layer_name}")

        # Final validation
        print("\n" + "=" * 80)
        print("FINAL VALIDATION")
        print("=" * 80)

        # Check final qualification
        final_qual = await db_adapter.fetch_one("SELECT * FROM companies WHERE symbol = 'TSLA'")

        if final_qual:
            print("TSLA Final Status:")
            print(f"  Active: {'✓' if final_qual['is_active'] else '✗'}")
            print(f"  Layer 0 Qualified: {'✓' if final_qual.get('layer0_qualified') else '✗'}")
            print(f"  Layer 1 Qualified: {'✓' if final_qual.get('layer1_qualified') else '✗'}")
            print(f"  Layer 2 Qualified: {'✓' if final_qual.get('layer2_qualified') else '✗'}")
            print(f"  Layer 3 Qualified: {'✓' if final_qual.get('layer3_qualified') else '✗'}")

        # Check for any generated alerts
        total_alerts = await db_adapter.fetch_one(
            "SELECT COUNT(*) as count FROM scanner_alerts WHERE symbol = 'TSLA'"
        )

        if total_alerts:
            print(f"\nTotal Alerts Generated: {total_alerts['count']}")

        # Determine success
        results["success"] = pipeline_result.success

        # Save results
        output_file = Path("data/validation/tsla_scanner_test.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        return results["success"]

    finally:
        await db_adapter.close()


if __name__ == "__main__":
    success = asyncio.run(test_tsla_scanner_pipeline())
    sys.exit(0 if success else 1)

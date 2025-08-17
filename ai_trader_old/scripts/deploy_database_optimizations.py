#!/usr/bin/env python3
"""
Database Optimization Deployment Script

This script deploys database indexes and sets up performance monitoring
for the AI trading system.
"""

# Standard library imports
import argparse
import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Local imports
from ai_trader.utils.database import (
    analyze_database_performance,
    deploy_trading_indexes,
    generate_performance_report,
    get_db_pool,
)
from main.config.config_manager import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_argument_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Deploy database optimizations for AI Trading System"
    )

    parser.add_argument(
        "--priority",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Index priority to deploy (1=high, 2=medium, 3=low)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze what would be deployed without actually creating indexes",
    )

    parser.add_argument(
        "--dashboard", action="store_true", help="Generate performance monitoring dashboard"
    )

    parser.add_argument(
        "--analyze", action="store_true", help="Analyze current database performance"
    )

    parser.add_argument("--output-file", type=str, help="Save results to JSON file")

    parser.add_argument(
        "--hours", type=int, default=24, help="Hours of performance data to analyze"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


async def deploy_indexes(priority: int, dry_run: bool) -> dict:
    """Deploy database indexes"""
    logger.info(f"Deploying indexes with priority {priority} (dry_run={dry_run})")

    try:
        # Initialize database pool
        config = get_config()
        db_pool = get_db_pool()
        db_pool.initialize(config=config)

        # Deploy indexes
        results = await deploy_trading_indexes(priority=priority, dry_run=dry_run)

        logger.info("Index deployment completed successfully")
        return {
            "success": True,
            "deployment_results": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Index deployment failed: {e}")
        return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}


async def create_dashboard() -> dict:
    """Create performance monitoring dashboard"""
    logger.info("Creating performance monitoring dashboard")

    try:
        # Initialize database pool
        config = get_config()
        db_pool = get_db_pool()
        db_pool.initialize(config=config)

        # Generate dashboard
        dashboard_data = await generate_performance_report()

        logger.info("Performance dashboard created successfully")
        return {
            "success": True,
            "dashboard": dashboard_data,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")
        return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}


async def analyze_performance(hours: int) -> dict:
    """Analyze database performance"""
    logger.info(f"Analyzing database performance for last {hours} hours")

    try:
        # Initialize database pool
        config = get_config()
        db_pool = get_db_pool()
        db_pool.initialize(config=config)

        # Analyze performance
        analysis_results = await analyze_database_performance(hours=hours)

        logger.info("Performance analysis completed successfully")
        return {
            "success": True,
            "analysis": analysis_results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}


def save_results(results: dict, output_file: str):
    """Save results to JSON file"""
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def print_summary(results: dict):
    """Print summary of results"""
    print("\n" + "=" * 60)
    print("DATABASE OPTIMIZATION SUMMARY")
    print("=" * 60)

    if "deployment_results" in results:
        deployment = results["deployment_results"]
        print("\nINDEX DEPLOYMENT:")
        print(f"  Total analyzed: {deployment.get('indexes_analyzed', 0)}")
        print(f"  Created: {deployment.get('indexes_created', 0)}")
        print(f"  Skipped: {deployment.get('indexes_skipped', 0)}")
        print(f"  Failed: {deployment.get('indexes_failed', 0)}")
        print(f"  Estimated space: {deployment.get('estimated_space_mb', 0):.1f} MB")
        print(f"  Total time: {deployment.get('total_time', 0):.2f} seconds")

        if deployment.get("results"):
            print("\n  Index Details:")
            for result in deployment["results"][:10]:  # Show first 10
                status = result.get("status", "unknown")
                index_name = result.get("index", "unknown")
                print(f"    {index_name}: {status}")

    if "dashboard" in results:
        dashboard = results["dashboard"]
        print("\nPERFORMANCE DASHBOARD:")

        # Connection pool info
        pool_info = dashboard.get("connection_pool", {}).get("pool_status", {})
        if pool_info:
            print(f"  Active connections: {pool_info.get('active_connections', 0)}")
            print(f"  Pool size: {pool_info.get('pool_size', 0)}")

        # Database health
        db_health = dashboard.get("database_health", {})
        cache_hit = db_health.get("cache_hit_ratio", 0)
        print(f"  Cache hit ratio: {cache_hit:.1f}%")

        # Memory usage
        memory_info = dashboard.get("memory_usage", {}).get("current", {})
        if memory_info:
            print(f"  Memory usage: {memory_info.get('rss_mb', 0):.1f} MB")

        # Recommendations
        recommendations = dashboard.get("recommendations", [])
        if recommendations:
            print(f"\n  Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"    {i}. {rec}")

    if "analysis" in results:
        analysis = results["analysis"]
        print("\nPERFORMANCE ANALYSIS:")

        slow_queries = analysis.get("slow_queries", [])
        print(f"  Slow queries found: {len(slow_queries)}")

        recommendations = analysis.get("recommendations", [])
        print(f"  Recommendations: {len(recommendations)}")

        if recommendations:
            print("\n  Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"    {i}. {rec}")

    print("\n" + "=" * 60)


async def main():
    """Main execution function"""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = {}

    try:
        # Deploy indexes if requested (default behavior)
        if not args.dashboard and not args.analyze:
            # Default: deploy indexes
            results.update(await deploy_indexes(args.priority, args.dry_run))

        # Create dashboard if requested
        if args.dashboard:
            dashboard_results = await create_dashboard()
            results.update(dashboard_results)

        # Analyze performance if requested
        if args.analyze:
            analysis_results = await analyze_performance(args.hours)
            results.update(analysis_results)

        # Save results if output file specified
        if args.output_file:
            save_results(results, args.output_file)

        # Print summary
        print_summary(results)

        # Exit with appropriate code
        if results.get("success", False):
            logger.info("Database optimization completed successfully")
            sys.exit(0)
        else:
            logger.error("Database optimization failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

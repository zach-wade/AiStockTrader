#!/usr/bin/env python3
"""
Storage Rotation Job

This job runs periodically to move old data from hot storage (PostgreSQL)
to cold storage (Data Lake) based on the configured lifecycle policy.

Can be run:
1. As a scheduled cron job
2. Manually via command line
3. As part of a larger workflow
"""

# Standard library imports
import argparse
import asyncio
import sys
from typing import Any

# Local imports
from main.config.config_manager import get_config
from main.data_pipeline.storage.archive_initializer import get_archive
from main.data_pipeline.storage.data_lifecycle_manager import DataLifecycleManager
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.utils.app.cli import error_message, info_message, success_message, warning_message
from main.utils.core import get_logger, timer

logger = get_logger(__name__)


async def run_storage_rotation(dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    """
    Execute the storage rotation job.

    Args:
        dry_run: If True, only simulate the rotation without making changes
        force: If True, run even if disabled in config

    Returns:
        Dictionary with job results
    """
    config = get_config()

    # Check if auto rotation is enabled
    auto_rotation_config = config.get("storage", {}).get("lifecycle", {}).get("auto_rotation", {})
    if not auto_rotation_config.get("enabled", True) and not force:
        warning_message(
            "Storage rotation disabled",
            "Auto rotation is disabled in configuration. Use --force to run anyway.",
        )
        return {"status": "skipped", "reason": "disabled_in_config"}

    with timer("storage_rotation") as job_timer:
        try:
            info_message(
                "Starting storage rotation job", f"Mode: {'DRY RUN' if dry_run else 'LIVE'}"
            )

            # Initialize database adapter
            db_factory = DatabaseFactory()
            db_adapter = db_factory.create_async_database(config)

            # Initialize archive
            archive = get_archive()

            # Create lifecycle manager
            lifecycle_manager = DataLifecycleManager(
                config=config, db_adapter=db_adapter, archive=archive
            )

            # Get current status before rotation
            info_message("Checking archival status...")
            status_before = await lifecycle_manager.get_archival_status()

            # Log current status
            for repo_name, repo_status in status_before.get("repository_status", {}).items():
                if "error" not in repo_status:
                    eligible = repo_status.get("archive_eligible", 0)
                    if eligible > 0:
                        info_message(
                            f"{repo_name}",
                            f"{eligible:,} records eligible for archival "
                            f"({repo_status.get('archive_percentage', 0):.1f}% of total)",
                        )

            # Run archival cycle
            info_message("Running archival cycle...")
            results = await lifecycle_manager.run_archival_cycle(dry_run=dry_run)

            # Process results
            if results["status"] == "success":
                records_archived = results.get("records_archived", 0)
                duration = job_timer.elapsed

                if records_archived > 0:
                    success_message(
                        "Storage rotation completed",
                        f"Archived {records_archived:,} records in {duration:.1f}s",
                    )
                else:
                    info_message("Storage rotation completed", "No records needed archiving")

                # Get status after rotation
                if not dry_run and records_archived > 0:
                    status_after = await lifecycle_manager.get_archival_status()

                    # Log changes
                    for repo_name, repo_status in status_after.get("repository_status", {}).items():
                        if "error" not in repo_status:
                            before_eligible = (
                                status_before.get("repository_status", {})
                                .get(repo_name, {})
                                .get("archive_eligible", 0)
                            )
                            after_eligible = repo_status.get("archive_eligible", 0)

                            if before_eligible > after_eligible:
                                info_message(
                                    f"{repo_name} cleanup",
                                    f"Archived {before_eligible - after_eligible:,} records",
                                )

                return {
                    "status": "success",
                    "records_archived": records_archived,
                    "duration_seconds": duration,
                    "dry_run": dry_run,
                }

            elif results["status"] == "noop":
                info_message("Storage rotation", "No data eligible for archiving")
                return {
                    "status": "noop",
                    "records_archived": 0,
                    "duration_seconds": job_timer.elapsed,
                }

            else:
                error_msg = results.get("message", "Unknown error")
                error_message("Storage rotation failed", error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "duration_seconds": job_timer.elapsed,
                }

        except Exception as e:
            error_message("Storage rotation error", str(e))
            logger.error(f"Storage rotation job failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": job_timer.elapsed if "job_timer" in locals() else 0,
            }


async def run_full_archival(dry_run: bool = False) -> dict[str, Any]:
    """
    Run archival for all configured repositories (not just market_data).

    Args:
        dry_run: If True, only simulate the archival

    Returns:
        Dictionary with archival results
    """
    config = get_config()

    with timer("full_archival") as job_timer:
        try:
            info_message(
                "Starting full repository archival", f"Mode: {'DRY RUN' if dry_run else 'LIVE'}"
            )

            # Initialize components
            db_factory = DatabaseFactory()
            db_adapter = db_factory.create_async_database(config)
            archive = get_archive()

            lifecycle_manager = DataLifecycleManager(
                config=config, db_adapter=db_adapter, archive=archive
            )

            # Run archival for all repositories
            results = await lifecycle_manager.archive_all_repositories(dry_run=dry_run)

            if results["status"] == "success":
                total_archived = results.get("total_archived", 0)
                duration = job_timer.elapsed

                # Show per-repository results
                for repo_name, repo_result in results.get("repository_results", {}).items():
                    if repo_result.get("status") == "success":
                        archived = repo_result.get("records_archived", 0)
                        if archived > 0:
                            info_message(f"{repo_name}", f"Archived {archived:,} records")
                    elif repo_result.get("status") == "error":
                        warning_message(
                            f"{repo_name} error", repo_result.get("message", "Unknown error")
                        )

                if total_archived > 0:
                    success_message(
                        "Full archival completed",
                        f"Archived {total_archived:,} total records in {duration:.1f}s",
                    )
                else:
                    info_message("Full archival completed", "No records needed archiving")

                return results

            else:
                error_message("Full archival failed", "See logs for details")
                return results

        except Exception as e:
            error_message("Full archival error", str(e))
            logger.error(f"Full archival job failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": job_timer.elapsed if "job_timer" in locals() else 0,
            }


def main():
    """Main entry point for command line execution."""
    parser = argparse.ArgumentParser(
        description="Storage Rotation Job - Move old data from hot to cold storage"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate the rotation without making changes"
    )
    parser.add_argument(
        "--force", action="store_true", help="Run even if disabled in configuration"
    )
    parser.add_argument(
        "--full", action="store_true", help="Archive all repository types, not just market_data"
    )

    args = parser.parse_args()

    # Run the job
    try:
        if args.full:
            results = asyncio.run(run_full_archival(dry_run=args.dry_run))
        else:
            results = asyncio.run(run_storage_rotation(dry_run=args.dry_run, force=args.force))

        # Exit with appropriate code
        if results["status"] == "success":
            sys.exit(0)
        elif results["status"] == "noop":
            sys.exit(0)  # No-op is still successful
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        warning_message("Job interrupted", "Storage rotation cancelled by user")
        sys.exit(130)
    except Exception as e:
        error_message("Fatal error", str(e))
        logger.error(f"Storage rotation job crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

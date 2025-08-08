#!/usr/bin/env python3
"""
AI Trading System - Master Job Scheduler CLI

Command-line interface for the job scheduler. The actual JobScheduler class
has been moved to main.orchestration.job_scheduler for proper module organization.

This script provides the CLI interface to run and manage scheduled jobs.

Author: AI Trading System
Created: July 12, 2025
"""

import os
import sys
import json
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import JobScheduler from its proper location in the orchestration module
from main.orchestration.job_scheduler import JobScheduler


def main():
    """Main entry point for the job scheduler CLI."""
    parser = argparse.ArgumentParser(description='AI Trading System Job Scheduler')
    parser.add_argument('--config', '-c', help='Path to job definitions YAML file')
    parser.add_argument('--run-job', help='Run a specific job once')
    parser.add_argument('--status', action='store_true', help='Show job status')
    parser.add_argument('--daemon', '-d', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = JobScheduler(config_path=args.config)
    
    if args.run_job:
        # Run single job
        success = scheduler.run_job(args.run_job)
        sys.exit(0 if success else 1)
    
    elif args.status:
        # Show status
        status = scheduler.get_job_status()
        print(json.dumps(status, indent=2, default=str))
        sys.exit(0)
    
    else:
        # Run scheduler daemon
        scheduler.run_scheduler()


if __name__ == '__main__':
    main()
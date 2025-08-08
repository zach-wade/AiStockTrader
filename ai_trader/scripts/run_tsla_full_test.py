#!/usr/bin/env python3
"""
Master script to run complete TSLA end-to-end test.
"""
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime
import json


def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ SUCCESS ({duration:.1f}s)")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ FAILED ({duration:.1f}s)")
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.stdout:
            print("STDOUT:", result.stdout)
            
    return result.returncode == 0


def main():
    """Run complete TSLA test suite."""
    
    print("=" * 80)
    print("TSLA COMPLETE END-TO-END TEST SUITE")
    print("=" * 80)
    print(f"Started: {datetime.now()}")
    
    results = {
        "start_time": datetime.now().isoformat(),
        "steps": []
    }
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    
    # Step 1: Validate database is empty
    step = {
        "name": "Validate Database Empty",
        "command": "python scripts/validate_db_empty.py"
    }
    success = run_command(
        f"cd {project_dir} && {step['command']}",
        step["name"]
    )
    step["success"] = success
    results["steps"].append(step)
    
    if not success:
        print("\n⚠️  Database is not empty. Please clean it before proceeding.")
        print("Run: TRUNCATE TABLE market_data_1m, market_data_5m, ...; or similar")
        return False
        
    # Step 2: Backfill TSLA data
    step = {
        "name": "Backfill TSLA Data (5 years)",
        "command": "python ai_trader.py backfill --symbols TSLA --days 1825 --stage all"
    }
    success = run_command(
        f"cd {project_dir} && {step['command']}",
        step["name"]
    )
    step["success"] = success
    results["steps"].append(step)
    
    if not success:
        print("\n❌ Backfill failed. Check logs for details.")
        return False
        
    # Wait a bit for data to settle
    print("\nWaiting 5 seconds for data to settle...")
    time.sleep(5)
    
    # Step 3: Validate backfill
    step = {
        "name": "Validate TSLA Backfill",
        "command": "python scripts/validate_tsla_backfill.py"
    }
    success = run_command(
        f"cd {project_dir} && {step['command']}",
        step["name"]
    )
    step["success"] = success
    results["steps"].append(step)
    
    # Step 4: Calculate features
    step = {
        "name": "Calculate Features",
        "command": "python ai_trader.py features --symbols TSLA --recalculate"
    }
    success = run_command(
        f"cd {project_dir} && {step['command']}",
        step["name"]
    )
    step["success"] = success
    results["steps"].append(step)
    
    # Step 5: Run scanner pipeline
    step = {
        "name": "Run Scanner Pipeline",
        "command": "python scripts/test_tsla_scanner_pipeline.py"
    }
    success = run_command(
        f"cd {project_dir} && {step['command']}",
        step["name"]
    )
    step["success"] = success
    results["steps"].append(step)
    
    # Step 6: Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    models = ["xgboost", "lstm", "ensemble"]
    for model_type in models:
        step = {
            "name": f"Train {model_type.upper()} Model",
            "command": f"python ai_trader.py train --symbols TSLA --model-type {model_type}"
        }
        success = run_command(
            f"cd {project_dir} && {step['command']}",
            step["name"]
        )
        step["success"] = success
        results["steps"].append(step)
        
    # Step 7: Run backtest
    step = {
        "name": "Run Backtest",
        "command": "python ai_trader.py backtest --symbols TSLA --model-type xgboost --start-date 2024-01-01"
    }
    success = run_command(
        f"cd {project_dir} && {step['command']}",
        step["name"]
    )
    step["success"] = success
    results["steps"].append(step)
    
    # Step 8: Generate comprehensive report
    step = {
        "name": "Generate Validation Report",
        "command": "python scripts/generate_tsla_validation_report.py"
    }
    success = run_command(
        f"cd {project_dir} && {step['command']}",
        step["name"]
    )
    step["success"] = success
    results["steps"].append(step)
    
    # Summary
    results["end_time"] = datetime.now().isoformat()
    results["total_duration"] = (
        datetime.fromisoformat(results["end_time"]) - 
        datetime.fromisoformat(results["start_time"])
    ).total_seconds()
    
    successful_steps = sum(1 for step in results["steps"] if step["success"])
    total_steps = len(results["steps"])
    
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    print(f"Total Steps: {total_steps}")
    print(f"Successful: {successful_steps}")
    print(f"Failed: {total_steps - successful_steps}")
    print(f"Duration: {results['total_duration']:.1f} seconds")
    
    print("\nStep Results:")
    for step in results["steps"]:
        status = "✅" if step["success"] else "❌"
        print(f"  {status} {step['name']}")
        
    # Save results
    output_file = Path("data/validation/tsla_test_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nTest results saved to: {output_file}")
    
    # Final status
    all_passed = successful_steps == total_steps
    if all_passed:
        print("\n✅ ALL TESTS PASSED! System is ready.")
    else:
        print("\n❌ SOME TESTS FAILED. Please review and fix issues.")
        
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
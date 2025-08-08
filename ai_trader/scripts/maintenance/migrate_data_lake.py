#!/usr/bin/env python3
"""
Data Lake Migration CLI

Simple command-line interface for migrating the data lake to standardized structure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_trader.data_pipeline.storage.migration_tool import DataLakeMigrator
import logging
import json

def main():
    """Main CLI function for data lake migration."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Data lake path (relative to script location)
    data_lake_path = Path(__file__).parent.parent / "data_lake"
    
    print("=" * 60)
    print("AI Trader Data Lake Migration Tool")
    print("=" * 60)
    print()
    
    if not data_lake_path.exists():
        print(f"❌ Data lake not found at: {data_lake_path}")
        return 1
    
    print(f"📁 Data lake location: {data_lake_path}")
    print()
    
    # Initialize migrator in dry-run mode
    migrator = DataLakeMigrator(
        data_lake_path=str(data_lake_path),
        dry_run=True,
        max_workers=4,
        backup_enabled=True
    )
    
    try:
        # Step 1: Analyze current structure
        print("🔍 Analyzing current data lake structure...")
        analysis = migrator.analyze_current_structure()
        
        print(f"\n📊 Analysis Results:")
        print(f"   • Total files: {analysis['total_files']:,}")
        print(f"   • Total size: {analysis['total_size_bytes'] / (1024**3):.2f} GB")
        print(f"   • Directories: {len(analysis['directories'])}")
        
        # Show data types found
        data_types = analysis.get('data_types', {})
        print(f"\n📂 Data Types Found:")
        for data_type, info in data_types.items():
            if info['directories']:
                print(f"   • {data_type}: {len(info['directories'])} directories ({info['migration_complexity']} complexity)")
        
        # Show issues
        if analysis.get('issues'):
            print(f"\n⚠️  Structural Issues ({len(analysis['issues'])}):")
            for issue in analysis['issues']:
                print(f"   • {issue}")
        
        # Step 2: Create migration plan
        print(f"\n📋 Creating migration plan...")
        migration_plan = migrator.create_migration_plan(analysis)
        
        print(f"\n🗺️  Migration Plan:")
        print(f"   • Steps: {len(migration_plan.migration_steps)}")
        print(f"   • Estimated duration: {migration_plan.estimated_duration}")
        print(f"   • Safety checks: {len(migration_plan.safety_checks)}")
        
        print(f"\n📝 Migration Steps:")
        for step in migration_plan.migration_steps:
            print(f"   {step['step']}. {step['name']} ({step['estimated_duration']})")
        
        # Step 3: Execute dry-run migration
        print(f"\n🧪 Running dry-run migration...")
        stats = migrator.execute_migration(migration_plan, confirm_destructive=False)
        
        print(f"\n📈 Dry-Run Results:")
        print(f"   • Files analyzed: {stats.files_analyzed:,}")
        print(f"   • Duration: {stats.duration_seconds:.1f} seconds")
        
        if stats.errors:
            print(f"   • Errors detected: {len(stats.errors)}")
            print(f"\n❌ First few errors:")
            for error in stats.errors[:3]:
                print(f"      • {error}")
        else:
            print(f"   • ✅ No errors detected")
        
        # Step 4: Generate report
        report = migrator.generate_migration_report(migration_plan, stats)
        
        # Save analysis report
        report_file = Path(__file__).parent / "data_lake_migration_analysis.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed analysis saved to: {report_file}")
        
        # Step 5: Ask user if they want to proceed
        print(f"\n" + "=" * 60)
        print(f"💡 Migration Analysis Complete")
        print(f"=" * 60)
        
        if stats.errors:
            print(f"⚠️  {len(stats.errors)} issues detected - review before proceeding")
        else:
            print(f"✅ No issues detected - migration should be safe")
        
        print(f"\nTo execute the actual migration:")
        print(f"   python migrate_data_lake.py --execute")
        print(f"\nTo create backup before migration:")
        print(f"   python migrate_data_lake.py --execute --backup")
        
        return 0
        
    except Exception as e:
        logger.error(f"Migration analysis failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1


def execute_migration():
    """Execute the actual migration (non-dry-run)."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    data_lake_path = Path(__file__).parent.parent / "data_lake"
    
    print("=" * 60)
    print("🚀 EXECUTING DATA LAKE MIGRATION")
    print("=" * 60)
    print()
    print("⚠️  This will modify your data lake structure!")
    print("⚠️  Make sure you have a backup if needed.")
    print()
    
    response = input("Are you sure you want to proceed? (yes/no): ").lower().strip()
    if response != 'yes':
        print("Migration cancelled.")
        return 1
    
    # Initialize migrator in execute mode
    migrator = DataLakeMigrator(
        data_lake_path=str(data_lake_path),
        dry_run=False,
        max_workers=4,
        backup_enabled=True
    )
    
    try:
        # Re-analyze and create plan
        analysis = migrator.analyze_current_structure()
        migration_plan = migrator.create_migration_plan(analysis)
        
        print(f"\n🔄 Executing migration...")
        stats = migrator.execute_migration(migration_plan, confirm_destructive=True)
        
        print(f"\n📈 Migration Results:")
        print(f"   • Files migrated: {stats.files_migrated:,}")
        print(f"   • Data migrated: {stats.bytes_migrated / (1024**3):.2f} GB")
        print(f"   • Success rate: {stats.success_rate:.1f}%")
        print(f"   • Duration: {stats.duration_seconds:.1f} seconds")
        
        if stats.errors:
            print(f"   • Errors: {len(stats.errors)}")
            for error in stats.errors[:5]:
                print(f"      • {error}")
        
        # Generate final report
        report = migrator.generate_migration_report(migration_plan, stats)
        report_file = Path(__file__).parent / "data_lake_migration_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Migration report saved to: {report_file}")
        
        if stats.success_rate > 95:
            print(f"\n✅ Migration completed successfully!")
        else:
            print(f"\n⚠️  Migration completed with some issues - check report")
        
        return 0
        
    except Exception as e:
        logger.error(f"Migration execution failed: {e}")
        print(f"\n❌ Migration failed: {e}")
        
        # Attempt rollback
        if hasattr(migrator, 'backup_path') and migrator.backup_path.exists():
            print(f"🔄 Attempting rollback...")
            try:
                migrator._rollback_migration()
                print(f"✅ Rollback completed")
            except Exception as rollback_error:
                print(f"❌ Rollback failed: {rollback_error}")
        
        return 1


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Lake Migration Tool')
    parser.add_argument('--execute', action='store_true', help='Execute actual migration (not dry-run)')
    parser.add_argument('--backup', action='store_true', help='Create backup before migration')
    
    args = parser.parse_args()
    
    if args.execute:
        sys.exit(execute_migration())
    else:
        sys.exit(main())
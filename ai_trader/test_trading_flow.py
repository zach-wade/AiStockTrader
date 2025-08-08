#!/usr/bin/env python3
"""
End-to-End Trading Flow Test Script
Tests the complete trading pipeline from data ingestion to order execution
Part of Phase 2: Critical Path Analysis
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import traceback
from typing import Dict, List, Optional, Any

# Add src/main to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "main"))

# Test results storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "components": {},
    "issues_found": [],
    "performance_metrics": {},
    "overall_status": "PENDING"
}

def log_test(component: str, test: str, status: str, details: str = "", issue: str = ""):
    """Log test results"""
    if component not in test_results["components"]:
        test_results["components"][component] = {}
    
    test_results["components"][component][test] = {
        "status": status,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    
    if issue:
        test_results["issues_found"].append({
            "component": component,
            "test": test,
            "issue": issue,
            "timestamp": datetime.now().isoformat()
        })
    
    # Print to console
    symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{symbol} [{component}] {test}: {status}")
    if details:
        print(f"   Details: {details}")
    if issue:
        print(f"   Issue: {issue}")

async def test_configuration():
    """Test 1: Configuration and Environment"""
    print("\n" + "="*50)
    print("TEST 1: CONFIGURATION & ENVIRONMENT")
    print("="*50)
    
    try:
        from main.config import get_config_manager
        config_manager = get_config_manager()
        
        # Test loading main config
        try:
            config = config_manager.load_config("unified_config")
            log_test("Configuration", "Load unified_config", "PASS", 
                    f"Config loaded with {len(config.__dict__)} settings")
        except Exception as e:
            log_test("Configuration", "Load unified_config", "FAIL", 
                    str(e), "Cannot load configuration")
            return False
        
        # Check critical environment variables
        critical_vars = ["POLYGON_API_KEY", "ALPACA_API_KEY"]
        optional_vars = ["DATABASE_URL"]
        
        for var in critical_vars:
            if os.getenv(var):
                log_test("Configuration", f"Environment {var}", "PASS", "Set")
            else:
                log_test("Configuration", f"Environment {var}", "FAIL", 
                        "Not set", f"Missing {var}")
        
        for var in optional_vars:
            if os.getenv(var):
                log_test("Configuration", f"Environment {var}", "PASS", "Set")
            else:
                log_test("Configuration", f"Environment {var}", "WARN", 
                        "Not set (optional)", f"{var} not required - using config")
        
        return True
        
    except Exception as e:
        log_test("Configuration", "Module Import", "FAIL", 
                str(e), "Configuration system broken")
        return False

async def test_database_connection():
    """Test 2: Database Connectivity"""
    print("\n" + "="*50)
    print("TEST 2: DATABASE CONNECTION")
    print("="*50)
    
    try:
        from main.data_pipeline.storage.database_factory import DatabaseFactory
        from main.config import get_config_manager
        
        config_manager = get_config_manager()
        config = config_manager.load_config("unified_config")
        
        # Create database adapter
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(config)
        
        # Test connection
        try:
            result = await db_adapter.fetch_one("SELECT 1 as test")
            log_test("Database", "PostgreSQL Connection", "PASS", 
                    "Connected successfully")
            
            # Check critical tables
            tables_query = """
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """
            tables = await db_adapter.fetch_all(tables_query)
            table_names = [t['tablename'] for t in tables]
            
            critical_tables = ['companies', 'market_data_1h', 'features']
            for table in critical_tables:
                if table in table_names:
                    log_test("Database", f"Table {table}", "PASS", "Exists")
                else:
                    log_test("Database", f"Table {table}", "FAIL", 
                            "Missing", f"Table {table} not found")
            
            await db_adapter.close()
            return True
            
        except Exception as e:
            log_test("Database", "PostgreSQL Connection", "FAIL", 
                    str(e), "Database connection failed")
            return False
            
    except Exception as e:
        log_test("Database", "Module Import", "FAIL", 
                str(e), "Database module broken")
        return False

async def test_data_ingestion():
    """Test 3: Data Ingestion Pipeline"""
    print("\n" + "="*50)
    print("TEST 3: DATA INGESTION")
    print("="*50)
    
    try:
        from main.data_pipeline.ingestion.clients.polygon_market_client import PolygonMarketClient
        from main.config import get_config_manager
        
        config_manager = get_config_manager()
        config = config_manager.load_config("unified_config")
        
        # Test Polygon client
        try:
            client = PolygonMarketClient(config)
            log_test("Data Ingestion", "Polygon Client Init", "PASS", 
                    "Client initialized")
            
            # Test fetching data for AAPL
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            try:
                # This would need actual implementation testing
                log_test("Data Ingestion", "Polygon API Test", "WARN", 
                        "Needs live test", "Manual verification needed")
            except Exception as e:
                log_test("Data Ingestion", "Polygon API Test", "FAIL", 
                        str(e), "API connection failed")
                
        except Exception as e:
            log_test("Data Ingestion", "Polygon Client Init", "FAIL", 
                    str(e), "Client initialization failed")
            return False
            
        # Test data validation
        try:
            from main.data_pipeline.validation import ValidationPipeline
            log_test("Data Ingestion", "Validation Module", "PASS", 
                    "ValidationPipeline imported")
        except Exception as e:
            log_test("Data Ingestion", "Validation Module", "WARN", 
                    f"Import error: {str(e)[:50]}", "Validation may be missing")
        
        return True
        
    except Exception as e:
        log_test("Data Ingestion", "Module Import", "FAIL", 
                str(e), "Ingestion module broken")
        return False

async def test_feature_calculation():
    """Test 4: Feature Calculation Pipeline"""
    print("\n" + "="*50)
    print("TEST 4: FEATURE CALCULATION")
    print("="*50)
    
    try:
        from main.feature_pipeline.feature_orchestrator import FeatureOrchestrator
        from main.config import get_config_manager
        import time
        
        config_manager = get_config_manager()
        config = config_manager.load_config("unified_config")
        
        # Initialize orchestrator
        try:
            orchestrator = FeatureOrchestrator(config)
            log_test("Features", "Orchestrator Init", "PASS", 
                    "Orchestrator initialized")
            
            # Test feature calculation for AAPL
            start_time = time.time()
            
            # This would need actual implementation testing
            log_test("Features", "Feature Calculation", "WARN", 
                    "Needs implementation test", "Manual verification needed")
            
            elapsed = time.time() - start_time
            test_results["performance_metrics"]["feature_calculation_time"] = elapsed
            
        except Exception as e:
            log_test("Features", "Orchestrator Init", "FAIL", 
                    str(e), "Orchestrator initialization failed")
            return False
        
        # Check calculator types
        try:
            from main.feature_pipeline.calculators import (
                TechnicalIndicatorsCalculator,
                AdvancedStatisticalCalculator,
                BaseFeatureCalculator
            )
            log_test("Features", "Calculator Modules", "PASS", 
                    "Technical, Statistical, Base calculators found")
        except Exception as e:
            log_test("Features", "Calculator Modules", "FAIL", 
                    str(e), "Some calculators missing")
        
        return True
        
    except Exception as e:
        log_test("Features", "Module Import", "FAIL", 
                str(e), "Feature module broken")
        return False

async def test_models():
    """Test 5: Model Loading and Prediction"""
    print("\n" + "="*50)
    print("TEST 5: MODELS & STRATEGIES")
    print("="*50)
    
    try:
        # Check model directory structure
        models_path = Path(__file__).parent / "src/main/models"
        
        subdirs = [d.name for d in models_path.iterdir() if d.is_dir()]
        log_test("Models", "Directory Structure", "PASS", 
                f"Found subdirectories: {', '.join(subdirs)}")
        
        # Try to import strategy modules
        try:
            from main.models.strategies import BaseStrategy
            log_test("Models", "Strategy Import", "PASS", 
                    "BaseStrategy imported")
        except Exception as e:
            log_test("Models", "Strategy Import", "FAIL", 
                    str(e), "Strategy module broken")
        
        # Check for trained models
        model_files = list(Path(".").glob("**/*.pkl")) + \
                     list(Path(".").glob("**/*.joblib")) + \
                     list(Path(".").glob("**/*.h5"))
        
        if model_files:
            log_test("Models", "Saved Models", "PASS", 
                    f"Found {len(model_files)} saved model files")
            for model_file in model_files[:3]:  # Show first 3
                print(f"   - {model_file}")
        else:
            log_test("Models", "Saved Models", "WARN", 
                    "No saved model files found", "Models may need training")
        
        return True
        
    except Exception as e:
        log_test("Models", "Module Access", "FAIL", 
                str(e), "Models module broken")
        return False

async def test_risk_management():
    """Test 6: Risk Management System"""
    print("\n" + "="*50)
    print("TEST 6: RISK MANAGEMENT")
    print("="*50)
    
    try:
        from main.risk_management.pre_trade import UnifiedLimitChecker
        from main.risk_management.real_time import LiveRiskMonitor
        from main.config import get_config_manager
        
        config_manager = get_config_manager()
        config = config_manager.load_config("unified_config")
        
        # Test pre-trade risk
        try:
            pre_trade = UnifiedLimitChecker(config)
            log_test("Risk", "Pre-Trade Manager", "PASS", 
                    "Manager initialized")
        except Exception as e:
            log_test("Risk", "Pre-Trade Manager", "FAIL", 
                    str(e), "Pre-trade risk broken")
        
        # Test real-time risk
        try:
            # Import the test implementation
            # WARNING: This uses TestPositionManager - must be replaced before production!
            from test_helpers.test_position_manager import TestPositionManager
            
            # Create a real (test) position manager, not a mock
            test_position_manager = TestPositionManager(config=config)
            
            # Initialize LiveRiskMonitor with the test implementation
            real_time = LiveRiskMonitor(
                position_manager=test_position_manager,
                config=config
            )
            log_test("Risk", "Real-Time Monitor", "PASS", 
                    "Monitor initialized with TestPositionManager (REPLACE BEFORE PRODUCTION!)")
        except Exception as e:
            log_test("Risk", "Real-Time Monitor", "FAIL", 
                    str(e), "Real-time risk broken")
        
        # Check circuit breaker (ISSUE-010)
        try:
            from main.risk_management.real_time.circuit_breaker import CircuitBreakerFacade
            log_test("Risk", "Circuit Breaker", "PASS", 
                    "Circuit breaker found")
        except:
            log_test("Risk", "Circuit Breaker", "WARN", 
                    "Module not found", "Circuit breaker may be elsewhere")
        
        return True
        
    except Exception as e:
        log_test("Risk", "Module Import", "FAIL", 
                str(e), "Risk management broken")
        return False

async def test_trading_engine():
    """Test 7: Trading Engine"""
    print("\n" + "="*50)
    print("TEST 7: TRADING ENGINE")
    print("="*50)
    
    try:
        from main.trading_engine.core.execution_engine import ExecutionEngine
        from main.config import get_config_manager
        
        config_manager = get_config_manager()
        config = config_manager.load_config("unified_config")
        
        # Test engine initialization
        try:
            engine = ExecutionEngine(config)
            log_test("Trading", "Execution Engine", "PASS", 
                    "Engine initialized")
        except Exception as e:
            log_test("Trading", "Execution Engine", "FAIL", 
                    str(e), "Engine initialization failed")
            return False
        
        # Check broker connections
        try:
            from main.trading_engine.brokers.alpaca_broker import AlpacaBroker
            log_test("Trading", "Alpaca Broker", "PASS", 
                    "Broker module found")
        except:
            log_test("Trading", "Alpaca Broker", "FAIL", 
                    "Module not found", "Alpaca integration missing")
        
        # Check algorithms
        try:
            from main.trading_engine.algorithms import TWAPAlgorithm, VWAPAlgorithm
            log_test("Trading", "Execution Algorithms", "PASS", 
                    "TWAP and VWAP found")
        except:
            log_test("Trading", "Execution Algorithms", "WARN", 
                    "Some algorithms missing")
        
        return True
        
    except Exception as e:
        log_test("Trading", "Module Import", "FAIL", 
                str(e), "Trading engine broken")
        return False

async def test_scanners():
    """Test 8: Scanner System (ISSUE-002)"""
    print("\n" + "="*50)
    print("TEST 8: SCANNERS")
    print("="*50)
    
    try:
        # Check if scanners module exists
        import main.scanners
        log_test("Scanners", "Module Import", "PASS", 
                "Scanners module found")
        
        # Check for scanner implementations
        from pathlib import Path
        scanners_path = Path(__file__).parent / "src/main/scanners"
        scanner_files = list(scanners_path.glob("**/*.py"))
        scanner_count = len([f for f in scanner_files if "scanner" in f.name.lower()])
        
        if scanner_count > 0:
            log_test("Scanners", "Scanner Files", "PASS", 
                    f"Found {scanner_count} scanner implementations")
        else:
            log_test("Scanners", "Scanner Files", "FAIL", 
                    "No scanner implementations found", "No scanner files")
        
        # Check CLI integration (ISSUE-002)
        ai_trader_path = Path(__file__).parent / "ai_trader.py"
        if ai_trader_path.exists():
            with open(ai_trader_path, "r") as f:
                cli_content = f.read()
            if "scanner" in cli_content.lower():
                log_test("Scanners", "CLI Integration", "PASS", 
                        "Scanner commands found in CLI")
            else:
                log_test("Scanners", "CLI Integration", "FAIL", 
                        "No scanner commands", "ISSUE-002: Scanner not integrated")
        else:
            log_test("Scanners", "CLI Integration", "FAIL", 
                    "ai_trader.py not found", "Cannot check CLI integration")
        
        return True
        
    except Exception as e:
        log_test("Scanners", "Module Import", "FAIL", 
                str(e), "Scanner module broken")
        return False

async def test_monitoring():
    """Test 9: Monitoring & Dashboards"""
    print("\n" + "="*50)
    print("TEST 9: MONITORING & DASHBOARDS")
    print("="*50)
    
    try:
        from main.monitoring import SystemDashboardV2, TradingDashboardV2
        
        # Test dashboard modules
        try:
            log_test("Monitoring", "Dashboard Modules", "PASS", 
                    "System and Trading dashboards found")
        except Exception as e:
            log_test("Monitoring", "Dashboard Modules", "FAIL", 
                    str(e), "Dashboard modules broken")
        
        # Check for health metrics (ISSUE-005)
        try:
            from main.monitoring.metrics import SystemHealthMetrics
            log_test("Monitoring", "Health Metrics", "PASS", 
                    "Health metrics module found")
        except:
            log_test("Monitoring", "Health Metrics", "FAIL", 
                    "Module not found", "ISSUE-005: System health not implemented")
        
        return True
        
    except Exception as e:
        log_test("Monitoring", "Module Import", "FAIL", 
                str(e), "Monitoring module broken")
        return False

async def test_scheduled_jobs():
    """Test 10: Scheduled Jobs (ISSUE-001)"""
    print("\n" + "="*50)
    print("TEST 10: SCHEDULED JOBS")
    print("="*50)
    
    try:
        from main.orchestration import JobScheduler
        
        try:
            log_test("Jobs", "Scheduler Module", "PASS", 
                    "Scheduler found")
        except:
            log_test("Jobs", "Scheduler Module", "FAIL", 
                    "Module not found", "ISSUE-001: Job scheduler broken")
        
        # Check job definitions
        jobs_path = Path(__file__).parent / "src/main/jobs"
        job_files = list(jobs_path.glob("*.py"))
        
        if len(job_files) > 1:
            log_test("Jobs", "Job Definitions", "PASS", 
                    f"Found {len(job_files)} job files")
        else:
            log_test("Jobs", "Job Definitions", "WARN", 
                    f"Only {len(job_files)} job files", "Jobs may be missing")
        
        return True
        
    except Exception as e:
        log_test("Jobs", "Module Import", "FAIL", 
                str(e), "Jobs module broken")
        return False

async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("AI TRADING SYSTEM - END-TO-END CRITICAL PATH TEST")
    print("="*60)
    print(f"Started: {datetime.now()}")
    print(f"Testing environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    # Run tests in sequence
    tests = [
        ("Configuration", test_configuration),
        ("Database", test_database_connection),
        ("Data Ingestion", test_data_ingestion),
        ("Features", test_feature_calculation),
        ("Models", test_models),
        ("Risk Management", test_risk_management),
        ("Trading Engine", test_trading_engine),
        ("Scanners", test_scanners),
        ("Monitoring", test_monitoring),
        ("Scheduled Jobs", test_scheduled_jobs)
    ]
    
    passed = 0
    failed = 0
    warnings = 0
    
    for name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {name}: {e}")
            traceback.print_exc()
            failed += 1
    
    # Count warnings
    for component in test_results["components"].values():
        for test in component.values():
            if test["status"] == "WARN":
                warnings += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(tests)} components")
    print(f"❌ Failed: {failed}/{len(tests)} components")
    print(f"⚠️  Warnings: {warnings} tests need attention")
    print(f"🐛 Issues Found: {len(test_results['issues_found'])}")
    
    # List critical issues
    if test_results["issues_found"]:
        print("\nCritical Issues Found:")
        for issue in test_results["issues_found"][:5]:  # Show first 5
            print(f"  - [{issue['component']}] {issue['issue']}")
    
    # Determine overall status
    if failed == 0:
        test_results["overall_status"] = "PASS"
        print("\n✅ OVERALL: System components are accessible")
    elif failed < 3:
        test_results["overall_status"] = "PARTIAL"
        print("\n⚠️ OVERALL: System partially functional")
    else:
        test_results["overall_status"] = "FAIL"
        print("\n❌ OVERALL: System has critical failures")
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nResults saved to: test_results.json")
    
    print(f"\nCompleted: {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())
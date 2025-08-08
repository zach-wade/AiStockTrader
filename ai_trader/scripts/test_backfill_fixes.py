#!/usr/bin/env python3
"""
Test script to verify backfill fixes are working properly.

This script tests:
1. Database migration execution 
2. S&P 500 data population
3. Symbol tier categorization with fallbacks
4. Session management and cleanup
5. Small-scale backfill functionality

Usage:
    python scripts/test_backfill_fixes.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main.config.config_manager import get_config
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.services.sp500_population_service import SP500PopulationService
from main.data_pipeline.backfill.symbol_tiers import SymbolTierManager
from main.data_pipeline.types import BackfillParams, DataSource, TimeInterval, DataType
from main.app.historical_backfill import create_historical_manager
from main.utils.core import get_logger
from main.utils.api.session_helpers import cleanup_orphaned_sessions
from datetime import datetime, timezone, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)


class BackfillFixesTester:
    """Test runner for backfill fixes."""
    
    def __init__(self):
        self.config = get_config()
        self.db_factory = DatabaseFactory()
        self.db_adapter = None
        self.test_results = {}
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.db_adapter = self.db_factory.create_async_database(self.config)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.db_adapter and hasattr(self.db_adapter, 'close'):
            await self.db_adapter.close()
        
        # Check for orphaned sessions
        await cleanup_orphaned_sessions()
    
    async def test_database_schema(self) -> Dict[str, Any]:
        """Test that the is_sp500 column exists and is queryable."""
        logger.info("🧪 Testing database schema...")
        
        try:
            # Test if is_sp500 column exists
            async with self.db_adapter.acquire() as conn:
                # First check if column exists
                column_check = await conn.fetchrow("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'companies' 
                    AND column_name = 'is_sp500'
                """)
                
                if not column_check:
                    return {
                        'success': False,
                        'error': 'is_sp500 column does not exist - run migration first'
                    }
                
                # Test querying the column
                test_query = await conn.fetchrow("""
                    SELECT COUNT(*) as total, 
                           COUNT(CASE WHEN is_sp500 = true THEN 1 END) as sp500_count
                    FROM companies
                """)
                
                return {
                    'success': True,
                    'column_exists': True,
                    'total_companies': test_query['total'],
                    'sp500_companies': test_query['sp500_count']
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_sp500_population(self) -> Dict[str, Any]:
        """Test S&P 500 data population service."""
        logger.info("🧪 Testing S&P 500 population service...")
        
        try:
            async with SP500PopulationService(self.db_adapter) as service:
                # Test data population
                result = await service.populate_sp500_data(force_refresh=True)
                
                if result['success']:
                    # Validate the data
                    validation = await service.validate_sp500_data()
                    return {
                        'success': True,
                        'population_result': result,
                        'validation_result': validation
                    }
                else:
                    return {
                        'success': False,
                        'error': result.get('error', 'Unknown error'),
                        'population_result': result
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_symbol_tiers(self) -> Dict[str, Any]:
        """Test symbol tier manager with fallback mechanisms."""
        logger.info("🧪 Testing symbol tier categorization...")
        
        try:
            tier_manager = SymbolTierManager(self.db_adapter)
            
            # Test with a mix of known and unknown symbols
            test_symbols = [
                'AAPL', 'MSFT', 'GOOGL',  # Should be priority tier
                'UNKNOWN_SYMBOL_123',      # Should fallback gracefully
                'TEST_SYMBOL_XYZ'         # Should fallback gracefully
            ]
            
            # Test categorization
            tier_results = await tier_manager.categorize_symbols(test_symbols)
            
            # Count symbols in each tier
            tier_counts = {tier.name: len(symbols) for tier, symbols in tier_results.items()}
            
            return {
                'success': True,
                'test_symbols': test_symbols,
                'tier_results': {tier.name: symbols for tier, symbols in tier_results.items()},
                'tier_counts': tier_counts
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_small_backfill(self) -> Dict[str, Any]:
        """Test a small backfill operation to verify session management."""
        logger.info("🧪 Testing small backfill operation...")
        
        try:
            # Create a small backfill params for testing
            test_symbols = ['AAPL', 'MSFT'] 
            backfill_params = BackfillParams(
                symbols=test_symbols,
                sources=[DataSource.POLYGON],
                intervals=[TimeInterval.DAY_1],
                start_date=datetime.now(timezone.utc) - timedelta(days=7),
                end_date=datetime.now(timezone.utc) - timedelta(days=1),
                data_types=[DataType.MARKET_DATA],
                max_concurrent=2,
                retry_failed=False
            )
            
            # Create historical manager
            historical_manager = await create_historical_manager(self.config)
            
            try:
                # Run small backfill
                results = await historical_manager.backfill_symbols(backfill_params)
                
                return {
                    'success': True,
                    'backfill_results': results,
                    'symbols_processed': results.get('symbols_processed', 0),
                    'records_downloaded': results.get('records_downloaded', 0)
                }
                
            finally:
                # Ensure cleanup
                await historical_manager.close()
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        logger.info("🚀 Starting backfill fixes test suite...")
        
        tests = [
            ('database_schema', self.test_database_schema),
            ('sp500_population', self.test_sp500_population),
            ('symbol_tiers', self.test_symbol_tiers),
            ('small_backfill', self.test_small_backfill)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                result = await test_func()
                results[test_name] = result
                
                if result.get('success'):
                    logger.info(f"✅ {test_name} PASSED")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name} FAILED: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"💥 {test_name} CRASHED: {e}")
                results[test_name] = {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # Summary
        success_rate = passed / total
        overall_success = success_rate == 1.0
        
        summary = {
            'overall_success': overall_success,
            'tests_passed': passed,
            'tests_total': total,
            'success_rate': success_rate,
            'test_results': results
        }
        
        logger.info(f"🏁 Test suite completed: {passed}/{total} tests passed ({success_rate:.1%})")
        
        return summary


async def main():
    """Main test runner."""
    async with BackfillFixesTester() as tester:
        results = await tester.run_all_tests()
        
        # Print detailed results
        print("\n" + "="*80)
        print("BACKFILL FIXES TEST RESULTS")
        print("="*80)
        
        for test_name, test_result in results['test_results'].items():
            status = "✅ PASS" if test_result.get('success') else "❌ FAIL"
            print(f"{test_name.upper()}: {status}")
            
            if not test_result.get('success'):
                print(f"  Error: {test_result.get('error', 'Unknown')}")
            else:
                # Print specific success details
                if test_name == 'database_schema':
                    print(f"  Total companies: {test_result.get('total_companies', 0)}")
                    print(f"  S&P 500 companies: {test_result.get('sp500_companies', 0)}")
                    
                elif test_name == 'sp500_population':
                    pop_result = test_result.get('population_result', {})
                    print(f"  Symbols updated: {pop_result.get('symbols_count', 0)}")
                    print(f"  Data source: {pop_result.get('data_source', 'unknown')}")
                    
                elif test_name == 'symbol_tiers':
                    tier_counts = test_result.get('tier_counts', {})
                    for tier, count in tier_counts.items():
                        if count > 0:
                            print(f"  {tier}: {count} symbols")
                            
                elif test_name == 'small_backfill':
                    print(f"  Symbols processed: {test_result.get('symbols_processed', 0)}")
                    print(f"  Records downloaded: {test_result.get('records_downloaded', 0)}")
            
            print()
        
        print("="*80)
        print(f"SUMMARY: {results['tests_passed']}/{results['tests_total']} tests passed")
        
        if results['overall_success']:
            print("🎉 All tests passed! Backfill fixes are working correctly.")
            return 0
        else:
            print("⚠️  Some tests failed. Check the detailed results above.")
            return 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
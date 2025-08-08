"""
System Validation Runner

This module provides comprehensive validation for all AI Trader system components.
It validates data pipeline, features, models, and trading components to ensure
the system is functioning correctly.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path

from main.utils.core import get_logger
from main.config.config_manager import get_config_manager
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.storage.archive import DataArchive
# Repository imports removed - using factory pattern
from main.feature_pipeline.feature_store import FeatureStore
from main.models.inference.model_registry import ModelRegistry
from main.trading_engine.brokers.broker_interface import BrokerInterface

logger = get_logger(__name__)


class ValidationRunner:
    """
    Validates system components and reports health status.
    
    This class performs comprehensive validation checks on:
    - Data pipeline components (database, archive, repositories)
    - Feature pipeline (feature store, calculators)
    - Model components (registry, inference)
    - Trading engine (broker connection, risk management)
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize validation runner."""
        self.config = config or get_config_manager().load_config('unified_config')
        self.logger = get_logger(f"{__name__}.ValidationRunner")
        self.results: Dict[str, Dict[str, Any]] = {}
        
    async def validate(self, component: str = 'all') -> Dict[str, Dict[str, Any]]:
        """
        Run validation for specified component(s).
        
        Args:
            component: Component to validate ('all', 'data', 'features', 'models', 'trading')
            
        Returns:
            Dictionary with validation results for each component
        """
        self.logger.info(f"Starting validation for: {component}")
        
        if component == 'all':
            await self._validate_data_pipeline()
            await self._validate_feature_pipeline()
            await self._validate_models()
            await self._validate_trading()
        elif component == 'data':
            await self._validate_data_pipeline()
        elif component == 'features':
            await self._validate_feature_pipeline()
        elif component == 'models':
            await self._validate_models()
        elif component == 'trading':
            await self._validate_trading()
        else:
            raise ValueError(f"Unknown component: {component}")
            
        return self.results
    
    async def _validate_data_pipeline(self):
        """Validate data pipeline components."""
        self.logger.info("Validating data pipeline...")
        
        errors = []
        warnings = []
        
        try:
            # Validate database connection
            db_factory = DatabaseFactory()
            db_adapter = db_factory.create_async_database(self.config)
            
            # Test basic query
            result = await db_adapter.fetch_one("SELECT 1 as test")
            if not result or result.get('test') != 1:
                errors.append("Database connectivity test failed")
            else:
                self.logger.info("✅ Database connection validated")
            
            # Validate repositories
            from main.data_pipeline.storage.repositories import get_repository_factory
            repo_factory = get_repository_factory()
            company_repo = repo_factory.create_company_repository(db_adapter)
            market_repo = repo_factory.create_market_data_repository(db_adapter)
            news_repo = repo_factory.create_news_repository(db_adapter)
            financials_repo = repo_factory.create_financials_repository(db_adapter)
            
            # Check company count
            company_count = await company_repo.get_total_count()
            if company_count == 0:
                warnings.append("No companies in database")
            else:
                self.logger.info(f"✅ Found {company_count} companies")
            
            # Check layer qualifications
            for layer in range(4):
                count = await company_repo.get_layer_count(layer)
                self.logger.info(f"  Layer {layer}: {count} symbols")
                
            # Check market data
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=7)
            
            # Get a sample symbol
            sample_symbols = await company_repo.get_layer_symbols(1, limit=1)
            if sample_symbols:
                symbol = sample_symbols[0]
                market_data = await market_repo.get_data(
                    symbol=symbol,
                    interval='1day',
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not market_data:
                    warnings.append(f"No recent market data for {symbol}")
                else:
                    self.logger.info(f"✅ Market data validated ({len(market_data)} records for {symbol})")
            
            # Validate archive
            archive_config = self.config.get('data_pipeline.storage.archive', {})
            archive = DataArchive(archive_config)
            
            # Check archive storage
            storage_path = Path(archive_config.get('local_path', 'data_lake'))
            if not storage_path.exists():
                errors.append(f"Archive storage path not found: {storage_path}")
            else:
                # Count archived files
                parquet_files = list(storage_path.rglob("*.parquet"))
                self.logger.info(f"✅ Archive validated ({len(parquet_files)} parquet files)")
            
            # Close database connection
            await db_adapter.close()
            
        except Exception as e:
            errors.append(f"Data pipeline validation error: {str(e)}")
            self.logger.error(f"Data pipeline validation failed: {e}", exc_info=True)
        
        self.results['data_pipeline'] = {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _validate_feature_pipeline(self):
        """Validate feature pipeline components."""
        self.logger.info("Validating feature pipeline...")
        
        errors = []
        warnings = []
        
        try:
            # Initialize feature store
            feature_store = FeatureStore(self.config)
            
            # Check if feature store is accessible
            db_factory = DatabaseFactory()
            db_adapter = db_factory.create_async_database(self.config)
            
            # Test feature store table exists
            table_check = await db_adapter.fetch_one("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'features'
                )
            """)
            
            if not table_check or not table_check.get('exists'):
                warnings.append("Features table does not exist")
            else:
                # Check feature count
                feature_count = await db_adapter.fetch_one(
                    "SELECT COUNT(*) as count FROM features"
                )
                count = feature_count.get('count', 0) if feature_count else 0
                
                if count == 0:
                    warnings.append("No features in feature store")
                else:
                    self.logger.info(f"✅ Feature store validated ({count} features)")
            
            # Validate calculator factory
            from main.feature_pipeline.calculator_factory import get_calculator_factory
            calculator_factory = get_calculator_factory()
            
            # Test calculator creation
            calculators = ['technical', 'sentiment', 'fundamental']
            for calc_type in calculators:
                try:
                    calculator = calculator_factory.create_calculator(calc_type)
                    if calculator:
                        self.logger.info(f"✅ {calc_type.capitalize()} calculator validated")
                    else:
                        warnings.append(f"Could not create {calc_type} calculator")
                except Exception as e:
                    errors.append(f"Calculator {calc_type} error: {str(e)}")
            
            await db_adapter.close()
            
        except Exception as e:
            errors.append(f"Feature pipeline validation error: {str(e)}")
            self.logger.error(f"Feature pipeline validation failed: {e}", exc_info=True)
        
        self.results['feature_pipeline'] = {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _validate_models(self):
        """Validate model components."""
        self.logger.info("Validating models...")
        
        errors = []
        warnings = []
        
        try:
            # Initialize model registry
            model_registry = ModelRegistry(self.config)
            
            # Check registered models
            registered_models = model_registry.list_models()
            
            if not registered_models:
                warnings.append("No models registered")
            else:
                self.logger.info(f"✅ Found {len(registered_models)} registered models")
                
                # Check production models
                production_models = model_registry.get_production_models()
                if not production_models:
                    warnings.append("No production models deployed")
                else:
                    self.logger.info(f"✅ {len(production_models)} production models deployed")
            
            # Validate model paths
            model_dir = Path(self.config.get('ml.model_storage_path', 'models'))
            if not model_dir.exists():
                warnings.append(f"Model storage directory not found: {model_dir}")
            else:
                model_files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))
                self.logger.info(f"✅ Model storage validated ({len(model_files)} model files)")
            
            # Test inference capability
            if production_models:
                try:
                    # Get a sample model
                    model_id = list(production_models.keys())[0]
                    model_version = production_models[model_id]
                    
                    # Try to load the model
                    model = await model_registry.load_model(model_id, model_version.version)
                    if model:
                        self.logger.info(f"✅ Model loading validated (model: {model_id})")
                    else:
                        errors.append(f"Could not load model: {model_id}")
                except Exception as e:
                    errors.append(f"Model loading error: {str(e)}")
                    
        except Exception as e:
            errors.append(f"Model validation error: {str(e)}")
            self.logger.error(f"Model validation failed: {e}", exc_info=True)
        
        self.results['models'] = {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _validate_trading(self):
        """Validate trading components."""
        self.logger.info("Validating trading engine...")
        
        errors = []
        warnings = []
        
        try:
            # Check broker configuration
            broker_config = self.config.get('trading.broker', {})
            
            if not broker_config:
                warnings.append("No broker configuration found")
            else:
                # Validate API keys
                api_key = broker_config.get('api_key')
                api_secret = broker_config.get('api_secret')
                
                if not api_key or not api_secret:
                    errors.append("Missing broker API credentials")
                else:
                    self.logger.info("✅ Broker credentials configured")
            
            # Initialize broker interface
            try:
                from main.trading_engine.brokers.alpaca_broker import AlpacaBroker
                broker = AlpacaBroker(self.config)
                
                # Test connection (without actually connecting in validation)
                if hasattr(broker, 'validate_connection'):
                    is_valid = await broker.validate_connection()
                    if is_valid:
                        self.logger.info("✅ Broker connection validated")
                    else:
                        warnings.append("Broker connection validation failed")
                else:
                    self.logger.info("⚠️ Broker connection validation not available")
                    
            except ImportError:
                errors.append("Broker interface not found")
            except Exception as e:
                errors.append(f"Broker initialization error: {str(e)}")
            
            # Check risk management configuration
            risk_config = self.config.get('trading.risk_management', {})
            
            required_risk_params = ['max_position_size', 'max_portfolio_risk', 'stop_loss_pct']
            missing_params = [p for p in required_risk_params if p not in risk_config]
            
            if missing_params:
                warnings.append(f"Missing risk parameters: {missing_params}")
            else:
                self.logger.info("✅ Risk management configured")
            
            # Check execution algorithms
            algo_config = self.config.get('trading.execution_algorithms', {})
            if not algo_config:
                warnings.append("No execution algorithms configured")
            else:
                algos = list(algo_config.keys())
                self.logger.info(f"✅ Execution algorithms configured: {algos}")
                
        except Exception as e:
            errors.append(f"Trading validation error: {str(e)}")
            self.logger.error(f"Trading validation failed: {e}", exc_info=True)
        
        self.results['trading'] = {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        total_passed = sum(1 for r in self.results.values() if r['passed'])
        total_components = len(self.results)
        
        return {
            'total_components': total_components,
            'passed': total_passed,
            'failed': total_components - total_passed,
            'success_rate': (total_passed / total_components * 100) if total_components > 0 else 0,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': self.results
        }


async def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate AI Trader system components')
    parser.add_argument(
        '--component',
        choices=['all', 'data', 'features', 'models', 'trading'],
        default='all',
        help='Component to validate'
    )
    parser.add_argument(
        '--output',
        help='Output file for validation results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Run validation
    runner = ValidationRunner()
    results = await runner.validate(args.component)
    
    # Print results
    print("\n=== Validation Results ===\n")
    for component, result in results.items():
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"{component}: {status}")
        
        if result.get('errors'):
            print("  Errors:")
            for error in result['errors']:
                print(f"    - {error}")
                
        if result.get('warnings'):
            print("  Warnings:")
            for warning in result['warnings']:
                print(f"    ⚠️ {warning}")
    
    # Get summary
    summary = runner.get_summary()
    print(f"\nOverall Success Rate: {summary['success_rate']:.1f}%")
    print(f"Components Passed: {summary['passed']}/{summary['total_components']}")
    
    # Save to file if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with error if validation failed
    if summary['failed'] > 0:
        exit(1)


if __name__ == '__main__':
    asyncio.run(main())
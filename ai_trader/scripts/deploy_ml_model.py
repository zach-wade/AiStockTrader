"""
Deploy ML Model Script

This script helps deploy trained ML models to production status
in the ModelRegistry for use with live trading.
"""

import asyncio
import logging
from pathlib import Path
import sys
import argparse
from datetime import datetime
import joblib
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from main.config.config_manager import get_config
from main.models.inference.model_registry import ModelRegistry
from main.models.inference.prediction_engine import PredictionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def list_available_models(model_registry: ModelRegistry):
    """List all available models in the registry."""
    models = model_registry.list_models()
    
    if not models:
        logger.info("No models found in registry")
        return
    
    logger.info(f"\nFound {len(models)} models in registry:")
    logger.info("-" * 80)
    logger.info(f"{'Model ID':<20} {'Version':<10} {'Status':<15} {'Created':<20} {'Metrics'}")
    logger.info("-" * 80)
    
    for model in models:
        metrics_str = ""
        if model.get('metrics'):
            mae = model['metrics'].get('mae', 'N/A')
            r2 = model['metrics'].get('r2', 'N/A')
            metrics_str = f"MAE: {mae:.4f}, R²: {r2:.4f}" if isinstance(mae, float) else "N/A"
        
        created = model.get('created_at', 'Unknown')
        if isinstance(created, str):
            created = created.split('T')[0]  # Just show date
        
        logger.info(f"{model['model_id']:<20} {model['version']:<10} {model['status']:<15} {created:<20} {metrics_str}")


async def deploy_model_from_file(model_registry: ModelRegistry, model_path: Path, model_id: str):
    """Deploy a model from a file path."""
    try:
        # Check if model file exists
        model_file = model_path / "model.pkl"
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return False
        
        # Load model and metadata
        model = joblib.load(model_file)
        
        # Load metadata if exists
        metadata_file = model_path / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Load metrics if exists
        metrics_file = model_path / "metrics.json"
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        
        # Extract features from metadata
        features = metadata.get('feature_columns', [])
        
        # Register model
        logger.info(f"Registering model {model_id} from {model_path}")
        
        model_version = model_registry.register_model(
            model=model,
            model_id=model_id,
            model_type=metadata.get('model_type', 'xgboost'),
            training_data_range=metadata.get('training_date_range', 'Unknown'),
            hyperparameters=metadata.get('hyperparameters', {}),
            metrics=metrics,
            features=features,
            metadata=metadata
        )
        
        logger.info(f"Model registered as {model_id} version {model_version.version}")
        return model_version
        
    except Exception as e:
        logger.error(f"Failed to deploy model from file: {e}")
        return None


async def deploy_model_to_production(model_registry: ModelRegistry, model_id: str, version: str):
    """Deploy a specific model version to production."""
    try:
        # Get the model version
        model_version = model_registry.get_model_version(model_id, version)
        if not model_version:
            logger.error(f"Model {model_id} version {version} not found")
            return False
        
        # Check current status
        if model_version.status == 'production':
            logger.info(f"Model {model_id} version {version} is already in production")
            return True
        
        # Update status to production
        model_version.status = 'production'
        model_version.deployment_pct = 100.0
        
        # Save updated registry
        model_registry._save_registry_state()
        
        # Update active deployments
        model_registry.active_deployments[model_id] = model_version
        
        logger.info(f"✓ Model {model_id} version {version} deployed to production")
        return True
        
    except Exception as e:
        logger.error(f"Failed to deploy model to production: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Deploy ML models for trading")
    parser.add_argument('--list', action='store_true', help='List all available models')
    parser.add_argument('--deploy', type=str, help='Deploy model by ID (e.g., aapl_xgboost)')
    parser.add_argument('--version', type=str, help='Specific version to deploy (default: latest)')
    parser.add_argument('--from-file', type=str, help='Deploy model from file path')
    parser.add_argument('--model-id', type=str, help='Model ID when deploying from file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Initialize components
    models_dir = Path(config.get('paths', {}).get('models', 'models/trained'))
    prediction_engine = PredictionEngine(config)
    model_registry = ModelRegistry(
        models_dir=models_dir,
        prediction_engine=prediction_engine,
        config=config
    )
    
    # Handle different commands
    if args.list:
        await list_available_models(model_registry)
    
    elif args.from_file:
        if not args.model_id:
            logger.error("--model-id required when using --from-file")
            return
        
        model_path = Path(args.from_file)
        if not model_path.exists():
            logger.error(f"Path not found: {model_path}")
            return
        
        model_version = await deploy_model_from_file(model_registry, model_path, args.model_id)
        if model_version:
            # Automatically deploy to production
            await deploy_model_to_production(model_registry, args.model_id, model_version.version)
    
    elif args.deploy:
        # Deploy existing model to production
        model_id = args.deploy
        
        if args.version:
            version = args.version
        else:
            # Get latest version
            latest = model_registry.get_latest_version(model_id)
            if not latest:
                logger.error(f"No versions found for model {model_id}")
                return
            version = latest.version
        
        await deploy_model_to_production(model_registry, model_id, version)
    
    else:
        # Default: check for AAPL model in default location
        logger.info("Checking for trained AAPL model...")
        
        aapl_model_path = Path("models/aapl_xgboost")
        if aapl_model_path.exists():
            logger.info(f"Found AAPL model at {aapl_model_path}")
            
            # Check if already registered
            aapl_models = [m for m in model_registry.list_models() if m['model_id'] == 'aapl_xgboost']
            
            if not aapl_models:
                logger.info("Registering AAPL model...")
                model_version = await deploy_model_from_file(model_registry, aapl_model_path, 'aapl_xgboost')
                if model_version:
                    await deploy_model_to_production(model_registry, 'aapl_xgboost', model_version.version)
            else:
                # Check if any in production
                production_models = [m for m in aapl_models if m['status'] == 'production']
                if not production_models:
                    # Deploy latest to production
                    latest = max(aapl_models, key=lambda x: x['created_at'])
                    await deploy_model_to_production(model_registry, 'aapl_xgboost', latest['version'])
                else:
                    logger.info("✓ AAPL model already deployed to production")
        else:
            logger.warning(f"No AAPL model found at {aapl_model_path}")
            logger.info("\nTo deploy a model:")
            logger.info("1. Train a model first")
            logger.info("2. Use --from-file to register it")
            logger.info("3. Or use --deploy to promote existing model")


if __name__ == "__main__":
    asyncio.run(main())
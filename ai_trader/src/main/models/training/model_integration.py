"""
Model Integration Utility Script

This script scans for newly trained model artifacts and registers them with the
official ModelRegistry. This is intended to be run from the command line as
part of the model deployment process.

Usage:
    python -m main.models.training.model_integration --path /path/to/trained_models
"""
import logging
import json
from pathlib import Path
import joblib
import argparse

# It's better to make scripts runnable as modules to handle Python paths correctly
from main.config.config_manager import get_config
from main.models.inference.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelIntegrator:
    """A utility to register trained model artifacts into the system."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_registry = ModelRegistry(config)

    def run(self, trained_models_dir: Path):
        """
        Scans the given directory, finds valid model artifacts and their
        metadata, and registers them.
        """
        if not trained_models_dir.exists():
            logger.error(f"Trained models directory not found: {trained_models_dir}")
            return
        
        logger.info(f"Scanning for models in: {trained_models_dir}")
        model_files = list(trained_models_dir.glob("*.pkl"))
        
        registered_count = 0
        for model_file in model_files:
            try:
                # Expect metadata file to exist with the same stem
                metadata_file = model_file.with_suffix(".json")
                if not metadata_file.exists():
                    logger.warning(f"Skipping model {model_file.name}: Missing metadata file {metadata_file.name}")
                    continue

                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Validate metadata has required fields
                required_keys = ['model_id', 'model_type', 'metrics', 'features']
                if not all(key in metadata for key in required_keys):
                    logger.warning(f"Skipping model {model_file.name}: Metadata is missing required keys.")
                    continue

                # Load the model artifact
                model = joblib.load(model_file)
                
                # Register with the central registry
                self.model_registry.register_model(
                    model=model,
                    model_id=metadata['model_id'],
                    model_type=metadata['model_type'],
                    metrics=metadata['metrics'],
                    features=metadata['features'],
                    hyperparameters=metadata.get('hyperparameters', {}),
                    training_data_info=metadata.get('training_data_info', {})
                )
                
                logger.info(f"✅ Registered model: {metadata['model_id']} (AUC: {metadata['metrics'].get('auc', 'N/A'):.4f})")
                registered_count += 1
                
            except Exception as e:
                logger.error(f"Failed to register model {model_file.name}: {e}", exc_info=True)
        
        logger.info(f"🎯 Model registration complete. Registered {registered_count} new models.")

def main():
    """Main entry point for the command-line utility."""
    parser = argparse.ArgumentParser(description="Register trained models into the system.")
    parser.add_argument(
        "--path",
        type=str,
        default="models/trained",
        help="Directory containing the trained model artifacts (.pkl and .json files)."
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("="*60)
    logger.info("🔧 MODEL INTEGRATION UTILITY")
    logger.info("="*60)
    
    config = get_config()
    integrator = ModelIntegrator(config)
    integrator.run(trained_models_dir=Path(args.path))
    
    logger.info("\nNext steps:")
    logger.info("1. Manually edit 'config.yaml' to enable strategies that use these models.")
    logger.info("2. Commit the configuration change to version control.")
    logger.info("3. Deploy the application with the new configuration.")

if __name__ == "__main__":
    main()
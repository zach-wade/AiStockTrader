# File: src/ai_trader/models/inference/model_registry_helpers/model_file_manager.py

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Use secure serializer instead of pickle for security
from main.utils.core.secure_serializer import SecureSerializer

# Corrected absolute import for ModelVersion (used for path generation)
from main.models.inference.model_registry_types import ModelVersion

logger = logging.getLogger(__name__)

class ModelFileManager:
    """
    Manages the persistent storage and retrieval of serialized model binary files
    (e.g., .pkl files) on disk.
    """

    def __init__(self, models_base_dir: Path):
        """
        Initializes the ModelFileManager.

        Args:
            models_base_dir: The base directory where actual model files are stored.
        """
        self.models_base_dir = models_base_dir
        self.models_base_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        logger.debug(f"ModelFileManager initialized for base directory: {models_base_dir}")

    def get_model_path(self, model_id: str, version: str) -> Path:
        """
        Generates the standard file path for a given model version.

        Args:
            model_id: The identifier of the model.
            version: The version string of the model.

        Returns:
            A Path object representing the expected file location.
        """
        return self.models_base_dir / f"{model_id}_{version}.pkl"

    def save_model(self, model_obj: Any, version_info: ModelVersion) -> Path:
        """
        Serializes and saves a trained model object to disk.

        Args:
            model_obj: The trained model object (e.g., scikit-learn, XGBoost model instance).
            version_info: The ModelVersion object associated with this model.

        Returns:
            The Path object where the model was saved.

        Raises:
            IOError: If the model file cannot be saved.
        """
        model_path = self.get_model_path(version_info.model_id, version_info.version)
        
        # Create a dictionary to save, which might include features for convenience on load
        model_data_to_save = {
            'model': model_obj,
            'version_info': version_info.to_dict(), # Store full metadata with model for self-containment
            'features': version_info.features # Redundant with version_info, but often useful for quick load
        }

        try:
            # Use secure serializer for model persistence
            serializer = SecureSerializer()
            serialized_data = serializer.serialize(model_data_to_save)
            
            with open(model_path, 'wb') as f:
                f.write(serialized_data)
            logger.info(f"Saved model '{version_info.model_id}' version '{version_info.version}' to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model {version_info.model_id} v{version_info.version} to {model_path}: {e}", exc_info=True)
            raise IOError(f"Could not save model file: {e}")

    def load_model(self, model_id: str, version: str) -> Tuple[Any, List[str], Dict[str, Any]]:
        """
        Loads a serialized model object from disk.

        Args:
            model_id: The identifier of the model.
            version: The version string of the model.

        Returns:
            A tuple: (model_object, list_of_features, metadata_dict).

        Raises:
            FileNotFoundError: If the model file does not exist.
            IOError: If the model file cannot be loaded/deserialized.
        """
        model_path = self.get_model_path(model_id, version)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            # Use secure serializer for model loading
            serializer = SecureSerializer()
            
            with open(model_path, 'rb') as f:
                serialized_data = f.read()
            
            model_data = serializer.deserialize(serialized_data)
            
            model_obj = model_data.get('model')
            features = model_data.get('features', [])
            version_metadata = model_data.get('version_info', {}) # Get full version info if saved
            
            if model_obj is None:
                raise ValueError(f"Model object is None after loading from {model_path}.")
            
            logger.debug(f"Loaded model '{model_id}' version '{version}' from {model_path}.")
            return model_obj, features, version_metadata
        except Exception as e:
            logger.error(f"Failed to load model {model_id} v{version} from {model_path}: {e}", exc_info=True)
            raise IOError(f"Could not load model file: {e}")

    def delete_model_file(self, model_id: str, version: str) -> bool:
        """
        Deletes a model's binary file from disk.

        Args:
            model_id: The identifier of the model.
            version: The version string of the model.

        Returns:
            True if deletion was successful or file did not exist, False on error.
        """
        model_path = self.get_model_path(model_id, version)
        if model_path.exists():
            try:
                model_path.unlink()
                logger.info(f"Deleted model file: {model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete model file {model_path}: {e}", exc_info=True)
                return False
        logger.debug(f"Model file not found for deletion: {model_path}")
        return True # Considered successful if file already doesn't exist
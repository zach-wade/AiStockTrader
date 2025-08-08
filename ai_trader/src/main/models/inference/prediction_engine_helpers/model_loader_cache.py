# File: src/ai_trader/models/inference/prediction_engine_helpers/model_loader_cache.py

import logging
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Use secure serializer instead of pickle for security
from main.utils.core.secure_serializer import SecureSerializer

logger = logging.getLogger(__name__)

class ModelLoaderCache:
    """
    Manages loading and in-memory caching of serialized model objects.
    Handles different serialization formats (pickle, joblib) and provides
    access to model metadata.
    """

    def __init__(self, models_base_dir: Path):
        """
        Initializes the ModelLoaderCache.

        Args:
            models_base_dir: The base directory where actual model files (.pkl, .joblib) are stored.
        """
        self.models_base_dir = models_base_dir
        self.models_base_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        self._loaded_models: Dict[str, Any] = {} # {model_name: model_object}
        self._model_metadata: Dict[str, Dict[str, Any]] = {} # {model_name: metadata_dict}
        
        logger.debug(f"ModelLoaderCache initialized for base directory: {models_base_dir}")

    def _get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Determines the correct model file path, trying common extensions.
        """
        for ext in ['.pkl', '.joblib']:
            model_path = self.models_base_dir / f"{model_name}{ext}"
            if model_path.exists():
                return model_path
        return None

    def load_model_object(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Loads a serialized model object from disk and caches it.
        If the model is already in cache, it's returned immediately.

        Args:
            model_name: The name or identifier of the model to load.

        Returns:
            A tuple: (model_object, metadata_dict).

        Raises:
            FileNotFoundError: If the model file does not exist.
            IOError: If the model file cannot be loaded/deserialized.
            ValueError: If loaded data is malformed.
        """
        if model_name in self._loaded_models:
            logger.debug(f"Model '{model_name}' found in cache.")
            return self._loaded_models[model_name], self._model_metadata[model_name]
        
        model_path = self._get_model_path(model_name)
        if not model_path:
            raise FileNotFoundError(f"Model file not found for '{model_name}' in {self.models_base_dir}")
        
        try:
            if model_path.suffix == '.joblib':
                model_data = joblib.load(model_path)
            else: # Assume .pkl
                # Use secure serializer for loading
                serializer = SecureSerializer()
                with open(model_path, 'rb') as f:
                    serialized_data = f.read()
                model_data = serializer.deserialize(serialized_data)
            
            # Extract model object and its metadata from loaded data
            model_obj: Any
            metadata: Dict[str, Any]
            
            if isinstance(model_data, dict):
                model_obj = model_data.get('model')
                metadata = model_data.get('metadata', {})
            else: # Assume legacy format where model object is directly the content
                model_obj = model_data
                metadata = {}
                logger.warning(f"Model '{model_name}' loaded in legacy format. No explicit metadata found.")

            if model_obj is None:
                raise ValueError(f"Model object is None after loading from {model_path}. File might be corrupted.")

            self._loaded_models[model_name] = model_obj
            self._model_metadata[model_name] = metadata
            
            logger.info(f"Loaded model '{model_name}' from disk: {model_path}")
            return model_obj, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}' from {model_path}: {e}", exc_info=True)
            raise IOError(f"Could not load model file for '{model_name}': {e}")

    def update_model_object_in_cache(self, model_name: str, model_obj: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Updates a model object and its metadata in the in-memory cache.
        Also saves the updated model to disk for persistence.

        Args:
            model_name: The name of the model to update.
            model_obj: The new trained model object.
            metadata: Optional. The new metadata for the model.
        """
        if not model_name:
            logger.error("Attempted to update model with empty model_name.")
            return
            
        self._loaded_models[model_name] = model_obj
        self._model_metadata[model_name] = metadata or {}
        
        # Persist the updated model to disk
        model_path = self.models_base_dir / f"{model_name}.pkl" # Default to .pkl for saving
        try:
            model_data_to_save = {
                'model': model_obj,
                'metadata': self._model_metadata[model_name]
            }
            # Use secure serializer for saving
            serializer = SecureSerializer()
            serialized_data = serializer.serialize(model_data_to_save)
            with open(model_path, 'wb') as f:
                f.write(serialized_data)
            logger.info(f"Updated model '{model_name}' in cache and saved to disk: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save updated model '{model_name}' to disk at {model_path}: {e}", exc_info=True)
            
    def unload_model_object(self, model_name: str):
        """Removes a model from the in-memory cache."""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            del self._model_metadata[model_name]
            logger.info(f"Model '{model_name}' unloaded from cache.")
        else:
            logger.debug(f"Model '{model_name}' not found in cache for unloading.")

    def get_loaded_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Gets information about loaded models.

        Args:
            model_name: Optional. If provided, returns info for that specific model.
                        Otherwise, returns info for all loaded models.

        Returns:
            A dictionary containing model information.
        """
        if model_name:
            if model_name not in self._loaded_models:
                return {'error': f'Model {model_name} not loaded'}
            
            return {
                'name': model_name,
                'loaded': True,
                'model_type': type(self._loaded_models[model_name]).__name__,
                **self._model_metadata[model_name] # Include all stored metadata
            }
        else:
            all_info = {}
            for name in self._loaded_models:
                all_info[name] = self.get_loaded_model_info(name)
            return all_info

    def get_all_loaded_model_names(self) -> List[str]:
        """Returns a list of names of all models currently loaded in cache."""
        return list(self._loaded_models.keys())
"""
Model loader utility for loading and caching trained ML models.

This module provides utilities for loading model artifacts,
validating compatibility, and caching loaded models for performance.
"""

import logging
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class ModelLoader:
    """
    Utility class for loading and caching ML models.
    
    Provides functionality to load model artifacts, validate them,
    and cache loaded models to avoid repeated disk I/O.
    """
    
    def __init__(self, cache_size: int = 10):
        """
        Initialize the model loader.
        
        Args:
            cache_size: Maximum number of models to keep in cache
        """
        self.cache_size = cache_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a model and its artifacts from disk.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Dictionary containing model, scaler, metadata, and other artifacts
            
        Raises:
            ModelLoadError: If model loading fails
        """
        model_path = Path(model_path)
        
        # Check cache first
        cache_key = self._get_cache_key(model_path)
        if cache_key in self._cache:
            logger.info(f"Loading model from cache: {model_path}")
            self._access_times[cache_key] = datetime.now()
            return self._cache[cache_key]
        
        # Load from disk
        logger.info(f"Loading model from disk: {model_path}")
        artifacts = self._load_from_disk(model_path)
        
        # Add to cache
        self._add_to_cache(cache_key, artifacts)
        
        return artifacts
    
    def _load_from_disk(self, model_path: Path) -> Dict[str, Any]:
        """Load model artifacts from disk."""
        if not model_path.exists():
            raise ModelLoadError(f"Model path does not exist: {model_path}")
        
        artifacts = {}
        
        # Load model
        model_file = model_path / 'model.pkl'
        if not model_file.exists():
            raise ModelLoadError(f"Model file not found: {model_file}")
        
        try:
            artifacts['model'] = joblib.load(model_file)
            artifacts['model_path'] = str(model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
        
        # Load scaler (optional)
        scaler_file = model_path / 'scaler.pkl'
        if scaler_file.exists():
            try:
                artifacts['scaler'] = joblib.load(scaler_file)
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}")
                artifacts['scaler'] = None
        else:
            artifacts['scaler'] = None
        
        # Load metadata (optional)
        metadata_file = model_path / 'metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    artifacts['metadata'] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                artifacts['metadata'] = {}
        else:
            artifacts['metadata'] = {}
        
        # Validate artifacts
        self._validate_artifacts(artifacts)
        
        return artifacts
    
    def _validate_artifacts(self, artifacts: Dict[str, Any]):
        """Validate loaded model artifacts."""
        model = artifacts.get('model')
        if model is None:
            raise ModelLoadError("Model object is None")
        
        # Check if model has required methods
        if not hasattr(model, 'predict'):
            raise ModelLoadError("Model does not have predict method")
        
        # Validate metadata if present
        metadata = artifacts.get('metadata', {})
        if metadata:
            # Check for required fields
            if 'model_type' not in metadata:
                logger.warning("Model metadata missing 'model_type' field")
            
            if 'feature_columns' not in metadata:
                logger.warning("Model metadata missing 'feature_columns' field")
            
            # Log model info
            model_type = metadata.get('model_type', 'unknown')
            features_count = len(metadata.get('feature_columns', []))
            logger.info(f"Loaded {model_type} model with {features_count} features")
    
    def _get_cache_key(self, model_path: Path) -> str:
        """Generate cache key for a model path."""
        # Use path and modification time for cache key
        stat = model_path.stat()
        key_string = f"{model_path}:{stat.st_mtime}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _add_to_cache(self, cache_key: str, artifacts: Dict[str, Any]):
        """Add model artifacts to cache."""
        # Check cache size
        if len(self._cache) >= self.cache_size:
            # Remove least recently used
            lru_key = min(self._access_times, key=self._access_times.get)
            del self._cache[lru_key]
            del self._access_times[lru_key]
            logger.debug(f"Evicted model from cache: {lru_key}")
        
        # Add to cache
        self._cache[cache_key] = artifacts
        self._access_times[cache_key] = datetime.now()
    
    def clear_cache(self):
        """Clear the model cache."""
        self._cache.clear()
        self._access_times.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        return {
            'size': len(self._cache),
            'max_size': self.cache_size,
            'models': list(self._cache.keys())
        }


def find_latest_model(model_type: str, models_dir: str = 'models') -> Optional[Path]:
    """
    Find the latest model of a given type.
    
    Args:
        model_type: Type of model (e.g., 'xgboost', 'lightgbm')
        models_dir: Base directory for models
        
    Returns:
        Path to latest model directory or None if not found
    """
    models_path = Path(models_dir)
    model_type_path = models_path / model_type
    
    if not model_type_path.exists():
        return None
    
    # Find all timestamp directories
    timestamp_dirs = [d for d in model_type_path.iterdir() if d.is_dir()]
    
    if not timestamp_dirs:
        return None
    
    # Sort by name (timestamp) and return latest
    latest = sorted(timestamp_dirs, key=lambda x: x.name)[-1]
    return latest


def list_available_models(models_dir: str = 'models') -> Dict[str, List[Dict[str, Any]]]:
    """
    List all available models.
    
    Args:
        models_dir: Base directory for models
        
    Returns:
        Dictionary mapping model types to list of model info
    """
    models_path = Path(models_dir)
    available_models = {}
    
    if not models_path.exists():
        return available_models
    
    for model_type_dir in models_path.iterdir():
        if not model_type_dir.is_dir():
            continue
        
        model_type = model_type_dir.name
        models = []
        
        for timestamp_dir in model_type_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue
            
            model_info = {
                'path': str(timestamp_dir),
                'timestamp': timestamp_dir.name,
                'has_metadata': (timestamp_dir / 'metadata.json').exists(),
                'has_scaler': (timestamp_dir / 'scaler.pkl').exists()
            }
            
            # Try to load metadata for more info
            metadata_file = timestamp_dir / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        model_info['metrics'] = metadata.get('metrics', {})
                        model_info['symbols'] = metadata.get('symbols', [])
                except (json.JSONDecodeError, FileNotFoundError, Exception):
                    pass  # Metadata file not found or invalid
            
            models.append(model_info)
        
        if models:
            # Sort by timestamp descending
            models.sort(key=lambda x: x['timestamp'], reverse=True)
            available_models[model_type] = models
    
    return available_models
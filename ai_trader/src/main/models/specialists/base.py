"""
Contains the BaseCatalystSpecialist Abstract Base Class.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Import the core data structure - will be imported later to avoid circular import
# from . import CatalystPrediction

logger = logging.getLogger(__name__)

# Define CatalystPrediction locally to avoid circular import
@dataclass
class CatalystPrediction:
    """Prediction result from a catalyst specialist."""
    probability: float
    confidence: float
    catalyst_type: str
    specialist_name: str
    metadata: Dict[str, Any] = None
    features_used: List[str] = None


class BaseCatalystSpecialist(ABC):
    """
    Base class for all catalyst specialists.
    
    Each specialist inherits from this and implements catalyst-specific logic.
    This ensures consistent interface and behavior across all specialists.
    """
    
    def __init__(self, config: Any, specialist_type: str):
        self.config = config
        self.specialist_type = specialist_type
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        
        ## REFACTOR: Get settings from the configuration object
        self.specialist_config = self.config['specialists'][self.specialist_type]
        self.model_version = self.specialist_config.get('model_version', 'v1.0')
        self.min_training_samples = self.config['training'].get('min_specialist_samples', 50)
        
        # Model persistence directory from config
        self.model_dir = Path(self.config['models']['save_directory'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_metrics = {}
        self.validation_metrics = {}
        
    @abstractmethod
    def extract_specialist_features(self, catalyst_features: Dict[str, Any]) -> Dict[str, float]:
        """Extract features relevant to this specialist."""
        pass
    
    @abstractmethod
    def get_minimum_catalyst_strength(self) -> float:
        """Return minimum catalyst strength for this specialist to activate."""
        # ## REFACTOR: Get this from config
        return self.specialist_config.get('min_catalyst_strength', 2.0)
    
    async def predict(self, catalyst_features: Dict[str, Any]) -> Optional[CatalystPrediction]:
        """Main prediction method for the specialist."""
        if not self.is_trained or self.model is None:
            logger.warning(f"{self.specialist_type} specialist not trained")
            return None
        
        try:
            catalyst_strength = self._get_catalyst_strength(catalyst_features)
            min_strength = self.get_minimum_catalyst_strength()
            
            if catalyst_strength < min_strength:
                return None
            
            specialist_features = self.extract_specialist_features(catalyst_features)
            if not specialist_features:
                return None
            
            feature_vector = self._prepare_feature_vector(specialist_features)
            if feature_vector is None:
                return None
            
            probability = self.model.predict_proba(feature_vector)[0][1]
            confidence = self._calculate_confidence(feature_vector, probability)
            feature_importances = self._get_feature_importances(specialist_features)
            
            return CatalystPrediction(
                probability=float(probability),
                confidence=float(confidence),
                specialist_type=self.specialist_type,
                catalyst_strength=float(catalyst_strength),
                features_used=list(specialist_features.keys()),
                model_version=self.model_version,
                prediction_timestamp=datetime.now(timezone.utc),
                feature_importances=feature_importances
            )
            
        except Exception as e:
            logger.error(f"Error in {self.specialist_type} prediction: {e}")
            raise
    
    def train(self, training_data: pd.DataFrame, target_column: str = 'successful_outcome') -> Dict[str, float]:
        """Train the specialist model on catalyst-specific data."""
        logger.info(f"Training {self.specialist_type} specialist on {len(training_data)} samples")
        
        try:
            relevant_data = self._filter_relevant_training_data(training_data)
            
            if len(relevant_data) < self.min_training_samples:
                logger.warning(f"{self.specialist_type}: insufficient training data ({len(relevant_data)} samples)")
                return {'error': 'insufficient_data', 'sample_count': len(relevant_data)}
            
            X = self._extract_training_features(relevant_data)
            y = self._prepare_training_targets(relevant_data, target_column)
            
            if X is None or len(X) == 0:
                logger.error(f"{self.specialist_type}: no valid features extracted")
                return {'error': 'no_features'}
            
            self.model = self._create_model()
            self.model.fit(X, y)
            
            train_score = accuracy_score(y, self.model.predict(X))
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
            
            self.is_trained = True
            metrics = {
                'training_accuracy': float(train_score),
                'cv_mean_accuracy': float(cv_scores.mean()),
                'cv_std_accuracy': float(cv_scores.std()),
                'training_samples': len(X),
                'positive_class_ratio': float(y.mean())
            }
            self.training_metrics = metrics
            
            self._save_model()
            
            logger.info(f"{self.specialist_type} training completed: {train_score:.3f} accuracy")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training {self.specialist_type}: {e}")
            return {'error': str(e)}
    
    def _get_catalyst_strength(self, catalyst_features: Dict[str, Any]) -> float:
        """Get the strength of catalyst signal for this specialist."""
        specialist_score_key = f'{self.specialist_type}_score'
        return catalyst_features.get(specialist_score_key, 0.0)
    
    def _prepare_feature_vector(self, specialist_features: Dict[str, float]) -> Optional[np.ndarray]:
        """## REFACTOR: Stricter feature vector preparation."""
        if not self.feature_columns:
            logger.error(f"{self.specialist_type} model has no feature_columns. Must be trained first.")
            return None
            
        try:
            feature_vector = [specialist_features.get(col, 0.0) for col in self.feature_columns]
            
            feature_array = np.array(feature_vector)
            if not np.all(np.isfinite(feature_array)):
                logger.warning(f"NaN/inf found in feature vector for {self.specialist_type}. Replacing with 0.")
                feature_array = np.nan_to_num(feature_array)
            
            return feature_array.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing feature vector for {self.specialist_type}: {e}")
            return None
    
    def _calculate_confidence(self, feature_vector: np.ndarray, probability: float) -> float:
        """Calculate confidence in the prediction."""
        distance_from_neutral = abs(probability - 0.5) * 2
        return min(distance_from_neutral, 1.0)
    
    def _get_feature_importances(self, specialist_features: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Get feature importances if available."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_columns, importances))
        return None
    
    def _create_model(self):
        """## REFACTOR: Create model using parameters from config."""
        model_params = self.specialist_config.get('model_params', {})
        # Provide sensible defaults if not in config
        return RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 8),
            min_samples_split=model_params.get('min_samples_split', 10),
            min_samples_leaf=model_params.get('min_samples_leaf', 5),
            random_state=self.config['settings']['random_seed'],
            class_weight='balanced',
            n_jobs=-1
        )
    
    def _filter_relevant_training_data(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Filter training data for this specialist."""
        catalyst_column = f'has_{self.specialist_type}_catalyst'
        if catalyst_column in training_data.columns:
            return training_data[training_data[catalyst_column] == True].copy()
        return training_data.copy()
    
    def _extract_training_features(self, training_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract features for training."""
        feature_data = [
            self.extract_specialist_features(row) for row in training_data.to_dict('records')
        ]
        
        if feature_data:
            df = pd.DataFrame(feature_data).fillna(0)
            self.feature_columns = sorted(df.columns.tolist())
            return df[self.feature_columns]
        return None
    
    def _prepare_training_targets(self, training_data: pd.DataFrame, target_column: str) -> np.ndarray:
        """Prepare training targets."""
        return training_data[target_column].values
    
    def _save_model(self):
        """## REFACTOR: Save model using joblib for efficiency."""
        try:
            model_file = self.model_dir / f'{self.specialist_type}_specialist_{self.model_version}.joblib'
            
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'model_version': self.model_version,
                'specialist_type': self.specialist_type,
                'training_metrics': self.training_metrics,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            joblib.dump(model_data, model_file, compress=3)
            logger.info(f"Saved {self.specialist_type} model to {model_file}")
            
        except Exception as e:
            logger.error(f"Error saving {self.specialist_type} model: {e}")
    
    def load_model(self, model_version: Optional[str] = None) -> bool:
        """## REFACTOR: Load model using joblib."""
        try:
            version = model_version or self.model_version
            model_file = self.model_dir / f'{self.specialist_type}_specialist_{version}.joblib'
            
            if not model_file.exists():
                logger.warning(f"Model file not found: {model_file}")
                return False
            
            model_data = joblib.load(model_file)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.model_version = model_data['model_version']
            self.training_metrics = model_data.get('training_metrics', {})
            self.is_trained = True
            
            logger.info(f"Loaded {self.specialist_type} model version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {self.specialist_type} model: {e}")
            return False
"""
A standardized, stateless pipeline for training a single machine learning model.
This module is a core component used by the ModelTrainingOrchestrator.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Handles the actual model training process for a given dataset and configuration.
    This class is stateless and receives all dependencies via its methods.
    """
    
    def __init__(self, config: Dict):
        """Initializes the pipeline with training configurations."""
        self.config = config
        self.training_params = config.get('ml', {}).get('training', {})
        self.model_configs = config.get('ml', {}).get('models', {})
        logger.info("ModelTrainingPipeline initialized.")

    def train_model(self, 
                    model_type: str, 
                    training_data: pd.DataFrame,
                    feature_columns: List[str],
                    target_column: str,
                    hyperparameters: Dict) -> Dict[str, Any]:
        """
        Trains a single model using the provided data and parameters.

        Args:
            model_type: The type of model to train (e.g., 'xgboost').
            training_data: A single, clean DataFrame containing all features and the target.
            feature_columns: A list of column names to be used as features.
            target_column: The name of the column to be used as the target.
            hyperparameters: A dictionary of hyperparameters for the model.

        Returns:
            A dictionary containing the trained model, metrics, and other artifacts.
        """
        logger.info(f"Starting training for model type: {model_type}")
        
        try:
            # 1. Prepare Data: Handle NaN values in targets (common for forward-looking targets)
            X = training_data[feature_columns]
            y = training_data[target_column]
            
            # Remove rows where target is NaN
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            logger.info(f"After removing NaN targets: {len(X)} samples remaining")
            
            if len(X) < 10:
                raise ValueError(f"Insufficient samples after removing NaN values: {len(X)}")
            
            # Use time-series split (no shuffling, train on earlier data, test on later data)
            train_size = self.training_params.get('train_split', 0.8)
            split_index = int(len(X) * train_size)
            
            X_train = X.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]
            
            logger.info(f"Time-series split: {len(X_train)} train samples, {len(X_test)} test samples")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 2. Train Model
            model_config = self.model_configs.get(model_type, {})
            # Override with optimized hyperparameters
            model_config.update(hyperparameters)

            model = self._create_model(model_type, model_config)
            model.fit(X_train_scaled, y_train)
            
            # 3. Evaluate Model
            y_pred = model.predict(X_test_scaled)
            
            metrics = self._calculate_metrics(y_test, y_pred)
            logger.info(f"Training complete for {model_type}. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

            # 4. Package and Return Artifacts
            # The orchestrator is responsible for saving these.
            return {
                "model_artifact": model,
                "scaler_artifact": scaler,
                "metrics": metrics,
                "feature_columns": feature_columns,
                "training_metadata": {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "hyperparameters_used": model_config
                }
            }

        except Exception as e:
            logger.error(f"Error during model training for {model_type}: {e}", exc_info=True)
            # Return a failure dictionary
            return {"error": str(e)}

    def _create_model(self, model_type: str, params: Dict[str, Any]) -> Any:
        """Creates a model instance from configuration."""
        if model_type == 'xgboost':
            return xgb.XGBRegressor(**params, random_state=42)
        elif model_type == 'lightgbm':
            return lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params, random_state=42)
        else:
            raise ValueError(f"Unknown model type specified: {model_type}")

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculates a standard set of regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy (for financial applications)
        directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'directional_accuracy': directional_accuracy,
            # Keep some legacy names for compatibility
            'accuracy': directional_accuracy,
            'f1_score': r2  # Use R2 as a proxy for F1 in regression context
        }
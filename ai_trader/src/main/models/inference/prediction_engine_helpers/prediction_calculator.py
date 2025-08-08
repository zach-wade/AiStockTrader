# File: src/ai_trader/models/inference/prediction_engine_helpers/prediction_calculator.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class PredictionCalculator:
    """
    Performs the core prediction logic: takes a loaded model and prepared features
    and generates predictions, probabilities, and confidence scores.
    """

    def __init__(self):
        logger.debug("PredictionCalculator initialized.")

    def calculate_prediction(self, 
                             model_obj: Any, 
                             features_df: pd.DataFrame, 
                             model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a prediction and associated metrics (probability, confidence)
        for a single input.

        Args:
            model_obj: The loaded model object (e.g., scikit-learn, XGBoost model).
            features_df: A Pandas DataFrame containing the features for a single prediction.
                         Must be in the format expected by the model (e.g., single row).
            model_metadata: Dictionary containing metadata about the model,
                            e.g., 'feature_names' or 'feature_config'.

        Returns:
            A dictionary with prediction, probability, confidence, and relevant metadata.
            Returns error details if prediction fails.
        """
        prediction_result: Dict[str, Any] = {
            'prediction': None,
            'probability': None,
            'confidence': 0.0,
            'error': None
        }

        if features_df.empty:
            prediction_result['error'] = 'No features available for prediction.'
            logger.warning("Empty features DataFrame provided for prediction.")
            return prediction_result

        # Ensure features match model expectations (reorder/filter if feature_names provided)
        expected_feature_names = model_metadata.get('features', []) # 'features' from ModelVersion
        
        if expected_feature_names:
            try:
                # Ensure all expected features are present, fill missing with 0 or NaN
                # Reindex to enforce order and fill missing features if necessary
                missing_cols = set(expected_feature_names) - set(features_df.columns)
                if missing_cols:
                    logger.warning(f"Missing {len(missing_cols)} features for prediction. Filling with 0.0: {missing_cols}")
                    # Create a new DataFrame to ensure correct order and fill missing
                    features_df = features_df.reindex(columns=expected_feature_names, fill_value=0.0)
                    
                # Filter to only expected features and ensure correct order
                features_df = features_df[expected_feature_names]

            except KeyError as e:
                prediction_result['error'] = f"Feature mismatch: {e}. Check expected_feature_names in model metadata."
                logger.error(f"Feature mismatch: {e} during reordering for prediction.", exc_info=True)
                return prediction_result
            except Exception as e:
                prediction_result['error'] = f"Error preprocessing features: {e}"
                logger.error(f"Error preprocessing features for prediction: {e}", exc_info=True)
                return prediction_result

        try:
            # Make prediction
            prediction = model_obj.predict(features_df)[0] # Assuming single prediction for single row DF
            prediction_result['prediction'] = int(prediction) # Ensure integer output
            
            # Get probability if model supports it
            if hasattr(model_obj, 'predict_proba'):
                probabilities = model_obj.predict_proba(features_df)[0] # Assuming probabilities for single row
                if len(probabilities) == 2: # Binary classification
                    probability = probabilities[1] # Probability of the positive class
                elif prediction_result['prediction'] is not None and prediction_result['prediction'] < len(probabilities):
                    probability = probabilities[prediction_result['prediction']] # Probability of the predicted class
                else:
                    probability = max(probabilities) # Fallback to max probability
                
                prediction_result['probability'] = float(probability)
                # Confidence: distance from 0.5 for binary, or max(prob) for multi-class
                prediction_result['confidence'] = abs(probability - 0.5) * 2 if len(probabilities) == 2 else float(probability)
            else:
                prediction_result['probability'] = None
                prediction_result['confidence'] = 0.5 # Default confidence if no proba

            prediction_result['error'] = None # Clear any error
            
        except Exception as e:
            prediction_result['prediction'] = None
            prediction_result['probability'] = None
            prediction_result['confidence'] = 0.0
            prediction_result['error'] = f"Prediction computation failed: {str(e)}"
            logger.error(f"Prediction computation failed: {e}", exc_info=True)
            
        return prediction_result
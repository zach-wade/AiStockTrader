# File: src/ai_trader/models/outcome_classifier_helpers/outcome_labeler.py

import logging
import numpy as np
from typing import Dict, Any, Tuple

# Corrected absolute import for OutcomeLabel Enum
from main.models.outcome_classifier_types import OutcomeLabel

logger = logging.getLogger(__name__)

class OutcomeLabeler:
    """
    Classifies a stock's price behavior after a catalyst into
    predefined outcome labels based on a set of thresholds and metrics.
    Assigns a confidence score to each classification.
    """

    def __init__(self, thresholds: Dict[str, Dict[str, float]]):
        """
        Initializes the OutcomeLabeler with specific classification thresholds.

        Args:
            thresholds: A dictionary defining the numerical criteria for each OutcomeLabel.
        """
        self.thresholds = thresholds
        logger.debug("OutcomeLabeler initialized with custom thresholds.")

    def classify_based_on_metrics(self, metrics: Dict[str, Any]) -> Tuple[OutcomeLabel, float]:
        """
        Classifies the outcome based on a dictionary of calculated metrics.
        Applies a hierarchical rule-based system to assign an OutcomeLabel
        and a confidence score.

        Args:
            metrics: A dictionary containing the computed outcome metrics.

        Returns:
            A tuple: (OutcomeLabel, confidence_score [0-1]).
            Defaults to VOLATILE_CHOP with moderate confidence if no clear pattern.
        """
        # Retrieve primary metrics for classification, providing defaults for robustness
        return_3d = metrics.get('return_3d', 0.0)
        max_favorable = metrics.get('max_favorable_move', 0.0)
        max_adverse = metrics.get('max_adverse_move', 0.0)
        momentum_score = metrics.get('momentum_score', 0.0)
        follow_through = metrics.get('follow_through_strength', 0.0)
        reversal_strength = metrics.get('reversal_strength', 0.0)
        volatility = metrics.get('volatility_realized', 0.0)
        
        # Ensure all necessary metrics are not None/NaN before comparison
        # If any key metrics are missing or NaN, return NO_DATA or CALCULATION_ERROR
        required_classification_metrics = [
            'return_3d', 'max_favorable_move', 'max_adverse_move',
            'momentum_score', 'follow_through_strength', 'reversal_strength',
            'volatility_realized'
        ]
        if any(metrics.get(k) is None or np.isnan(metrics.get(k)) for k in required_classification_metrics):
             logger.warning(f"Missing or NaN critical classification metrics. Returning NO_DATA. Metrics: {metrics.keys()}")
             return OutcomeLabel.NO_DATA, 0.0

        # --- Classification Logic (Hierarchical) ---

        # 1. Successful Breakout (High confidence if met)
        thresh_sb = self.thresholds.get('successful_breakout', {})
        if (return_3d >= thresh_sb.get('min_return_3d', 0.05) and
            max_favorable >= thresh_sb.get('min_max_favorable', 0.08) and
            momentum_score >= thresh_sb.get('min_follow_through', 0.3)):
            
            confidence = min(
                (return_3d / thresh_sb.get('min_return_3d', 0.05) * 0.4) +  # Return component
                (max_favorable / thresh_sb.get('min_max_favorable', 0.08) * 0.3) +  # Peak move component
                (momentum_score * 0.3),  # Consistency component
                1.0
            )
            logger.debug(f"Classified as SUCCESSFUL_BREAKOUT for return_3d={return_3d:.2f}, max_fav={max_favorable:.2f}, momentum={momentum_score:.2f}")
            return OutcomeLabel.SUCCESSFUL_BREAKOUT, confidence
        
        # 2. Failed Breakdown (Clear downside)
        thresh_fb = self.thresholds.get('failed_breakdown', {})
        if (return_3d <= thresh_fb.get('max_return_3d', -0.03) and
            max_adverse <= thresh_fb.get('min_adverse_move', -0.05)):
            
            confidence = min(
                (abs(return_3d) / abs(thresh_fb.get('max_return_3d', -0.03)) * 0.5) +
                (abs(max_adverse) / abs(thresh_fb.get('min_adverse_move', -0.05)) * 0.5),
                1.0
            )
            logger.debug(f"Classified as FAILED_BREAKDOWN for return_3d={return_3d:.2f}, max_adv={max_adverse:.2f}")
            return OutcomeLabel.FAILED_BREAKDOWN, confidence
        
        # 3. Reversal Pattern (Initial strong move then reversal)
        thresh_rev = self.thresholds.get('reversal_pattern', {})
        if (reversal_strength > thresh_rev.get('min_reversal_strength', 0.4) and
            max_favorable > thresh_rev.get('min_initial_move', 0.03) and # Initial upward move
            abs(return_3d) <= thresh_rev.get('max_final_return', 0.01)): # Final close to start
            
            confidence = min(reversal_strength * 0.7 + (max_favorable / thresh_rev.get('min_initial_move', 0.03)) * 0.3, 1.0)
            logger.debug(f"Classified as REVERSAL_PATTERN for reversal={reversal_strength:.2f}, max_fav={max_favorable:.2f}")
            return OutcomeLabel.REVERSAL_PATTERN, confidence

        # 4. Modest Gain (Specific range, before general volatile)
        thresh_mg = self.thresholds.get('modest_gain', {})
        if (thresh_mg.get('min_return_3d', 0.02) <= return_3d <= 
            thresh_mg.get('max_return_3d', 0.05)):
            
            confidence = 0.7 # Moderate confidence for modest gains
            logger.debug(f"Classified as MODEST_GAIN for return_3d={return_3d:.2f}")
            return OutcomeLabel.MODEST_GAIN, confidence
        
        # 5. Sideways Fizzle (Low movement, low volatility)
        thresh_sf = self.thresholds.get('sideways_fizzle', {})
        if (abs(return_3d) <= thresh_sf.get('max_abs_return_3d', 0.02) and
            volatility <= thresh_sf.get('max_volatility', 0.03) and # Volatility must be low
            max_favorable <= thresh_sf.get('max_max_move', 0.04) and # Overall range must be limited
            abs(max_adverse) <= thresh_sf.get('max_max_move', 0.04)):
            
            confidence = 0.8 # High confidence for clear sideways action
            logger.debug(f"Classified as SIDEWAYS_FIZZLE for return_3d={return_3d:.2f}, volatility={volatility:.2f}")
            return OutcomeLabel.SIDEWAYS_FIZZLE, confidence
        
        # 6. Default: Volatile Chop (if nothing else matches, but still some movement/volatility)
        # This is a fallback, implying no clear directional pattern but active trading
        confidence = 0.5 # Lower confidence as pattern is ambiguous
        logger.debug(f"Classified as VOLATILE_CHOP (default).")
        return OutcomeLabel.VOLATILE_CHOP, confidence
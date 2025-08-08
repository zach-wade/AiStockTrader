from typing import Dict, Any
from .base import BaseCatalystSpecialist

class NewsSpecialist(BaseCatalystSpecialist):
    """Specialist for predicting news catalyst outcomes."""
    def __init__(self, config: Any):
        super().__init__(config, 'news')

    def extract_specialist_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        return {
            'news_score': features.get('news_score', 0.0),
            'total_catalyst_score': features.get('total_catalyst_score', 0.0),
            'rvol': features.get('rvol', 1.0),
            'social_score': features.get('social_score', 0.0),
            'volume_score': features.get('volume_score', 0.0),
        }
"""
Specialist models module.

This module provides specialized models for analyzing specific aspects
of market behavior and generating catalyst features.
"""

from .base import BaseCatalystSpecialist, CatalystPrediction
from .earnings import EarningsSpecialist
from .ensemble import CatalystSpecialistEnsemble
from .news import NewsSpecialist
from .options import OptionsSpecialist
from .social import SocialSpecialist
from .technical import TechnicalSpecialist

__all__ = [
    'BaseCatalystSpecialist',
    'CatalystPrediction',
    'EarningsSpecialist',
    'CatalystSpecialistEnsemble',
    'NewsSpecialist',
    'OptionsSpecialist',
    'SocialSpecialist',
    'TechnicalSpecialist'
]
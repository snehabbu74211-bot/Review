"""
Fake Review Archaeology - Source Package
========================================

A comprehensive system for detecting AI-generated and fraudulent product reviews.

Modules:
    data_pipeline: Data loading, cleaning, and preprocessing
    feature_engineering: Linguistic and metadata feature extraction
    ensemble_model: Two-layer ensemble architecture (RoBERTa + XGBoost)
    business_intelligence: Fraud analysis and impact quantification
    utils: Helper functions and utilities

Example:
    >>> from src.data_pipeline import DataPipeline
    >>> from src.ensemble_model import StackingEnsemble
    >>> 
    >>> pipeline = DataPipeline()
    >>> df = pipeline.load_yelp_data('reviews.json')
    >>> train_df, test_df = pipeline.split_data(df)
"""

__version__ = '1.0.0'
__author__ = 'Data Analytics Team'

from .data_pipeline import DataPipeline
from .feature_engineering import FeaturePipeline, LinguisticFeatureExtractor, MetadataFeatureExtractor
from .ensemble_model import StackingEnsemble, RoBERTaClassifier, XGBoostClassifier
from .business_intelligence import FraudAnalyzer, RiskVisualizer
from .utils import (
    setup_logging,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    get_risk_level,
    get_risk_color,
    format_currency,
    Timer
)

__all__ = [
    'DataPipeline',
    'FeaturePipeline',
    'LinguisticFeatureExtractor',
    'MetadataFeatureExtractor',
    'StackingEnsemble',
    'RoBERTaClassifier',
    'XGBoostClassifier',
    'FraudAnalyzer',
    'RiskVisualizer',
    'setup_logging',
    'save_json',
    'load_json',
    'save_pickle',
    'load_pickle',
    'get_risk_level',
    'get_risk_color',
    'format_currency',
    'Timer',
]

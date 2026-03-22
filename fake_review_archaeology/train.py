#!/usr/bin/env python3
"""
Fake Review Archaeology - Training Script
=========================================
End-to-end training pipeline for the fake review detection ensemble model.

Usage:
    python train.py --data_path data/yelp_reviews.json --output_dir models/
    python train.py --config config.yaml

Author: Data Analytics Team
Date: 2026-03-22
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_pipeline import DataPipeline
from feature_engineering import FeaturePipeline
from ensemble_model import StackingEnsemble
from business_intelligence import FraudAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_pipeline(args):
    """Execute full training pipeline."""
    
    logger.info("=" * 60)
    logger.info("Fake Review Archaeology - Training Pipeline")
    logger.info("=" * 60)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info("Loaded configuration from %s", args.config)
    else:
        config = {
            'data': {
                'test_size': 0.2,
                'random_state': 42,
                'min_review_length': 20,
                'max_review_length': 2000,
                'balance_method': 'undersample'
            },
            'features': {
                'include_perplexity': True,
                'include_burstiness': True,
                'include_coherence': True,
                'sample_size': None
            },
            'model': {
                'roberta_epochs': 3,
                'roberta_batch_size': 16,
                'roberta_learning_rate': 2e-5,
                'xgboost_max_depth': 6,
                'xgboost_n_estimators': 200,
                'xgboost_learning_rate': 0.1
            }
        }
    
    # Step 1: Data Pipeline
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Data Pipeline")
    logger.info("=" * 60)
    
    data_pipeline = DataPipeline(config=config.get('data', {}))
    
    # Load data
    if args.data_path.endswith('.json'):
        df = data_pipeline.load_yelp_data(args.data_path)
    elif args.data_path.endswith('.csv'):
        df = data_pipeline.load_amazon_data(args.data_path)
    else:
        raise ValueError("Unsupported file format. Use .json or .csv")
    
    # Clean data
    df = data_pipeline.clean_data(df)
    
    # Split data
    train_df, test_df = data_pipeline.split_data(
        balance_train=config['data'].get('balance_method') is not None
    )
    
    # Save processed data
    if args.output_dir:
        data_pipeline.save_processed_data(Path(args.output_dir) / 'data')
    
    logger.info("Data pipeline complete. Train: %d, Test: %d", 
                len(train_df), len(test_df))
    
    # Step 2: Feature Engineering
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Feature Engineering")
    logger.info("=" * 60)
    
    feature_pipeline = FeaturePipeline(device=args.device)
    
    # Extract features for training set
    logger.info("Extracting training features...")
    X_train = feature_pipeline.extract_features(
        train_df,
        include_perplexity=config['features'].get('include_perplexity', True),
        include_burstiness=config['features'].get('include_burstiness', True),
        include_coherence=config['features'].get('include_coherence', True),
        sample_size=config['features'].get('sample_size')
    )
    
    # Extract features for test set
    logger.info("Extracting test features...")
    X_test = feature_pipeline.extract_features(
        test_df,
        include_perplexity=config['features'].get('include_perplexity', True),
        include_burstiness=config['features'].get('include_burstiness', True),
        include_coherence=config['features'].get('include_coherence', True),
        sample_size=config['features'].get('sample_size')
    )
    
    logger.info("Feature extraction complete. Features: %d", len(X_train.columns))
    
    # Save features
    if args.output_dir:
        X_train.to_parquet(Path(args.output_dir) / 'features_train.parquet')
        X_test.to_parquet(Path(args.output_dir) / 'features_test.parquet')
    
    # Step 3: Model Training
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Model Training")
    logger.info("=" * 60)
    
    ensemble = StackingEnsemble()
    
    # Train ensemble
    ensemble.train(
        train_texts=train_df['review_text_clean'].tolist(),
        train_features=X_train.values,
        train_labels=train_df['label'].values,
        val_texts=test_df['review_text_clean'].tolist(),
        val_features=X_test.values,
        val_labels=test_df['label'].values,
        feature_names=X_train.columns.tolist(),
        roberta_epochs=config['model'].get('roberta_epochs', 3)
    )
    
    # Save ensemble
    if args.output_dir:
        ensemble.save_ensemble(Path(args.output_dir) / 'ensemble')
    
    logger.info("Model training complete")
    
    # Step 4: Evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Evaluation")
    logger.info("=" * 60)
    
    results = ensemble.evaluate(
        test_df['review_text_clean'].tolist(),
        X_test.values,
        test_df['label'].values
    )
    
    # Print results
    logger.info("\nEvaluation Results:")
    for model, metrics in results.items():
        if model != 'confusion_matrices':
            logger.info(f"\n{model.upper()}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    if args.output_dir:
        with open(Path(args.output_dir) / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Step 5: Business Intelligence (Optional)
    if args.run_bi:
        logger.info("\n" + "=" * 60)
        logger.info("Step 5: Business Intelligence Analysis")
        logger.info("=" * 60)
        
        # Get predictions for full dataset
        all_texts = df['review_text_clean'].tolist()
        all_features = feature_pipeline.extract_features(df)
        predictions = ensemble.predict(all_texts, all_features.values)
        
        df['fraud_probability'] = predictions
        
        # Run analysis
        analyzer = FraudAnalyzer(risk_threshold=0.7)
        
        # Category analysis
        if 'category' in df.columns:
            cat_analysis = analyzer.analyze_by_category(df)
            logger.info("\nTop Risky Categories:")
            logger.info(cat_analysis.head().to_string())
        
        # Suspicious accounts
        susp_accounts = analyzer.identify_suspicious_accounts(df)
        logger.info("\nSuspicious Accounts: %d", susp_accounts['is_suspicious'].sum())
        
        # Business impact
        impact = analyzer.calculate_business_impact(df)
        logger.info("\nBusiness Impact:")
        logger.info(json.dumps(impact, indent=2, default=str))
        
        # Save analysis
        if args.output_dir:
            analyzer.save_analysis(Path(args.output_dir) / 'business_intelligence')
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info("=" * 60)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train Fake Review Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to training data (JSON or CSV)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device for training'
    )
    
    parser.add_argument(
        '--run_bi',
        action='store_true',
        help='Run business intelligence analysis'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    try:
        results = train_pipeline(args)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"\nEnsemble F1-Score: {results['ensemble']['f1']:.4f}")
        print(f"Ensemble Precision: {results['ensemble']['precision']:.4f}")
        print(f"Ensemble Recall: {results['ensemble']['recall']:.4f}")
        print(f"Ensemble ROC-AUC: {results['ensemble']['roc_auc']:.4f}")
        print(f"\nModels saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error("Training failed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

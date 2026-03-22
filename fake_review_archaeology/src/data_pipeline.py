"""
Fake Review Archaeology - Data Pipeline Module
==============================================
Handles data loading, cleaning, preprocessing, and train/test splitting
for the fake review detection system.

Author: Data Analytics Team
Date: 2026-03-22
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from typing import Tuple, Optional, Dict, List
import logging
from pathlib import Path
import json
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Data pipeline for fake review detection.
    
    Handles loading of Yelp/Amazon review data, cleaning, feature extraction,
    and preparation for model training with proper class balancing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data pipeline.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        # Default configuration
        self.random_state = self.config.get('random_state', 42)
        self.test_size = self.config.get('test_size', 0.2)
        self.min_review_length = self.config.get('min_review_length', 20)
        self.max_review_length = self.config.get('max_review_length', 2000)
        
        logger.info("DataPipeline initialized with config: %s", self.config)
    
    def load_yelp_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Yelp dataset from JSON file.
        
        Yelp dataset contains 'recommended' field where:
        - recommended = 1: Likely genuine review
        - recommended = 0: Likely fake/spam review (not recommended by Yelp)
        
        Args:
            file_path: Path to Yelp JSON dataset
            
        Returns:
            DataFrame with loaded data
        """
        logger.info("Loading Yelp data from %s", file_path)
        
        try:
            # Yelp dataset is typically in JSON format, one JSON object per line
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            
            df = pd.DataFrame(data)
            logger.info("Loaded %d records from Yelp dataset", len(df))
            
            # Map Yelp fields to our schema
            df = self._standardize_yelp_schema(df)
            self.raw_data = df
            
            return df
            
        except Exception as e:
            logger.error("Error loading Yelp data: %s", str(e))
            raise
    
    def load_amazon_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Amazon review dataset.
        
        For Amazon, we use heuristics to identify potential fake reviews:
        - Extremely short reviews with 5 stars
        - Reviews from accounts with very few reviews
        - Duplicate/similar content
        
        Args:
            file_path: Path to Amazon dataset (CSV or JSON)
            
        Returns:
            DataFrame with loaded data
        """
        logger.info("Loading Amazon data from %s", file_path)
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, lines=True)
            else:
                raise ValueError("Unsupported file format. Use .csv or .json")
            
            logger.info("Loaded %d records from Amazon dataset", len(df))
            
            df = self._standardize_amazon_schema(df)
            self.raw_data = df
            
            return df
            
        except Exception as e:
            logger.error("Error loading Amazon data: %s", str(e))
            raise
    
    def _standardize_yelp_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Yelp dataset to common schema.
        
        Args:
            df: Raw Yelp DataFrame
            
        Returns:
            Standardized DataFrame
        """
        # Map Yelp columns to standard schema
        column_mapping = {
            'review_id': 'review_id',
            'user_id': 'reviewer_id',
            'business_id': 'product_id',
            'stars': 'rating',
            'text': 'review_text',
            'date': 'review_date',
            'useful': 'useful_votes',
            'funny': 'funny_votes',
            'cool': 'cool_votes',
            'recommended': 'is_recommended'
        }
        
        # Select and rename columns that exist
        available_cols = [col for col in column_mapping.keys() if col in df.columns]
        df = df[available_cols].copy()
        df.rename(columns=column_mapping, inplace=True)
        
        # Create target label: 1 = genuine (recommended), 0 = fake (not recommended)
        if 'is_recommended' in df.columns:
            df['label'] = df['is_recommended'].astype(int)
        else:
            # If no recommendation field, use heuristics
            logger.warning("No 'recommended' field found. Using heuristics for labeling.")
            df['label'] = self._apply_heuristic_labeling(df)
        
        # Add metadata
        df['platform'] = 'yelp'
        df['review_length'] = df['review_text'].str.len()
        df['word_count'] = df['review_text'].str.split().str.len()
        
        return df
    
    def _standardize_amazon_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Amazon dataset to common schema.
        
        Args:
            df: Raw Amazon DataFrame
            
        Returns:
            Standardized DataFrame
        """
        # Common Amazon column mappings
        column_mapping = {
            'reviewerID': 'reviewer_id',
            'asin': 'product_id',
            'reviewText': 'review_text',
            'overall': 'rating',
            'reviewTime': 'review_date',
            'unixReviewTime': 'review_timestamp',
            'helpful': 'helpful_votes',
            'summary': 'review_summary'
        }
        
        available_cols = [col for col in column_mapping.keys() if col in df.columns]
        df = df[available_cols].copy()
        df.rename(columns=column_mapping, inplace=True)
        
        # Apply heuristic labeling for Amazon (no explicit fake label)
        df['label'] = self._apply_heuristic_labeling(df)
        df['platform'] = 'amazon'
        df['review_length'] = df['review_text'].str.len()
        df['word_count'] = df['review_text'].str.split().str.len()
        
        return df
    
    def _apply_heuristic_labeling(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply heuristic rules to identify potential fake reviews.
        
        Heuristics:
        1. Very short reviews ( < 30 chars) with 5-star rating
        2. Reviews with excessive capitalization (>50%)
        3. Duplicate reviews from same user
        4. Reviews with repetitive content
        
        Args:
            df: DataFrame with review data
            
        Returns:
            Series with labels (1 = genuine, 0 = fake)
        """
        labels = pd.Series(1, index=df.index)  # Default to genuine
        
        # Heuristic 1: Very short 5-star reviews
        if 'rating' in df.columns and 'review_length' in df.columns:
            short_extreme = (df['review_length'] < 30) & (df['rating'] == 5)
            labels[short_extreme] = 0
        
        # Heuristic 2: Excessive capitalization
        if 'review_text' in df.columns:
            def cap_ratio(text):
                if pd.isna(text) or len(text) == 0:
                    return 0
                return sum(1 for c in text if c.isupper()) / len(text)
            
            cap_ratios = df['review_text'].apply(cap_ratio)
            labels[cap_ratios > 0.5] = 0
        
        # Heuristic 3: Duplicate reviews from same user
        if 'reviewer_id' in df.columns and 'review_text' in df.columns:
            duplicates = df.duplicated(subset=['reviewer_id', 'review_text'], keep=False)
            labels[duplicates] = 0
        
        # Heuristic 4: Repetitive content (same word repeated many times)
        if 'review_text' in df.columns:
            def has_repetition(text):
                if pd.isna(text):
                    return False
                words = text.lower().split()
                if len(words) < 5:
                    return False
                from collections import Counter
                word_counts = Counter(words)
                most_common = word_counts.most_common(1)[0][1]
                return most_common > len(words) * 0.4  # Same word > 40% of text
            
            repetitive = df['review_text'].apply(has_repetition)
            labels[repetitive] = 0
        
        logger.info("Heuristic labeling: %d fake, %d genuine", 
                   (labels == 0).sum(), (labels == 1).sum())
        
        return labels
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and preprocess the review data.
        
        Cleaning steps:
        1. Remove null values
        2. Filter by review length
        3. Remove HTML tags
        4. Normalize whitespace
        5. Remove non-printable characters
        
        Args:
            df: DataFrame to clean (uses self.raw_data if None)
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.raw_data.copy()
        else:
            df = df.copy()
        
        logger.info("Starting data cleaning. Initial records: %d", len(df))
        
        # Remove rows with null review text
        initial_count = len(df)
        df = df.dropna(subset=['review_text'])
        logger.info("Removed %d rows with null review text", initial_count - len(df))
        
        # Filter by review length
        df = df[
            (df['review_length'] >= self.min_review_length) & 
            (df['review_length'] <= self.max_review_length)
        ]
        logger.info("After length filtering: %d records", len(df))
        
        # Clean review text
        df['review_text_clean'] = df['review_text'].apply(self._clean_text)
        
        # Remove empty reviews after cleaning
        df = df[df['review_text_clean'].str.len() > 0]
        
        # Convert date columns
        if 'review_date' in df.columns:
            df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        
        self.processed_data = df
        logger.info("Data cleaning complete. Final records: %d", len(df))
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """
        Clean individual review text.
        
        Args:
            text: Raw review text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_data(self, df: Optional[pd.DataFrame] = None, 
                   stratify: bool = True,
                   balance_train: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets with optional stratification and balancing.
        
        Args:
            df: DataFrame to split (uses self.processed_data if None)
            stratify: Whether to stratify by label
            balance_train: Whether to balance the training set
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if df is None:
            df = self.processed_data.copy()
        else:
            df = df.copy()
        
        logger.info("Splitting data. Test size: %.2f", self.test_size)
        
        # Prepare stratification
        stratify_col = df['label'] if stratify else None
        
        # Split data
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )
        
        logger.info("Initial split - Train: %d, Test: %d", len(train_df), len(test_df))
        logger.info("Train label distribution:\n%s", train_df['label'].value_counts())
        logger.info("Test label distribution:\n%s", test_df['label'].value_counts())
        
        # Balance training set if requested
        if balance_train:
            train_df = self._balance_classes(train_df)
        
        self.train_data = train_df
        self.test_data = test_df
        
        return train_df, test_df
    
    def _balance_classes(self, df: pd.DataFrame, 
                        method: str = 'undersample') -> pd.DataFrame:
        """
        Balance classes in the dataset.
        
        Args:
            df: DataFrame to balance
            method: Balancing method ('undersample', 'oversample', or 'smote')
            
        Returns:
            Balanced DataFrame
        """
        logger.info("Balancing classes using method: %s", method)
        
        class_counts = df['label'].value_counts()
        logger.info("Before balancing: %s", class_counts.to_dict())
        
        if method == 'undersample':
            # Undersample majority class
            min_class = class_counts.idxmin()
            min_count = class_counts.min()
            
            balanced_dfs = []
            for label in class_counts.index:
                class_df = df[df['label'] == label]
                if len(class_df) > min_count:
                    class_df = resample(class_df, 
                                       n_samples=min_count,
                                       random_state=self.random_state)
                balanced_dfs.append(class_df)
            
            df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif method == 'oversample':
            # Oversample minority class
            max_class = class_counts.idxmax()
            max_count = class_counts.max()
            
            balanced_dfs = []
            for label in class_counts.index:
                class_df = df[df['label'] == label]
                if len(class_df) < max_count:
                    class_df = resample(class_df,
                                       n_samples=max_count,
                                       random_state=self.random_state,
                                       replace=True)
                balanced_dfs.append(class_df)
            
            df = pd.concat(balanced_dfs, ignore_index=True)
        
        logger.info("After balancing: %s", df['label'].value_counts().to_dict())
        
        return df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
    
    def get_data_summary(self) -> Dict:
        """
        Generate summary statistics of the data.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'raw_records': len(self.raw_data) if self.raw_data is not None else 0,
            'processed_records': len(self.processed_data) if self.processed_data is not None else 0,
            'train_records': len(self.train_data) if self.train_data is not None else 0,
            'test_records': len(self.test_data) if self.test_data is not None else 0,
        }
        
        if self.processed_data is not None:
            df = self.processed_data
            summary['label_distribution'] = df['label'].value_counts().to_dict()
            summary['rating_distribution'] = df['rating'].value_counts().to_dict() if 'rating' in df.columns else None
            summary['avg_review_length'] = df['review_length'].mean()
            summary['avg_word_count'] = df['word_count'].mean()
            summary['platform_distribution'] = df['platform'].value_counts().to_dict() if 'platform' in df.columns else None
        
        return summary
    
    def save_processed_data(self, output_dir: str):
        """
        Save processed datasets to disk.
        
        Args:
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.train_data is not None:
            self.train_data.to_parquet(output_path / 'train.parquet')
            logger.info("Saved training data to %s", output_path / 'train.parquet')
        
        if self.test_data is not None:
            self.test_data.to_parquet(output_path / 'test.parquet')
            logger.info("Saved test data to %s", output_path / 'test.parquet')
        
        # Save summary
        summary = self.get_data_summary()
        with open(output_path / 'data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'review_id': range(1000),
        'reviewer_id': np.random.choice(['user_' + str(i) for i in range(100)], 1000),
        'product_id': np.random.choice(['prod_' + str(i) for i in range(50)], 1000),
        'review_text': [
            'This product is amazing! Highly recommend it to everyone. ' * np.random.randint(1, 5)
            for _ in range(1000)
        ],
        'rating': np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.05, 0.1, 0.15, 0.3, 0.4]),
        'review_date': pd.date_range('2023-01-01', periods=1000, freq='H'),
        'is_recommended': np.random.choice([0, 1], 1000, p=[0.2, 0.8])
    })
    
    # Initialize pipeline
    pipeline = DataPipeline(config={'test_size': 0.2, 'random_state': 42})
    
    # Process data
    pipeline.raw_data = sample_data
    processed = pipeline.clean_data()
    train, test = pipeline.split_data(balance_train=True)
    
    # Print summary
    print("\nData Summary:")
    print(json.dumps(pipeline.get_data_summary(), indent=2, default=str))

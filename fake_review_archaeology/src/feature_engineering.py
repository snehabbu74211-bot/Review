"""
Fake Review Archaeology - Feature Engineering Module
====================================================
Extracts linguistic and metadata features for fake review detection.

Features:
- Perplexity scores (GPT-2 based)
- Text burstiness (sentence length variance)
- Semantic coherence metrics
- Metadata features (review velocity, account age)

Author: Data Analytics Team
Date: 2026-03-22
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import Counter
import re
from datetime import datetime, timedelta
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinguisticFeatureExtractor:
    """
    Extracts linguistic features from review text for fake review detection.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the feature extractor.
        
        Args:
            device: Device to use for model inference ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        self.roberta_model = None
        self.roberta_tokenizer = None
        
        logger.info("LinguisticFeatureExtractor initialized on device: %s", self.device)
    
    def _load_gpt2(self):
        """Lazy load GPT-2 model for perplexity calculation."""
        if self.gpt2_model is None:
            logger.info("Loading GPT-2 model...")
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_model.to(self.device)
            self.gpt2_model.eval()
            logger.info("GPT-2 model loaded successfully")
    
    def _load_roberta(self):
        """Lazy load RoBERTa model for semantic features."""
        if self.roberta_model is None:
            logger.info("Loading RoBERTa model...")
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.roberta_model = RobertaModel.from_pretrained('roberta-base')
            self.roberta_model.to(self.device)
            self.roberta_model.eval()
            logger.info("RoBERTa model loaded successfully")
    
    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        Calculate perplexity scores using GPT-2.
        
        AI-generated text typically has lower perplexity (more predictable)
        while human text has higher perplexity (more creative/varied).
        
        Args:
            texts: List of review texts
            batch_size: Batch size for inference
            
        Returns:
            List of perplexity scores
        """
        self._load_gpt2()
        perplexities = []
        
        logger.info("Calculating perplexity for %d texts", len(texts))
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_perplexities = []
            
            for text in batch:
                try:
                    # Encode text
                    encodings = self.gpt2_tokenizer(
                        text, 
                        return_tensors='pt',
                        truncation=True,
                        max_length=512
                    )
                    input_ids = encodings.input_ids.to(self.device)
                    
                    # Calculate perplexity
                    with torch.no_grad():
                        outputs = self.gpt2_model(input_ids, labels=input_ids)
                        loss = outputs.loss
                        perplexity = torch.exp(loss).item()
                    
                    batch_perplexities.append(perplexity)
                    
                except Exception as e:
                    logger.warning("Error calculating perplexity: %s", str(e))
                    batch_perplexities.append(np.nan)
            
            perplexities.extend(batch_perplexities)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info("Processed %d/%d batches", i // batch_size + 1, len(texts) // batch_size + 1)
        
        return perplexities
    
    def calculate_burstiness(self, texts: List[str]) -> List[float]:
        """
        Calculate text burstiness (sentence length variance).
        
        AI-generated text tends to have more uniform sentence lengths,
        while human writing shows more variation (burstiness).
        
        Burstiness = (std_dev_sentence_length / mean_sentence_length)
        
        Args:
            texts: List of review texts
            
        Returns:
            List of burstiness scores
        """
        logger.info("Calculating burstiness for %d texts", len(texts))
        
        burstiness_scores = []
        
        for text in texts:
            try:
                sentences = sent_tokenize(str(text))
                if len(sentences) < 2:
                    burstiness_scores.append(0.0)
                    continue
                
                sentence_lengths = [len(sent.split()) for sent in sentences]
                mean_length = np.mean(sentence_lengths)
                std_length = np.std(sentence_lengths)
                
                if mean_length > 0:
                    burstiness = std_length / mean_length
                else:
                    burstiness = 0.0
                
                burstiness_scores.append(burstiness)
                
            except Exception as e:
                logger.warning("Error calculating burstiness: %s", str(e))
                burstiness_scores.append(np.nan)
        
        return burstiness_scores
    
    def calculate_semantic_coherence(self, texts: List[str], 
                                    batch_size: int = 16) -> List[float]:
        """
        Calculate semantic coherence between sentences using RoBERTa embeddings.
        
        AI-generated text may have lower coherence between sentences
        (abrupt topic changes) compared to human writing.
        
        Args:
            texts: List of review texts
            batch_size: Batch size for inference
            
        Returns:
            List of coherence scores (average cosine similarity between consecutive sentences)
        """
        self._load_roberta()
        logger.info("Calculating semantic coherence for %d texts", len(texts))
        
        coherence_scores = []
        
        for text in texts:
            try:
                sentences = sent_tokenize(str(text))
                if len(sentences) < 2:
                    coherence_scores.append(1.0)  # Single sentence, perfect coherence
                    continue
                
                # Get embeddings for each sentence
                embeddings = []
                for sent in sentences:
                    inputs = self.roberta_tokenizer(
                        sent,
                        return_tensors='pt',
                        truncation=True,
                        max_length=128,
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.roberta_model(**inputs)
                        # Use CLS token embedding
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        embeddings.append(embedding[0])
                
                # Calculate pairwise cosine similarities
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[i + 1].reshape(1, -1)
                    )[0, 0]
                    similarities.append(sim)
                
                avg_coherence = np.mean(similarities) if similarities else 1.0
                coherence_scores.append(avg_coherence)
                
            except Exception as e:
                logger.warning("Error calculating coherence: %s", str(e))
                coherence_scores.append(np.nan)
        
        return coherence_scores
    
    def extract_linguistic_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract comprehensive linguistic features from texts.
        
        Args:
            texts: List of review texts
            
        Returns:
            DataFrame with linguistic features
        """
        logger.info("Extracting linguistic features for %d texts", len(texts))
        
        features = pd.DataFrame()
        
        # Basic text statistics
        features['char_count'] = [len(str(t)) for t in texts]
        features['word_count'] = [len(str(t).split()) for t in texts]
        features['sentence_count'] = [len(sent_tokenize(str(t))) for t in texts]
        features['avg_word_length'] = [
            np.mean([len(w) for w in str(t).split()]) if str(t).split() else 0
            for t in texts
        ]
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'].clip(lower=1)
        
        # Punctuation features
        features['exclamation_count'] = [str(t).count('!') for t in texts]
        features['question_count'] = [str(t).count('?') for t in texts]
        features['comma_count'] = [str(t).count(',') for t in texts]
        features['punctuation_ratio'] = [
            sum(1 for c in str(t) if c in '.,!?;:') / max(len(str(t)), 1)
            for t in texts
        ]
        
        # Capitalization features
        features['uppercase_ratio'] = [
            sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
            for t in texts
        ]
        features['all_caps_words'] = [
            len([w for w in str(t).split() if w.isupper() and len(w) > 1])
            for t in texts
        ]
        
        # Readability scores
        features['flesch_reading_ease'] = [
            flesch_reading_ease(str(t)) if len(str(t)) > 10 else 0
            for t in texts
        ]
        features['flesch_kincaid_grade'] = [
            flesch_kincaid_grade(str(t)) if len(str(t)) > 10 else 0
            for t in texts
        ]
        features['automated_readability_index'] = [
            automated_readability_index(str(t)) if len(str(t)) > 10 else 0
            for t in texts
        ]
        
        # Vocabulary diversity
        features['unique_word_ratio'] = [
            len(set(str(t).lower().split())) / max(len(str(t).split()), 1)
            for t in texts
        ]
        
        # Repetition features
        features['repeated_word_ratio'] = [
            self._calculate_repetition(str(t))
            for t in texts
        ]
        
        # Sentiment indicators (simple lexicon-based)
        features['positive_word_count'] = [
            self._count_sentiment_words(str(t), 'positive')
            for t in texts
        ]
        features['negative_word_count'] = [
            self._count_sentiment_words(str(t), 'negative')
            for t in texts
        ]
        
        return features
    
    def _calculate_repetition(self, text: str) -> float:
        """Calculate word repetition ratio."""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        word_counts = Counter(words)
        repeated = sum(1 for count in word_counts.values() if count > 1)
        return repeated / len(word_counts)
    
    def _count_sentiment_words(self, text: str, sentiment: str) -> int:
        """Count positive/negative words in text."""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic',
            'wonderful', 'perfect', 'love', 'best', 'recommend', 'happy',
            'satisfied', 'quality', 'nice', 'beautiful', 'outstanding'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disappointed', 'waste', 'poor', 'cheap', 'broken', 'defective',
            'useless', 'return', 'refund', 'problem', 'issue', 'complaint'
        }
        
        words = set(text.lower().split())
        if sentiment == 'positive':
            return len(words & positive_words)
        else:
            return len(words & negative_words)


class MetadataFeatureExtractor:
    """
    Extracts metadata-based features for fake review detection.
    """
    
    def __init__(self):
        """Initialize metadata feature extractor."""
        logger.info("MetadataFeatureExtractor initialized")
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from review timestamps.
        
        Features:
        - Review velocity (reviews per day by user)
        - Time since account creation
        - Review timing patterns (hour of day, day of week)
        
        Args:
            df: DataFrame with review data including 'review_date' and 'reviewer_id'
            
        Returns:
            DataFrame with temporal features
        """
        logger.info("Extracting temporal features")
        
        features = pd.DataFrame(index=df.index)
        
        if 'review_date' not in df.columns:
            logger.warning("No review_date column found")
            return features
        
        # Convert to datetime if needed
        df = df.copy()
        df['review_date'] = pd.to_datetime(df['review_date'])
        
        # Hour of day (0-23)
        features['review_hour'] = df['review_date'].dt.hour
        
        # Day of week (0=Monday, 6=Sunday)
        features['review_dayofweek'] = df['review_date'].dt.dayofweek
        
        # Is weekend
        features['is_weekend'] = features['review_dayofweek'].isin([5, 6]).astype(int)
        
        # Review velocity (reviews per day by user)
        if 'reviewer_id' in df.columns:
            reviewer_counts = df.groupby('reviewer_id').size()
            date_range = df.groupby('reviewer_id')['review_date'].agg(['min', 'max'])
            date_range['days_active'] = (date_range['max'] - date_range['min']).dt.days + 1
            
            velocity = reviewer_counts / date_range['days_active'].clip(lower=1)
            features['review_velocity'] = df['reviewer_id'].map(velocity).fillna(0)
            
            # Account age in days (assuming first review is account creation proxy)
            first_review = df.groupby('reviewer_id')['review_date'].min()
            features['account_age_days'] = (
                df['review_date'] - df['reviewer_id'].map(first_review)
            ).dt.days
        
        # Time since first review in dataset (as proxy for platform tenure)
        first_overall = df['review_date'].min()
        features['days_since_first_review'] = (df['review_date'] - first_overall).dt.days
        
        return features
    
    def extract_reviewer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract reviewer-level behavioral features.
        
        Features:
        - Total reviews by reviewer
        - Average rating given
        - Rating variance
        - Review length patterns
        
        Args:
            df: DataFrame with review data
            
        Returns:
            DataFrame with reviewer features
        """
        logger.info("Extracting reviewer features")
        
        features = pd.DataFrame(index=df.index)
        
        if 'reviewer_id' not in df.columns:
            logger.warning("No reviewer_id column found")
            return features
        
        # Total reviews by reviewer
        review_counts = df.groupby('reviewer_id').size()
        features['reviewer_total_reviews'] = df['reviewer_id'].map(review_counts)
        
        # Average rating given
        if 'rating' in df.columns:
            avg_rating = df.groupby('reviewer_id')['rating'].mean()
            features['reviewer_avg_rating'] = df['reviewer_id'].map(avg_rating)
            
            # Rating variance
            rating_var = df.groupby('reviewer_id')['rating'].var()
            features['reviewer_rating_variance'] = df['reviewer_id'].map(rating_var).fillna(0)
            
            # Extreme rating ratio (% of 1-star or 5-star reviews)
            extreme_ratings = df[df['rating'].isin([1, 5])].groupby('reviewer_id').size()
            features['reviewer_extreme_ratio'] = (
                df['reviewer_id'].map(extreme_ratings) / review_counts
            ).fillna(0)
        
        # Average review length
        if 'review_length' in df.columns:
            avg_length = df.groupby('reviewer_id')['review_length'].mean()
            features['reviewer_avg_length'] = df['reviewer_id'].map(avg_length)
            
            # Review length variance
            length_var = df.groupby('reviewer_id')['review_length'].var()
            features['reviewer_length_variance'] = df['reviewer_id'].map(length_var).fillna(0)
        
        # Product diversity (unique products / total reviews)
        if 'product_id' in df.columns:
            unique_products = df.groupby('reviewer_id')['product_id'].nunique()
            features['reviewer_product_diversity'] = (
                df['reviewer_id'].map(unique_products) / review_counts
            )
        
        return features
    
    def extract_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract product-level features.
        
        Features:
        - Total reviews for product
        - Average product rating
        - Review acquisition rate
        
        Args:
            df: DataFrame with review data
            
        Returns:
            DataFrame with product features
        """
        logger.info("Extracting product features")
        
        features = pd.DataFrame(index=df.index)
        
        if 'product_id' not in df.columns:
            logger.warning("No product_id column found")
            return features
        
        # Total reviews for product
        product_reviews = df.groupby('product_id').size()
        features['product_total_reviews'] = df['product_id'].map(product_reviews)
        
        # Average product rating
        if 'rating' in df.columns:
            product_avg_rating = df.groupby('product_id')['rating'].mean()
            features['product_avg_rating'] = df['product_id'].map(product_avg_rating)
            
            # Rating distribution entropy (higher = more diverse ratings)
            def rating_entropy(group):
                counts = group['rating'].value_counts(normalize=True)
                return entropy(counts.values)
            
            product_entropy = df.groupby('product_id').apply(rating_entropy)
            features['product_rating_entropy'] = df['product_id'].map(product_entropy)
        
        # Review acquisition velocity (reviews per day)
        if 'review_date' in df.columns:
            df_temp = df.copy()
            df_temp['review_date'] = pd.to_datetime(df_temp['review_date'])
            
            date_range = df_temp.groupby('product_id')['review_date'].agg(['min', 'max'])
            date_range['days_listed'] = (date_range['max'] - date_range['min']).dt.days + 1
            
            acquisition_velocity = product_reviews / date_range['days_listed'].clip(lower=1)
            features['product_review_velocity'] = df['product_id'].map(acquisition_velocity)
        
        return features
    
    def extract_all_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all metadata features.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            DataFrame with all metadata features
        """
        temporal = self.extract_temporal_features(df)
        reviewer = self.extract_reviewer_features(df)
        product = self.extract_product_features(df)
        
        # Combine all features
        all_features = pd.concat([temporal, reviewer, product], axis=1)
        
        # Fill NaN values
        all_features = all_features.fillna(0)
        
        return all_features


class FeaturePipeline:
    """
    Complete feature extraction pipeline combining linguistic and metadata features.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the feature pipeline.
        
        Args:
            device: Device for model inference
        """
        self.linguistic_extractor = LinguisticFeatureExtractor(device=device)
        self.metadata_extractor = MetadataFeatureExtractor()
        logger.info("FeaturePipeline initialized")
    
    def extract_features(self, df: pd.DataFrame, 
                        include_perplexity: bool = True,
                        include_burstiness: bool = True,
                        include_coherence: bool = True,
                        sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Extract all features from the dataset.
        
        Args:
            df: DataFrame with review data
            include_perplexity: Whether to calculate perplexity (computationally expensive)
            include_burstiness: Whether to calculate burstiness
            include_coherence: Whether to calculate semantic coherence (computationally expensive)
            sample_size: If set, only process a random sample of this size
            
        Returns:
            DataFrame with all features
        """
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info("Sampled %d records for feature extraction", sample_size)
        
        logger.info("Starting feature extraction on %d records", len(df))
        
        all_features = pd.DataFrame(index=df.index)
        
        # Linguistic features
        logger.info("Extracting linguistic features...")
        linguistic = self.linguistic_extractor.extract_linguistic_features(
            df['review_text_clean'].tolist()
        )
        all_features = pd.concat([all_features, linguistic], axis=1)
        
        # Perplexity (optional, expensive)
        if include_perplexity:
            logger.info("Calculating perplexity...")
            all_features['perplexity'] = self.linguistic_extractor.calculate_perplexity(
                df['review_text_clean'].tolist()
            )
        
        # Burstiness
        if include_burstiness:
            logger.info("Calculating burstiness...")
            all_features['burstiness'] = self.linguistic_extractor.calculate_burstiness(
                df['review_text_clean'].tolist()
            )
        
        # Semantic coherence (optional, expensive)
        if include_coherence:
            logger.info("Calculating semantic coherence...")
            all_features['semantic_coherence'] = self.linguistic_extractor.calculate_semantic_coherence(
                df['review_text_clean'].tolist()
            )
        
        # Metadata features
        logger.info("Extracting metadata features...")
        metadata = self.metadata_extractor.extract_all_metadata(df)
        all_features = pd.concat([all_features, metadata], axis=1)
        
        # Handle any remaining NaN values
        all_features = all_features.fillna(0)
        
        logger.info("Feature extraction complete. Total features: %d", len(all_features.columns))
        
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Linguistic features
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'avg_sentence_length', 'exclamation_count', 'question_count',
            'comma_count', 'punctuation_ratio', 'uppercase_ratio', 'all_caps_words',
            'flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability_index',
            'unique_word_ratio', 'repeated_word_ratio', 'positive_word_count',
            'negative_word_count', 'perplexity', 'burstiness', 'semantic_coherence',
            # Metadata features
            'review_hour', 'review_dayofweek', 'is_weekend', 'review_velocity',
            'account_age_days', 'days_since_first_review', 'reviewer_total_reviews',
            'reviewer_avg_rating', 'reviewer_rating_variance', 'reviewer_extreme_ratio',
            'reviewer_avg_length', 'reviewer_length_variance', 'reviewer_product_diversity',
            'product_total_reviews', 'product_avg_rating', 'product_rating_entropy',
            'product_review_velocity'
        ]


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_df = pd.DataFrame({
        'review_text_clean': [
            'This product is absolutely amazing! I love it so much. Highly recommend to everyone!',
            'Terrible product. Complete waste of money. Do not buy.',
            'Good quality for the price. Works as expected. Nothing special but decent.',
        ] * 10,
        'review_date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'reviewer_id': ['user_' + str(i % 5) for i in range(30)],
        'product_id': ['prod_' + str(i % 3) for i in range(30)],
        'rating': [5, 1, 3] * 10,
        'review_length': [80, 45, 55] * 10
    })
    
    # Initialize pipeline
    pipeline = FeaturePipeline()
    
    # Extract features (without expensive features for quick test)
    features = pipeline.extract_features(
        sample_df,
        include_perplexity=False,
        include_coherence=False
    )
    
    print("\nExtracted Features:")
    print(features.head())
    print(f"\nTotal features: {len(features.columns)}")
    print(f"Feature names: {list(features.columns)}")

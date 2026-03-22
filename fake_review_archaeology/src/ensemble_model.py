"""
Fake Review Archaeology - Ensemble Model Module
===============================================
Two-layer ensemble architecture for fake review detection.

Layer 1: 
- RoBERTa for semantic analysis
- XGBoost on statistical features

Layer 2:
- Meta-classifier (Logistic Regression) stacking predictions

Author: Data Analytics Team
Date: 2026-03-22
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
from pathlib import Path
import json
from datetime import datetime

# ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import xgboost as xgb

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewDataset(Dataset):
    """PyTorch Dataset for review text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class RoBERTaClassifier:
    """
    RoBERTa-based classifier for semantic analysis of reviews.
    """
    
    def __init__(self, model_name: str = 'roberta-base', 
                 num_labels: int = 2,
                 device: Optional[str] = None):
        """
        Initialize RoBERTa classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of classification labels
            device: Device for training/inference
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = None
        self.training_history = []
        
        logger.info("RoBERTaClassifier initialized on device: %s", self.device)
    
    def build_model(self):
        """Build the RoBERTa classification model."""
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        logger.info("RoBERTa model built with %d labels", self.num_labels)
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: Optional[List[str]] = None,
              val_labels: Optional[List[int]] = None,
              epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 100,
              save_path: Optional[str] = None):
        """
        Train the RoBERTa model.
        
        Args:
            train_texts: Training review texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
            save_path: Path to save best model
        """
        if self.model is None:
            self.build_model()
        
        # Create datasets
        train_dataset = ReviewDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_texts is not None and val_labels is not None:
                val_loss, val_metrics = self.evaluate(val_texts, val_labels, batch_size)
                logger.info(f"Validation loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
                
                # Save best model
                if save_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(save_path)
                    logger.info(f"Saved best model to {save_path}")
                
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_metrics['accuracy']
                })
    
    def evaluate(self, texts: List[str], labels: List[int], 
                batch_size: int = 16) -> Tuple[float, Dict]:
        """
        Evaluate the model.
        
        Args:
            texts: Review texts
            labels: True labels
            batch_size: Batch size
            
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.eval()
        dataset = ReviewDataset(texts, labels, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                
                total_loss += outputs.loss.item()
                
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
        }
        
        return avg_loss, metrics
    
    def predict(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Get predictions for texts.
        
        Args:
            texts: Review texts
            batch_size: Batch size
            
        Returns:
            Array of predicted probabilities (class 1)
        """
        self.model.eval()
        
        # Create dummy labels
        dummy_labels = [0] * len(texts)
        dataset = ReviewDataset(texts, dummy_labels, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        return np.array(all_probs)
    
    def save_model(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Model saved to %s", path)
    
    def load_model(self, path: str):
        """Load model from disk."""
        self.tokenizer = RobertaTokenizer.from_pretrained(path)
        self.model = RobertaForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        logger.info("Model loaded from %s", path)


class XGBoostClassifier:
    """
    XGBoost classifier for statistical features.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize XGBoost classifier.
        
        Args:
            params: XGBoost parameters
        """
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc', 'error'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_history = []
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None):
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names
        """
        self.feature_names = feature_names
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Build model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Prepare eval set
        eval_set = [(X_train_scaled, y_train)]
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set.append((X_val_scaled, y_val))
        
        # Train
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Store feature importance
        importance = self.model.feature_importances_
        if feature_names:
            self.feature_importance = dict(zip(feature_names, importance))
        else:
            self.feature_importance = {f'feature_{i}': imp for i, imp in enumerate(importance)}
        
        logger.info("XGBoost training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Metrics dictionary
        """
        probs = self.predict(X)
        preds = (probs > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y, preds),
            'precision': precision_score(y, preds, zero_division=0),
            'recall': recall_score(y, preds, zero_division=0),
            'f1': f1_score(y, preds, zero_division=0),
            'roc_auc': roc_auc_score(y, probs) if len(set(y)) > 1 else 0.5
        }
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance.items()
        ])
        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df.head(top_n)
    
    def save_model(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'params': self.params
            }, f)
        logger.info("XGBoost model saved to %s", path)
    
    def load_model(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.feature_importance = data['feature_importance']
            self.params = data['params']
        logger.info("XGBoost model loaded from %s", path)


class StackingEnsemble:
    """
    Two-layer stacking ensemble combining RoBERTa and XGBoost.
    """
    
    def __init__(self, 
                 roberta_model: Optional[RoBERTaClassifier] = None,
                 xgb_model: Optional[XGBoostClassifier] = None,
                 meta_classifier: Optional[Any] = None):
        """
        Initialize stacking ensemble.
        
        Args:
            roberta_model: Pre-trained RoBERTa classifier
            xgb_model: Pre-trained XGBoost classifier
            meta_classifier: Meta-classifier for stacking
        """
        self.roberta = roberta_model or RoBERTaClassifier()
        self.xgb = xgb_model or XGBoostClassifier()
        self.meta_classifier = meta_classifier or LogisticRegression(
            C=1.0, class_weight='balanced', random_state=42
        )
        
        self.is_trained = False
        self.cv_folds = 5
        
        logger.info("StackingEnsemble initialized")
    
    def train_base_models(self, 
                         train_texts: List[str],
                         train_features: np.ndarray,
                         train_labels: np.ndarray,
                         val_texts: Optional[List[str]] = None,
                         val_features: Optional[np.ndarray] = None,
                         val_labels: Optional[List[int]] = None,
                         feature_names: Optional[List[str]] = None,
                         roberta_epochs: int = 3,
                         roberta_batch_size: int = 16):
        """
        Train base models (RoBERTa and XGBoost).
        
        Args:
            train_texts: Training review texts
            train_features: Training statistical features
            train_labels: Training labels
            val_texts: Validation texts
            val_features: Validation features
            val_labels: Validation labels
            feature_names: Feature names
            roberta_epochs: Epochs for RoBERTa training
            roberta_batch_size: Batch size for RoBERTa
        """
        logger.info("Training base models...")
        
        # Train RoBERTa
        logger.info("Training RoBERTa...")
        self.roberta.train(
            train_texts, train_labels,
            val_texts, val_labels,
            epochs=roberta_epochs,
            batch_size=roberta_batch_size
        )
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        self.xgb.train(
            train_features, train_labels,
            val_features, val_labels,
            feature_names=feature_names
        )
        
        logger.info("Base models training complete")
    
    def generate_meta_features(self, 
                              texts: List[str],
                              features: np.ndarray,
                              labels: np.ndarray,
                              use_cv: bool = True) -> np.ndarray:
        """
        Generate meta-features for stacking using cross-validation predictions.
        
        Args:
            texts: Review texts
            features: Statistical features
            labels: True labels (for stratified CV)
            use_cv: Whether to use cross-validation
            
        Returns:
            Meta-feature matrix
        """
        logger.info("Generating meta-features...")
        
        if use_cv and not self.is_trained:
            # Use cross-validation to generate out-of-fold predictions
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            roberta_preds = np.zeros(len(labels))
            xgb_preds = np.zeros(len(labels))
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
                logger.info(f"CV Fold {fold + 1}/{self.cv_folds}")
                
                # Split data
                fold_train_texts = [texts[i] for i in train_idx]
                fold_val_texts = [texts[i] for i in val_idx]
                fold_train_features = features[train_idx]
                fold_val_features = features[val_idx]
                fold_train_labels = labels[train_idx]
                
                # Train temporary models
                temp_roberta = RoBERTaClassifier()
                temp_roberta.train(
                    fold_train_texts, fold_train_labels,
                    epochs=2, batch_size=16
                )
                roberta_preds[val_idx] = temp_roberta.predict(fold_val_texts)
                
                temp_xgb = XGBoostClassifier()
                temp_xgb.train(fold_train_features, fold_train_labels)
                xgb_preds[val_idx] = temp_xgb.predict(fold_val_features)
            
            meta_features = np.column_stack([roberta_preds, xgb_preds])
        else:
            # Use trained models for predictions
            roberta_preds = self.roberta.predict(texts)
            xgb_preds = self.xgb.predict(features)
            meta_features = np.column_stack([roberta_preds, xgb_preds])
        
        return meta_features
    
    def train_meta_classifier(self, 
                             meta_features: np.ndarray,
                             labels: np.ndarray):
        """
        Train meta-classifier on base model predictions.
        
        Args:
            meta_features: Meta-feature matrix
            labels: True labels
        """
        logger.info("Training meta-classifier...")
        
        self.meta_classifier.fit(meta_features, labels)
        self.is_trained = True
        
        logger.info("Meta-classifier training complete")
    
    def train(self,
             train_texts: List[str],
             train_features: np.ndarray,
             train_labels: np.ndarray,
             val_texts: Optional[List[str]] = None,
             val_features: Optional[np.ndarray] = None,
             val_labels: Optional[List[int]] = None,
             feature_names: Optional[List[str]] = None,
             roberta_epochs: int = 3):
        """
        Complete training pipeline for the ensemble.
        
        Args:
            train_texts: Training review texts
            train_features: Training statistical features
            train_labels: Training labels
            val_texts: Validation texts
            val_features: Validation features
            val_labels: Validation labels
            feature_names: Feature names
            roberta_epochs: Epochs for RoBERTa
        """
        # Train base models
        self.train_base_models(
            train_texts, train_features, train_labels,
            val_texts, val_features, val_labels,
            feature_names, roberta_epochs
        )
        
        # Generate meta-features
        meta_features = self.generate_meta_features(
            train_texts, train_features, train_labels, use_cv=True
        )
        
        # Train meta-classifier
        self.train_meta_classifier(meta_features, train_labels)
        
        logger.info("Ensemble training complete")
    
    def predict(self, texts: List[str], features: np.ndarray) -> np.ndarray:
        """
        Get ensemble predictions.
        
        Args:
            texts: Review texts
            features: Statistical features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")
        
        # Get base model predictions
        roberta_preds = self.roberta.predict(texts)
        xgb_preds = self.xgb.predict(features)
        
        # Stack predictions
        meta_features = np.column_stack([roberta_preds, xgb_preds])
        
        # Meta-classifier prediction
        ensemble_probs = self.meta_classifier.predict_proba(meta_features)[:, 1]
        
        return ensemble_probs
    
    def evaluate(self, texts: List[str], features: np.ndarray, 
                labels: np.ndarray) -> Dict:
        """
        Evaluate ensemble and individual models.
        
        Args:
            texts: Review texts
            features: Statistical features
            labels: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # RoBERTa evaluation
        roberta_probs = self.roberta.predict(texts)
        roberta_preds = (roberta_probs > 0.5).astype(int)
        results['roberta'] = {
            'accuracy': accuracy_score(labels, roberta_preds),
            'precision': precision_score(labels, roberta_preds, zero_division=0),
            'recall': recall_score(labels, roberta_preds, zero_division=0),
            'f1': f1_score(labels, roberta_preds, zero_division=0),
            'roc_auc': roc_auc_score(labels, roberta_probs)
        }
        
        # XGBoost evaluation
        xgb_probs = self.xgb.predict(features)
        xgb_preds = (xgb_probs > 0.5).astype(int)
        results['xgboost'] = {
            'accuracy': accuracy_score(labels, xgb_preds),
            'precision': precision_score(labels, xgb_preds, zero_division=0),
            'recall': recall_score(labels, xgb_preds, zero_division=0),
            'f1': f1_score(labels, xgb_preds, zero_division=0),
            'roc_auc': roc_auc_score(labels, xgb_probs)
        }
        
        # Ensemble evaluation
        ensemble_probs = self.predict(texts, features)
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        results['ensemble'] = {
            'accuracy': accuracy_score(labels, ensemble_preds),
            'precision': precision_score(labels, ensemble_preds, zero_division=0),
            'recall': recall_score(labels, ensemble_preds, zero_division=0),
            'f1': f1_score(labels, ensemble_preds, zero_division=0),
            'roc_auc': roc_auc_score(labels, ensemble_probs)
        }
        
        # Confusion matrices
        results['confusion_matrices'] = {
            'roberta': confusion_matrix(labels, roberta_preds).tolist(),
            'xgboost': confusion_matrix(labels, xgb_preds).tolist(),
            'ensemble': confusion_matrix(labels, ensemble_preds).tolist()
        }
        
        return results
    
    def save_ensemble(self, output_dir: str):
        """
        Save all ensemble components.
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save RoBERTa
        self.roberta.save_model(str(output_path / 'roberta'))
        
        # Save XGBoost
        self.xgb.save_model(str(output_path / 'xgboost.pkl'))
        
        # Save meta-classifier
        with open(output_path / 'meta_classifier.pkl', 'wb') as f:
            pickle.dump(self.meta_classifier, f)
        
        # Save ensemble config
        config = {
            'is_trained': self.is_trained,
            'cv_folds': self.cv_folds,
            'timestamp': datetime.now().isoformat()
        }
        with open(output_path / 'ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Ensemble saved to %s", output_dir)
    
    def load_ensemble(self, output_dir: str):
        """
        Load all ensemble components.
        
        Args:
            output_dir: Directory containing saved models
        """
        output_path = Path(output_dir)
        
        # Load RoBERTa
        self.roberta.load_model(str(output_path / 'roberta'))
        
        # Load XGBoost
        self.xgb.load_model(str(output_path / 'xgboost.pkl'))
        
        # Load meta-classifier
        with open(output_path / 'meta_classifier.pkl', 'rb') as f:
            self.meta_classifier = pickle.load(f)
        
        # Load config
        with open(output_path / 'ensemble_config.json', 'r') as f:
            config = json.load(f)
            self.is_trained = config['is_trained']
        
        logger.info("Ensemble loaded from %s", output_dir)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    sample_texts = [
        'This product is absolutely amazing! I love it so much.',
        'Terrible product. Complete waste of money.',
        'Good quality for the price. Works as expected.',
        'Outstanding! Best purchase I have ever made.',
        'Do not buy this. Broke after one day.',
    ] * (n_samples // 5)
    
    sample_features = np.random.randn(n_samples, 20)
    sample_labels = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Split data
    split_idx = int(0.8 * n_samples)
    train_texts = sample_texts[:split_idx]
    train_features = sample_features[:split_idx]
    train_labels = sample_labels[:split_idx]
    
    val_texts = sample_texts[split_idx:]
    val_features = sample_features[split_idx:]
    val_labels = sample_labels[split_idx:]
    
    # Initialize and train ensemble
    ensemble = StackingEnsemble()
    
    # Quick test with minimal training
    ensemble.train(
        train_texts, train_features, train_labels,
        val_texts, val_features, val_labels,
        roberta_epochs=1
    )
    
    # Evaluate
    results = ensemble.evaluate(val_texts, val_features, val_labels)
    
    print("\nEvaluation Results:")
    for model, metrics in results.items():
        if model != 'confusion_matrices':
            print(f"\n{model.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

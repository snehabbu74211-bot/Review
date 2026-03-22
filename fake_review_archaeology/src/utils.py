"""
Fake Review Archaeology - Utility Functions
===========================================
Helper functions and utilities for the fake review detection system.

Author: Data Analytics Team
Date: 2026-03-22
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional file path for logging
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def save_json(data: Dict, filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: str):
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_experiment_dir(base_dir: str = 'experiments') -> Path:
    """
    Create timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(base_dir) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'models').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    (exp_dir / 'data').mkdir(exist_ok=True)
    
    return exp_dir


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap = plt.cm.Blues,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: Class names
        normalize: Whether to normalize
        title: Plot title
        cmap: Colormap
        save_path: Optional path to save figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = 'ROC Curve',
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        title: Plot title
        save_path: Optional path to save figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = 'Precision-Recall Curve',
    save_path: Optional[str] = None
):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        title: Plot title
        save_path: Optional path to save figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
            label=f'PR Curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def get_risk_level(probability: float) -> str:
    """
    Get risk level from probability score.
    
    Args:
        probability: Fraud probability (0-1)
        
    Returns:
        Risk level string
    """
    if probability >= 0.9:
        return 'Critical'
    elif probability >= 0.7:
        return 'High'
    elif probability >= 0.5:
        return 'Medium'
    elif probability >= 0.3:
        return 'Low'
    else:
        return 'Minimal'


def get_risk_color(probability: float) -> str:
    """
    Get risk color from probability score.
    
    Args:
        probability: Fraud probability (0-1)
        
    Returns:
        Hex color code
    """
    if probability >= 0.9:
        return '#7f1d1d'  # Dark red
    elif probability >= 0.7:
        return '#dc2626'  # Red
    elif probability >= 0.5:
        return '#f59e0b'  # Orange
    elif probability >= 0.3:
        return '#10b981'  # Green
    else:
        return '#3b82f6'  # Blue


def format_currency(value: float, precision: int = 1) -> str:
    """
    Format value as currency string.
    
    Args:
        value: Numeric value
        precision: Decimal precision
        
    Returns:
        Formatted currency string
    """
    if value >= 1e9:
        return f'${value/1e9:.{precision}f}B'
    elif value >= 1e6:
        return f'${value/1e6:.{precision}f}M'
    elif value >= 1e3:
        return f'${value/1e3:.{precision}f}K'
    else:
        return f'${value:.{precision}f}'


def calculate_sample_size(
    population: int,
    confidence: float = 0.95,
    margin_error: float = 0.05,
    proportion: float = 0.5
) -> int:
    """
    Calculate required sample size for statistical significance.
    
    Args:
        population: Population size
        confidence: Confidence level (0-1)
        margin_error: Margin of error (0-1)
        proportion: Expected proportion (0-1)
        
    Returns:
        Required sample size
    """
    from scipy.stats import norm
    
    z_score = norm.ppf((1 + confidence) / 2)
    numerator = (z_score ** 2) * proportion * (1 - proportion)
    denominator = margin_error ** 2
    sample_size = numerator / denominator
    
    # Adjust for finite population
    sample_size = sample_size / (1 + (sample_size - 1) / population)
    
    return int(np.ceil(sample_size))


def memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Calculate DataFrame memory usage in MB.
    
    Args:
        df: DataFrame
        
    Returns:
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting types.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    df_optimized = df.copy()
    
    # Downcast integers
    int_cols = df_optimized.select_dtypes(include=['int']).columns
    df_optimized[int_cols] = df_optimized[int_cols].apply(
        pd.to_numeric, downcast='integer'
    )
    
    # Downcast floats
    float_cols = df_optimized.select_dtypes(include=['float']).columns
    df_optimized[float_cols] = df_optimized[float_cols].apply(
        pd.to_numeric, downcast='float'
    )
    
    # Convert object columns to category if beneficial
    obj_cols = df_optimized.select_dtypes(include=['object']).columns
    for col in obj_cols:
        num_unique = df_optimized[col].nunique()
        num_total = len(df_optimized[col])
        if num_unique / num_total < 0.5:
            df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.name}...")
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"{self.name} completed in {elapsed:.2f} seconds")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0


# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(log_file='logs/utils_test.log')
    
    # Test timer
    with Timer("Sample Operation", logger):
        import time
        time.sleep(1)
    
    # Test DataFrame optimization
    df = pd.DataFrame({
        'a': np.random.randint(0, 100, 100000),
        'b': np.random.randn(100000),
        'c': np.random.choice(['x', 'y', 'z'], 100000)
    })
    
    print(f"Before optimization: {memory_usage_mb(df):.2f} MB")
    df_opt = optimize_dataframe(df)
    print(f"After optimization: {memory_usage_mb(df_opt):.2f} MB")
    
    # Test currency formatting
    print(format_currency(1234567.89))
    print(format_currency(1234567890))

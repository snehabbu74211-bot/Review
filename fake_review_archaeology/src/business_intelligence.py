"""
Fake Review Archaeology - Business Intelligence Layer
=====================================================
Analyzes fraud patterns, identifies high-risk segments,
and quantifies business impact.

Author: Data Analytics Team
Date: 2026-03-22
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudAnalyzer:
    """
    Analyzes fraud patterns and generates business insights.
    """
    
    def __init__(self, risk_threshold: float = 0.7):
        """
        Initialize fraud analyzer.
        
        Args:
            risk_threshold: Probability threshold for high-risk classification
        """
        self.risk_threshold = risk_threshold
        self.analysis_results = {}
        
        logger.info("FraudAnalyzer initialized with threshold: %.2f", risk_threshold)
    
    def analyze_by_category(self, 
                           df: pd.DataFrame,
                           category_col: str = 'category',
                           risk_col: str = 'fraud_probability',
                           gmv_col: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze fraud rates by product category.
        
        Args:
            df: DataFrame with predictions
            category_col: Column containing category information
            risk_col: Column containing fraud probability
            gmv_col: Column containing GMV (Gross Merchandise Value)
            
        Returns:
            DataFrame with category analysis
        """
        logger.info("Analyzing fraud by category...")
        
        if category_col not in df.columns:
            logger.warning("Category column not found: %s", category_col)
            return pd.DataFrame()
        
        results = []
        
        for category in df[category_col].unique():
            if pd.isna(category):
                continue
            
            cat_data = df[df[category_col] == category]
            
            # Calculate metrics
            total_reviews = len(cat_data)
            high_risk_reviews = (cat_data[risk_col] >= self.risk_threshold).sum()
            synthetic_rate = high_risk_reviews / total_reviews if total_reviews > 0 else 0
            
            # Average risk score
            avg_risk = cat_data[risk_col].mean()
            
            # GMV at risk
            gmv_at_risk = 0
            if gmv_col and gmv_col in df.columns:
                gmv_at_risk = cat_data[cat_data[risk_col] >= self.risk_threshold][gmv_col].sum()
                total_gmv = cat_data[gmv_col].sum()
            else:
                total_gmv = 0
            
            results.append({
                'category': category,
                'total_reviews': total_reviews,
                'high_risk_reviews': high_risk_reviews,
                'synthetic_rate': synthetic_rate,
                'avg_risk_score': avg_risk,
                'total_gmv': total_gmv,
                'gmv_at_risk': gmv_at_risk,
                'risk_level': self._categorize_risk(synthetic_rate)
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('synthetic_rate', ascending=False)
        
        self.analysis_results['by_category'] = results_df
        
        return results_df
    
    def analyze_by_rating(self,
                         df: pd.DataFrame,
                         rating_col: str = 'rating',
                         risk_col: str = 'fraud_probability') -> pd.DataFrame:
        """
        Analyze fraud patterns by rating.
        
        Args:
            df: DataFrame with predictions
            rating_col: Column containing ratings
            risk_col: Column containing fraud probability
            
        Returns:
            DataFrame with rating analysis
        """
        logger.info("Analyzing fraud by rating...")
        
        if rating_col not in df.columns:
            logger.warning("Rating column not found: %s", rating_col)
            return pd.DataFrame()
        
        results = []
        
        for rating in sorted(df[rating_col].unique()):
            rating_data = df[df[rating_col] == rating]
            
            total_reviews = len(rating_data)
            high_risk = (rating_data[risk_col] >= self.risk_threshold).sum()
            fraud_rate = high_risk / total_reviews if total_reviews > 0 else 0
            avg_risk = rating_data[risk_col].mean()
            
            results.append({
                'rating': rating,
                'total_reviews': total_reviews,
                'high_risk_count': high_risk,
                'fraud_rate': fraud_rate,
                'avg_risk_score': avg_risk
            })
        
        results_df = pd.DataFrame(results)
        self.analysis_results['by_rating'] = results_df
        
        return results_df
    
    def analyze_temporal_patterns(self,
                                  df: pd.DataFrame,
                                  date_col: str = 'review_date',
                                  risk_col: str = 'fraud_probability') -> pd.DataFrame:
        """
        Analyze fraud patterns over time.
        
        Args:
            df: DataFrame with predictions
            date_col: Column containing dates
            risk_col: Column containing fraud probability
            
        Returns:
            DataFrame with temporal analysis
        """
        logger.info("Analyzing temporal patterns...")
        
        if date_col not in df.columns:
            logger.warning("Date column not found: %s", date_col)
            return pd.DataFrame()
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['date'] = df[date_col].dt.date
        
        daily_stats = df.groupby('date').agg({
            risk_col: ['mean', 'std', 'count'],
        }).reset_index()
        
        daily_stats.columns = ['date', 'avg_risk', 'risk_std', 'review_count']
        daily_stats['high_risk_count'] = df[df[risk_col] >= self.risk_threshold].groupby('date').size().values
        daily_stats['fraud_rate'] = daily_stats['high_risk_count'] / daily_stats['review_count']
        
        self.analysis_results['temporal'] = daily_stats
        
        return daily_stats
    
    def identify_suspicious_accounts(self,
                                    df: pd.DataFrame,
                                    reviewer_col: str = 'reviewer_id',
                                    risk_col: str = 'fraud_probability',
                                    min_reviews: int = 3) -> pd.DataFrame:
        """
        Identify suspicious reviewer accounts.
        
        Args:
            df: DataFrame with predictions
            reviewer_col: Column containing reviewer IDs
            risk_col: Column containing fraud probability
            min_reviews: Minimum reviews for analysis
            
        Returns:
            DataFrame with suspicious accounts
        """
        logger.info("Identifying suspicious accounts...")
        
        if reviewer_col not in df.columns:
            logger.warning("Reviewer column not found: %s", reviewer_col)
            return pd.DataFrame()
        
        reviewer_stats = df.groupby(reviewer_col).agg({
            risk_col: ['mean', 'std', 'count'],
            'review_length': 'mean',
            'rating': ['mean', 'std']
        }).reset_index()
        
        reviewer_stats.columns = [
            'reviewer_id', 'avg_risk', 'risk_std', 'review_count',
            'avg_review_length', 'avg_rating', 'rating_std'
        ]
        
        # Filter accounts with minimum reviews
        reviewer_stats = reviewer_stats[reviewer_stats['review_count'] >= min_reviews]
        
        # Calculate suspicion score
        reviewer_stats['suspicion_score'] = (
            reviewer_stats['avg_risk'] * 0.4 +
            (reviewer_stats['review_count'] / reviewer_stats['review_count'].max()) * 0.2 +
            (1 - reviewer_stats['rating_std'].fillna(0) / 2) * 0.2 +
            (reviewer_stats['avg_review_length'] < 50).astype(float) * 0.2
        )
        
        # Flag high-risk accounts
        reviewer_stats['is_suspicious'] = (
            (reviewer_stats['avg_risk'] >= self.risk_threshold) |
            (reviewer_stats['suspicion_score'] >= 0.7)
        )
        
        reviewer_stats = reviewer_stats.sort_values('suspicion_score', ascending=False)
        
        self.analysis_results['suspicious_accounts'] = reviewer_stats
        
        return reviewer_stats
    
    def calculate_business_impact(self,
                                  df: pd.DataFrame,
                                  risk_col: str = 'fraud_probability',
                                  gmv_col: Optional[str] = None,
                                  conversion_impact: float = 0.15) -> Dict:
        """
        Calculate business impact of fake reviews.
        
        Args:
            df: DataFrame with predictions
            risk_col: Column containing fraud probability
            gmv_col: Column containing GMV data
            conversion_impact: Estimated conversion rate impact
            
        Returns:
            Dictionary with business impact metrics
        """
        logger.info("Calculating business impact...")
        
        total_reviews = len(df)
        high_risk_reviews = (df[risk_col] >= self.risk_threshold).sum()
        fraud_rate = high_risk_reviews / total_reviews
        
        impact = {
            'total_reviews': total_reviews,
            'high_risk_reviews': int(high_risk_reviews),
            'overall_fraud_rate': fraud_rate,
            'estimated_fake_reviews': int(high_risk_reviews),
            'analysis_date': datetime.now().isoformat()
        }
        
        # GMV impact
        if gmv_col and gmv_col in df.columns:
            total_gmv = df[gmv_col].sum()
            gmv_at_risk = df[df[risk_col] >= self.risk_threshold][gmv_col].sum()
            
            impact['total_gmv'] = total_gmv
            impact['gmv_at_risk'] = gmv_at_risk
            impact['gmv_at_risk_percentage'] = gmv_at_risk / total_gmv if total_gmv > 0 else 0
            impact['estimated_revenue_impact'] = gmv_at_risk * conversion_impact
        
        # Trust score impact
        impact['trust_score_impact'] = fraud_rate * 100  # Simple linear model
        
        # Customer acquisition cost impact
        impact['estimated_cac_increase'] = fraud_rate * 0.25  # 25% CAC increase per fraud rate
        
        self.analysis_results['business_impact'] = impact
        
        return impact
    
    def generate_risk_segments(self,
                              df: pd.DataFrame,
                              risk_col: str = 'fraud_probability',
                              category_col: Optional[str] = None) -> pd.DataFrame:
        """
        Generate risk segments for targeted action.
        
        Args:
            df: DataFrame with predictions
            risk_col: Column containing fraud probability
            category_col: Optional category column
            
        Returns:
            DataFrame with risk segments
        """
        logger.info("Generating risk segments...")
        
        # Define risk segments
        def assign_segment(prob):
            if prob >= 0.9:
                return 'Critical'
            elif prob >= 0.7:
                return 'High'
            elif prob >= 0.5:
                return 'Medium'
            elif prob >= 0.3:
                return 'Low'
            else:
                return 'Minimal'
        
        df = df.copy()
        df['risk_segment'] = df[risk_col].apply(assign_segment)
        
        segment_order = ['Critical', 'High', 'Medium', 'Low', 'Minimal']
        
        if category_col and category_col in df.columns:
            segments = df.groupby([category_col, 'risk_segment']).size().unstack(fill_value=0)
            # Reorder columns
            segments = segments[[s for s in segment_order if s in segments.columns]]
        else:
            segments = df.groupby('risk_segment').size().to_frame('count')
            segments = segments.reindex([s for s in segment_order if s in segments.index])
        
        self.analysis_results['risk_segments'] = segments
        
        return segments
    
    def _categorize_risk(self, fraud_rate: float) -> str:
        """Categorize risk level based on fraud rate."""
        if fraud_rate >= 0.5:
            return 'Critical'
        elif fraud_rate >= 0.3:
            return 'High'
        elif fraud_rate >= 0.15:
            return 'Medium'
        elif fraud_rate >= 0.05:
            return 'Low'
        else:
            return 'Minimal'
    
    def generate_insights_report(self) -> Dict:
        """
        Generate comprehensive insights report.
        
        Returns:
            Dictionary with all insights
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'risk_threshold': self.risk_threshold,
            'insights': []
        }
        
        # Category insights
        if 'by_category' in self.analysis_results:
            cat_df = self.analysis_results['by_category']
            top_risky = cat_df.head(3)
            
            for _, row in top_risky.iterrows():
                insight = {
                    'type': 'category_risk',
                    'category': row['category'],
                    'metric': 'synthetic_rate',
                    'value': row['synthetic_rate'],
                    'description': f"{row['category']}: {row['synthetic_rate']*100:.1f}% synthetic rate"
                }
                
                if row['gmv_at_risk'] > 0:
                    insight['description'] += f", ${row['gmv_at_risk']/1e6:.1f}M GMV at risk"
                
                report['insights'].append(insight)
        
        # Account insights
        if 'suspicious_accounts' in self.analysis_results:
            susp_df = self.analysis_results['suspicious_accounts']
            suspicious_count = susp_df['is_suspicious'].sum()
            
            report['insights'].append({
                'type': 'account_risk',
                'metric': 'suspicious_accounts',
                'value': int(suspicious_count),
                'description': f"{int(suspicious_count)} suspicious accounts identified requiring review"
            })
        
        # Business impact insights
        if 'business_impact' in self.analysis_results:
            impact = self.analysis_results['business_impact']
            
            report['insights'].append({
                'type': 'business_impact',
                'metric': 'fraud_rate',
                'value': impact['overall_fraud_rate'],
                'description': f"Overall fraud rate: {impact['overall_fraud_rate']*100:.1f}%"
            })
            
            if 'gmv_at_risk' in impact:
                report['insights'].append({
                    'type': 'business_impact',
                    'metric': 'gmv_at_risk',
                    'value': impact['gmv_at_risk'],
                    'description': f"${impact['gmv_at_risk']/1e6:.1f}M GMV at risk from fake reviews"
                })
        
        return report
    
    def save_analysis(self, output_dir: str):
        """
        Save all analysis results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each analysis component
        for name, data in self.analysis_results.items():
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path / f'{name}.csv', index=True)
            elif isinstance(data, dict):
                with open(output_path / f'{name}.json', 'w') as f:
                    json.dump(data, f, indent=2, default=str)
        
        # Save insights report
        report = self.generate_insights_report()
        with open(output_path / 'insights_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Analysis saved to %s", output_dir)


class RiskVisualizer:
    """
    Creates visualizations for fraud analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize visualizer."""
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
    
    def plot_fraud_heatmap(self,
                          category_data: pd.DataFrame,
                          save_path: Optional[str] = None):
        """
        Create fraud rate heatmap by category.
        
        Args:
            category_data: DataFrame with category analysis
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        heatmap_data = category_data.set_index('category')[['synthetic_rate']].T
        
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.2%',
                   cmap='Reds',
                   cbar_kws={'label': 'Fraud Rate'},
                   ax=ax)
        
        ax.set_title('Fraud Rate Heatmap by Category', fontsize=14, fontweight='bold')
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Heatmap saved to %s", save_path)
        
        return fig
    
    def plot_risk_distribution(self,
                              df: pd.DataFrame,
                              risk_col: str = 'fraud_probability',
                              save_path: Optional[str] = None):
        """
        Plot distribution of risk scores.
        
        Args:
            df: DataFrame with predictions
            risk_col: Column containing risk scores
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(df[risk_col], bins=50, color='#3498db', edgecolor='white', alpha=0.7)
        axes[0].axvline(x=0.7, color='#e74c3c', linestyle='--', label='High Risk Threshold')
        axes[0].set_xlabel('Fraud Probability', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Distribution of Fraud Probabilities', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        # Box plot by category if available
        if 'category' in df.columns:
            sns.boxplot(data=df, x='category', y=risk_col, ax=axes[1])
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
            axes[1].set_title('Risk Distribution by Category', fontsize=12, fontweight='bold')
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Risk distribution plot saved to %s", save_path)
        
        return fig
    
    def plot_temporal_trends(self,
                            temporal_data: pd.DataFrame,
                            save_path: Optional[str] = None):
        """
        Plot temporal fraud trends.
        
        Args:
            temporal_data: DataFrame with temporal analysis
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Fraud rate over time
        axes[0].plot(temporal_data['date'], temporal_data['fraud_rate'], 
                    color='#e74c3c', linewidth=2)
        axes[0].fill_between(temporal_data['date'], temporal_data['fraud_rate'], 
                            alpha=0.3, color='#e74c3c')
        axes[0].set_ylabel('Fraud Rate', fontsize=12)
        axes[0].set_title('Fraud Rate Trends Over Time', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Review volume over time
        axes[1].bar(temporal_data['date'], temporal_data['review_count'], 
                   color='#3498db', alpha=0.7)
        axes[1].set_ylabel('Review Count', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_title('Review Volume Over Time', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Temporal trends plot saved to %s", save_path)
        
        return fig
    
    def plot_business_impact(self,
                            impact_data: Dict,
                            save_path: Optional[str] = None):
        """
        Create business impact visualization.
        
        Args:
            impact_data: Dictionary with business impact metrics
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Fraud breakdown pie chart
        labels = ['Genuine Reviews', 'Suspicious Reviews']
        sizes = [
            impact_data['total_reviews'] - impact_data['high_risk_reviews'],
            impact_data['high_risk_reviews']
        ]
        colors = ['#2ecc71', '#e74c3c']
        
        axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, explode=(0, 0.1))
        axes[0].set_title('Review Authenticity Breakdown', fontsize=12, fontweight='bold')
        
        # GMV impact bar chart
        if 'gmv_at_risk' in impact_data:
            categories = ['Total GMV', 'GMV at Risk']
            values = [impact_data['total_gmv'] / 1e6, impact_data['gmv_at_risk'] / 1e6]
            
            bars = axes[1].bar(categories, values, color=['#3498db', '#e74c3c'])
            axes[1].set_ylabel('GMV ($ Millions)', fontsize=12)
            axes[1].set_title('GMV Impact Analysis', fontsize=12, fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:.1f}M',
                           ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Business impact plot saved to %s", save_path)
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_df = pd.DataFrame({
        'review_id': range(n_samples),
        'reviewer_id': np.random.choice(['user_' + str(i) for i in range(100)], n_samples),
        'product_id': np.random.choice(['prod_' + str(i) for i in range(50)], n_samples),
        'category': np.random.choice(
            ['Electronics', 'Supplements', 'Clothing', 'Home & Garden', 'Books'],
            n_samples
        ),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
        'review_date': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'fraud_probability': np.random.beta(2, 5, n_samples),
        'gmv': np.random.exponential(100, n_samples)
    })
    
    # Initialize analyzer
    analyzer = FraudAnalyzer(risk_threshold=0.7)
    
    # Run analyses
    category_analysis = analyzer.analyze_by_category(
        sample_df, gmv_col='gmv'
    )
    print("\nCategory Analysis:")
    print(category_analysis.head())
    
    rating_analysis = analyzer.analyze_by_rating(sample_df)
    print("\nRating Analysis:")
    print(rating_analysis)
    
    suspicious_accounts = analyzer.identify_suspicious_accounts(sample_df)
    print("\nSuspicious Accounts:")
    print(suspicious_accounts.head())
    
    business_impact = analyzer.calculate_business_impact(
        sample_df, gmv_col='gmv'
    )
    print("\nBusiness Impact:")
    print(json.dumps(business_impact, indent=2, default=str))
    
    # Generate insights
    insights = analyzer.generate_insights_report()
    print("\nKey Insights:")
    for insight in insights['insights']:
        print(f"  - {insight['description']}")

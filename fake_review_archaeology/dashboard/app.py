"""
Fake Review Archaeology - Interactive Dashboard
===============================================
Streamlit application for fraud detection visualization and real-time scoring.

Features:
- Fraud heatmaps by category
- t-SNE linguistic fingerprint visualization
- Real-time risk scoring
- Business impact analysis

Author: Data Analytics Team
Date: 2026-03-22
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# ML imports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Configure page
st.set_page_config(
    page_title="Fake Review Archaeology",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== DATA LOADING ====================

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    np.random.seed(42)
    n_samples = 2000
    
    categories = ['Electronics', 'Supplements', 'Clothing', 'Home & Garden', 
                  'Books', 'Beauty', 'Sports', 'Toys']
    
    data = {
        'review_id': range(n_samples),
        'reviewer_id': [f'user_{np.random.randint(1, 300)}' for _ in range(n_samples)],
        'product_id': [f'prod_{np.random.randint(1, 200)}' for _ in range(n_samples)],
        'category': np.random.choice(categories, n_samples),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.08, 0.12, 0.2, 0.3, 0.3]),
        'review_date': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'review_text': [
            'This product exceeded my expectations! Highly recommend to everyone.',
            'Average quality, nothing special but decent for the price.',
            'Terrible experience, would not recommend to anyone.',
            'Amazing value for money, fast shipping too!',
            'Disappointed with the quality, broke after a week.',
        ] * (n_samples // 5) + ['Good product, works as described.'] * (n_samples % 5),
        'fraud_probability': np.random.beta(2, 5, n_samples),
        'gmv': np.random.exponential(80, n_samples),
        'word_count': np.random.poisson(50, n_samples),
        'avg_word_length': np.random.normal(4.5, 0.8, n_samples),
        'sentence_count': np.random.poisson(5, n_samples),
        'punctuation_ratio': np.random.beta(2, 8, n_samples),
        'uppercase_ratio': np.random.beta(1.5, 50, n_samples),
        'unique_word_ratio': np.random.beta(7, 3, n_samples),
        'flesch_reading_ease': np.random.normal(60, 15, n_samples),
        'perplexity': np.random.lognormal(3, 0.5, n_samples),
        'burstiness': np.random.gamma(2, 0.3, n_samples),
        'semantic_coherence': np.random.beta(6, 2, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Adjust fraud probability based on category (some categories have higher fraud)
    fraud_multipliers = {
        'Supplements': 1.8,
        'Electronics': 1.4,
        'Beauty': 1.3,
        'Books': 0.6,
        'Clothing': 0.9,
        'Home & Garden': 0.8,
        'Sports': 0.7,
        'Toys': 0.75
    }
    
    for cat, mult in fraud_multipliers.items():
        mask = df['category'] == cat
        df.loc[mask, 'fraud_probability'] = (df.loc[mask, 'fraud_probability'] * mult).clip(0, 1)
    
    # Create risk segments
    df['risk_segment'] = pd.cut(
        df['fraud_probability'],
        bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
        labels=['Minimal', 'Low', 'Medium', 'High', 'Critical']
    )
    
    return df


@st.cache_data
def get_category_analysis(df):
    """Get fraud analysis by category."""
    analysis = df.groupby('category').agg({
        'fraud_probability': ['mean', 'std', 'count'],
        'gmv': 'sum'
    }).reset_index()
    
    analysis.columns = ['category', 'avg_fraud_prob', 'fraud_std', 'review_count', 'total_gmv']
    
    # Calculate high-risk count
    high_risk = df[df['fraud_probability'] >= 0.7].groupby('category').size()
    analysis['high_risk_count'] = analysis['category'].map(high_risk).fillna(0).astype(int)
    analysis['synthetic_rate'] = analysis['high_risk_count'] / analysis['review_count']
    analysis['gmv_at_risk'] = analysis['total_gmv'] * analysis['synthetic_rate']
    
    return analysis.sort_values('synthetic_rate', ascending=False)


# ==================== VISUALIZATION FUNCTIONS ====================

def create_fraud_heatmap(df):
    """Create fraud rate heatmap by category."""
    cat_analysis = get_category_analysis(df)
    
    fig = px.imshow(
        [cat_analysis['synthetic_rate'].values],
        x=cat_analysis['category'].values,
        y=['Fraud Rate'],
        color_continuous_scale='Reds',
        aspect='auto',
        title='Fraud Rate Heatmap by Category'
    )
    
    fig.update_traces(text=[[f'{v:.1%}' for v in cat_analysis['synthetic_rate'].values]],
                     texttemplate="%{text}")
    
    fig.update_layout(
        height=300,
        xaxis_title='Category',
        yaxis_title='',
        coloraxis_colorbar_title='Fraud Rate'
    )
    
    return fig


def create_tsne_visualization(df, sample_size=500):
    """Create t-SNE visualization of linguistic fingerprints."""
    # Sample data for performance
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Select linguistic features
    feature_cols = [
        'word_count', 'avg_word_length', 'sentence_count',
        'punctuation_ratio', 'uppercase_ratio', 'unique_word_ratio',
        'flesch_reading_ease', 'perplexity', 'burstiness', 'semantic_coherence'
    ]
    
    features = sample_df[feature_cols].fillna(0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(features)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'fraud_probability': sample_df['fraud_probability'].values,
        'risk_segment': sample_df['risk_segment'].values,
        'category': sample_df['category'].values,
        'review_text': sample_df['review_text'].values[:100]  # Limit for hover
    })
    
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='fraud_probability',
        hover_data=['risk_segment', 'category'],
        color_continuous_scale='RdYlGn_r',
        title='Linguistic Fingerprints (t-SNE)',
        labels={'fraud_probability': 'Fraud Probability'}
    )
    
    fig.update_layout(
        height=500,
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2'
    )
    
    return fig


def create_risk_distribution(df):
    """Create risk distribution visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Fraud Probability Distribution', 'Risk Segment Breakdown'),
        specs=[[{"secondary_y": False}, {"type": "domain"}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=df['fraud_probability'],
            nbinsx=50,
            name='Fraud Probability',
            marker_color='#3498db',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add threshold line
    fig.add_vline(x=0.7, line_dash="dash", line_color="#e74c3c", 
                 annotation_text="High Risk Threshold", row=1, col=1)
    
    # Pie chart
    segment_counts = df['risk_segment'].value_counts()
    colors = {'Critical': '#7f1d1d', 'High': '#dc2626', 'Medium': '#f59e0b', 
              'Low': '#10b981', 'Minimal': '#3b82f6'}
    
    fig.add_trace(
        go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            marker_colors=[colors.get(s, '#999') for s in segment_counts.index],
            textinfo='label+percent',
            hole=0.4
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Risk Distribution Analysis"
    )
    
    return fig


def create_temporal_trends(df):
    """Create temporal fraud trends."""
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['review_date']).dt.date
    
    daily_stats = df_temp.groupby('date').agg({
        'fraud_probability': 'mean',
        'review_id': 'count'
    }).reset_index()
    daily_stats.columns = ['date', 'avg_fraud_prob', 'review_count']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Fraud Probability Over Time', 'Review Volume Over Time'),
        shared_xaxes=True
    )
    
    # Fraud probability trend
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['avg_fraud_prob'],
            mode='lines',
            name='Avg Fraud Probability',
            line=dict(color='#e74c3c', width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ),
        row=1, col=1
    )
    
    # Review volume
    fig.add_trace(
        go.Bar(
            x=daily_stats['date'],
            y=daily_stats['review_count'],
            name='Review Count',
            marker_color='#3498db'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Fraud Probability", row=1, col=1)
    fig.update_yaxes(title_text="Review Count", row=2, col=1)
    
    return fig


def create_category_comparison(df):
    """Create category comparison chart."""
    cat_analysis = get_category_analysis(df)
    
    fig = go.Figure()
    
    # Synthetic rate bars
    fig.add_trace(go.Bar(
        x=cat_analysis['category'],
        y=cat_analysis['synthetic_rate'],
        name='Synthetic Rate',
        marker_color='#e74c3c',
        text=[f'{v:.1%}' for v in cat_analysis['synthetic_rate']],
        textposition='auto'
    ))
    
    # GMV at risk line
    fig.add_trace(go.Scatter(
        x=cat_analysis['category'],
        y=cat_analysis['gmv_at_risk'] / 1000,
        name='GMV at Risk ($K)',
        mode='lines+markers',
        line=dict(color='#f39c12', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Fraud Rate & GMV at Risk by Category',
        xaxis_title='Category',
        yaxis_title='Synthetic Rate',
        yaxis2=dict(
            title='GMV at Risk ($K)',
            overlaying='y',
            side='right'
        ),
        height=450,
        legend=dict(x=0.01, y=0.99),
        barmode='group'
    )
    
    return fig


# ==================== REAL-TIME SCORING ====================

def calculate_risk_score(text, features=None):
    """
    Calculate risk score for a new review.
    This is a simplified version for demonstration.
    """
    if features is None:
        features = {}
    
    # Simple heuristic scoring
    score = 0.3  # Base score
    
    # Text length factor
    text_len = len(text)
    if text_len < 30:
        score += 0.2
    elif text_len > 500:
        score -= 0.1
    
    # Capitalization factor
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.3:
        score += 0.15
    
    # Exclamation factor
    excl_count = text.count('!')
    if excl_count > 3:
        score += 0.1
    
    # Repetitive content
    words = text.lower().split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            score += 0.15
    
    # Generic phrases
    generic_phrases = ['amazing product', 'highly recommend', 'five stars', 
                      'best ever', 'love it', 'great product']
    for phrase in generic_phrases:
        if phrase in text.lower():
            score += 0.05
    
    return min(score, 1.0)


def get_risk_level(score):
    """Get risk level from score."""
    if score >= 0.9:
        return 'Critical', '#7f1d1d'
    elif score >= 0.7:
        return 'High', '#dc2626'
    elif score >= 0.5:
        return 'Medium', '#f59e0b'
    elif score >= 0.3:
        return 'Low', '#10b981'
    else:
        return 'Minimal', '#3b82f6'


# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Fake Review Archaeology</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Fraud Detection & Business Intelligence</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dashboard Overview", "🔥 Fraud Heatmap", "🧬 Linguistic Fingerprints", 
         "⚡ Real-time Scoring", "📈 Business Impact", "🔍 Deep Dive Analysis"]
    )
    
    # Load data
    df = load_sample_data()
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=df['category'].unique(),
        default=df['category'].unique()
    )
    
    risk_threshold = st.sidebar.slider(
        "Risk Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df['review_date'].min().date(), df['review_date'].max().date()),
        min_value=df['review_date'].min().date(),
        max_value=df['review_date'].max().date()
    )
    
    # Apply filters
    filtered_df = df[
        (df['category'].isin(selected_categories)) &
        (df['fraud_probability'] >= risk_threshold if risk_threshold > 0 else True)
    ]
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['review_date'].dt.date >= date_range[0]) &
            (filtered_df['review_date'].dt.date <= date_range[1])
        ]
    
    # Page content
    if page == "📊 Dashboard Overview":
        st.header("Dashboard Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_reviews = len(filtered_df)
            st.metric("Total Reviews", f"{total_reviews:,}")
        
        with col2:
            high_risk = (filtered_df['fraud_probability'] >= 0.7).sum()
            fraud_rate = high_risk / total_reviews if total_reviews > 0 else 0
            st.metric("High Risk Reviews", f"{high_risk:,}", f"{fraud_rate:.1%}")
        
        with col3:
            total_gmv = filtered_df['gmv'].sum()
            gmv_at_risk = filtered_df[filtered_df['fraud_probability'] >= 0.7]['gmv'].sum()
            st.metric("GMV at Risk", f"${gmv_at_risk/1e6:.2f}M")
        
        with col4:
            avg_risk = filtered_df['fraud_probability'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.2f}")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_risk_distribution(filtered_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_category_comparison(filtered_df), use_container_width=True)
        
        # Temporal trends
        st.plotly_chart(create_temporal_trends(filtered_df), use_container_width=True)
        
        # Category breakdown table
        st.subheader("Category Breakdown")
        cat_analysis = get_category_analysis(filtered_df)
        st.dataframe(
            cat_analysis.style.format({
                'avg_fraud_prob': '{:.2%}',
                'synthetic_rate': '{:.1%}',
                'total_gmv': '${:,.0f}',
                'gmv_at_risk': '${:,.0f}'
            }),
            use_container_width=True
        )
    
    elif page == "🔥 Fraud Heatmap":
        st.header("Fraud Heatmap Analysis")
        
        # Heatmap
        st.plotly_chart(create_fraud_heatmap(filtered_df), use_container_width=True)
        
        # Detailed category analysis
        st.subheader("Category Risk Analysis")
        
        cat_analysis = get_category_analysis(filtered_df)
        
        for _, row in cat_analysis.head(5).iterrows():
            risk_class = 'risk-high' if row['synthetic_rate'] > 0.3 else \
                        'risk-medium' if row['synthetic_rate'] > 0.15 else 'risk-low'
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h4>{row['category']}</h4>
                <p>Synthetic Rate: <span class="{risk_class}">{row['synthetic_rate']:.1%}</span> | 
                   GMV at Risk: ${row['gmv_at_risk']/1e6:.2f}M | 
                   Reviews: {row['review_count']:,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "🧬 Linguistic Fingerprints":
        st.header("Linguistic Fingerprint Analysis")
        
        st.markdown("""
        This visualization uses t-SNE dimensionality reduction to map reviews based on their 
        linguistic characteristics. Reviews with similar linguistic patterns cluster together.
        
        **Key Insights:**
        - Red clusters indicate high fraud probability
        - Green clusters indicate low fraud probability
        - Mixed clusters suggest sophisticated fraud attempts
        """)
        
        # Sample size selector
        sample_size = st.slider("Sample Size", 100, 1000, 500, 100)
        
        # t-SNE plot
        st.plotly_chart(create_tsne_visualization(filtered_df, sample_size), 
                       use_container_width=True)
        
        # Feature correlation
        st.subheader("Linguistic Feature Correlations")
        
        feature_cols = ['word_count', 'punctuation_ratio', 'uppercase_ratio',
                       'unique_word_ratio', 'flesch_reading_ease', 'fraud_probability']
        
        corr_matrix = filtered_df[feature_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Feature Correlation Matrix'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "⚡ Real-time Scoring":
        st.header("Real-time Risk Scoring")
        
        st.markdown("""
        Enter a review text to get an instant fraud risk assessment.
        The model analyzes linguistic patterns, writing style, and content characteristics.
        """)
        
        # Input
        review_text = st.text_area(
            "Review Text",
            height=150,
            placeholder="Enter review text here..."
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🔍 Analyze Review", type="primary"):
                if review_text:
                    # Calculate risk score
                    risk_score = calculate_risk_score(review_text)
                    risk_level, risk_color = get_risk_level(risk_score)
                    
                    # Display result
                    st.markdown(f"""
                    <div style="background-color: {risk_color}20; padding: 20px; 
                                border-radius: 10px; border-left: 5px solid {risk_color};
                                text-align: center;">
                        <h2 style="color: {risk_color}; margin: 0;">{risk_level} Risk</h2>
                        <h1 style="color: {risk_color}; margin: 10px 0;">{risk_score:.1%}</h1>
                        <p>Fraud Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk factors
                    st.subheader("Risk Factors")
                    
                    factors = []
                    if len(review_text) < 30:
                        factors.append("⚠️ Very short review")
                    if review_text.count('!') > 3:
                        factors.append("⚠️ Excessive exclamation marks")
                    if sum(1 for c in review_text if c.isupper()) / max(len(review_text), 1) > 0.3:
                        factors.append("⚠️ High capitalization ratio")
                    
                    words = review_text.lower().split()
                    if len(words) > 5 and len(set(words)) / len(words) < 0.5:
                        factors.append("⚠️ Repetitive content detected")
                    
                    if factors:
                        for factor in factors:
                            st.markdown(factor)
                    else:
                        st.markdown("✅ No major risk factors detected")
                else:
                    st.warning("Please enter review text to analyze.")
        
        with col2:
            # Recent high-risk reviews
            st.subheader("Recent High-Risk Reviews")
            
            high_risk_reviews = filtered_df[filtered_df['fraud_probability'] >= 0.7].head(5)
            
            for _, review in high_risk_reviews.iterrows():
                with st.expander(f"🚨 {review['category']} - Risk: {review['fraud_probability']:.1%}"):
                    st.write(f"**Review:** {review['review_text'][:200]}...")
                    st.write(f"**Rating:** {'⭐' * int(review['rating'])}")
                    st.write(f"**Date:** {review['review_date']}")
    
    elif page == "📈 Business Impact":
        st.header("Business Impact Analysis")
        
        # Calculate impact metrics
        total_reviews = len(filtered_df)
        high_risk_reviews = (filtered_df['fraud_probability'] >= 0.7).sum()
        fraud_rate = high_risk_reviews / total_reviews if total_reviews > 0 else 0
        
        total_gmv = filtered_df['gmv'].sum()
        gmv_at_risk = filtered_df[filtered_df['fraud_probability'] >= 0.7]['gmv'].sum()
        
        # Impact cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Reviews at Risk</h3>
                <h2 style="color: #e74c3c;">{high_risk_reviews:,}</h2>
                <p>{fraud_rate:.1%} of total reviews</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>GMV at Risk</h3>
                <h2 style="color: #e74c3c;">${gmv_at_risk/1e6:.2f}M</h2>
                <p>{gmv_at_risk/total_gmv:.1%} of total GMV</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            estimated_impact = gmv_at_risk * 0.15  # 15% conversion impact
            st.markdown(f"""
            <div class="metric-card">
                <h3>Est. Revenue Impact</h3>
                <h2 style="color: #f39c12;">${estimated_impact/1e6:.2f}M</h2>
                <p>Assuming 15% conversion impact</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Category impact table
        st.subheader("Impact by Category")
        
        cat_analysis = get_category_analysis(filtered_df)
        
        impact_data = []
        for _, row in cat_analysis.iterrows():
            impact_data.append({
                'Category': row['category'],
                'Synthetic Rate': f"{row['synthetic_rate']:.1%}",
                'Reviews at Risk': f"{int(row['high_risk_count']):,}",
                'GMV at Risk': f"${row['gmv_at_risk']/1e6:.2f}M",
                'Risk Level': '🔴 Critical' if row['synthetic_rate'] > 0.3 else
                             '🟠 High' if row['synthetic_rate'] > 0.15 else
                             '🟡 Medium' if row['synthetic_rate'] > 0.05 else '🟢 Low'
            })
        
        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True)
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        
        top_risky = cat_analysis.head(3)
        
        for _, cat in top_risky.iterrows():
            st.info(f"""
            **{cat['category']}** - Synthetic Rate: {cat['synthetic_rate']:.1%}
            
            - Implement additional verification for reviews in this category
            - Review {int(cat['high_risk_count']):,} flagged reviews manually
            - Potential savings: ${cat['gmv_at_risk'] * 0.15 / 1e6:.2f}M in protected revenue
            """)
    
    elif page == "🔍 Deep Dive Analysis":
        st.header("Deep Dive Analysis")
        
        # Suspicious accounts
        st.subheader("Suspicious Account Analysis")
        
        reviewer_stats = filtered_df.groupby('reviewer_id').agg({
            'fraud_probability': ['mean', 'count'],
            'review_length': 'mean',
            'rating': 'mean'
        }).reset_index()
        
        reviewer_stats.columns = ['reviewer_id', 'avg_risk', 'review_count', 'avg_length', 'avg_rating']
        reviewer_stats = reviewer_stats[reviewer_stats['review_count'] >= 3]
        reviewer_stats['suspicion_score'] = (
            reviewer_stats['avg_risk'] * 0.5 +
            (reviewer_stats['review_count'] / reviewer_stats['review_count'].max()) * 0.3
        )
        reviewer_stats = reviewer_stats.sort_values('suspicion_score', ascending=False)
        
        st.dataframe(
            reviewer_stats.head(20).style.format({
                'avg_risk': '{:.2%}',
                'suspicion_score': '{:.2f}',
                'avg_rating': '{:.1f}'
            }),
            use_container_width=True
        )
        
        # Rating vs Fraud correlation
        st.subheader("Rating vs Fraud Correlation")
        
        rating_fraud = filtered_df.groupby('rating').agg({
            'fraud_probability': 'mean',
            'review_id': 'count'
        }).reset_index()
        rating_fraud.columns = ['rating', 'avg_fraud_prob', 'count']
        
        fig = px.bar(
            rating_fraud,
            x='rating',
            y='avg_fraud_prob',
            text=[f'{v:.1%}' for v in rating_fraud['avg_fraud_prob']],
            labels={'avg_fraud_prob': 'Average Fraud Probability', 'rating': 'Star Rating'},
            title='Fraud Probability by Rating',
            color='avg_fraud_prob',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export High-Risk Reviews"):
                high_risk = filtered_df[filtered_df['fraud_probability'] >= 0.7]
                csv = high_risk.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"high_risk_reviews_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Full Analysis"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()

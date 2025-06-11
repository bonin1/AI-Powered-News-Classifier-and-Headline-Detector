"""
Analytics Dashboard for News Classification System
Provides comprehensive analytics and insights.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import sqlite3
import os
from typing import Dict, List, Optional
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from textstat import flesch_reading_ease, flesch_kincaid_grade
import re

class NewsAnalytics:
    """Analytics engine for news data."""
    
    def __init__(self, db_path: str = "data/news_monitor.db"):
        self.db_path = db_path
        
    def get_article_data(self, days: int = 30) -> pd.DataFrame:
        """Load article data from database."""
        if not os.path.exists(self.db_path):
            return pd.DataFrame()
        
        query = """
            SELECT title, description, published, source, category,
                   created_at
            FROM articles 
            WHERE published > datetime('now', '-{} days')
            ORDER BY published DESC
        """.format(days)
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    # Handle datetime parsing with mixed formats
                    try:
                        df['published'] = pd.to_datetime(df['published'], format='mixed')
                    except:
                        df['published'] = pd.to_datetime(df['published'], errors='coerce')
                    
                    try:
                        df['created_at'] = pd.to_datetime(df['created_at'], format='mixed')
                    except:
                        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                    
                    # Extract time components safely - use numpy array conversion to avoid warnings
                    df['hour'] = df['published'].dt.hour
                    df['day_of_week'] = df['published'].dt.day_name()
                    df['date'] = df['published'].dt.date
                return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze sentiment trends over time (placeholder for integration)."""
        # This would integrate with the sentiment analyzer
        return {
            'positive_trend': np.random.rand(len(df)),
            'negative_trend': np.random.rand(len(df)),
            'neutral_trend': np.random.rand(len(df))
        }
    
    def extract_keywords(self, texts: List[str], top_n: int = 20) -> Dict[str, int]:
        """Extract top keywords from text data."""
        # Simple keyword extraction (can be enhanced with NLP)
        all_text = ' '.join(texts).lower()
        
        # Remove common stop words and clean text
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                     'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 
                     'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'must',
                     'this', 'that', 'these', 'those', 'said', 'says', 'new', 'news'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
        words = [word for word in words if word not in stop_words]
        
        return dict(Counter(words).most_common(top_n))
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability scores for text."""
        try:
            return {
                'flesch_reading_ease': flesch_reading_ease(text),
                'flesch_kincaid_grade': flesch_kincaid_grade(text)
            }
        except:
            return {'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0}

def create_analytics_dashboard():
    """Create the main analytics dashboard."""
    
    st.title("📊 News Analytics Dashboard")
    st.markdown("Comprehensive insights into news classification and trends")
    
    # Initialize analytics
    analytics = NewsAnalytics()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    days = st.sidebar.slider("Analysis Period (Days)", 1, 90, 30)
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", False)
    if auto_refresh:
        st.sidebar.info("Dashboard will refresh every 30 seconds")
        # Auto-refresh every 30 seconds
        import time
        time.sleep(30)
        st.rerun()
    
    # Load data
    with st.spinner("Loading analytics data..."):
        df = analytics.get_article_data(days)
    
    if df.empty:
        st.warning("No data available. Make sure the RSS monitor is collecting articles.")
        st.info("To start collecting data, run: `python -c 'from utils.rss_monitor import RSSMonitor; m=RSSMonitor(); m.check_all_feeds()'`")
        return
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", len(df))
    
    with col2:
        unique_sources = df['source'].nunique()
        st.metric("News Sources", unique_sources)
    
    with col3:
        categories = df['category'].nunique()
        st.metric("Categories", categories)
    
    with col4:
        recent_24h = len(df[df['published'] > datetime.now() - timedelta(days=1)])
        st.metric("Last 24 Hours", recent_24h)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Articles Over Time")
        daily_counts = df.groupby('date').size().reset_index(name='count')
        # Convert to datetime properly to avoid warnings
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        fig = px.line(daily_counts, x='date', y='count', 
                     title="Daily Article Volume",
                     labels={'count': 'Number of Articles', 'date': 'Date'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Category Distribution")
        category_counts = df['category'].value_counts()
        
        fig = px.pie(values=category_counts.values, names=category_counts.index,
                    title="Articles by Category")
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕒 Publishing Patterns")
        hourly_counts = df.groupby(['hour', 'category']).size().reset_index(name='count')
        
        fig = px.bar(hourly_counts, x='hour', y='count', color='category',
                    title="Articles by Hour of Day",
                    labels={'hour': 'Hour of Day', 'count': 'Number of Articles'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📰 Top News Sources")
        source_counts = df['source'].value_counts().head(10)
        
        fig = px.bar(x=source_counts.values, y=source_counts.index, 
                    orientation='h', title="Most Active News Sources",
                    labels={'x': 'Number of Articles', 'y': 'News Source'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Word Cloud Section
    st.subheader("☁️ Trending Keywords")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate word cloud
        all_titles = ' '.join(df['title'].fillna('').astype(str))
        keywords = analytics.extract_keywords(df['title'].tolist())
        
        if keywords:
            # Create word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                colormap='viridis').generate_from_frequencies(keywords)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No keywords to display")
    
    with col2:
        st.subheader("Top Keywords")
        if keywords:
            keywords_df = pd.DataFrame(list(keywords.items()), 
                                     columns=['Keyword', 'Frequency'])
            st.dataframe(keywords_df.head(15), hide_index=True)
    
    # Detailed Analysis Section
    st.subheader("🔍 Detailed Analysis")
    
    tabs = st.tabs(["Category Trends", "Source Analysis", "Content Quality", "Recent Articles"])
    
    with tabs[0]:
        # Category trends over time
        daily_category = df.groupby(['date', 'category']).size().reset_index(name='count')
        daily_category['date'] = pd.to_datetime(daily_category['date'])
        
        fig = px.line(daily_category, x='date', y='count', color='category',
                     title="Category Trends Over Time",
                     labels={'count': 'Number of Articles', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Category performance metrics
        st.subheader("Category Metrics")
        category_metrics = df.groupby('category').agg({
            'title': 'count',
            'published': ['min', 'max']
        }).round(2)
        category_metrics.columns = ['Total Articles', 'First Article', 'Latest Article']
        st.dataframe(category_metrics)
    
    with tabs[1]:
        # Source reliability and activity
        source_analysis = df.groupby('source').agg({
            'title': 'count',
            'published': 'max',
            'category': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
        }).round(2)
        source_analysis.columns = ['Total Articles', 'Latest Article', 'Primary Category']
        source_analysis = source_analysis.sort_values('Total Articles', ascending=False)
        
        st.subheader("Source Performance")
        st.dataframe(source_analysis)
        
        # Source diversity chart
        source_category = df.groupby(['source', 'category']).size().reset_index(name='count')
        fig = px.treemap(source_category, path=['source', 'category'], values='count',
                        title="Source-Category Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # Content quality analysis
        st.subheader("Content Quality Metrics")
        
        # Calculate average title length by category
        df['title_length'] = df['title'].str.len()
        title_stats = df.groupby('category')['title_length'].agg(['mean', 'std']).round(1)
        title_stats.columns = ['Avg Title Length', 'Title Length Std']
        
        st.dataframe(title_stats)
        
        # Title length distribution
        fig = px.box(df, x='category', y='title_length',
                    title="Title Length Distribution by Category")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample readability analysis (if description available)
        if 'description' in df.columns and not df['description'].isna().all():
            st.subheader("Readability Analysis (Sample)")
            sample_text = df['description'].dropna().iloc[0] if not df['description'].dropna().empty else "No description available"
            readability = analytics.calculate_readability(sample_text)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Flesch Reading Ease", f"{readability['flesch_reading_ease']:.1f}")
            with col2:
                st.metric("Flesch-Kincaid Grade", f"{readability['flesch_kincaid_grade']:.1f}")
    
    with tabs[3]:
        # Recent articles table
        st.subheader("Recent Articles")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_category = st.selectbox("Filter by Category", 
                                           ['All'] + list(df['category'].unique()))
        with col2:
            selected_source = st.selectbox("Filter by Source",
                                         ['All'] + list(df['source'].unique()))
        
        # Apply filters
        filtered_df = df.copy()
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        if selected_source != 'All':
            filtered_df = filtered_df[filtered_df['source'] == selected_source]
        
        # Display recent articles
        recent_articles = filtered_df.head(20)[['title', 'source', 'category', 'published']]
        recent_articles['published'] = recent_articles['published'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(recent_articles, hide_index=True)
    
    # Export Options
    st.subheader("📤 Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "news_analytics.csv", "text/csv")
    
    with col2:
        if st.button("Export Keywords"):
            keywords_df = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Frequency'])
            csv = keywords_df.to_csv(index=False)
            st.download_button("Download Keywords CSV", csv, "keywords.csv", "text/csv")
    
    with col3:
        if st.button("Export Summary Report"):
            # Create summary report
            summary = f"""
# News Analytics Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Metrics
- Total Articles: {len(df)}
- Date Range: {df['published'].min().strftime('%Y-%m-%d')} to {df['published'].max().strftime('%Y-%m-%d')}
- News Sources: {df['source'].nunique()}
- Categories: {df['category'].nunique()}

## Category Distribution
{category_counts.to_string()}

## Top Sources
{source_counts.head(10).to_string()}

## Top Keywords
{pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Frequency']).head(10).to_string(index=False)}
            """
            st.download_button("Download Report", summary, "analytics_report.md", "text/markdown")

# Main execution
if __name__ == "__main__":
    create_analytics_dashboard()

"""
Enhanced Streamlit Web Application for AI-Powered News System
Includes RSS monitoring, analytics, and bias detection.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from news_processor import NewsProcessor
from utils.rss_monitor import RSSMonitor, RSSFeed
from models.bias_detector import BiasDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI News Intelligence System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .category-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin: 0.25rem;
    }
    .sports { background-color: #28a745; }
    .business { background-color: #007bff; }
    .technology { background-color: #6f42c1; }
    .politics { background-color: #dc3545; }
    .general { background-color: #6c757d; }
    
    .bias-low { color: #28a745; font-weight: bold; }
    .bias-medium { color: #ffc107; font-weight: bold; }
    .bias-high { color: #dc3545; font-weight: bold; }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_processor():
    """Load the news processor with caching."""
    try:
        return NewsProcessor(enable_sentiment=True, enable_export=True)
    except Exception as e:
        st.error(f"Error loading news processor: {e}")
        return None

@st.cache_resource
def load_bias_detector():
    """Load the bias detector with caching."""
    try:
        return BiasDetector()
    except Exception as e:
        st.error(f"Error loading bias detector: {e}")
        return None

@st.cache_resource
def load_rss_monitor():
    """Load the RSS monitor with caching."""
    try:
        return RSSMonitor()
    except Exception as e:
        st.error(f"Error loading RSS monitor: {e}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🔬 AI News Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced News Classification, Headline Generation & Analytics</div>', unsafe_allow_html=True)
    
    # Initialize components
    processor = load_processor()
    bias_detector = load_bias_detector()
    rss_monitor = load_rss_monitor()
    
    if not processor:
        st.error("Failed to load news processor. Please check your installation.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Choose a feature:",
        [
            "🏠 Home", 
            "📰 Process Article", 
            "📊 Analytics Dashboard",
            "📡 RSS Monitor", 
            "🎯 Bias Analysis",
            "⚙️ Settings"
        ]
    )
      # Home Page
    if page == "🏠 Home":
        show_home_page(rss_monitor, processor)
    
    # Process Article Page
    elif page == "📰 Process Article":
        show_process_page(processor, bias_detector)
    
    # Analytics Dashboard
    elif page == "📊 Analytics Dashboard":
        show_analytics_page()
    
    # RSS Monitor Page
    elif page == "📡 RSS Monitor":
        show_rss_page(rss_monitor)
    
    # Bias Analysis Page
    elif page == "🎯 Bias Analysis":
        show_bias_page(bias_detector)
    
    # Settings Page
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page(rss_monitor, processor):
    """Show the home page with system overview."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚀 Welcome to AI News Intelligence")
        
        st.markdown("""
        This advanced system provides comprehensive news analysis with cutting-edge AI:
        
        ### 🔥 Key Features
        - **Smart Classification**: Automatically categorize news into Sports, Business, Technology, and Politics
        - **Headline Generation**: Create engaging headlines using state-of-the-art AI models
        - **Sentiment Analysis**: Understand the emotional tone of articles
        - **Bias Detection**: Identify potential bias and reliability issues
        - **RSS Monitoring**: Real-time news collection from multiple sources
        - **Analytics Dashboard**: Deep insights and trend analysis
        
        ### 🎯 How It Works
        1. **Input**: Provide a URL or paste article text
        2. **AI Analysis**: Our models classify, analyze sentiment, and detect bias
        3. **Smart Headlines**: Generate multiple engaging headline options
        4. **Export Results**: Download your analysis in multiple formats
        """)
        
        # Quick demo section
        st.subheader("🎬 Quick Demo")
        demo_text = st.text_area(
            "Try it now! Paste a news article excerpt:",
            placeholder="Enter news text here for instant analysis...",
            height=100
        )
        
        if st.button("🔍 Analyze Demo Text") and demo_text:
            with st.spinner("Analyzing..."):
                result = processor.process_text(demo_text)
                
                col_demo1, col_demo2 = st.columns(2)
                with col_demo1:
                    st.success(f"**Category**: {result['classification']['category']}")
                    st.info(f"**Sentiment**: {result['sentiment']['sentiment']}")
                
                with col_demo2:
                    st.write("**Generated Headlines**:")
                    for i, headline in enumerate(result['headlines'][:2], 1):
                        st.write(f"{i}. {headline}")
    
    with col2:
        st.subheader("📈 System Status")
        
        # System metrics
        if rss_monitor:
            stats = rss_monitor.get_stats()
            
            # Create metric cards
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats['total_articles']}</h3>
                <p>Total Articles Collected</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats['active_feeds']}</h3>
                <p>Active RSS Feeds</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats['recent_articles_24h']}</h3>
                <p>Articles (Last 24h)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Category distribution
            if stats['category_distribution']:
                st.subheader("📊 Category Distribution")
                categories = list(stats['category_distribution'].keys())
                values = list(stats['category_distribution'].values())
                
                fig = px.pie(values=values, names=categories, 
                           title="Article Categories")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
          # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Refresh RSS Feeds"):
            if rss_monitor:
                with st.spinner("Checking RSS feeds..."):
                    count = rss_monitor.check_all_feeds()
                    st.success(f"Found {count} articles from RSS feeds!")
            else:
                st.error("RSS monitor not available")
        
        if st.button("📊 View Analytics"):
            st.rerun()

def show_process_page(processor, bias_detector):
    """Show the article processing page."""
    
    st.header("📰 Process News Article")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["URL", "Text Input", "File Upload"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if input_method == "URL":
            url = st.text_input(
                "🔗 Enter article URL:",
                placeholder="https://example.com/news-article"
            )
            process_button = st.button("🚀 Process URL", type="primary")
            
            if process_button and url:
                process_article_url(url, processor, bias_detector)
        
        elif input_method == "Text Input":
            text = st.text_area(
                "📝 Paste article text:",
                placeholder="Enter the news article content here...",
                height=200
            )
            process_button = st.button("🚀 Process Text", type="primary")
            
            if process_button and text:
                process_article_text(text, processor, bias_detector)
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "📁 Upload text file:",
                type=['txt', 'md']
            )
            
            if uploaded_file:
                text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", text, height=200, disabled=True)
                
                if st.button("🚀 Process File", type="primary"):
                    process_article_text(text, processor, bias_detector)
    
    with col2:
        st.subheader("💡 Tips")
        st.markdown("""
        **For best results:**
        - Use complete article URLs
        - Include full article text
        - Ensure text is in English
        - Articles should be news-related
        
        **Analysis includes:**
        - 🏷️ Category classification
        - 😊 Sentiment analysis
        - 🎯 Bias detection
        - ✨ Headline generation
        - 📤 Export options
        """)
        
        # Recent processed articles
        if hasattr(st.session_state, 'recent_articles'):
            st.subheader("📋 Recent Articles")
            for article in st.session_state.recent_articles[-3:]:
                st.text(f"• {article[:50]}...")

def process_article_url(url, processor, bias_detector):
    """Process article from URL."""
    try:
        with st.spinner("Extracting and analyzing article..."):
            result = processor.process_url(url)
            display_results(result, processor, bias_detector, source_url=url)
            
            # Store in session state
            if 'recent_articles' not in st.session_state:
                st.session_state.recent_articles = []
            st.session_state.recent_articles.append(result.get('title', url))
            
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        st.info("Please check the URL and try again.")

def process_article_text(text, processor, bias_detector):
    """Process article from text input."""
    try:
        with st.spinner("Analyzing article..."):
            result = processor.process_text(text)
            display_results(result, processor, bias_detector)
            
            # Store in session state
            if 'recent_articles' not in st.session_state:
                st.session_state.recent_articles = []
            st.session_state.recent_articles.append(text[:100] + "...")
            
    except Exception as e:
        st.error(f"Error processing text: {e}")

def display_results(result, processor, bias_detector=None, source_url=None):
    """Display processing results with enhanced formatting."""
    
    if not result:
        st.error("No results to display")
        return
    
    # Main results section
    st.success("✅ Analysis Complete!")
    
    # Title and source
    if result.get('title'):
        st.markdown(f"### 📰 {result['title']}")
    
    if source_url:
        st.markdown(f"**Source**: [{source_url}]({source_url})")
    
    # Analysis results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Classification
        category = result['classification']['category']
        confidence = result['classification']['confidence']
        
        st.markdown(f"""
        <div class="result-box">
            <h4>🏷️ Classification</h4>
            <span class="category-badge {category.lower()}">{category.upper()}</span>
            <br><br>
            <strong>Confidence:</strong> {confidence:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Sentiment Analysis
        sentiment = result.get('sentiment', {})
        if sentiment:
            sentiment_score = sentiment.get('confidence', 0)
            sentiment_label = sentiment.get('sentiment', 'Unknown')
            
            color = "#28a745" if sentiment_label == "POSITIVE" else "#dc3545" if sentiment_label == "NEGATIVE" else "#6c757d"
            
            st.markdown(f"""
            <div class="result-box">
                <h4>😊 Sentiment</h4>
                <span style="color: {color}; font-weight: bold; font-size: 1.2em;">{sentiment_label}</span>
                <br>
                <strong>Confidence:</strong> {sentiment_score:.1%}
                <br>
                <strong>Emotions:</strong>
            """, unsafe_allow_html=True)            # Top emotions
            emotions = sentiment.get('emotions', {})
            if emotions and isinstance(emotions, dict):
                # Handle the correct emotion structure
                all_emotions = emotions.get('all_emotions', [])
                if all_emotions:
                    # Display top 3 emotions
                    for emotion_data in all_emotions[:3]:
                        emotion_name = emotion_data.get('emotion', 'unknown')
                        emotion_score = emotion_data.get('confidence', 0)
                        st.text(f"• {emotion_name}: {emotion_score:.1%}")
                else:
                    # Fallback: display primary emotion
                    primary_emotion = emotions.get('emotion', 'neutral')
                    emotion_confidence = emotions.get('confidence', 0)
                    st.text(f"• {primary_emotion}: {emotion_confidence:.1%}")
            else:
                st.text("• No emotion data available")
            
            st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        # Bias Analysis (re-enabled after fixing emotion display issue)
        if bias_detector and result.get('content'):
            try:
                with st.spinner("Analyzing bias..."):
                    bias_result = bias_detector.comprehensive_bias_analysis(
                        result.get('title', ''),
                        result.get('content', ''),
                        source_url or 'Unknown'
                    )
                    
                    bias_level = bias_result['bias_level']
                    bias_score = bias_result['overall_bias_score']
                    
                    bias_class = f"bias-{bias_level.lower()}"
                    
                    st.markdown(f"""
                    <div class="result-box">
                        <h4>🎯 Bias Analysis</h4>
                        <span class="{bias_class}">{bias_level} Bias</span>
                        <br>
                        <strong>Score:</strong> {bias_score:.1f}/10
                        <br>
                        <strong>Political:</strong> {bias_result['political_bias']['dominant_bias']}
                        <br>
                        <strong>Source:</strong> {bias_result['source_analysis']['reliability']}
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Bias analysis error: {str(e)}")
                st.markdown("""
                <div class="result-box">
                    <h4>🎯 Bias Analysis</h4>
                    <p>Bias analysis temporarily unavailable</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box">
                <h4>🎯 Bias Analysis</h4>
                <p>Bias analysis not available</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Generated Headlines
    st.subheader("✨ Generated Headlines")
    headlines = result.get('headlines', [])
    
    if headlines:
        for i, headline in enumerate(headlines, 1):
            col_h1, col_h2 = st.columns([6, 1])
            with col_h1:
                st.write(f"**{i}.** {headline}")
            with col_h2:
                if st.button("📋", key=f"copy_{i}", help="Copy to clipboard"):
                    st.session_state[f'copied_headline_{i}'] = headline
                    st.success("Copied!")
    else:
        st.info("No headlines generated")
    
    # Export Options
    st.subheader("📤 Export Results")
    export_col1, export_col2, export_col3, export_col4 = st.columns(4)
    
    with export_col1:
        if st.button("📄 Export JSON"):
            json_data = processor.export_results([result], 'json')
            st.download_button(
                "Download JSON",
                json_data,
                f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    
    with export_col2:
        if st.button("📊 Export CSV"):
            csv_data = processor.export_results([result], 'csv')
            st.download_button(
                "Download CSV",
                csv_data,
                f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    with export_col3:
        if st.button("📋 Export TXT"):
            txt_data = processor.export_results([result], 'txt')
            st.download_button(
                "Download TXT",
                txt_data,
                f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )
    
    with export_col4:
        if st.button("🌐 Export HTML"):
            html_data = processor.export_results([result], 'html')
            st.download_button(
                "Download HTML",
                html_data,
                f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html"
            )

def show_analytics_page():
    """Show the analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    # Import and run analytics dashboard
    try:
        from analytics_dashboard import create_analytics_dashboard
        create_analytics_dashboard()
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        st.info("Make sure the RSS monitor has collected some articles first.")

def show_rss_page(rss_monitor):
    """Show RSS monitoring page."""
    st.header("📡 RSS Feed Monitor")
    
    if not rss_monitor:
        st.error("RSS Monitor not available")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Monitor Status")
        
        # Monitor controls
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            if st.button("▶️ Start Monitoring"):
                rss_monitor.start_monitoring()
                st.success("RSS monitoring started!")
        
        with control_col2:
            if st.button("⏹️ Stop Monitoring"):
                rss_monitor.stop_monitoring()
                st.info("RSS monitoring stopped")
        
        with control_col3:
            if st.button("🔄 Check Now"):
                with st.spinner("Checking feeds..."):
                    count = rss_monitor.check_all_feeds()
                    st.success(f"Collected {count} articles!")
          # Recent articles
        st.subheader("📰 Recent Articles")
        recent_articles = rss_monitor.get_recent_articles(hours=24, limit=20)
        
        if recent_articles:
            df = pd.DataFrame(recent_articles)
            # Handle datetime parsing with mixed formats
            try:
                df['published'] = pd.to_datetime(df['published'], format='mixed').dt.strftime('%Y-%m-%d %H:%M')
            except:
                # Fallback: try to parse without format specification
                df['published'] = pd.to_datetime(df['published'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            
            # Display with pagination
            page_size = 10
            total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
            page = st.selectbox("Page", range(1, total_pages + 1)) - 1
            
            start_idx = page * page_size
            end_idx = start_idx + page_size
            
            for _, article in df.iloc[start_idx:end_idx].iterrows():
                with st.expander(f"[{article['category']}] {article['title'][:80]}..."):
                    st.write(f"**Source**: {article['source']}")
                    st.write(f"**Published**: {article['published']}")
                    st.write(f"**Description**: {article['description'][:200]}...")
                    if st.button(f"🔗 Read Full Article", key=f"read_{article['link']}"):
                        st.markdown(f"[Open Article]({article['link']})")
        else:
            st.info("No recent articles found. Start the monitor to collect articles.")
    
    with col2:
        st.subheader("⚙️ Feed Management")
        
        # Add new feed
        with st.expander("➕ Add New Feed"):
            feed_name = st.text_input("Feed Name")
            feed_url = st.text_input("RSS URL")
            feed_category = st.selectbox("Category", 
                                       ["general", "sports", "business", "technology", "politics"])
            
            if st.button("Add Feed"):
                if feed_name and feed_url:
                    new_feed = RSSFeed(feed_name, feed_url, feed_category)
                    if rss_monitor.add_feed(new_feed):
                        st.success(f"Added feed: {feed_name}")
                    else:
                        st.error("Feed already exists")
                else:
                    st.error("Please fill all fields")
        
        # Feed statistics
        stats = rss_monitor.get_stats()
        st.metric("Active Feeds", stats['active_feeds'])
        st.metric("Total Articles", stats['total_articles'])
        st.metric("Recent (24h)", stats['recent_articles_24h'])

def show_bias_page(bias_detector):
    """Show bias analysis page."""
    st.header("🎯 Bias Analysis")
    
    if not bias_detector:
        st.error("Bias detector not available")
        return
    
    st.markdown("""
    Analyze news articles for potential bias, including:
    - **Political bias** (liberal, conservative, neutral)
    - **Emotional bias** (sensational language, loaded words)
    - **Source reliability** (based on known source ratings)
    - **Language complexity** (readability and accessibility)
    """)
    
    # Input for bias analysis
    analysis_text = st.text_area(
        "Enter article text for bias analysis:",
        height=200,
        placeholder="Paste article content here..."
    )
    
    source_name = st.text_input("Source name (optional):", placeholder="e.g., CNN, Fox News, BBC")
    
    if st.button("🔍 Analyze Bias") and analysis_text:
        with st.spinner("Analyzing bias..."):
            # Extract title (first sentence) and content
            sentences = analysis_text.split('.')
            title = sentences[0] if sentences else analysis_text[:100]
            
            bias_result = bias_detector.comprehensive_bias_analysis(
                title, analysis_text, source_name
            )
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 Overall Assessment")
                bias_level = bias_result['bias_level']
                bias_score = bias_result['overall_bias_score']
                
                if bias_level == "Low":
                    st.success(f"✅ {bias_level} Bias (Score: {bias_score:.1f})")
                elif bias_level == "Medium":
                    st.warning(f"⚠️ {bias_level} Bias (Score: {bias_score:.1f})")
                else:
                    st.error(f"🚨 {bias_level} Bias (Score: {bias_score:.1f})")
            
            with col2:
                st.subheader("🏛️ Political Bias")
                political = bias_result['political_bias']
                st.write(f"**Dominant**: {political['dominant_bias']}")
                st.write(f"**Confidence**: {political['confidence']:.2f}")
                
                # Political scores
                for bias_type, score in political['scores'].items():
                    if score > 0:
                        st.write(f"• {bias_type}: {score:.2f}%")
            
            with col3:
                st.subheader("😤 Emotional Bias")
                emotional = bias_result['emotional_bias']
                st.write(f"**Total Score**: {emotional['total_emotional_bias']:.2f}")
                st.write(f"**Loaded Language**: {emotional['loaded_language']:.2f}%")
                
                if emotional['is_sensational']:
                    st.warning("⚠️ Sensational language detected")
            
            # Detailed analysis
            st.subheader("📋 Detailed Analysis")
            
            # Recommendations
            if bias_result['recommendations']:
                st.warning("**Recommendations**:")
                for rec in bias_result['recommendations']:
                    st.write(f"• {rec}")
            else:
                st.success("✅ No major bias concerns detected")
            
            # Headline analysis
            headline_analysis = bias_result['headline_analysis']
            if headline_analysis['bias_indicators']:
                st.subheader("📰 Headline Issues")
                for indicator in headline_analysis['bias_indicators']:
                    st.write(f"• {indicator}")
            
            # Language analysis
            lang_analysis = bias_result['language_analysis']
            st.subheader("📖 Language Analysis")
            col_lang1, col_lang2 = st.columns(2)
            
            with col_lang1:
                st.metric("Reading Ease", f"{lang_analysis['reading_ease']:.1f}")
                st.metric("Avg Sentence Length", f"{lang_analysis['avg_sentence_length']:.1f}")
            
            with col_lang2:
                st.metric("Complexity Ratio", f"{lang_analysis['complexity_ratio']:.1f}%")
                st.metric("Accessibility", lang_analysis['accessibility'])

def show_settings_page():
    """Show settings and configuration page."""
    st.header("⚙️ Settings & Configuration")
    
    # Model settings
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text("Classification Model")
        st.code("textattack/distilbert-base-uncased-ag-news")
        
        st.text("Headline Model") 
        st.code("google/pegasus-xsum")
    
    with col2:
        st.text("Sentiment Model")
        st.code("cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        st.text("Emotion Model")
        st.code("j-hartmann/emotion-english-distilroberta-base")
    
    # System settings
    st.subheader("🔧 System Settings")
    
    # RSS Monitor settings
    st.text("RSS Monitor Interval")
    interval = st.slider("Check interval (minutes)", 1, 60, 5)
    
    # Cache settings
    st.text("Model Caching")
    st.checkbox("Enable model caching", True, disabled=True, help="Improves performance")
    
    # Export settings
    st.subheader("📤 Export Settings")
    st.checkbox("Include metadata in exports", True)
    st.checkbox("Include bias analysis in exports", True)
    st.checkbox("Include sentiment analysis in exports", True)
    
    # About section
    st.subheader("ℹ️ About")
    st.markdown("""
    **AI News Intelligence System v2.0**
    
    Enhanced with:
    - Real-time RSS monitoring
    - Advanced bias detection
    - Comprehensive analytics
    - Multi-format export
    
    Built with Python, Streamlit, and state-of-the-art AI models.
    """)

if __name__ == "__main__":
    main()

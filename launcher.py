"""
Multi-App Launcher for AI News Intelligence System
Choose between different interfaces and features.
"""

import os
import sys
import subprocess
import streamlit as st
from datetime import datetime
import webbrowser
import time

def check_dependencies():
    """Check if all required packages are installed."""
    try:
        import transformers
        import torch
        import newspaper
        import feedparser
        import textstat
        import wordcloud
        import plotly
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies automatically")
            print("Please run: pip install -r requirements.txt")
            return False

def create_launcher_app():
    """Create the Streamlit launcher interface."""
    
    st.set_page_config(
        page_title="AI News Intelligence Launcher",
        page_icon="🚀",
        layout="wide"
    )
    
    # Custom CSS for launcher
    st.markdown("""
    <style>
        .launcher-header {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }
        .app-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .app-card:hover {
            transform: translateY(-5px);
        }
        .feature-list {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .status-good { color: #28a745; font-weight: bold; }
        .status-warning { color: #ffc107; font-weight: bold; }
        .status-error { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="launcher-header">🚀 AI News Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Choose your interface and start analyzing news with AI</p>', unsafe_allow_html=True)
    
    # System status check
    st.subheader("🔧 System Status")
    
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        # Check Python version
        python_version = sys.version.split()[0]
        if sys.version_info >= (3, 8):
            st.markdown(f'<p class="status-good">✅ Python {python_version}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="status-error">❌ Python {python_version} (Need 3.8+)</p>', unsafe_allow_html=True)
    
    with col_status2:
        # Check dependencies
        try:
            import transformers
            st.markdown('<p class="status-good">✅ Dependencies OK</p>', unsafe_allow_html=True)
        except ImportError:
            st.markdown('<p class="status-error">❌ Missing Dependencies</p>', unsafe_allow_html=True)
            if st.button("🔧 Install Dependencies"):
                with st.spinner("Installing packages..."):
                    if check_dependencies():
                        st.success("Dependencies installed!")
                        st.experimental_rerun()
                    else:
                        st.error("Installation failed")
    
    with col_status3:
        # Check data directory
        data_dir = "data"
        if os.path.exists(data_dir):
            st.markdown('<p class="status-good">✅ Data Directory</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-warning">⚠️ No Data Directory</p>', unsafe_allow_html=True)
            if st.button("📁 Create Data Directory"):
                os.makedirs(data_dir, exist_ok=True)
                st.success("Data directory created!")
    
    st.divider()
    
    # Application options
    st.subheader("🎯 Choose Your Interface")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="app-card">
            <h3>🔬 Enhanced Intelligence App</h3>
            <p>Full-featured web interface with all advanced capabilities</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-list">
            <h4>✨ Features Include:</h4>
            <ul>
                <li>🏷️ AI News Classification</li>
                <li>✨ Smart Headline Generation</li>
                <li>😊 Advanced Sentiment Analysis</li>
                <li>🎯 Bias Detection System</li>
                <li>📡 Real-time RSS Monitoring</li>
                <li>📊 Comprehensive Analytics</li>
                <li>📤 Multi-format Export</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Enhanced App", type="primary", key="enhanced"):
            st.info("Starting Enhanced Intelligence App...")
            st.markdown("**Starting enhanced_app.py...**")
            st.code("streamlit run enhanced_app.py")
            
    with col2:
        st.markdown("""
        <div class="app-card">
            <h3>📰 Classic News Processor</h3>
            <p>Simple, fast interface for basic news processing</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-list">
            <h4>📋 Features Include:</h4>
            <ul>
                <li>🏷️ News Classification</li>
                <li>✨ Headline Generation</li>
                <li>😊 Sentiment Analysis</li>
                <li>📤 Basic Export Options</li>
                <li>🌐 URL Processing</li>
                <li>📝 Text Input</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📰 Launch Classic App", type="secondary", key="classic"):
            st.info("Starting Classic News Processor...")
            st.markdown("**Starting app.py...**")
            st.code("streamlit run app.py")
    
    st.divider()
    
    # Additional tools
    st.subheader("🛠️ Additional Tools")
    
    tool_col1, tool_col2, tool_col3 = st.columns(3)
    
    with tool_col1:
        st.markdown("### 📊 Analytics Dashboard")
        st.markdown("View comprehensive analytics and insights")
        if st.button("📊 Open Analytics"):
            st.info("Opening Analytics Dashboard...")
            st.code("streamlit run analytics_dashboard.py")
    
    with tool_col2:
        st.markdown("### 💻 Command Line Interface")
        st.markdown("Use the CLI for batch processing")
        if st.button("💻 CLI Instructions"):
            st.code("python cli.py --help")
            st.info("Use CLI for automated processing and scripting")
    
    with tool_col3:
        st.markdown("### 📓 Jupyter Demo")
        st.markdown("Interactive notebook demonstration")
        if st.button("📓 Open Notebook"):
            st.info("Opening Jupyter Notebook...")
            st.code("jupyter notebook demo.ipynb")
    
    st.divider()
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🔄 Update RSS Feeds"):
            try:
                from utils.rss_monitor import RSSMonitor
                monitor = RSSMonitor()
                with st.spinner("Checking RSS feeds..."):
                    count = monitor.check_all_feeds()
                    st.success(f"Collected {count} articles!")
            except Exception as e:
                st.error(f"RSS update failed: {e}")
    
    with action_col2:
        if st.button("🧪 Run Tests"):
            st.info("Running system tests...")
            st.code("python test.py")
    
    with action_col3:
        if st.button("📖 View Documentation"):
            if os.path.exists("README.md"):
                st.info("Opening README.md...")
                with open("README.md", "r", encoding="utf-8") as f:
                    readme_content = f.read()
                with st.expander("📖 Documentation", expanded=True):
                    st.markdown(readme_content)
            else:
                st.error("README.md not found")
    
    with action_col4:
        if st.button("🔧 System Info"):
            st.info("System Information:")
            st.text(f"Python: {sys.version}")
            st.text(f"Platform: {sys.platform}")
            st.text(f"Working Directory: {os.getcwd()}")
            st.text(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🔬 AI News Intelligence System v2.0 | Built with Python & Streamlit</p>
        <p>For help and documentation, visit the project repository</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function for command line usage."""
    print("🚀 AI News Intelligence System Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("\nAvailable interfaces:")
    print("1. 🔬 Enhanced Intelligence App (Recommended)")
    print("2. 📰 Classic News Processor")
    print("3. 📊 Analytics Dashboard")
    print("4. 💻 Command Line Interface")
    print("5. 🌐 Web Launcher Interface")
    print("6. 📓 Jupyter Notebook Demo")
    
    try:
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == "1":
            print("🚀 Starting Enhanced Intelligence App...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "enhanced_app.py"])
        elif choice == "2":
            print("📰 Starting Classic News Processor...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        elif choice == "3":
            print("📊 Starting Analytics Dashboard...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "analytics_dashboard.py"])
        elif choice == "4":
            print("💻 Starting CLI interface...")
            subprocess.run([sys.executable, "cli.py", "--help"])
        elif choice == "5":
            print("🌐 Starting Web Launcher...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "launcher.py"])
        elif choice == "6":
            print("📓 Starting Jupyter Notebook...")
            subprocess.run(["jupyter", "notebook", "demo.ipynb"])
        else:
            print("❌ Invalid choice. Please select 1-6.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # If running as Streamlit app
    if "streamlit" in sys.modules:
        create_launcher_app()
    else:
        # If running from command line
        main()

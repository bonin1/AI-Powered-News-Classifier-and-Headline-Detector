# 🤖 AI-Powered News Classifier & Headline Detector

> **Enhanced Edition** - An intelligent, comprehensive news analysis system with advanced AI capabilities

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29+-red.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/transformers-4.0+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An state-of-the-art system that provides comprehensive news analysis including classification, sentiment analysis, bias detection, headline generation, and automated RSS monitoring with a modern web interface.

## 🚀 Key Features

### 🔍 **Core Analysis**
- **📰 Article Classification**: Advanced categorization into 8+ categories (Sports, Politics, Technology, Business, Science, etc.)
- **😊 Sentiment Analysis**: Multi-dimensional emotion detection with confidence scores
- **🎯 Bias Detection**: Political, emotional, and source bias analysis
- **✨ Headline Generation**: AI-powered engaging headline creation
- **📊 Content Analytics**: Readability scores, keyword extraction, and quality metrics

### 🌐 **Data Sources**
- **🔗 URL Processing**: Extract and analyze articles from any news website
- **📝 Text Input**: Direct text analysis and processing
- **📡 RSS Monitoring**: Automated collection from 20+ major news sources
- **📁 File Upload**: Support for text and markdown files

### 📈 **Analytics & Insights**
- **📊 Real-time Dashboard**: Comprehensive analytics with interactive visualizations
- **📅 Trend Analysis**: Historical patterns and content distribution
- **🏷️ Category Insights**: Performance metrics across news categories
- **📰 Source Analysis**: Reliability ratings and bias assessments

### 🖥️ **Interface Options**
- **🌟 Enhanced Web App**: Modern multi-page Streamlit interface
- **📱 Responsive Design**: Mobile-friendly layouts and components
- **🎨 Professional UI**: Clean, intuitive design with real-time updates
- **⚡ Fast Processing**: Optimized performance with caching

## 🛠️ Technology Stack

### **AI/ML Models**
- **Classification**: `textattack/distilbert-base-uncased-ag-news` (94.5% accuracy)
- **Sentiment**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Emotions**: `j-hartmann/emotion-english-distilroberta-base`
- **Headlines**: `google/pegasus-xsum` (SOTA summarization)

### **Core Technologies**
- **Backend**: Python 3.8+, Transformers, NumPy, Pandas
- **Frontend**: Streamlit, Plotly, Matplotlib
- **Database**: SQLite for RSS data storage
- **Processing**: newspaper3k, textstat, NLTK

## 📦 Installation & Setup

### **🚀 Quick Start (Recommended)**
```bash
git clone https://github.com/bonin1/AI-Powered-News-Classifier-and-Headline-Detector.git
cd AI-Powered-News-Classifier-and-Headline-Detector
python start.py
```

### **🔧 Manual Installation**
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py

# Start the enhanced application
streamlit run enhanced_app.py
```

### **📋 System Requirements**
- Python 3.8 or higher
- 4GB+ RAM (for ML models)
- Internet connection (for model downloads)
- Modern web browser

## 🎮 Usage Guide

### **🌟 Enhanced Web Interface**
```bash
streamlit run enhanced_app.py
```

**Features include:**
- 🏠 **Home Dashboard**: System overview and quick stats
- 📰 **Process Article**: URL/text analysis with full AI pipeline
- 📊 **Analytics**: Interactive charts and insights
- 📡 **RSS Monitor**: Real-time feed monitoring and collection
- 🎯 **Bias Analysis**: Comprehensive bias detection and reporting
- ⚙️ **Settings**: System configuration and preferences

### **💻 Command Line Interface**
```python
from news_processor import NewsProcessor
from models.bias_detector import BiasDetector

# Initialize system
processor = NewsProcessor(enable_sentiment=True, enable_export=True)
bias_detector = BiasDetector()

# Process article from URL
result = processor.process_url("https://example-news-site.com/article")

# Get comprehensive results
print(f"Category: {result['classification']['category']}")
print(f"Confidence: {result['classification']['confidence']:.1%}")
print(f"Sentiment: {result['sentiment']['sentiment']}")
print(f"Headlines: {result['headlines']}")

# Bias analysis
bias_result = bias_detector.comprehensive_bias_analysis(
    result['title'], result['content'], "Source Name"
)
print(f"Bias Level: {bias_result['bias_level']}")
```

### **📡 RSS Monitoring**
```python
from utils.rss_monitor import RSSMonitor

monitor = RSSMonitor()
# Collect from 20+ news sources
articles_collected = monitor.check_all_feeds()
print(f"Collected {articles_collected} new articles")

# Get recent articles
recent = monitor.get_recent_articles(hours=24)
```

## 📁 Enhanced Project Structure

```
📦 AI-Powered-News-Classifier-and-Headline-Detector/
├── 🌟 enhanced_app.py          # Enhanced multi-page Streamlit app
├── 📰 news_processor.py        # Core processing engine
├── 📊 analytics_dashboard.py   # Advanced analytics dashboard
├── 🏗️ models/
│   ├── 🔍 classifier.py        # News classification
│   ├── ✨ headline_generator.py # AI headline generation
│   ├── 😊 sentiment_analyzer.py # Emotion & sentiment analysis
│   └── 🎯 bias_detector.py     # Bias detection system
├── 🛠️ utils/
│   ├── 🔗 article_extractor.py # Web article extraction
│   ├── 📡 rss_monitor.py       # RSS feed monitoring
│   ├── 📝 text_preprocessor.py # Text processing utilities
│   └── 📤 export_manager.py    # Multi-format export
├── 💾 data/
│   └── 📊 news_monitor.db      # RSS articles database
├── 🔧 setup.py                 # Automated setup script
├── 📋 requirements.txt         # Dependencies
└── 📖 README.md               # This file
```

## 🎯 API Reference

### **NewsProcessor**
```python
class NewsProcessor:
    def process_url(url: str) -> Dict
    def process_text(text: str) -> Dict
    def export_results(results: List, format: str) -> str
```

### **BiasDetector**
```python
class BiasDetector:
    def comprehensive_bias_analysis(title: str, content: str, source: str) -> Dict
    def detect_political_bias(text: str) -> Dict
    def detect_emotional_bias(text: str) -> Dict
```

### **RSSMonitor**
```python
class RSSMonitor:
    def check_all_feeds() -> int
    def get_recent_articles(hours: int) -> List[Dict]
    def start_monitoring() -> None
```

## 📊 Performance Metrics

### **Analysis Accuracy**
- **Classification**: 94.5% accuracy across categories
- **Sentiment Analysis**: 85%+ confidence on clear sentiment
- **Bias Detection**: Multi-dimensional scoring (0-10 scale)
- **Processing Speed**: ~30 seconds per article (full pipeline)

### **Data Collection**
- **RSS Sources**: 20+ major news outlets
- **Categories**: 8 news categories supported
- **Collection Rate**: 150+ articles per monitoring cycle
- **Database**: SQLite with 270+ stored articles

## 🔧 Configuration

### **RSS Sources**
The system monitors major news sources including:
- **General**: BBC, Reuters, Associated Press, NPR, CNN
- **Technology**: TechCrunch, Ars Technica, The Verge, Wired
- **Business**: Financial Times, MarketWatch
- **Science**: Science Daily, NASA News
- **Sports**: ESPN, Sky Sports

### **Export Formats**
- **JSON**: Structured data for APIs
- **CSV**: Spreadsheet-compatible format
- **HTML**: Web-ready reports
- **TXT**: Plain text summaries

## 🚀 Recent Enhancements (v2.0)

### **✅ Major Fixes**
- Fixed float/list comparison errors in bias analysis
- Resolved newspaper3k article extraction issues
- Updated deprecated Streamlit and pandas functions
- Enhanced RSS feed reliability and error handling

### **🌟 New Features**
- Multi-page enhanced web interface
- Real-time RSS monitoring with SQLite storage
- Comprehensive bias detection system
- Advanced analytics dashboard with visualizations
- Multi-format export functionality
- Professional UI with responsive design

### **📈 Performance Improvements**
- Increased article collection from 121 to 270+ articles
- Added 20+ reliable RSS sources
- Improved error handling and user feedback
- Optimized model loading with caching

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Development Setup**
```bash
# Clone and setup development environment
git clone https://github.com/bonin1/AI-Powered-News-Classifier-and-Headline-Detector.git
cd AI-Powered-News-Classifier-and-Headline-Detector
pip install -r requirements.txt
python setup.py

# Run tests
python comprehensive_test.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for providing state-of-the-art NLP models
- **Streamlit** for the excellent web framework
- **The open-source community** for the amazing libraries and tools

---

**🎉 Ready to analyze news with AI? Get started with `python start.py`!**

### Command Line

```python
from news_processor import NewsProcessor

processor = NewsProcessor()

# Process from URL
result = processor.process_url("https://example-news-site.com/article")

# Process text directly
result = processor.process_text("Your news article text here...")

print(f"Category: {result['category']}")
print(f"Original Headline: {result['original_headline']}")
print(f"Generated Headline: {result['generated_headline']}")
```

## Project Structure

```
├── app.py                 # Streamlit web interface
├── news_processor.py      # Main processing logic
├── models/               # Model utilities
│   ├── classifier.py     # Text classification
│   └── headline_generator.py  # Headline generation
├── utils/               # Utility functions
│   ├── article_extractor.py  # Web scraping
│   └── text_preprocessor.py  # Text preprocessing
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## API Reference

### NewsProcessor

Main class for processing news articles.

#### Methods

- `process_url(url: str)` - Process article from URL
- `process_text(text: str, title: str = "")` - Process raw text
- `classify_text(text: str)` - Get article category
- `generate_headline(text: str)` - Generate new headline

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

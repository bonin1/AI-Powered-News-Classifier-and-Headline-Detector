"""
Enhanced Setup script for AI News Intelligence System
Handles installation, configuration, and initial setup.
"""

import subprocess
import sys
import os
import platform
import json
from datetime import datetime
import urllib.request
import zipfile

class NewsSystemSetup:
    """Enhanced setup class for the News Intelligence System."""
    
    def __init__(self):
        self.system_info = {
            'python_version': sys.version,
            'platform': platform.system(),
            'architecture': platform.architecture()[0],
            'setup_date': datetime.now().isoformat()
        }
        
    def check_python_version(self):
        """Check if Python version is compatible."""
        print("🐍 Checking Python version...")
        if sys.version_info < (3, 8):
            print(f"❌ Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is compatible")
        return True
    
    def create_directories(self):
        """Create necessary directories."""
        print("📁 Creating directories...")
        directories = ['data', 'exports', 'logs', 'cache']
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Created directory: {directory}")
            except Exception as e:
                print(f"❌ Error creating directory {directory}: {e}")
                return False
        return True
    
    def install_requirements(self):
        """Install required packages with progress tracking."""
        print("📦 Installing required packages...")
        try:
            # Upgrade pip first
            print("  ⬆️ Upgrading pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                                capture_output=True)
            
            # Install requirements
            print("  📋 Installing from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing requirements: {e}")
            print("💡 Try running: pip install -r requirements.txt manually")
            return False
    
    def download_nltk_data(self):
        """Download required NLTK data."""
        print("📚 Downloading NLTK data...")
        try:
            import nltk
            
            # Download required NLTK data
            nltk_data = ['punkt', 'stopwords', 'vader_lexicon', 'wordnet', 'averaged_perceptron_tagger']
            
            for data in nltk_data:
                print(f"  📥 Downloading {data}...")
                nltk.download(data, quiet=True)
            
            print("✅ NLTK data downloaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error downloading NLTK data: {e}")
            return False
    
    def test_models(self):
        """Test if AI models can be loaded."""
        print("🤖 Testing AI models...")
        
        try:
            # Test transformers installation
            print("  🧠 Testing Transformers...")
            from transformers import pipeline
            
            # Test classification model
            print("  🏷️ Testing Classification model...")
            classifier = pipeline("text-classification", 
                                 model="textattack/distilbert-base-uncased-ag-news")
            test_result = classifier("This is a test news article about technology.")
            print(f"    ✅ Classification: {test_result[0]['label']}")
            
            # Test headline generation model (lighter test)
            print("  ✨ Testing Headline model...")
            # Just verify the model exists without loading (to save time)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
            print("    ✅ Headline model accessible")
            
            print("✅ AI models tested successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error testing models: {e}")
            print("💡 Models will be downloaded on first use")
            return True  # Non-critical error
    
    def create_config_file(self):
        """Create configuration file."""
        print("⚙️ Creating configuration file...")
        
        config = {
            'system_info': self.system_info,
            'models': {
                'classification': 'textattack/distilbert-base-uncased-ag-news',
                'headline_generation': 'google/pegasus-xsum',
                'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'emotion': 'j-hartmann/emotion-english-distilroberta-base'
            },
            'rss_feeds': {
                'check_interval': 300,  # 5 minutes
                'max_articles': 1000
            },
            'export': {
                'formats': ['json', 'csv', 'html', 'txt'],
                'include_metadata': True
            },
            'cache': {
                'enabled': True,
                'ttl': 3600  # 1 hour
            }
        }
        
        try:
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ Configuration file created: config.json")
            return True
        except Exception as e:
            print(f"❌ Error creating config file: {e}")
            return False
    
    def setup_sample_data(self):
        """Set up sample data for testing."""
        print("📄 Setting up sample data...")
        
        sample_articles = [
            {
                'title': 'Tech Giants Report Strong Q4 Earnings',
                'content': 'Major technology companies have reported strong fourth-quarter earnings, driven by cloud computing and AI investments.',
                'category': 'business',
                'url': 'https://example.com/tech-earnings'
            },
            {
                'title': 'Championship Game Draws Record Viewership',
                'content': 'The championship game attracted record television viewership, with millions of fans tuning in worldwide.',
                'category': 'sports',
                'url': 'https://example.com/championship'
            },
            {
                'title': 'New AI Model Achieves Breakthrough in Language Understanding',
                'content': 'Researchers have developed a new artificial intelligence model that demonstrates unprecedented language comprehension abilities.',
                'category': 'technology',
                'url': 'https://example.com/ai-breakthrough'
            }
        ]
        
        try:
            with open('data/sample_articles.json', 'w') as f:
                json.dump(sample_articles, f, indent=2)
            print("✅ Sample data created: data/sample_articles.json")
            return True
        except Exception as e:
            print(f"❌ Error creating sample data: {e}")
            return False
    
    def run_quick_test(self):
        """Run a quick functionality test."""
        print("🧪 Running quick functionality test...")
        
        try:
            # Test basic imports
            from news_processor import NewsProcessor
            from utils.rss_monitor import RSSMonitor
            from models.bias_detector import BiasDetector
            
            print("  ✅ All modules imported successfully")
            
            # Test basic functionality
            processor = NewsProcessor()
            test_text = "This is a test article about technology and innovation."
            result = processor.process_text(test_text)
            
            if result and 'classification' in result:
                print(f"  ✅ Text processing works: {result['classification']['category']}")
            else:
                print("  ⚠️ Text processing returned unexpected result")
            
            print("✅ Quick test completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            return False
    
    def print_next_steps(self):
        """Print next steps for the user."""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE!")
        print("="*60)
        print()
        print("🚀 Ready to start! Choose your interface:")
        print()
        print("1. 🔬 Enhanced Intelligence App (Recommended):")
        print("   streamlit run enhanced_app.py")
        print()
        print("2. 📰 Classic Interface:")
        print("   streamlit run app.py")
        print()
        print("3. 🌐 Launcher Interface:")
        print("   streamlit run launcher.py")
        print()
        print("4. 💻 Command Line:")
        print("   python cli.py --help")
        print()
        print("5. 📓 Jupyter Notebook:")
        print("   jupyter notebook demo.ipynb")
        print()
        print("📚 For documentation, check README.md")
        print("🐛 For issues, check the logs in the 'logs' directory")
        print()
    
    def run_setup(self):
        """Run the complete setup process."""
        print("🚀 AI News Intelligence System Setup")
        print("="*50)
        print()
        
        setup_steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating directories", self.create_directories),
            ("Installing requirements", self.install_requirements),
            ("Downloading NLTK data", self.download_nltk_data),
            ("Testing AI models", self.test_models),
            ("Creating configuration", self.create_config_file),
            ("Setting up sample data", self.setup_sample_data),
            ("Running quick test", self.run_quick_test)
        ]
        
        failed_steps = []
        
        for step_name, step_function in setup_steps:
            print(f"\n📋 {step_name}...")
            try:
                if not step_function():
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                failed_steps.append(step_name)
        
        print(f"\n📊 Setup Summary:")
        print(f"  ✅ Completed: {len(setup_steps) - len(failed_steps)}/{len(setup_steps)} steps")
        
        if failed_steps:
            print(f"  ❌ Failed steps: {', '.join(failed_steps)}")
            print("  💡 You can retry failed steps manually")
        
        if len(failed_steps) <= 2:  # Allow some non-critical failures
            self.print_next_steps()
        else:
            print("  ⚠️ Too many setup failures. Please check the errors above.")

def main():
    """Main setup function."""
    setup = NewsSystemSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()

def test_models():
    """Test if models can be loaded."""
    print("Testing model loading...")
    try:
        from news_processor import NewsProcessor
        processor = NewsProcessor()
        
        # Test with sample text
        test_text = "Apple announced a new iPhone model with advanced AI capabilities."
        result = processor.process_text(test_text, "Apple News")
        
        print(f"✅ Models loaded and tested successfully!")
        print(f"   Test classification: {result['category']}")
        print(f"   Test headline: {result['generated_headline']}")
        
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        return False
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up AI-Powered News Classifier and Headline Detector")
    print("=" * 60)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("Setup failed at requirements installation.")
        sys.exit(1)
    
    print()
    
    # Step 2: Download NLTK data
    if not download_nltk_data():
        print("Setup failed at NLTK data download.")
        sys.exit(1)
    
    print()
    
    # Step 3: Test models (this will download them if needed)
    print("⚠️  Note: Model loading may take several minutes on first run...")
    if not test_models():
        print("Setup failed at model testing.")
        sys.exit(1)
    
    print()
    print("🎉 Setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Run the web app: streamlit run app.py")
    print("2. Or use the CLI: python cli.py --help")
    print("3. Check the README.md for more information")

if __name__ == "__main__":
    main()

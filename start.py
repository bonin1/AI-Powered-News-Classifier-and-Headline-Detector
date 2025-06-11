#!/usr/bin/env python3
"""
Quick Start Script for AI-Powered News Classifier and Headline Detector
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version {sys.version.split()[0]} is compatible")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'transformers', 'torch', 'streamlit', 'newspaper3k', 
        'pandas', 'numpy', 'nltk', 'requests', 'beautifulsoup4'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_quick_demo():
    """Run a quick demonstration."""
    print("\n🎯 Running quick demonstration...")
    try:
        subprocess.run([sys.executable, "demo.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        return False

def show_usage_options():
    """Show different ways to use the system."""
    print("\n🚀 How to use your AI News Classifier:")
    print("=" * 50)
    
    print("\n1. 🌐 Web Interface (Recommended):")
    print("   streamlit run app.py")
    print("   Then open: http://localhost:8501")
    
    print("\n2. 💻 Command Line:")
    print("   python cli.py --url \"https://example.com/news\"")
    print("   python cli.py --text \"Your article content...\"")
    print("   python cli.py --help")
    
    print("\n3. 📓 Jupyter Notebook:")
    print("   jupyter notebook demo.ipynb")
    
    print("\n4. 🐍 Python API:")
    print("   from news_processor import NewsProcessor")
    print("   processor = NewsProcessor()")
    print("   result = processor.process_text('Article content...')")
    
    print("\n📚 Documentation:")
    print("   • README.md - Full documentation")
    print("   • PROJECT_SUMMARY.md - Project overview")
    print("   • examples/ - Sample files")

def main():
    """Main startup function."""
    print("🚀 AI-Powered News Classifier and Headline Detector")
    print("   Quick Start Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Please run this script from the project root directory")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        print("\n🔧 Installing missing dependencies...")
        if not install_dependencies():
            return 1
    
    # Run quick demo
    if not run_quick_demo():
        print("⚠️  Demo failed, but you can still use the system")
    
    # Show usage options
    show_usage_options()
    
    print("\n🎉 Setup complete! Your AI News Classifier is ready to use!")
    
    # Ask user what they want to do
    print("\n❓ What would you like to do now?")
    print("   1. Start web interface")
    print("   2. Run command line help")
    print("   3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🌐 Starting web interface...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        elif choice == "2":
            print("\n💻 Command line help:")
            subprocess.run([sys.executable, "cli.py", "--help"])
        else:
            print("\n👋 Thank you for using AI News Classifier!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using AI News Classifier!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

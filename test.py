"""
Test script for AI-Powered News Classifier and Headline Detector
Run this script to test all components of the system.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import newspaper
        import streamlit
        import pandas
        import numpy
        import nltk
        import requests
        from bs4 import BeautifulSoup
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_models():
    """Test if models can be loaded and used."""
    print("\nTesting model loading...")
    try:
        from models.classifier import NewsClassifier
        from models.headline_generator import HeadlineGenerator
        
        # Test classifier
        print("  Loading classifier...")
        classifier = NewsClassifier()
        
        test_text = "Apple announced a new iPhone with advanced features."
        result = classifier.classify(test_text)
        print(f"  ✅ Classifier working - Category: {result['category']}")
        
        # Test headline generator
        print("  Loading headline generator...")
        generator = HeadlineGenerator()
        
        headline = generator.generate(test_text)
        print(f"  ✅ Headline generator working - Headline: {headline}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        traceback.print_exc()
        return False

def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    try:
        from utils.text_preprocessor import TextPreprocessor
        from utils.article_extractor import ArticleExtractor
        
        # Test preprocessor
        preprocessor = TextPreprocessor()
        test_text = "  This is a test article with extra   spaces.  "
        cleaned = preprocessor.preprocess(test_text)
        print(f"  ✅ Text preprocessor working")
        
        # Test extractor (basic validation only)
        extractor = ArticleExtractor()
        is_valid = extractor._is_valid_url("https://example.com")
        print(f"  ✅ Article extractor basic functions working")
        
        return True
    except Exception as e:
        print(f"❌ Utilities error: {e}")
        traceback.print_exc()
        return False

def test_main_processor():
    """Test the main news processor."""
    print("\nTesting main processor...")
    try:
        from news_processor import NewsProcessor
        
        processor = NewsProcessor()
        
        # Test text processing
        test_article = """
        Apple Inc. announced today that they are releasing a new iPhone model with advanced 
        artificial intelligence capabilities. The device features improved camera technology 
        and faster processing speeds.
        """
        
        result = processor.process_text(test_article, "Apple Announces New iPhone")
        
        print(f"  ✅ Text processing working")
        print(f"     Category: {result['category']}")
        print(f"     Confidence: {result['confidence']:.2%}")
        print(f"     Generated Headline: {result['generated_headline']}")
        
        return True
    except Exception as e:
        print(f"❌ Main processor error: {e}")
        traceback.print_exc()
        return False

def test_cli():
    """Test CLI functionality."""
    print("\nTesting CLI...")
    try:
        import subprocess
        import sys
        
        # Test CLI help
        result = subprocess.run([sys.executable, "cli.py", "--help"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("  ✅ CLI help working")
            return True
        else:
            print(f"  ❌ CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False

def test_streamlit_app():
    """Test if Streamlit app can be imported."""
    print("\nTesting Streamlit app...")
    try:
        # Just test if the app file can be imported without running it
        import app
        print("  ✅ Streamlit app imports successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit app error: {e}")
        return False

def run_sample_tests():
    """Run sample processing tests."""
    print("\nRunning sample tests...")
    try:
        from news_processor import NewsProcessor
        
        processor = NewsProcessor()
        
        # Test different categories
        test_cases = [
            ("Technology", "Google unveiled new AI features for search engines with machine learning."),
            ("Sports", "The Lakers beat the Warriors 110-105 in last night's basketball game."),
            ("Business", "Stock markets fell today due to inflation concerns and Federal Reserve policy."),
            ("World", "The United Nations called for international cooperation on climate change.")
        ]
        
        print("  Sample test results:")
        for expected_category, text in test_cases:
            result = processor.classify_text(text)
            print(f"    Expected: {expected_category}, Got: {result['category']} "
                  f"(Confidence: {result['confidence']:.2%})")
        
        return True
    except Exception as e:
        print(f"❌ Sample tests error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running AI News Classifier Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Utilities Test", test_utilities),
        ("Model Loading Test", test_models),
        ("Main Processor Test", test_main_processor),
        ("CLI Test", test_cli),
        ("Streamlit App Test", test_streamlit_app),
        ("Sample Processing Test", run_sample_tests)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🧪 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run the web app: streamlit run app.py")
        print("2. Try the CLI: python cli.py --help")
        print("3. Check out the demo notebook: jupyter notebook demo.ipynb")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

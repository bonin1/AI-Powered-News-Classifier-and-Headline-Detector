#!/usr/bin/env python3
"""
Simple demonstration of the AI News Classifier and Headline Detector
"""

def demo():
    print("🚀 AI-Powered News Classifier and Headline Detector - Demo")
    print("=" * 60)
    
    try:
        # Import the main processor
        from news_processor import NewsProcessor
        
        print("Loading AI models (this may take a moment)...")
        processor = NewsProcessor()
        print("✅ Models loaded successfully!\n")
        
        # Test cases representing different news categories
        test_articles = [
            {
                "title": "Technology News",
                "content": """
                Apple Inc. announced today the launch of their revolutionary new iPhone 15 Pro with 
                advanced artificial intelligence capabilities. The device features a state-of-the-art 
                A17 Pro chip, enhanced camera system with computational photography, and improved 
                battery life that lasts up to 29 hours of video playback. The company expects this 
                to be their most innovative product release this year, with pre-orders starting 
                next Friday and general availability beginning the following week.
                """
            },
            {
                "title": "Sports News", 
                "content": """
                The Los Angeles Lakers secured a thrilling 118-112 overtime victory against the 
                Golden State Warriors at Crypto.com Arena last night. LeBron James led all scorers 
                with 35 points, 8 rebounds, and 12 assists, while Anthony Davis contributed 28 points 
                and 15 rebounds. Stephen Curry scored 31 points for the Warriors but missed a crucial 
                three-pointer in the final seconds of overtime. This victory moves the Lakers to 
                third place in the Western Conference standings.
                """
            },
            {
                "title": "Business News",
                "content": """
                Global stock markets experienced significant volatility today as investors reacted to 
                the Federal Reserve's latest interest rate announcement. The Dow Jones Industrial 
                Average fell 2.1%, while the S&P 500 dropped 1.8% and the Nasdaq declined 2.4%. 
                Technology stocks were particularly affected, with major companies like Microsoft, 
                Apple, and Google seeing declines of 3-4%. Financial analysts attribute the sell-off 
                to concerns about inflation and tighter monetary policy.
                """
            }
        ]
        
        print("📊 Processing Sample Articles:")
        print("-" * 60)
        
        for i, article in enumerate(test_articles, 1):
            print(f"\n🔍 Test {i}: {article['title']}")
            print("-" * 30)
            
            # Process the article
            result = processor.process_text(article['content'], article['title'])
            
            # Display results
            print(f"📋 Classification:")
            print(f"   Category: {result['category']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            
            print(f"📰 Headlines:")
            print(f"   Original: {result['original_headline']}")
            print(f"   AI Generated: {result['generated_headline']}")
            
            print(f"📈 Statistics:")
            print(f"   Word Count: {result['word_count']}")
            
            # Show content preview
            preview = result['content_preview'][:150] + "..." if len(result['content_preview']) > 150 else result['content_preview']
            print(f"   Content Preview: {preview}")
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\n💡 Next steps:")
        print("   • Run the web app: streamlit run app.py")
        print("   • Try the CLI: python cli.py --help") 
        print("   • Explore the notebook: jupyter notebook demo.ipynb")
        print("   • Check the documentation: README.md")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   • Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   • Run the setup script: python setup.py")
        print("   • Check the test script: python test.py")

if __name__ == "__main__":
    demo()

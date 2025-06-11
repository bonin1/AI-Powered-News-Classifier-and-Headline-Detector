"""
Main News Processing Module
Handles the complete workflow of news classification and headline generation.
"""

import logging
from typing import Dict, List, Optional
from models.classifier import NewsClassifier
from models.headline_generator import HeadlineGenerator
from models.sentiment_analyzer import SentimentAnalyzer
from utils.article_extractor import ArticleExtractor
from utils.text_preprocessor import TextPreprocessor
from utils.export_manager import ResultExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsProcessor:
    """
    Main class for processing news articles.
    Combines classification and headline generation capabilities.
    """
    def __init__(self, enable_sentiment=True, enable_export=True):
        """Initialize the news processor with all required models."""
        logger.info("Initializing NewsProcessor...")
        
        self.classifier = NewsClassifier()
        self.headline_generator = HeadlineGenerator()
        self.article_extractor = ArticleExtractor()
        self.text_preprocessor = TextPreprocessor()
        
        # Optional components
        self.sentiment_analyzer = None
        self.export_manager = None
        
        if enable_sentiment:
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                logger.info("Sentiment analysis enabled")
            except Exception as e:
                logger.warning(f"Could not load sentiment analyzer: {e}")
        
        if enable_export:
            try:
                self.export_manager = ResultExporter()
                logger.info("Export functionality enabled")
            except Exception as e:
                logger.warning(f"Could not load export manager: {e}")
        
        logger.info("NewsProcessor initialized successfully!")
    
    def process_url(self, url: str) -> Dict[str, str]:
        """
        Process a news article from a URL.
        
        Args:
            url (str): The URL of the news article
            
        Returns:
            Dict[str, str]: Dictionary containing classification and headline results
        """
        try:
            logger.info(f"Processing URL: {url}")
            
            # Extract article content
            article_data = self.article_extractor.extract_from_url(url)
            
            if not article_data['content']:
                raise ValueError("Could not extract article content from URL")
            
            # Process the extracted content
            return self._process_article_data(article_data)
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return {
                'error': str(e),
                'category': 'unknown',
                'confidence': 0.0,
                'original_headline': '',
                'generated_headline': 'Error processing article',
                'content_preview': ''
            }
    
    def process_text(self, text: str, title: str = "") -> Dict[str, str]:
        """
        Process raw text content.
        
        Args:
            text (str): The article text content
            title (str): Optional original title
            
        Returns:
            Dict[str, str]: Dictionary containing classification and headline results        """
        try:
            logger.info("Processing raw text content")
            
            if not text.strip():
                raise ValueError("Text content cannot be empty")
            
            article_data = {
                'title': title,
                'content': text,
                'url': 'direct_input'
            }
            
            return self._process_article_data(article_data)
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return {
                'error': str(e),
                'classification': {
                    'category': 'unknown',
                    'confidence': 0.0,
                    'all_scores': []
                },
                'title': title,
                'headlines': ['Error processing article'],
                'content_preview': text[:200] + '...' if len(text) > 200 else text
            }
    
    def _process_article_data(self, article_data: Dict[str, str]) -> Dict[str, str]:
        """
        Internal method to process extracted article data.
        
        Args:
            article_data (Dict[str, str]): Dictionary with title, content, and url
            
        Returns:
            Dict[str, str]: Processing results
        """
        # Preprocess the content
        processed_content = self.text_preprocessor.preprocess(article_data['content'])
          # Classify the article
        classification_result = self.classifier.classify(processed_content)
        
        # Generate new headline
        generated_headline = self.headline_generator.generate(processed_content)
        
        # Analyze sentiment if available
        sentiment_result = None
        if self.sentiment_analyzer:
            sentiment_result = self.sentiment_analyzer.analyze_complete(processed_content)
          # Create content preview
        content_preview = processed_content[:300] + '...' if len(processed_content) > 300 else processed_content
        
        result = {
            'classification': {
                'category': classification_result['category'],
                'confidence': classification_result['confidence'],
                'all_scores': classification_result.get('all_scores', [])
            },
            'title': article_data['title'],
            'content': processed_content,
            'headlines': [generated_headline] if generated_headline else [],
            'content_preview': content_preview,
            'url': article_data.get('url', ''),
            'word_count': len(processed_content.split())
        }        # Add sentiment analysis if available
        if sentiment_result:
            result['sentiment'] = {
                'sentiment': sentiment_result['sentiment']['sentiment'],
                'confidence': sentiment_result['sentiment']['confidence'],
                'emotions': sentiment_result['emotion'],
                'all_scores': sentiment_result['sentiment'].get('all_scores', [])
            }
        
        return result
    
    def classify_text(self, text: str) -> Dict[str, any]:
        """
        Classify text content only.
        
        Args:
            text (str): Text to classify
            
        Returns:
            Dict[str, any]: Classification results
        """
        processed_text = self.text_preprocessor.preprocess(text)
        return self.classifier.classify(processed_text)
    
    def generate_headline(self, text: str) -> str:
        """
        Generate headline only.
        
        Args:
            text (str): Text to generate headline for
            
        Returns:
            str: Generated headline
        """
        processed_text = self.text_preprocessor.preprocess(text)
        return self.headline_generator.generate(processed_text)
    
    def export_result(self, result: Dict[str, any], filename: str, format_type: str = 'json') -> str:
        """
        Export a single result to file.
        
        Args:
            result (Dict[str, any]): Result to export
            filename (str): Output filename (without extension)
            format_type (str): Export format ('json', 'csv', 'html', 'txt')
            
        Returns:
            str: Path to exported file
        """
        if not self.export_manager:
            raise ValueError("Export functionality not enabled")
        
        return self.export_manager.export_single_result(result, filename, format_type)
    
    def export_batch_results(self, results: List[Dict[str, any]], filename: str, format_type: str = 'json') -> str:
        """
        Export multiple results to file.
        
        Args:
            results (List[Dict[str, any]]): Results to export
            filename (str): Output filename (without extension)
            format_type (str): Export format
            
        Returns:
            str: Path to exported file
        """
        if not self.export_manager:
            raise ValueError("Export functionality not enabled")
        
        return self.export_manager.export_batch_results(results, filename, format_type)
    def export_results(self, results: List[Dict[str, any]], format_type: str = 'json') -> str:
        """
        Export multiple results.
        
        Args:
            results (List[Dict[str, any]]): List of results to export
            format_type (str): Export format ('json', 'csv', 'html', 'txt')
            
        Returns:
            str: Exported data as string
        """
        if not self.export_manager:
            raise ValueError("Export manager not available. Initialize with enable_export=True")
        
        return self.export_manager.export_batch_results(results, "news_analysis", format_type)
    
    def process_and_export(self, text: str, title: str = "", filename: str = "analysis", 
                          format_type: str = 'json') -> tuple:
        """
        Process text and immediately export the result.
        
        Args:
            text (str): Text to process
            title (str): Optional title
            filename (str): Output filename
            format_type (str): Export format
            
        Returns:
            tuple: (result, export_path)
        """
        result = self.process_text(text, title)
        export_path = self.export_result(result, filename, format_type)
        return result, export_path


if __name__ == "__main__":
    # Example usage
    processor = NewsProcessor()
    
    # Test with sample text
    sample_text = """
    Apple Inc. announced today that they are releasing a new iPhone model with advanced AI capabilities.
    The device features improved camera technology and faster processing speeds. The company expects
    this to be their best-selling product this year, with pre-orders starting next week.
    """
    
    result = processor.process_text(sample_text, "Apple Announces New iPhone")
    print("Sample Processing Result:")
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Original Headline: {result['original_headline']}")
    print(f"Generated Headline: {result['generated_headline']}")

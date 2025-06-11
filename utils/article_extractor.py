"""
Article Extraction Module
Extracts article content from URLs using newspaper3k library.
"""

import logging
from typing import Dict, Optional
from newspaper import Article
import requests
from urllib.parse import urlparse
import time

logger = logging.getLogger(__name__)


class ArticleExtractor:
    """
    Extracts article content from web URLs.
    Uses newspaper3k for intelligent content extraction.
    """
    
    def __init__(self):
        """Initialize the article extractor."""
        self.timeout = 30  # Request timeout in seconds
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        
    def extract_from_url(self, url: str) -> Dict[str, str]:
        """
        Extract article content from a given URL.
        
        Args:
            url (str): URL of the article to extract
            
        Returns:
            Dict[str, str]: Dictionary containing title, content, authors, and other metadata
        """
        try:
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError("Invalid URL provided")
            
            logger.info(f"Extracting article from: {url}")
              # Create article object with configuration
            article = Article(url)
            
            # Set custom headers for better access
            article.config.browser_user_agent = self.user_agent
            article.config.request_timeout = self.timeout
            
            # Download and parse article
            article.download()
            article.parse()
            
            # Validate extracted content
            if not article.text or len(article.text.strip()) < 100:
                # Try alternative extraction method
                return self._extract_with_fallback(url)
            
            # Extract metadata
            result = {
                'title': article.title or 'No title found',
                'content': article.text,
                'authors': ', '.join(article.authors) if article.authors else 'Unknown',
                'publish_date': str(article.publish_date) if article.publish_date else 'Unknown',
                'url': url,
                'summary': article.summary or '',
                'top_image': article.top_image or '',
                'word_count': len(article.text.split()) if article.text else 0
            }
            
            logger.info(f"Successfully extracted article: {len(result['content'])} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting article from {url}: {str(e)}")
            
            # Try fallback method
            try:
                return self._extract_with_fallback(url)
            except Exception as fallback_error:
                logger.error(f"Fallback extraction also failed: {str(fallback_error)}")
                return {
                    'title': 'Extraction Error',
                    'content': '',
                    'authors': 'Unknown',
                    'publish_date': 'Unknown',
                    'url': url,
                    'summary': '',
                    'top_image': '',
                    'word_count': 0,
                    'error': str(e)
                }
    
    def _extract_with_fallback(self, url: str) -> Dict[str, str]:
        """
        Fallback extraction method using basic web scraping.
        
        Args:
            url (str): URL to extract content from
            
        Returns:
            Dict[str, str]: Extracted content
        """
        logger.info("Trying fallback extraction method...")
        
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        
        # Create new article object for parsing HTML content
        article = Article('')
        article.set_html(response.content)
        article.parse()
        
        if not article.text or len(article.text.strip()) < 50:
            # Last resort: basic HTML parsing
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find title
            title_tag = soup.find('title') or soup.find('h1')
            title = title_tag.get_text().strip() if title_tag else 'No title found'
            
            # Try to extract main content
            content_candidates = []
            for tag in ['article', 'main', 'div[class*="content"]', 'div[class*="article"]']:
                elements = soup.select(tag)
                for element in elements:
                    text = element.get_text().strip()
                    if len(text) > 200:  # Minimum content length
                        content_candidates.append(text)
            
            content = max(content_candidates, key=len) if content_candidates else "Could not extract content"
            
            return {
                'title': title,
                'content': content,
                'authors': 'Unknown',
                'publish_date': 'Unknown',
                'url': url,
                'summary': content[:200] + '...' if len(content) > 200 else content,
                'top_image': '',
                'word_count': len(content.split()) if content else 0
            }
        
        return {
            'title': article.title or 'No title found',
            'content': article.text,
            'authors': ', '.join(article.authors) if article.authors else 'Unknown',
            'publish_date': str(article.publish_date) if article.publish_date else 'Unknown',
            'url': url,
            'summary': article.summary or '',
            'top_image': article.top_image or '',
            'word_count': len(article.text.split()) if article.text else 0
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Validate if the given string is a valid URL.
        
        Args:
            url (str): URL string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def extract_multiple_urls(self, urls: list) -> Dict[str, Dict[str, str]]:
        """
        Extract articles from multiple URLs.
        
        Args:
            urls (list): List of URLs to extract
            
        Returns:
            Dict[str, Dict[str, str]]: Dictionary mapping URLs to extracted content
        """
        results = {}
        
        for url in urls:
            try:
                logger.info(f"Processing URL {len(results) + 1}/{len(urls)}: {url}")
                results[url] = self.extract_from_url(url)
                
                # Add small delay to be respectful to servers
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to extract from {url}: {str(e)}")
                results[url] = {
                    'title': 'Extraction Failed',
                    'content': '',
                    'error': str(e),
                    'url': url
                }
        
        return results
    
    def is_article_url(self, url: str) -> bool:
        """
        Check if URL likely contains an article.
        
        Args:
            url (str): URL to check
            
        Returns:
            bool: True if likely an article URL
        """
        if not self._is_valid_url(url):
            return False
        
        # Common patterns that suggest article URLs
        article_patterns = [
            '/article/', '/news/', '/story/', '/post/', '/blog/',
            '/articles/', '/stories/', '/posts/', '/blogs/',
            '/2024/', '/2025/'  # Year patterns
        ]
        
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in article_patterns)


if __name__ == "__main__":
    # Test the article extractor
    extractor = ArticleExtractor()
    
    # Test URLs (you can replace these with actual news URLs)
    test_urls = [
        "https://www.bbc.com/news",  # This will likely fail as it's not a specific article
        "https://example.com/fake-news-article"  # This will definitely fail
    ]
    
    print("Testing Article Extractor:")
    print("-" * 50)
    
    for url in test_urls:
        print(f"\nTesting URL: {url}")
        result = extractor.extract_from_url(url)
        
        print(f"Title: {result['title']}")
        print(f"Content length: {len(result['content'])} characters")
        print(f"Word count: {result['word_count']}")
        print(f"Authors: {result['authors']}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        
        if result['content']:
            print(f"Content preview: {result['content'][:200]}...")
    
    print(f"\nURL validation tests:")
    test_urls_validation = [
        "https://www.example.com/article/123",
        "http://news.site.com/story/456",
        "not-a-url",
        "https://malformed-url"
    ]
    
    for url in test_urls_validation:
        is_valid = extractor._is_valid_url(url)
        is_article = extractor.is_article_url(url)
        print(f"{url}: Valid={is_valid}, Article={is_article}")

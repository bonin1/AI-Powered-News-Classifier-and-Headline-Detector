"""
Text Preprocessing Module
Handles text cleaning and preprocessing for better model performance.
"""

import logging
import re
import nltk
from typing import Optional

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing utilities for news articles.
    Cleans and normalizes text for better model performance.
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def preprocess(self, text: str, preserve_structure: bool = True) -> str:
        """
        Main preprocessing function.
        
        Args:
            text (str): Raw text to preprocess
            preserve_structure (bool): Whether to preserve paragraph structure
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = self._clean_basic(text)
        
        # Remove unwanted elements
        text = self._remove_unwanted_elements(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text, preserve_structure)
        
        # Fix common encoding issues
        text = self._fix_encoding_issues(text)
        
        return text.strip()
    
    def _clean_basic(self, text: str) -> str:
        """
        Basic text cleaning operations.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation (3 or more consecutive punctuation marks)
        text = re.sub(r'[!?.]{3,}', '...', text)
        
        return text
    
    def _remove_unwanted_elements(self, text: str) -> str:
        """
        Remove unwanted elements from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove social media handles
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove common article footers/headers
        unwanted_patterns = [
            r'subscribe to our newsletter',
            r'follow us on',
            r'share this article',
            r'advertisement',
            r'read more:',
            r'related articles?:',
            r'source:',
            r'photo credit:',
            r'image credit:',
            r'reuters/',
            r'ap photo',
            r'getty images'
        ]
        
        for pattern in unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove standalone numbers (likely page numbers or timestamps)
        text = re.sub(r'\b\d{1,3}\b', '', text)
        
        # Remove common navigation text
        nav_patterns = [
            r'next page',
            r'previous page',
            r'page \d+ of \d+',
            r'continue reading',
            r'read full story'
        ]
        
        for pattern in nav_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text
    
    def truncate_to_token_limit(self, text: str, max_tokens: int = 500) -> str:
        """
        Truncate text to fit within token limits for models.
        
        Args:
            text (str): Input text
            max_tokens (int): Maximum number of tokens (approximate)
            
        Returns:
            str: Truncated text
        """
        if not text:
            return ""
        
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Try to truncate at sentence boundaries
        sentences = text.split('.')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence + ".") <= max_chars:
                truncated += sentence + "."
            else:
                break
        
        # If no complete sentences fit, just truncate by character count
        if not truncated:
            truncated = text[:max_chars]
        
        return truncated.strip()
    
    def _normalize_whitespace(self, text: str, preserve_structure: bool = True) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text (str): Input text
            preserve_structure (bool): Whether to preserve paragraph breaks
            
        Returns:
            str: Text with normalized whitespace
        """
        if preserve_structure:
            # Replace multiple newlines with double newline (paragraph break)
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Replace single newlines with space (unless it's a paragraph break)
            text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        else:
            # Replace all newlines with spaces
            text = re.sub(r'\n+', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        return text
    
    def _fix_encoding_issues(self, text: str) -> str:
        """
        Fix common encoding issues in text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with fixed encoding
        """
        # Common encoding fixes
        encoding_fixes = {
            ''': "'",  # Right single quotation mark
            ''': "'",  # Left single quotation mark
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '…': '...',  # Horizontal ellipsis
            '®': '(R)',  # Registered trademark
            '™': '(TM)',  # Trademark
            '©': '(C)',  # Copyright
            '°': ' degrees',  # Degree symbol
            '½': '1/2',  # Fraction one half
            '¼': '1/4',  # Fraction one quarter
            '¾': '3/4',  # Fraction three quarters
        }
        
        for old, new in encoding_fixes.items():
            text = text.replace(old, new)
        
        return text
    
    def clean_for_classification(self, text: str) -> str:
        """
        Preprocessing specifically optimized for classification.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text for classification
        """
        # Use standard preprocessing
        text = self.preprocess(text, preserve_structure=False)
        
        # Additional classification-specific cleaning
        # Remove quotes and parenthetical statements that might confuse classifier
        text = re.sub(r'"[^"]*"', '', text)
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Normalize contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def clean_for_headline_generation(self, text: str) -> str:
        """
        Preprocessing specifically optimized for headline generation.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text for headline generation
        """
        # Use standard preprocessing with structure preservation
        text = self.preprocess(text, preserve_structure=True)
        
        # For headline generation, focus on the first few sentences
        sentences = text.split('.')
        if len(sentences) > 5:
            # Take first 5 sentences for better headline generation
            text = '. '.join(sentences[:5]) + '.'
        
        # Remove bylines and datelines that might interfere
        byline_patterns = [
            r'^by .+? \|',
            r'^\w+, \w+ \d+',  # Date patterns like "NEW YORK, Jan 15"
            r'^\([^)]+\)',  # Location in parentheses
        ]
        
        for pattern in byline_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        return text
    
    def extract_sentences(self, text: str, max_sentences: int = 5) -> list:
        """
        Extract individual sentences from text.
        
        Args:
            text (str): Input text
            max_sentences (int): Maximum number of sentences to return
            
        Returns:
            list: List of sentences
        """
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            return sentences[:max_sentences] if max_sentences else sentences
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}, using simple split")
            sentences = text.split('.')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
            return sentences[:max_sentences] if max_sentences else sentences
    
    def get_word_count(self, text: str) -> int:
        """
        Get accurate word count.
        
        Args:
            text (str): Input text
            
        Returns:
            int: Word count
        """
        # Remove extra whitespace and count words
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        return len(cleaned_text.split()) if cleaned_text else 0


if __name__ == "__main__":
    # Test the text preprocessor
    preprocessor = TextPreprocessor()
    
    # Test sample with various issues
    test_text = """
    NEW YORK, Jan 15 (Reuters) - Apple Inc. announced today that they're releasing a new iPhone model 
    with advanced AI capabilities.    The device features improved camera technology and faster 
    processing speeds. 
    
    
    The company expects this to be their best-selling product this year, with pre-orders starting 
    next week. "We're excited about this launch," said CEO Tim Cook.
    
    Follow us on Twitter @TechNews    Share this article    Advertisement
    
    Related articles: Previous iPhone models, Tech industry news
    Photo credit: Getty Images
    """
    
    print("Testing Text Preprocessor:")
    print("-" * 50)
    print(f"Original text length: {len(test_text)} characters")
    print(f"Original text:\n{test_text}")
    print("\n" + "=" * 50)
    
    # Test standard preprocessing
    cleaned = preprocessor.preprocess(test_text)
    print(f"Preprocessed text length: {len(cleaned)} characters")
    print(f"Preprocessed text:\n{cleaned}")
    print("\n" + "=" * 50)
    
    # Test classification preprocessing
    classification_text = preprocessor.clean_for_classification(test_text)
    print(f"Classification text:\n{classification_text}")
    print("\n" + "=" * 50)
    
    # Test headline generation preprocessing
    headline_text = preprocessor.clean_for_headline_generation(test_text)
    print(f"Headline generation text:\n{headline_text}")
    print("\n" + "=" * 50)
    
    # Test sentence extraction
    sentences = preprocessor.extract_sentences(test_text, max_sentences=3)
    print("Extracted sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    print(f"\nWord count: {preprocessor.get_word_count(cleaned)}")

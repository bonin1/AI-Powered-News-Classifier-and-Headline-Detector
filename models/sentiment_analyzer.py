"""
Sentiment Analysis Module
Adds emotion and sentiment detection to news articles.
"""

import logging
from typing import Dict, List
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer for news articles.
    Provides emotion detection and sentiment scoring.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        logger.info("Loading sentiment analysis models...")
        
        try:
            # Load sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Load emotion detection pipeline
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            logger.info("Sentiment analysis models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading sentiment models: {str(e)}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of the given text.
        
        Args:
            text (str): Text to analyze
              Returns:
            Dict[str, any]: Sentiment analysis results
        """
        try:
            if not text.strip():
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'all_scores': []
                }
            
            # Truncate text if too long (RoBERTa has a 512 token limit)
            # Use character-based approximation: 1 token ≈ 4 characters
            max_chars = 1800  # Approximately 450 tokens, leaving room for special tokens
            if len(text) > max_chars:
                # Try to truncate at sentence boundary
                sentences = text[:max_chars].split('.')
                if len(sentences) > 1:
                    text = '.'.join(sentences[:-1]) + '.'
                else:
                    text = text[:max_chars]
            
            # Get sentiment predictions
            sentiment_results = self.sentiment_pipeline(text)
            
            # Process results
            if sentiment_results and len(sentiment_results) > 0:
                best_sentiment = max(sentiment_results[0], key=lambda x: x['score'])
                
                # Map labels to readable names
                sentiment_mapping = {
                    'LABEL_0': 'negative',
                    'LABEL_1': 'neutral', 
                    'LABEL_2': 'positive'
                }
                
                sentiment = sentiment_mapping.get(best_sentiment['label'], best_sentiment['label'].lower())
                confidence = best_sentiment['score']
                
                # Format all scores
                all_scores = []
                for result in sentiment_results[0]:
                    mapped_label = sentiment_mapping.get(result['label'], result['label'].lower())
                    all_scores.append({
                        'sentiment': mapped_label,
                        'confidence': result['score']
                    })
                
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'all_scores': all_scores
                }
            
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'all_scores': []
            }
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            return {
                'sentiment': 'error',
                'confidence': 0.0,
                'all_scores': [],
                'error': str(e)
            }
    
    def analyze_emotions(self, text: str) -> Dict[str, any]:
        """
        Analyze emotions in the given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, any]: Emotion analysis results        """
        try:
            if not text.strip():
                return {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'all_emotions': []
                }
            
            # Truncate text if too long (DistilRoBERTa has a 512 token limit)
            # Use character-based approximation: 1 token ≈ 4 characters
            max_chars = 1800  # Approximately 450 tokens, leaving room for special tokens
            if len(text) > max_chars:
                # Try to truncate at sentence boundary
                sentences = text[:max_chars].split('.')
                if len(sentences) > 1:
                    text = '.'.join(sentences[:-1]) + '.'
                else:
                    text = text[:max_chars]
            
            # Get emotion predictions
            emotion_results = self.emotion_pipeline(text)
            
            # Process results
            if emotion_results and len(emotion_results) > 0:
                best_emotion = max(emotion_results[0], key=lambda x: x['score'])
                
                emotion = best_emotion['label'].lower()
                confidence = best_emotion['score']
                
                # Format all emotions
                all_emotions = []
                for result in emotion_results[0]:
                    all_emotions.append({
                        'emotion': result['label'].lower(),
                        'confidence': result['score']
                    })
                
                # Sort by confidence
                all_emotions.sort(key=lambda x: x['confidence'], reverse=True)
                
                return {
                    'emotion': emotion,
                    'confidence': confidence,
                    'all_emotions': all_emotions
                }
            
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': []
            }
            
        except Exception as e:
            logger.error(f"Error during emotion analysis: {str(e)}")
            return {
                'emotion': 'error',
                'confidence': 0.0,
                'all_emotions': [],
                'error': str(e)
            }
    
    def analyze_complete(self, text: str) -> Dict[str, any]:
        """
        Perform complete sentiment and emotion analysis.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, any]: Complete analysis results
        """
        sentiment_result = self.analyze_sentiment(text)
        emotion_result = self.analyze_emotions(text)
        
        return {
            'sentiment': sentiment_result,
            'emotion': emotion_result,
            'summary': {
                'overall_sentiment': sentiment_result['sentiment'],
                'primary_emotion': emotion_result['emotion'],
                'sentiment_confidence': sentiment_result['confidence'],
                'emotion_confidence': emotion_result['confidence']
            }
        }
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded models.
        
        Returns:
            Dict[str, str]: Model information
        """
        return {
            'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'emotion_model': 'j-hartmann/emotion-english-distilroberta-base',
            'sentiment_categories': ['positive', 'negative', 'neutral'],
            'emotion_categories': ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'love'],
            'description': 'Advanced sentiment and emotion analysis for news articles'
        }


if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test samples
    test_texts = [
        "This is wonderful news! The company exceeded all expectations and delivered amazing results.",
        "Unfortunately, the accident caused significant damage and several people were injured.",
        "The meeting was scheduled for 3 PM to discuss the quarterly budget review.",
        "Breaking: Scientists make groundbreaking discovery that could change everything we know about space travel!"
    ]
    
    print("Testing Sentiment Analyzer:")
    print("-" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:50]}...")
        
        # Complete analysis
        result = analyzer.analyze_complete(text)
        
        print(f"Sentiment: {result['summary']['overall_sentiment']} "
              f"({result['summary']['sentiment_confidence']:.1%})")
        print(f"Emotion: {result['summary']['primary_emotion']} "
              f"({result['summary']['emotion_confidence']:.1%})")
        
        # Show top emotions
        if result['emotion']['all_emotions']:
            print("Top emotions:")
            for emotion in result['emotion']['all_emotions'][:3]:
                print(f"  {emotion['emotion']}: {emotion['confidence']:.1%}")
    
    print(f"\nModel info: {analyzer.get_model_info()}")

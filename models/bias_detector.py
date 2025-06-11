"""
Bias Detection System for News Articles
Detects potential political, ideological, and source bias in news content.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
from textstat import flesch_reading_ease
from collections import Counter
import logging

class BiasDetector:
    """Detect various types of bias in news articles."""
    
    def __init__(self):
        """Initialize bias detection models and classifiers."""
        self.logger = logging.getLogger(__name__)
        
        # Political bias indicators
        self.political_keywords = {
            'liberal': ['progressive', 'democrat', 'liberal', 'left-wing', 'socialism', 
                       'environmental', 'diversity', 'inclusion', 'gun control'],
            'conservative': ['republican', 'conservative', 'right-wing', 'traditional', 
                           'capitalism', 'freedom', 'liberty', 'second amendment', 'fiscal'],
            'neutral': ['bipartisan', 'nonpartisan', 'independent', 'moderate', 'centrist']
        }
        
        # Emotional bias words
        self.emotional_words = {
            'positive': ['amazing', 'excellent', 'outstanding', 'brilliant', 'fantastic', 
                        'wonderful', 'incredible', 'magnificent', 'superb', 'exceptional'],
            'negative': ['terrible', 'awful', 'horrible', 'disgusting', 'outrageous', 
                        'shocking', 'devastating', 'catastrophic', 'disastrous', 'appalling'],
            'sensational': ['breaking', 'explosive', 'bombshell', 'shocking', 'stunning', 
                          'exclusive', 'unprecedented', 'dramatic', 'incredible', 'unbelievable']
        }
        
        # Loaded bias words (words that carry emotional weight)
        self.loaded_words = [
            'terrorist', 'extremist', 'radical', 'militant', 'activist', 'regime', 
            'dictator', 'tyrant', 'hero', 'victim', 'freedom fighter', 'insurgent',
            'propaganda', 'conspiracy', 'cover-up', 'scandal', 'corruption'
        ]
        
        # Source reliability categories (can be expanded)
        self.source_reliability = {
            'high': ['Reuters', 'Associated Press', 'BBC', 'NPR', 'PBS'],
            'medium': ['CNN', 'Fox News', 'MSNBC', 'Wall Street Journal'],
            'low': ['Breitbart', 'InfoWars', 'RT', 'Daily Mail']
        }
        
        # Initialize sentiment model for emotional bias
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
        except Exception as e:
            self.logger.warning(f"Could not load sentiment model: {e}")
            self.sentiment_analyzer = None
    
    def detect_political_bias(self, text: str) -> Dict[str, float]:
        """Detect political bias in text based on keyword analysis."""
        text_lower = text.lower()
        word_count = len(text.split())
        
        bias_scores = {}
        
        for bias_type, keywords in self.political_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by text length
            bias_scores[bias_type] = (count / word_count) * 100 if word_count > 0 else 0
        
        # Determine dominant bias
        max_bias = max(bias_scores.values())
        dominant_bias = 'neutral'
        if max_bias > 0.5:  # Threshold for bias detection
            dominant_bias = max(bias_scores, key=bias_scores.get)
        
        return {
            'scores': bias_scores,
            'dominant_bias': dominant_bias,
            'confidence': max_bias
        }
    
    def detect_emotional_bias(self, text: str) -> Dict[str, float]:
        """Detect emotional bias and sensationalism."""
        text_lower = text.lower()
        word_count = len(text.split())
        
        emotional_scores = {}
        
        for emotion_type, words in self.emotional_words.items():
            count = sum(1 for word in words if word in text_lower)
            emotional_scores[emotion_type] = (count / word_count) * 100 if word_count > 0 else 0
        
        # Check for loaded language
        loaded_count = sum(1 for word in self.loaded_words if word in text_lower)
        loaded_score = (loaded_count / word_count) * 100 if word_count > 0 else 0
        
        # Overall emotional bias score
        total_emotional = sum(emotional_scores.values()) + loaded_score
        return {
            'emotional_scores': emotional_scores,
            'loaded_language': loaded_score,
            'total_emotional_bias': total_emotional,
            'is_sensational': emotional_scores.get('sensational', 0) > 1.0
        }
    
    def analyze_language_complexity(self, text: str) -> Dict[str, float]:
        """Analyze language complexity and readability."""
        try:
            # Reading ease score
            reading_ease = flesch_reading_ease(text)
            
            # Sentence length analysis
            sentences = re.split(r'[.!?]+', text)
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
            
            # Word complexity (syllable estimation)
            words = re.findall(r'\b\w+\b', text.lower())
            complex_words = [w for w in words if len(w) > 6]  # Simple proxy for complexity
            complexity_ratio = len(complex_words) / len(words) if words else 0
            
            return {
                'reading_ease': float(reading_ease),
                'avg_sentence_length': float(avg_sentence_length),
                'complexity_ratio': float(complexity_ratio * 100),
                'accessibility': 'high' if reading_ease > 70 else 'medium' if reading_ease > 30 else 'low'
            }
        except Exception as e:
            self.logger.error(f"Error analyzing language complexity: {e}")
            return {
                'reading_ease': 50.0,
                'avg_sentence_length': 15.0,
                'complexity_ratio': 20.0,
                'accessibility': 'medium'
            }
    
    def detect_source_bias(self, source: str) -> Dict[str, str]:
        """Assess source reliability and potential bias."""
        reliability = 'unknown'
        
        for level, sources in self.source_reliability.items():
            if any(source_name.lower() in source.lower() for source_name in sources):
                reliability = level
                break
        
        # Additional source analysis could include:
        # - Domain analysis (.com vs .org vs .gov)
        # - Known bias databases
        # - Fact-checking ratings
        
        return {
            'reliability': reliability,
            'assessment': f"Source reliability: {reliability}"
        }
    
    def analyze_headline_bias(self, headline: str) -> Dict[str, any]:
        """Specific analysis for headlines which often contain more bias."""
        # Check for question marks (often used for speculation)
        has_question = '?' in headline
        
        # Check for sensational punctuation
        exclamation_count = headline.count('!')
        
        # Check for ALL CAPS words (shouting)
        caps_words = re.findall(r'\b[A-Z]{2,}\b', headline)
        caps_ratio = len(caps_words) / len(headline.split()) if headline.split() else 0
        
        # Check for quotation marks (often used for emphasis or skepticism)
        quote_count = headline.count('"') + headline.count("'")
        
        # Emotional words in headline
        emotional_analysis = self.detect_emotional_bias(headline)
        
        bias_indicators = []
        if has_question:
            bias_indicators.append("Speculative (contains question)")
        if exclamation_count > 0:
            bias_indicators.append(f"Sensational punctuation ({exclamation_count} exclamations)")
        if caps_ratio > 0.2:
            bias_indicators.append("Excessive capitalization")
        if emotional_analysis['total_emotional_bias'] > 2.0:
            bias_indicators.append("High emotional language")
        
        return {
            'bias_indicators': bias_indicators,
            'sensational_score': emotional_analysis['emotional_scores'].get('sensational', 0),
            'emotional_score': emotional_analysis['total_emotional_bias'],
            'has_speculation': has_question,
            'caps_ratio': caps_ratio * 100
        }
    
    def comprehensive_bias_analysis(self, title: str, content: str = "", source: str = "") -> Dict[str, any]:
        """Perform comprehensive bias analysis on an article."""
        
        # Combine title and content for full analysis
        full_text = f"{title}. {content}".strip()
        
        # Perform all bias analyses
        political_bias = self.detect_political_bias(full_text)
        emotional_bias = self.detect_emotional_bias(full_text)
        language_analysis = self.analyze_language_complexity(full_text)
        headline_bias = self.analyze_headline_bias(title)
        source_analysis = self.detect_source_bias(source) if source else {'reliability': 'unknown', 'assessment': 'No source provided'}
          # Calculate overall bias score
        overall_bias_factors = [
            float(political_bias['confidence']),
            float(emotional_bias['total_emotional_bias']),
            float(headline_bias['emotional_score']),
            float((100 - language_analysis['reading_ease']) / 10)  # Lower readability might indicate bias
        ]
        
        # Ensure all factors are valid numbers
        valid_factors = [f for f in overall_bias_factors if not (np.isnan(f) or np.isinf(f))]
        overall_bias_score = np.mean(valid_factors) if valid_factors else 0.0
        
        # Determine bias level
        if overall_bias_score < 2:
            bias_level = "Low"
        elif overall_bias_score < 5:
            bias_level = "Medium"
        else:
            bias_level = "High"
        
        # Generate recommendations
        recommendations = []
        if political_bias['confidence'] > 1:
            recommendations.append(f"Consider political bias: {political_bias['dominant_bias']}")
        if emotional_bias['total_emotional_bias'] > 3:
            recommendations.append("High emotional language detected - verify objectivity")
        if headline_bias['bias_indicators']:
            recommendations.append("Headline shows signs of bias")
        if source_analysis['reliability'] == 'low':
            recommendations.append("Source has low reliability rating")
        if language_analysis['accessibility'] == 'low':
            recommendations.append("Complex language may indicate attempt to obscure facts")
        
        return {
            'overall_bias_score': round(overall_bias_score, 2),
            'bias_level': bias_level,
            'political_bias': political_bias,
            'emotional_bias': emotional_bias,
            'headline_analysis': headline_bias,
            'language_analysis': language_analysis,
            'source_analysis': source_analysis,
            'recommendations': recommendations,
            'summary': {
                'dominant_political_bias': political_bias['dominant_bias'],
                'emotional_bias_level': bias_level,
                'source_reliability': source_analysis['reliability'],
                'headline_bias_indicators': len(headline_bias['bias_indicators'])
            }
        }
    
    def batch_analyze_bias(self, articles: List[Dict[str, str]]) -> pd.DataFrame:
        """Analyze bias for multiple articles and return summary DataFrame."""
        results = []
        
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', article.get('description', ''))
            source = article.get('source', '')
            
            analysis = self.comprehensive_bias_analysis(title, content, source)
            
            result = {
                'title': title[:50] + '...' if len(title) > 50 else title,
                'source': source,
                'overall_bias_score': analysis['overall_bias_score'],
                'bias_level': analysis['bias_level'],
                'political_bias': analysis['political_bias']['dominant_bias'],
                'emotional_bias': analysis['emotional_bias']['total_emotional_bias'],
                'source_reliability': analysis['source_analysis']['reliability'],
                'headline_indicators': len(analysis['headline_analysis']['bias_indicators']),
                'reading_ease': analysis['language_analysis']['reading_ease']
            }
            results.append(result)
        
        return pd.DataFrame(results)

# Example usage and testing
if __name__ == "__main__":
    detector = BiasDetector()
    
    # Test articles with different bias levels
    test_articles = [
        {
            'title': 'BREAKING: Shocking revelations expose corrupt politicians!',
            'content': 'This devastating report reveals the terrible truth about political corruption...',
            'source': 'Daily Mail'
        },
        {
            'title': 'Senate passes bipartisan infrastructure bill',
            'content': 'The Senate voted to approve the infrastructure legislation with support from both parties...',
            'source': 'Reuters'
        },
        {
            'title': 'Liberal activists push radical agenda?',
            'content': 'Progressive groups continue their campaign for environmental protection and social justice...',
            'source': 'Unknown'
        }
    ]
    
    print("=== Bias Analysis Results ===")
    for i, article in enumerate(test_articles):
        print(f"\n--- Article {i+1} ---")
        analysis = detector.comprehensive_bias_analysis(
            article['title'], 
            article['content'], 
            article['source']
        )
        
        print(f"Title: {article['title']}")
        print(f"Overall Bias Score: {analysis['overall_bias_score']}")
        print(f"Bias Level: {analysis['bias_level']}")
        print(f"Political Bias: {analysis['political_bias']['dominant_bias']}")
        print(f"Source Reliability: {analysis['source_analysis']['reliability']}")
        print(f"Recommendations: {', '.join(analysis['recommendations']) if analysis['recommendations'] else 'None'}")
    
    # Batch analysis
    print("\n=== Batch Analysis ===")
    df = detector.batch_analyze_bias(test_articles)
    print(df.to_string(index=False))

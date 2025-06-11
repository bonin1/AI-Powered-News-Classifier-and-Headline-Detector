"""
News Classification Module
Uses DistilBERT model fine-tuned on AG News dataset for text classification.
"""

import logging
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

logger = logging.getLogger(__name__)


class NewsClassifier:
    """    News article classifier using DistilBERT model.
    Classifies articles into: World, Sports, Business, Sci/Tech categories.
    """
    
    def __init__(self):
        """Initialize the classifier with pre-trained model."""
        self.model_name = "textattack/distilbert-base-uncased-ag-news"
        self.category_mapping = {
            0: "World",
            1: "Sports", 
            2: "Business",
            3: "Sci/Tech"
        }
        
        logger.info(f"Loading classification model: {self.model_name}")
        try:
            # Initialize the classification pipeline with a working model
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True
            )
            logger.info("Classification model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading classification model: {str(e)}")
            raise
    def classify(self, text: str) -> Dict[str, any]:
        """
        Classify the given text into news categories.
        
        Args:
            text (str): Text content to classify
            
        Returns:
            Dict[str, any]: Classification results with category, confidence, and all scores
        """
        try:
            if not text.strip():
                return {
                    'category': 'Unknown',
                    'confidence': 0.0,
                    'all_scores': []
                }
            
            # Truncate text if too long (DistilBERT has a 512 token limit)
            # Use character-based approximation: 1 token ≈ 4 characters
            max_chars = 1800  # Approximately 450 tokens, leaving room for special tokens
            if len(text) > max_chars:
                # Try to truncate at sentence boundary
                sentences = text[:max_chars].split('.')
                if len(sentences) > 1:
                    text = '.'.join(sentences[:-1]) + '.'
                else:
                    text = text[:max_chars]
            
            # Get predictions
            results = self.classifier(text)
            
            # Process results
            if results and len(results) > 0:
                # Get the prediction with highest score
                best_prediction = max(results[0], key=lambda x: x['score'])
                
                # Map label to readable category
                category = self._map_label_to_category(best_prediction['label'])
                confidence = best_prediction['score']
                
                # Format all scores for transparency
                all_scores = []
                for pred in results[0]:
                    all_scores.append({
                        'category': self._map_label_to_category(pred['label']),
                        'confidence': pred['score']
                    })
                
                return {
                    'category': category,
                    'confidence': confidence,
                    'all_scores': all_scores
                }
            else:
                return {
                    'category': 'Unknown',
                    'confidence': 0.0,
                    'all_scores': []
                }
                
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            return {
                'category': 'Error',
                'confidence': 0.0,
                'all_scores': [],
                'error': str(e)
            }
    
    def _map_label_to_category(self, label: str) -> str:
        """
        Map model output labels to readable categories.
        
        Args:
            label (str): Model output label
            
        Returns:
            str: Readable category name
        """
        # The AG News labels are typically "LABEL_0", "LABEL_1", etc.
        if label.startswith("LABEL_"):
            try:
                label_num = int(label.split("_")[1])
                return self.category_mapping.get(label_num, "Unknown")
            except (IndexError, ValueError):
                return "Unknown"
        
        # Sometimes the model might return direct category names
        return label.title()
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Classify multiple texts at once.
        
        Args:
            texts (List[str]): List of texts to classify
            
        Returns:
            List[Dict[str, any]]: List of classification results
        """
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, str]: Model information
        """
        return {
            'model_name': self.model_name,
            'categories': list(self.category_mapping.values()),
            'description': 'DistilBERT model fine-tuned on AG News dataset for news classification'
        }


if __name__ == "__main__":
    # Test the classifier
    classifier = NewsClassifier()
    
    # Test samples
    test_texts = [
        "Apple announced a new iPhone with advanced AI features and improved camera technology.",
        "The Lakers won against the Warriors 112-108 in an exciting basketball game last night.",
        "Scientists have discovered a new exoplanet that could potentially support life.",
        "The stock market reached new highs today as investors showed confidence in tech companies."
    ]
    
    print("Testing News Classifier:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        result = classifier.classify(text)
        print(f"\nTest {i}:")
        print(f"Text: {text[:100]}...")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        if 'all_scores' in result and result['all_scores']:
            print("All scores:")
            for score in result['all_scores']:
                print(f"  {score['category']}: {score['confidence']:.3f}")

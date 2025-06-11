"""
Headline Generation Module
Uses Google's PEGASUS model for generating catchy news headlines.
"""

import logging
from typing import List, Optional
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
import re

logger = logging.getLogger(__name__)


class HeadlineGenerator:
    """
    Headline generator using Google's PEGASUS model fine-tuned on XSum dataset.
    Generates concise and catchy headlines from article content.
    """
    
    def __init__(self):
        """Initialize the headline generator with PEGASUS model."""
        self.model_name = "google/pegasus-xsum"
        
        logger.info(f"Loading headline generation model: {self.model_name}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
            
            # Set device (use GPU if available)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            logger.info(f"Headline generation model loaded successfully on {self.device}!")
            
        except Exception as e:
            logger.error(f"Error loading headline generation model: {str(e)}")
            raise
    
    def generate(self, text: str, max_length: int = 32, num_beams: int = 4, 
                early_stopping: bool = True) -> str:
        """
        Generate a headline from the given text.
        
        Args:
            text (str): Article content to generate headline for
            max_length (int): Maximum length of generated headline
            num_beams (int): Number of beams for beam search
            early_stopping (bool): Whether to stop early when possible
            
        Returns:
            str: Generated headline
        """
        try:
            if not text.strip():
                return "No content provided"
            
            # Preprocess the text
            processed_text = self._preprocess_text(text)
            
            # Tokenize input
            inputs = self.tokenizer(
                processed_text,
                max_length=1024,  # PEGASUS input limit
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate headline
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    no_repeat_ngram_size=2,
                    do_sample=False
                )
            
            # Decode the generated headline
            headline = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process the headline
            headline = self._postprocess_headline(headline)
            
            return headline
            
        except Exception as e:
            logger.error(f"Error generating headline: {str(e)}")
            return f"Error generating headline: {str(e)}"
    
    def generate_multiple(self, text: str, num_headlines: int = 3, 
                         max_length: int = 32) -> List[str]:
        """
        Generate multiple headline options.
        
        Args:
            text (str): Article content
            num_headlines (int): Number of headlines to generate
            max_length (int): Maximum length of each headline
            
        Returns:
            List[str]: List of generated headlines
        """
        try:
            if not text.strip():
                return ["No content provided"] * num_headlines
            
            processed_text = self._preprocess_text(text)
            
            inputs = self.tokenizer(
                processed_text,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            headlines = []
            
            # Generate multiple headlines with different parameters
            for i in range(num_headlines):
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        temperature=0.7 + (i * 0.1),  # Vary temperature for diversity
                        top_p=0.9
                    )
                
                headline = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                headline = self._postprocess_headline(headline)
                headlines.append(headline)
            
            # Remove duplicates while preserving order
            unique_headlines = []
            seen = set()
            for headline in headlines:
                if headline.lower() not in seen:
                    unique_headlines.append(headline)
                    seen.add(headline.lower())
            
            # Fill with variants if we don't have enough unique headlines
            while len(unique_headlines) < num_headlines:
                unique_headlines.append(f"Alternative headline {len(unique_headlines)}")
            
            return unique_headlines[:num_headlines]
            
        except Exception as e:
            logger.error(f"Error generating multiple headlines: {str(e)}")
            return [f"Error generating headline {i+1}" for i in range(num_headlines)]
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before headline generation.
        
        Args:
            text (str): Raw text content
            
        Returns:
            str: Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Focus on the first few sentences for better headline generation
        sentences = text.split('. ')
        if len(sentences) > 5:
            text = '. '.join(sentences[:5]) + '.'
        
        # Limit length to avoid token limit issues
        max_words = 400
        words = text.split()
        if len(words) > max_words:
            text = ' '.join(words[:max_words])
        
        return text
    
    def _postprocess_headline(self, headline: str) -> str:
        """
        Post-process generated headline.
        
        Args:
            headline (str): Raw generated headline
            
        Returns:
            str: Cleaned headline
        """
        # Remove extra whitespace
        headline = re.sub(r'\s+', ' ', headline.strip())
        
        # Capitalize first letter
        if headline:
            headline = headline[0].upper() + headline[1:]
        
        # Remove trailing periods if present
        headline = headline.rstrip('.')
        
        # Ensure reasonable length
        if len(headline) > 100:
            # Try to cut at a reasonable point
            words = headline.split()
            if len(words) > 15:
                headline = ' '.join(words[:15])
        
        return headline
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'description': 'PEGASUS model fine-tuned on XSum dataset for headline generation',
            'max_input_length': 1024,
            'typical_output_length': '10-32 tokens'
        }


if __name__ == "__main__":
    # Test the headline generator
    generator = HeadlineGenerator()
    
    # Test sample
    test_text = """
    Apple Inc. announced today that they are releasing a new iPhone model with advanced artificial 
    intelligence capabilities. The device features an improved camera system with computational 
    photography, faster processing speeds with the new A17 chip, and enhanced battery life. 
    The company expects this to be their best-selling product this year, with pre-orders starting 
    next week. CEO Tim Cook highlighted the revolutionary AI features that will change how users 
    interact with their phones. The starting price will be $999 for the base model.
    """
    
    print("Testing Headline Generator:")
    print("-" * 50)
    print(f"Original text: {test_text[:200]}...")
    print()
    
    # Generate single headline
    headline = generator.generate(test_text)
    print(f"Generated headline: {headline}")
    print()
    
    # Generate multiple headlines
    multiple_headlines = generator.generate_multiple(test_text, num_headlines=3)
    print("Multiple headline options:")
    for i, h in enumerate(multiple_headlines, 1):
        print(f"{i}. {h}")
    
    print(f"\nModel info: {generator.get_model_info()}")

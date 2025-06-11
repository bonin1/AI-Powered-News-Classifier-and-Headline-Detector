#!/usr/bin/env python3
"""
Command Line Interface for AI-Powered News Classifier and Headline Detector
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from news_processor import NewsProcessor

def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def process_url(processor, url, output_format='json'):
    """Process a single URL."""
    print(f"Processing URL: {url}")
    result = processor.process_url(url)
    
    if output_format == 'json':
        print(json.dumps(result, indent=2))
    else:
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Original Headline: {result['original_headline']}")
        print(f"Generated Headline: {result['generated_headline']}")
        if 'error' in result:
            print(f"Error: {result['error']}")

def process_text(processor, text, title='', output_format='json'):
    """Process text content."""
    print("Processing text content...")
    result = processor.process_text(text, title)
    
    if output_format == 'json':
        print(json.dumps(result, indent=2))
    else:
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Generated Headline: {result['generated_headline']}")
        if 'error' in result:
            print(f"Error: {result['error']}")

def process_file(processor, file_path, output_format='json'):
    """Process text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Processing file: {file_path}")
        process_text(processor, content, '', output_format)
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def process_batch_urls(processor, urls_file, output_file=None):
    """Process multiple URLs from a file."""
    try:
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(urls)} URLs from {urls_file}")
        
        results = []
        for i, url in enumerate(urls, 1):
            print(f"Processing {i}/{len(urls)}: {url}")
            result = processor.process_url(url)
            results.append({
                'url': url,
                'result': result
            })
        
        # Output results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        else:
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        print(f"Error processing batch URLs: {e}")
        sys.exit(1)

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='AI-Powered News Classifier and Headline Detector CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single URL
  python cli.py --url "https://example.com/news-article"
  
  # Process text content
  python cli.py --text "Apple announced a new iPhone model today..."
  
  # Process text with original title
  python cli.py --text "Article content..." --title "Original Headline"
  
  # Process text from file
  python cli.py --file "article.txt"
  
  # Batch process URLs
  python cli.py --batch-urls "urls.txt" --output "results.json"
  
  # Get only classification
  python cli.py --url "https://example.com/article" --classify-only
  
  # Get only headline generation
  python cli.py --text "Article content..." --headline-only
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--url', help='URL of news article to process')
    input_group.add_argument('--text', help='Text content to process')
    input_group.add_argument('--file', help='File containing text content')
    input_group.add_argument('--batch-urls', help='File containing URLs (one per line)')
    
    # Optional arguments
    parser.add_argument('--title', help='Original headline (for text input)')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--format', choices=['json', 'text'], default='json',
                       help='Output format (default: json)')
    
    # Processing options
    parser.add_argument('--classify-only', action='store_true',
                       help='Only perform classification')
    parser.add_argument('--headline-only', action='store_true',
                       help='Only generate headline')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--version', action='version', version='1.0.0')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Initialize processor
    try:
        print("Loading AI models...")
        processor = NewsProcessor()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)
    
    # Process based on input type
    try:
        if args.url:
            if args.classify_only:
                # Extract content and classify only
                result = processor.process_url(args.url)
                classification = {
                    'category': result['category'],
                    'confidence': result['confidence']
                }
                if args.format == 'json':
                    print(json.dumps(classification, indent=2))
                else:
                    print(f"Category: {classification['category']}")
                    print(f"Confidence: {classification['confidence']:.3f}")
            elif args.headline_only:
                # Extract content and generate headline only
                from utils.article_extractor import ArticleExtractor
                extractor = ArticleExtractor()
                article_data = extractor.extract_from_url(args.url)
                headline = processor.generate_headline(article_data['content'])
                if args.format == 'json':
                    print(json.dumps({'generated_headline': headline}, indent=2))
                else:
                    print(f"Generated Headline: {headline}")
            else:
                process_url(processor, args.url, args.format)
        
        elif args.text:
            if args.classify_only:
                classification = processor.classify_text(args.text)
                if args.format == 'json':
                    print(json.dumps(classification, indent=2))
                else:
                    print(f"Category: {classification['category']}")
                    print(f"Confidence: {classification['confidence']:.3f}")
            elif args.headline_only:
                headline = processor.generate_headline(args.text)
                if args.format == 'json':
                    print(json.dumps({'generated_headline': headline}, indent=2))
                else:
                    print(f"Generated Headline: {headline}")
            else:
                process_text(processor, args.text, args.title or '', args.format)
        
        elif args.file:
            process_file(processor, args.file, args.format)
        
        elif args.batch_urls:
            process_batch_urls(processor, args.batch_urls, args.output)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

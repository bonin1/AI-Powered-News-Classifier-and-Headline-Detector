"""
Export Module
Handles exporting analysis results to various formats.
"""

import json
import csv
import logging
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultExporter:
    """
    Export analysis results to various formats (JSON, CSV, HTML, etc.).
    """
    
    def __init__(self):
        """Initialize the exporter."""
        self.supported_formats = ['json', 'csv', 'html', 'txt']
    
    def export_single_result(self, result: Dict[str, Any], filename: str, format_type: str = 'json') -> str:
        """
        Export a single analysis result.
        
        Args:
            result (Dict[str, Any]): Analysis result to export
            filename (str): Output filename (without extension)
            format_type (str): Export format ('json', 'csv', 'html', 'txt')
            
        Returns:
            str: Path to exported file
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. Supported: {self.supported_formats}")
        
        # Add timestamp and format extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{filename}_{timestamp}.{format_type}"
        
        try:
            if format_type == 'json':
                self._export_json_single(result, output_path)
            elif format_type == 'csv':
                self._export_csv_single(result, output_path)
            elif format_type == 'html':
                self._export_html_single(result, output_path)
            elif format_type == 'txt':
                self._export_txt_single(result, output_path)
            
            logger.info(f"Successfully exported result to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting result: {str(e)}")
            raise
    
    def export_batch_results(self, results: List[Dict[str, Any]], filename: str, format_type: str = 'json') -> str:
        """
        Export multiple analysis results.
        
        Args:
            results (List[Dict[str, Any]]): List of analysis results
            filename (str): Output filename (without extension)
            format_type (str): Export format
            
        Returns:
            str: Path to exported file
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. Supported: {self.supported_formats}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{filename}_batch_{timestamp}.{format_type}"
        
        try:
            if format_type == 'json':
                self._export_json_batch(results, output_path)
            elif format_type == 'csv':
                self._export_csv_batch(results, output_path)
            elif format_type == 'html':
                self._export_html_batch(results, output_path)
            elif format_type == 'txt':
                self._export_txt_batch(results, output_path)
            
            logger.info(f"Successfully exported {len(results)} results to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting batch results: {str(e)}")
            raise
    
    def _export_json_single(self, result: Dict[str, Any], output_path: str):
        """Export single result as JSON."""
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'export_type': 'single_article',
                'version': '1.0'
            },
            'result': result
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_json_batch(self, results: List[Dict[str, Any]], output_path: str):
        """Export batch results as JSON."""
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'export_type': 'batch_articles',
                'count': len(results),
                'version': '1.0'
            },
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_csv_single(self, result: Dict[str, Any], output_path: str):
        """Export single result as CSV."""
        # Flatten the result for CSV format
        flattened = self._flatten_result(result)
        
        df = pd.DataFrame([flattened])
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def _export_csv_batch(self, results: List[Dict[str, Any]], output_path: str):
        """Export batch results as CSV."""
        # Flatten all results
        flattened_results = [self._flatten_result(result) for result in results]
        
        df = pd.DataFrame(flattened_results)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def _export_html_single(self, result: Dict[str, Any], output_path: str):
        """Export single result as HTML."""
        html_content = self._generate_html_single(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _export_html_batch(self, results: List[Dict[str, Any]], output_path: str):
        """Export batch results as HTML."""
        html_content = self._generate_html_batch(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _export_txt_single(self, result: Dict[str, Any], output_path: str):
        """Export single result as plain text."""
        text_content = self._generate_text_single(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
    
    def _export_txt_batch(self, results: List[Dict[str, Any]], output_path: str):
        """Export batch results as plain text."""
        text_content = self._generate_text_batch(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
    
    def _flatten_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested result dictionary for CSV export."""
        flattened = {}
        
        # Basic fields
        basic_fields = ['category', 'confidence', 'original_headline', 'generated_headline', 
                       'word_count', 'url']
        
        for field in basic_fields:
            flattened[field] = result.get(field, '')
        
        # Add sentiment if available
        if 'sentiment' in result:
            sentiment = result['sentiment']
            if isinstance(sentiment, dict):
                flattened['sentiment'] = sentiment.get('sentiment', '')
                flattened['sentiment_confidence'] = sentiment.get('confidence', 0)
            else:
                flattened['sentiment'] = sentiment
        
        # Add emotion if available
        if 'emotion' in result:
            emotion = result['emotion']
            if isinstance(emotion, dict):
                flattened['emotion'] = emotion.get('emotion', '')
                flattened['emotion_confidence'] = emotion.get('confidence', 0)
        
        # Add timestamp
        flattened['export_timestamp'] = datetime.now().isoformat()
        
        return flattened
    
    def _generate_html_single(self, result: Dict[str, Any]) -> str:
        """Generate HTML for single result."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>News Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f2f6; padding: 20px; border-radius: 8px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #1f77b4; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .confidence {{ color: #666; font-size: 0.9em; }}
        .content-preview {{ background-color: #f8f9fa; padding: 15px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📰 News Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="section">
        <h2>Article Information</h2>
        <p><strong>Original Headline:</strong> {result.get('original_headline', 'N/A')}</p>
        <p><strong>Generated Headline:</strong> {result.get('generated_headline', 'N/A')}</p>
        <p><strong>Word Count:</strong> {result.get('word_count', 0)}</p>
        {f'<p><strong>Source URL:</strong> <a href="{result.get("url")}">{result.get("url")}</a></p>' if result.get('url') and result.get('url') != 'direct_input' else ''}
    </div>
    
    <div class="section">
        <h2>Classification Results</h2>
        <div class="metric">
            <strong>Category:</strong> {result.get('category', 'Unknown')}
            <div class="confidence">Confidence: {result.get('confidence', 0):.1%}</div>
        </div>
    </div>
    
    {self._add_sentiment_html(result) if 'sentiment' in result else ''}
    
    {f'<div class="section"><h2>Content Preview</h2><div class="content-preview">{result.get("content_preview", "No preview available")}</div></div>' if result.get('content_preview') else ''}
    
</body>
</html>
"""
        return html
    
    def _generate_html_batch(self, results: List[Dict[str, Any]]) -> str:
        """Generate HTML for batch results."""
        # Create summary statistics
        categories = [r.get('category', 'Unknown') for r in results]
        category_counts = pd.Series(categories).value_counts()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Batch News Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f2f6; padding: 20px; border-radius: 8px; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .article {{ border: 1px solid #ddd; margin: 15px 0; padding: 15px; border-radius: 4px; }}
        .article:nth-child(even) {{ background-color: #f9f9f9; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Batch News Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Total Articles Processed: {len(results)}</p>
    </div>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <h3>Category Distribution:</h3>
        <ul>
        {''.join([f'<li>{cat}: {count} articles</li>' for cat, count in category_counts.items()])}
        </ul>
    </div>
    
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>#</th>
            <th>Original Headline</th>
            <th>Generated Headline</th>
            <th>Category</th>
            <th>Confidence</th>
            <th>Word Count</th>
        </tr>
        {''.join([f'''<tr>
            <td>{i+1}</td>
            <td>{result.get('original_headline', 'N/A')[:50]}{'...' if len(str(result.get('original_headline', ''))) > 50 else ''}</td>
            <td>{result.get('generated_headline', 'N/A')[:50]}{'...' if len(str(result.get('generated_headline', ''))) > 50 else ''}</td>
            <td>{result.get('category', 'Unknown')}</td>
            <td>{result.get('confidence', 0):.1%}</td>
            <td>{result.get('word_count', 0)}</td>
        </tr>''' for i, result in enumerate(results)])}
    </table>
    
</body>
</html>
"""
        return html
    
    def _generate_text_single(self, result: Dict[str, Any]) -> str:
        """Generate plain text for single result."""
        text = f"""
NEWS ANALYSIS REPORT
====================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

ARTICLE INFORMATION
-------------------
Original Headline: {result.get('original_headline', 'N/A')}
Generated Headline: {result.get('generated_headline', 'N/A')}
Word Count: {result.get('word_count', 0)}
Source URL: {result.get('url', 'N/A')}

CLASSIFICATION
--------------
Category: {result.get('category', 'Unknown')}
Confidence: {result.get('confidence', 0):.1%}

{self._add_sentiment_text(result) if 'sentiment' in result else ''}

CONTENT PREVIEW
---------------
{result.get('content_preview', 'No preview available')}
"""
        return text
    
    def _generate_text_batch(self, results: List[Dict[str, Any]]) -> str:
        """Generate plain text for batch results."""
        categories = [r.get('category', 'Unknown') for r in results]
        category_counts = pd.Series(categories).value_counts()
        
        text = f"""
BATCH NEWS ANALYSIS REPORT
==========================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Articles: {len(results)}

SUMMARY STATISTICS
------------------
Category Distribution:
{chr(10).join([f"  {cat}: {count} articles" for cat, count in category_counts.items()])}

DETAILED RESULTS
----------------
"""
        
        for i, result in enumerate(results, 1):
            text += f"""
Article {i}:
  Original Headline: {result.get('original_headline', 'N/A')}
  Generated Headline: {result.get('generated_headline', 'N/A')}
  Category: {result.get('category', 'Unknown')} ({result.get('confidence', 0):.1%})
  Word Count: {result.get('word_count', 0)}
"""
        
        return text
    
    def _add_sentiment_html(self, result: Dict[str, Any]) -> str:
        """Add sentiment information to HTML."""
        sentiment = result.get('sentiment', {})
        if not isinstance(sentiment, dict):
            return ""
        
        return f"""
    <div class="section">
        <h2>Sentiment Analysis</h2>
        <div class="metric">
            <strong>Sentiment:</strong> {sentiment.get('sentiment', 'Unknown').title()}
            <div class="confidence">Confidence: {sentiment.get('confidence', 0):.1%}</div>
        </div>
        {f'<div class="metric"><strong>Emotion:</strong> {result.get("emotion", {}).get("emotion", "Unknown").title()}<div class="confidence">Confidence: {result.get("emotion", {}).get("confidence", 0):.1%}</div></div>' if 'emotion' in result else ''}
    </div>
"""
    
    def _add_sentiment_text(self, result: Dict[str, Any]) -> str:
        """Add sentiment information to text."""
        sentiment = result.get('sentiment', {})
        if not isinstance(sentiment, dict):
            return ""
        
        text = f"""
SENTIMENT ANALYSIS
------------------
Sentiment: {sentiment.get('sentiment', 'Unknown').title()}
Confidence: {sentiment.get('confidence', 0):.1%}
"""
        
        if 'emotion' in result:
            emotion = result.get('emotion', {})
            text += f"Emotion: {emotion.get('emotion', 'Unknown').title()}\n"
            text += f"Emotion Confidence: {emotion.get('confidence', 0):.1%}\n"
        
        return text


if __name__ == "__main__":
    # Test the exporter
    exporter = ResultExporter()
    
    # Sample result
    sample_result = {
        'category': 'Sci/Tech',
        'confidence': 0.95,
        'original_headline': 'Apple Announces New iPhone',
        'generated_headline': 'Apple Unveils Revolutionary iPhone with AI Features',
        'word_count': 150,
        'url': 'https://example.com/article',
        'content_preview': 'Apple Inc. announced today...'
    }
    
    # Test exports
    for format_type in ['json', 'csv', 'html', 'txt']:
        try:
            output_path = exporter.export_single_result(sample_result, 'test_export', format_type)
            print(f"✅ {format_type.upper()} export successful: {output_path}")
        except Exception as e:
            print(f"❌ {format_type.upper()} export failed: {e}")

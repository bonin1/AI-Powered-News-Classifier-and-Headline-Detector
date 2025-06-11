"""
RSS Feed Monitor for automatic news collection and processing.
"""

import feedparser
import requests
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import os

@dataclass
class RSSFeed:
    """RSS Feed configuration."""
    name: str
    url: str
    category: str = "general"
    active: bool = True
    last_checked: Optional[datetime] = None

@dataclass
class NewsItem:
    """News item from RSS feed."""
    title: str
    link: str
    description: str
    published: datetime
    source: str
    category: str
    processed: bool = False

class RSSMonitor:
    """Monitor RSS feeds and collect news articles."""
    
    def __init__(self, db_path: str = "data/news_monitor.db"):
        """Initialize RSS monitor."""
        self.db_path = db_path
        self.feeds: List[RSSFeed] = []
        self.running = False
        self.check_interval = 300  # 5 minutes
        self.logger = logging.getLogger(__name__)
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        self._load_default_feeds()
    
    def _init_database(self):
        """Initialize SQLite database for storing articles."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    link TEXT UNIQUE NOT NULL,
                    description TEXT,
                    published TIMESTAMP,
                    source TEXT,
                    category TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    category TEXT DEFAULT 'general',
                    active BOOLEAN DEFAULT TRUE,
                    last_checked TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP                )
            """)
    
    def _load_default_feeds(self):
        """Load default RSS feeds."""
        default_feeds = [
            # Major news sources with reliable RSS feeds
            RSSFeed("BBC News", "http://feeds.bbci.co.uk/news/rss.xml", "general"),
            RSSFeed("Reuters Top News", "https://feeds.reuters.com/reuters/topNews", "general"),
            RSSFeed("Associated Press", "https://feeds.apnews.com/rss/apf-topnews", "general"),
            RSSFeed("NPR News", "https://feeds.npr.org/1001/rss.xml", "general"),
            RSSFeed("CNN Top Stories", "http://rss.cnn.com/rss/edition.rss", "general"),
            
            # Technology
            RSSFeed("TechCrunch", "https://techcrunch.com/feed/", "technology"),
            RSSFeed("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index", "technology"),
            RSSFeed("The Verge", "https://www.theverge.com/rss/index.xml", "technology"),
            RSSFeed("Wired", "https://www.wired.com/feed/rss", "technology"),
            
            # Business
            RSSFeed("Reuters Business", "https://feeds.reuters.com/reuters/businessNews", "business"),
            RSSFeed("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/", "business"),
            RSSFeed("Financial Times", "https://www.ft.com/rss/home", "business"),
            
            # Sports
            RSSFeed("ESPN", "https://www.espn.com/espn/rss/news", "sports"),
            RSSFeed("Sky Sports", "https://www.skysports.com/rss/12040", "sports"),
            
            # Science
            RSSFeed("Science Daily", "https://www.sciencedaily.com/rss/all.xml", "science"),
            RSSFeed("NASA News", "https://www.nasa.gov/rss/dyn/breaking_news.rss", "science"),
            
            # Entertainment
            RSSFeed("Entertainment Weekly", "https://ew.com/feed/", "entertainment"),
            RSSFeed("Variety", "https://variety.com/feed/", "entertainment"),
            
            # Health
            RSSFeed("WebMD Health", "https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC", "health"),
            
            # World News
            RSSFeed("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", "world"),
            RSSFeed("Deutsche Welle", "https://rss.dw.com/rdf/rss-en-all", "world"),
        ]
        
        # Load existing feeds from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name, url, category, active, last_checked FROM feeds")
            existing_feeds = cursor.fetchall()
            
            if not existing_feeds:
                # Insert default feeds
                for feed in default_feeds:
                    conn.execute(
                        "INSERT OR IGNORE INTO feeds (name, url, category, active) VALUES (?, ?, ?, ?)",
                        (feed.name, feed.url, feed.category, feed.active)
                    )
                self.feeds = default_feeds
            else:
                # Load from database
                self.feeds = [
                    RSSFeed(
                        name=row[0], 
                        url=row[1], 
                        category=row[2], 
                        active=bool(row[3]),
                        last_checked=datetime.fromisoformat(row[4]) if row[4] else None
                    )
                    for row in existing_feeds
                ]
    
    def add_feed(self, feed: RSSFeed) -> bool:
        """Add a new RSS feed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO feeds (name, url, category, active) VALUES (?, ?, ?, ?)",
                    (feed.name, feed.url, feed.category, feed.active)
                )
            self.feeds.append(feed)
            self.logger.info(f"Added RSS feed: {feed.name}")
            return True
        except sqlite3.IntegrityError:
            self.logger.warning(f"Feed already exists: {feed.url}")
            return False
    
    def remove_feed(self, feed_url: str) -> bool:
        """Remove an RSS feed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM feeds WHERE url = ?", (feed_url,))
            if cursor.rowcount > 0:
                self.feeds = [f for f in self.feeds if f.url != feed_url]
                self.logger.info(f"Removed RSS feed: {feed_url}")
                return True
        return False
    
    def fetch_feed(self, feed: RSSFeed) -> List[NewsItem]:
        """Fetch articles from a single RSS feed."""
        try:
            self.logger.info(f"Fetching feed: {feed.name}")
            
            # Set user agent and allow redirects
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Try to fetch with custom headers first
            try:
                response = requests.get(feed.url, headers=headers, timeout=30, allow_redirects=True)
                if response.status_code == 200:
                    parsed_feed = feedparser.parse(response.content)
                else:
                    # Fallback to feedparser's built-in fetching
                    parsed_feed = feedparser.parse(feed.url)
            except:
                # Final fallback
                parsed_feed = feedparser.parse(feed.url)
            
            # Check if we got valid feed data
            if hasattr(parsed_feed, 'status') and parsed_feed.status not in [200, 301, 302]:
                self.logger.warning(f"Failed to fetch {feed.name}: HTTP {parsed_feed.status}")
                return []
            
            if not hasattr(parsed_feed, 'entries') or not parsed_feed.entries:
                self.logger.warning(f"No entries found in feed: {feed.name}")
                return []
            
            articles = []
            for entry in parsed_feed.entries:
                try:
                    # Parse published date
                    published = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published = datetime(*entry.updated_parsed[:6])
                    
                    # Create news item
                    article = NewsItem(
                        title=entry.get('title', '').strip(),
                        link=entry.get('link', '').strip(),
                        description=entry.get('description', '').strip(),
                        published=published,
                        source=feed.name,
                        category=feed.category
                    )
                    
                    # Only add if we have title and link
                    if article.title and article.link:
                        articles.append(article)
                        
                except Exception as entry_error:
                    self.logger.warning(f"Error processing entry in {feed.name}: {str(entry_error)}")
                    continue
            
            self.logger.info(f"Fetched {len(articles)} articles from {feed.name}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching feed {feed.name}: {e}")
            return []
    
    def save_articles(self, articles: List[NewsItem]) -> int:
        """Save articles to database."""
        saved_count = 0
        with sqlite3.connect(self.db_path) as conn:
            for article in articles:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO articles 
                        (title, link, description, published, source, category)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        article.title, article.link, article.description,
                        article.published, article.source, article.category
                    ))
                    if conn.total_changes > 0:
                        saved_count += 1
                except Exception as e:
                    self.logger.error(f"Error saving article: {e}")
        
        if saved_count > 0:
            self.logger.info(f"Saved {saved_count} new articles")
        return saved_count
    
    def get_recent_articles(self, hours: int = 24, category: str = None, limit: int = 100) -> List[Dict]:
        """Get recent articles from database."""
        query = """
            SELECT title, link, description, published, source, category
            FROM articles 
            WHERE published > datetime('now', '-{} hours')
        """.format(hours)
        
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        
        query += " ORDER BY published DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            articles = cursor.fetchall()
            
            return [
                {
                    'title': row[0],
                    'link': row[1], 
                    'description': row[2],
                    'published': row[3],
                    'source': row[4],
                    'category': row[5]
                }
                for row in articles
            ]
    
    def check_all_feeds(self):
        """Check all active feeds for new articles."""
        all_articles = []
        
        # Use thread pool for parallel fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for feed in self.feeds:
                if feed.active:
                    future = executor.submit(self.fetch_feed, feed)
                    futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    articles = future.result(timeout=30)
                    all_articles.extend(articles)
                except Exception as e:
                    self.logger.error(f"Error in feed fetch: {e}")
        
        # Save articles
        if all_articles:
            saved_count = self.save_articles(all_articles)
            self.logger.info(f"Monitoring cycle complete: {saved_count} new articles saved")
        
        # Update last checked time
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE feeds SET last_checked = ?", (now,))
        
        return len(all_articles)
    
    def start_monitoring(self):
        """Start background monitoring of RSS feeds."""
        if self.running:
            self.logger.warning("Monitor is already running")
            return
        
        self.running = True
        self.logger.info("Starting RSS monitoring...")
        
        def monitor_loop():
            while self.running:
                try:
                    self.check_all_feeds()
                    time.sleep(self.check_interval)
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        # Start monitoring in background thread
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"RSS monitoring started (checking every {self.check_interval} seconds)")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping RSS monitoring...")
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("RSS monitoring stopped")
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Total articles
            total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            
            # Articles by category
            category_stats = dict(conn.execute("""
                SELECT category, COUNT(*) FROM articles 
                GROUP BY category
            """).fetchall())
            
            # Recent articles (last 24 hours)
            recent_count = conn.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE published > datetime('now', '-24 hours')
            """).fetchone()[0]
            
            # Active feeds
            active_feeds = conn.execute("""
                SELECT COUNT(*) FROM feeds WHERE active = TRUE
            """).fetchone()[0]
        
        return {
            'total_articles': total_articles,
            'category_distribution': category_stats,
            'recent_articles_24h': recent_count,
            'active_feeds': active_feeds,
            'monitoring_active': self.running
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    monitor = RSSMonitor()
    
    # Check feeds once
    monitor.check_all_feeds()
    
    # Get recent articles
    recent_articles = monitor.get_recent_articles(hours=24)
    print(f"Found {len(recent_articles)} recent articles")
    
    # Get stats
    stats = monitor.get_stats()
    print("Monitoring stats:", stats)

"""
Utility functions for web scraping
"""
import requests
import time
import random
import hashlib
import json
import os
from typing import Dict, List, Optional, Callable
from functools import wraps
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                     backoff: float = 2.0):
    """
    Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay after each retry
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator


def rate_limit(calls_per_second: float = 1.0):
    """
    Decorator to rate limit function calls.
    
    Args:
        calls_per_second: Maximum calls per second
    """
    min_interval = 1.0 / calls_per_second
    last_call = [0.0]  # Use list to allow modification in closure
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_call[0] = time.time()
            return result
        return wrapper
    return decorator


class CacheManager:
    """Simple file-based cache for scraped data."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def get(self, url: str) -> Optional[str]:
        """Get cached content for URL."""
        cache_file = self.cache_dir / f"{self._get_cache_key(url)}.html"
        if cache_file.exists():
            logger.debug(f"Cache hit for {url}")
            return cache_file.read_text(encoding='utf-8')
        return None
    
    def set(self, url: str, content: str) -> None:
        """Cache content for URL."""
        cache_file = self.cache_dir / f"{self._get_cache_key(url)}.html"
        cache_file.write_text(content, encoding='utf-8')
        logger.debug(f"Cached {url}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        for f in self.cache_dir.glob("*.html"):
            f.unlink()
        logger.info("Cache cleared")


class UserAgentRotator:
    """Rotate user agents to avoid detection."""
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    def __init__(self):
        self.index = 0
    
    def get_random(self) -> str:
        """Get a random user agent."""
        return random.choice(self.USER_AGENTS)
    
    def get_next(self) -> str:
        """Get the next user agent in rotation."""
        ua = self.USER_AGENTS[self.index]
        self.index = (self.index + 1) % len(self.USER_AGENTS)
        return ua


def clean_text(text: str) -> str:
    """Clean and normalize text from web pages."""
    if not text:
        return ""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    return text.strip()


def extract_cas_number(text: str) -> Optional[str]:
    """Extract CAS registry number from text."""
    import re
    # CAS format: 1-7 digits, hyphen, 2 digits, hyphen, 1 digit
    pattern = r'\b(\d{1,7}-\d{2}-\d)\b'
    match = re.search(pattern, text)
    return match.group(1) if match else None


def extract_molecular_formula(text: str) -> Optional[str]:
    """Extract molecular formula from text."""
    import re
    # Match patterns like C10H16O, CH3COOH
    pattern = r'\b([A-Z][a-z]?\d*)+\b'
    matches = re.findall(pattern, text)
    for match in matches:
        # Validate it looks like a formula
        if any(c.isupper() for c in match) and any(c.isdigit() for c in match):
            return match
    return None


def save_checkpoint(data: List[Dict], checkpoint_file: str) -> None:
    """Save scraping progress to checkpoint file."""
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved checkpoint with {len(data)} items")


def load_checkpoint(checkpoint_file: str) -> List[Dict]:
    """Load scraping progress from checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded checkpoint with {len(data)} items")
        return data
    return []


def validate_url(url: str) -> bool:
    """Validate if a URL is properly formatted."""
    import re
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(pattern.match(url))

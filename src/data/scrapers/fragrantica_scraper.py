"""
Scraper for Fragrantica perfume database.
Extracts perfume names, notes, accords, descriptions, and reviews.
"""
import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FragranticaScraper:
    """Scraper for Fragrantica perfume data."""
    
    def __init__(self, base_url="https://www.fragrantica.com", rate_limit=2.0):
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/120.0.0.0 Safari/537.36'),
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
    
    def scrape_perfume(self, perfume_id):
        """Scrape single perfume data by ID."""
        url = f"{self.base_url}/perfume/-/{perfume_id}.html"
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Request failed for perfume {perfume_id}: {e}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        data = {
            'id': perfume_id,
            'name': self._extract_name(soup),
            'brand': self._extract_brand(soup),
            'notes': self._extract_notes(soup),
            'top_notes': self._extract_notes_by_type(soup, 'top'),
            'middle_notes': self._extract_notes_by_type(soup, 'middle'),
            'base_notes': self._extract_notes_by_type(soup, 'base'),
            'description': self._extract_description(soup),
            'accords': self._extract_accords(soup),
            'reviews': self._extract_reviews(soup),
            'url': url,
        }
        
        return data
    
    def scrape_batch(self, start_id, end_id, output_file):
        """Scrape multiple perfumes and save to JSON."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results = []
        
        for perfume_id in tqdm(range(start_id, end_id), desc="Scraping perfumes"):
            try:
                data = self.scrape_perfume(perfume_id)
                if data and data['name']:
                    results.append(data)
                time.sleep(self.rate_limit)
            except Exception as e:
                logger.error(f"Error scraping {perfume_id}: {e}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scraped {len(results)} perfumes, saved to {output_file}")
        return results
    
    # --- Extraction helpers ---
    
    def _extract_name(self, soup):
        """Extract perfume name from page."""
        tag = soup.find('h1')
        if tag:
            # Fragrantica h1 usually has "Name by Brand"
            text = tag.get_text(strip=True)
            # Split on " by " to get just the name
            parts = text.split(' by ')
            return parts[0].strip() if parts else text
        
        # Fallback: look for meta og:title
        meta = soup.find('meta', property='og:title')
        if meta and meta.get('content'):
            return meta['content'].split(' by ')[0].strip()
        
        return None
    
    def _extract_brand(self, soup):
        """Extract brand/house name."""
        tag = soup.find('h1')
        if tag:
            text = tag.get_text(strip=True)
            parts = text.split(' by ')
            if len(parts) > 1:
                return parts[1].strip()
        
        # Look for brand link
        brand_link = soup.find('a', href=re.compile(r'/designers/'))
        if brand_link:
            return brand_link.get_text(strip=True)
        
        return None
    
    def _extract_notes(self, soup):
        """Extract all fragrance notes."""
        notes = set()
        
        # Look for note links in pyramid
        note_links = soup.find_all('a', href=re.compile(r'/notes/'))
        for link in note_links:
            note_text = link.get_text(strip=True)
            if note_text and len(note_text) > 1:
                notes.add(note_text)
        
        # Fallback: look for spans with note class
        note_spans = soup.find_all('span', class_=re.compile(r'note', re.I))
        for span in note_spans:
            text = span.get_text(strip=True)
            if text and len(text) > 1:
                notes.add(text)
        
        return list(notes) if notes else []
    
    def _extract_notes_by_type(self, soup, note_type):
        """Extract notes by pyramid position (top/middle/base)."""
        notes = []
        
        # Look for pyramid divs
        pyramid = soup.find('div', id=re.compile(f'{note_type}', re.I))
        if not pyramid:
            pyramid = soup.find('div', class_=re.compile(f'{note_type}', re.I))
        
        if pyramid:
            links = pyramid.find_all('a', href=re.compile(r'/notes/'))
            for link in links:
                text = link.get_text(strip=True)
                if text:
                    notes.append(text)
        
        return notes
    
    def _extract_description(self, soup):
        """Extract perfume description text."""
        # Look for description div
        desc_div = soup.find('div', itemprop='description')
        if desc_div:
            return desc_div.get_text(separator=' ', strip=True)
        
        # Look for main content paragraphs
        content = soup.find('div', class_=re.compile(r'description|content|main', re.I))
        if content:
            paras = content.find_all('p')
            if paras:
                return ' '.join(p.get_text(strip=True) for p in paras[:3])
        
        # Fallback: meta description
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta and meta.get('content'):
            return meta['content']
        
        return ""
    
    def _extract_accords(self, soup):
        """Extract fragrance accords (e.g., citrus, woody, floral)."""
        accords = []
        
        # Look for accord bars
        accord_divs = soup.find_all('div', class_=re.compile(r'accord', re.I))
        for div in accord_divs:
            text = div.get_text(strip=True)
            if text and len(text) > 1 and len(text) < 30:
                accords.append(text)
        
        # Look for accord spans
        accord_spans = soup.find_all('span', class_=re.compile(r'accord', re.I))
        for span in accord_spans:
            text = span.get_text(strip=True)
            if text and len(text) > 1:
                accords.append(text)
        
        return accords
    
    def _extract_reviews(self, soup, max_reviews=20):
        """Extract user reviews/comments."""
        reviews = []
        
        # Look for review containers
        review_divs = soup.find_all('div', class_=re.compile(r'review|comment', re.I))
        for div in review_divs[:max_reviews]:
            text = div.get_text(separator=' ', strip=True)
            if text and len(text) > 10:
                reviews.append(text[:500])  # Cap length
        
        # Fallback: look for p tags in review section
        if not reviews:
            review_section = soup.find('div', id=re.compile(r'review', re.I))
            if review_section:
                for p in review_section.find_all('p')[:max_reviews]:
                    text = p.get_text(strip=True)
                    if text and len(text) > 10:
                        reviews.append(text[:500])
        
        return reviews


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    scraper = FragranticaScraper()
    
    # Test with a single perfume
    result = scraper.scrape_perfume(1)
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Could not scrape perfume.")

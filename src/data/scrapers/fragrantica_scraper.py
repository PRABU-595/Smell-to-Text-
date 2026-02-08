"""
Scraper for Fragrantica perfume database
"""
import requests
from bs4 import BeautifulSoup
import json
import time
from tqdm import tqdm

class FragranticaScraper:
    def __init__(self, base_url="https://www.fragrantica.com"):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
    
    def scrape_perfume(self, perfume_id):
        """Scrape single perfume data"""
        url = f"{self.base_url}/perfume/{perfume_id}"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        data = {
            'id': perfume_id,
            'name': self._extract_name(soup),
            'brand': self._extract_brand(soup),
            'notes': self._extract_notes(soup),
            'description': self._extract_description(soup),
            'accords': self._extract_accords(soup),
            'reviews': self._extract_reviews(soup)
        }
        
        return data
    
    def scrape_batch(self, start_id, end_id, output_file):
        """Scrape multiple perfumes"""
        results = []
        for perfume_id in tqdm(range(start_id, end_id)):
            try:
                data = self.scrape_perfume(perfume_id)
                results.append(data)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error scraping {perfume_id}: {e}")
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    # Helper methods
    def _extract_name(self, soup): ...
    def _extract_brand(self, soup): ...
    def _extract_notes(self, soup): ...

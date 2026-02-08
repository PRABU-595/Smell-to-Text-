"""
Scraper for The Good Scents Company database
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from tqdm import tqdm
from typing import Dict, List, Optional
import json


class GoodScentsScraper:
    """Scraper for thegoodscentscompany.com chemical database."""
    
    def __init__(self, base_url: str = "http://www.thegoodscentscompany.com"):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        self.chemicals_data = []
    
    def get_chemical_page(self, chemical_id: str) -> Optional[BeautifulSoup]:
        """Fetch a single chemical page."""
        url = f"{self.base_url}/data/rw{chemical_id}.html"
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
            return None
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_chemical_data(self, soup: BeautifulSoup, chemical_id: str) -> Dict:
        """Extract chemical information from page."""
        data = {
            'id': chemical_id,
            'name': None,
            'cas_number': None,
            'formula': None,
            'molecular_weight': None,
            'odor_description': None,
            'odor_type': None,
            'taste_description': None,
            'flavor_type': None,
            'occurrence': [],
            'synonyms': []
        }
        
        # Extract name
        title = soup.find('title')
        if title:
            data['name'] = title.text.strip().split(' - ')[0]
        
        # Extract from tables
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    label = cells[0].text.strip().lower()
                    value = cells[1].text.strip()
                    
                    if 'cas' in label:
                        data['cas_number'] = value
                    elif 'formula' in label:
                        data['formula'] = value
                    elif 'molecular weight' in label:
                        data['molecular_weight'] = value
                    elif 'odor' in label and 'type' in label:
                        data['odor_type'] = value
                    elif 'odor' in label:
                        data['odor_description'] = value
                    elif 'taste' in label and 'type' not in label:
                        data['taste_description'] = value
                    elif 'flavor' in label:
                        data['flavor_type'] = value
        
        return data
    
    def scrape_chemical_list(self, letter: str) -> List[str]:
        """Get list of chemical IDs starting with a letter."""
        url = f"{self.base_url}/odor/{letter}.html"
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            chemical_ids = []
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                match = re.search(r'rw(\d+)\.html', href)
                if match:
                    chemical_ids.append(match.group(1))
            
            return list(set(chemical_ids))
        except requests.RequestException as e:
            print(f"Error fetching chemical list for {letter}: {e}")
            return []
    
    def scrape_all_chemicals(self, output_file: str, delay: float = 1.0) -> pd.DataFrame:
        """Scrape all chemicals from A-Z."""
        all_data = []
        
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            print(f"\nScraping chemicals starting with '{letter.upper()}'...")
            chemical_ids = self.scrape_chemical_list(letter)
            
            for chem_id in tqdm(chemical_ids, desc=f"Letter {letter.upper()}"):
                soup = self.get_chemical_page(chem_id)
                if soup:
                    data = self.extract_chemical_data(soup, chem_id)
                    if data['name'] and data['odor_description']:
                        all_data.append(data)
                time.sleep(delay)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        
        print(f"\nScraped {len(df)} chemicals with odor descriptions")
        return df
    
    def scrape_sample(self, n_samples: int = 100, output_file: str = None) -> pd.DataFrame:
        """Scrape a sample of chemicals for testing."""
        all_ids = []
        for letter in 'abc':  # Just first few letters
            ids = self.scrape_chemical_list(letter)
            all_ids.extend(ids[:n_samples // 3])
        
        all_data = []
        for chem_id in tqdm(all_ids[:n_samples]):
            soup = self.get_chemical_page(chem_id)
            if soup:
                data = self.extract_chemical_data(soup, chem_id)
                if data['name']:
                    all_data.append(data)
            time.sleep(0.5)
        
        df = pd.DataFrame(all_data)
        if output_file:
            df.to_csv(output_file, index=False)
        
        return df


if __name__ == '__main__':
    scraper = GoodScentsScraper()
    
    # Test with sample
    print("Testing scraper with sample...")
    df = scraper.scrape_sample(n_samples=10)
    print(df.head())

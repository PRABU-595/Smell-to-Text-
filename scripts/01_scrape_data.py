#!/usr/bin/env python3
"""
Script to scrape perfume data from Fragrantica and chemical data
"""
import sys
sys.path.append('.')

from src.data.scrapers.fragrantica_scraper import FragranticaScraper
from src.data.scrapers.pubchem_api import PubChemAPI
import yaml

def main():
    # Load config
    with open('configs/scraping_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Scrape Fragrantica
    print("Scraping Fragrantica...")
    scraper = FragranticaScraper()
    perfumes = scraper.scrape_batch(
        start_id=config['fragrantica']['start_id'],
        end_id=config['fragrantica']['end_id'],
        output_file='data/raw/fragrantica/perfumes_raw.json'
    )
    print(f"Scraped {len(perfumes)} perfumes")
    
    # Get chemical data from PubChem
    print("\nFetching chemical data...")
    pubchem = PubChemAPI()
    # ... implementation
    
    print("Data scraping complete!")

if __name__ == '__main__':
    main()

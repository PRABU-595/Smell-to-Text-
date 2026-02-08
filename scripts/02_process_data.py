#!/usr/bin/env python3
"""Script to process raw data into training format."""
import sys
sys.path.append('.')

import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.preprocessing.text_cleaner import TextCleaner
from src.data.preprocessing.chemical_mapper import ChemicalMapper

def main():
    print("Processing data...")
    
    # Load raw data
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    processed_dir.mkdir(exist_ok=True)
    
    cleaner = TextCleaner()
    mapper = ChemicalMapper()
    
    samples = []
    
    # Process Fragrantica data
    fragrantica_file = raw_dir / 'fragrantica/perfumes_raw.json'
    if fragrantica_file.exists():
        with open(fragrantica_file) as f:
            perfumes = json.load(f)
        
        for p in perfumes:
            desc = p.get('description', '') or ' '.join(p.get('notes', []))
            if desc:
                cleaned = cleaner.clean_description(desc)
                chemicals = mapper.map_description(cleaned)
                if chemicals:
                    samples.append({'description': cleaned, 'chemicals': json.dumps(list(chemicals.keys()))})
    
    if not samples:
        print("No data found. Creating sample data...")
        samples = [
            {'description': 'Fresh citrus bergamot lemon', 'chemicals': json.dumps(['Limonene', 'Citral'])},
            {'description': 'Warm woody sandalwood cedar', 'chemicals': json.dumps(['Santalol', 'Cedrene'])},
            {'description': 'Sweet vanilla caramel', 'chemicals': json.dumps(['Vanillin', 'Maltol'])},
        ]
    
    df = pd.DataFrame(samples)
    
    # Split data
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    train.to_csv(processed_dir / 'train.csv', index=False)
    val.to_csv(processed_dir / 'val.csv', index=False)
    test.to_csv(processed_dir / 'test.csv', index=False)
    
    # Save statistics
    stats = {'total': len(df), 'train': len(train), 'val': len(val), 'test': len(test)}
    with open(processed_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Processed {len(df)} samples: train={len(train)}, val={len(val)}, test={len(test)}")

if __name__ == '__main__':
    main()

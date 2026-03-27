#!/usr/bin/env python3
"""
Script 02: Process raw data into train/val/test splits.

Reads the generated/scraped JSON data, cleans descriptions,
maps chemicals to indices, and saves as CSV files ready for training.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.dataset import CHEMICAL_LIST, CHEMICAL_TO_IDX


def main():
    print("=" * 60)
    print("Step 2: Processing raw data into training format")
    print("=" * 60)
    
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Load raw data ----
    raw_file = raw_dir / 'fragrantica' / 'perfumes_raw.json'
    if not raw_file.exists():
        print(f"ERROR: Raw data not found at {raw_file}")
        print("Run  python scripts/01_scrape_data.py  first.")
        sys.exit(1)
    
    with open(raw_file, 'r', encoding='utf-8') as f:
        raw_samples = json.load(f)
    
    print(f"Loaded {len(raw_samples)} raw samples")
    
    # ---- Clean & transform ----
    rows = []
    skipped = 0
    for sample in raw_samples:
        desc = sample.get('description', '').strip()
        chems = sample.get('chemicals', [])
        
        # Skip empty descriptions
        if not desc or len(desc) < 5:
            skipped += 1
            continue
        
        # Keep only chemicals that are in our index
        valid_chems = []
        for c in chems:
            name = c if isinstance(c, str) else c.get('name', '')
            if name in CHEMICAL_TO_IDX:
                if isinstance(c, dict):
                    valid_chems.append({'name': name, 'weight': c.get('weight', 1.0)})
                else:
                    valid_chems.append({'name': name, 'weight': 1.0})
        
        if not valid_chems:
            skipped += 1
            continue
        
        # Clean description
        desc = desc.replace('\n', ' ').replace('\r', ' ')
        desc = ' '.join(desc.split())  # normalize whitespace
        
        rows.append({
            'description': desc,
            'chemicals': json.dumps(valid_chems),
        })
    
    print(f"Valid samples: {len(rows)}  (skipped: {skipped})")
    
    if len(rows) < 10:
        print("ERROR: Too few valid samples!")
        sys.exit(1)
    
    df = pd.DataFrame(rows)
    
    # ---- Split data (70/15/15) ----
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    # Save
    train_df.to_csv(processed_dir / 'train.csv', index=False)
    val_df.to_csv(processed_dir / 'val.csv', index=False)
    test_df.to_csv(processed_dir / 'test.csv', index=False)
    
    # ---- Statistics ----
    stats = {
        'total_samples': len(df),
        'train': len(train_df),
        'val': len(val_df),
        'test': len(test_df),
        'num_chemicals': len(CHEMICAL_LIST),
        'chemical_list': CHEMICAL_LIST,
    }
    with open(processed_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Data saved to {processed_dir}/")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Chemical classes: {len(CHEMICAL_LIST)}")


if __name__ == '__main__':
    main()

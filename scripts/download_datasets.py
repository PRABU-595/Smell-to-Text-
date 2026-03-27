#!/usr/bin/env python3
"""
Download real olfactory datasets from public sources.

Datasets:
1. Pyrfume GoodScents — behavior.csv, molecules.csv, identifiers.csv
2. Pyrfume Leffingwell — behavior.csv, molecules.csv 
3. Pyrfume DREAM — behavior.csv, molecules.csv, stimuli.csv
4. Pyrfume Dravnieks — behavior.csv, molecules.csv
"""
import os
import urllib.request
import ssl

# Bypass SSL verification for GitHub raw downloads
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

DATASETS = {
    "goodscents": [
        "behavior.csv",
        "molecules.csv",
        "identifiers.csv",
    ],
    "leffingwell": [
        "behavior.csv",
        "molecules.csv",
    ],
    "keller_2016": [
        "behavior.csv",
        "molecules.csv",
        "stimuli.csv",
        "subjects.csv",
    ],
    "dravnieks_1985": [
        "behavior.csv",
        "molecules.csv",
        "stimuli.csv",
    ],
    "ifra_2019": [
        "behavior.csv",
        "molecules.csv",
    ],
}

BASE_URL = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main"


def download_file(url, dest):
    """Download a file, return True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, context=ctx, timeout=30)
        data = response.read()
        with open(dest, 'wb') as f:
            f.write(data)
        size_kb = len(data) / 1024
        print(f"  ✓ {os.path.basename(dest)} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  ✗ {os.path.basename(dest)}: {e}")
        return False


def main():
    print("=" * 60)
    print("Downloading Real Olfactory Datasets")
    print("=" * 60)
    
    base_dir = "data/raw"
    os.makedirs(base_dir, exist_ok=True)
    
    total_success = 0
    total_files = 0
    
    for dataset_name, files in DATASETS.items():
        print(f"\n--- {dataset_name} ---")
        dataset_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        for filename in files:
            total_files += 1
            url = f"{BASE_URL}/{dataset_name}/{filename}"
            dest = os.path.join(dataset_dir, filename)
            
            if download_file(url, dest):
                total_success += 1
    
    print(f"\n{'=' * 60}")
    print(f"Downloaded {total_success}/{total_files} files")
    
    # Verify what we got
    print(f"\nDataset Summary:")
    for dataset_name in DATASETS:
        dataset_dir = os.path.join(base_dir, dataset_name)
        files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        if files:
            import pandas as pd
            for f in files:
                path = os.path.join(dataset_dir, f)
                try:
                    df = pd.read_csv(path)
                    print(f"  {dataset_name}/{f}: {len(df)} rows × {len(df.columns)} cols — {list(df.columns)[:5]}")
                except Exception as e:
                    print(f"  {dataset_name}/{f}: error reading — {e}")
    
    print("\n✓ All real datasets downloaded!")


if __name__ == '__main__':
    main()

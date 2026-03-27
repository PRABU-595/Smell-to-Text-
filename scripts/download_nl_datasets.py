#!/usr/bin/env python3
"""
Download 4 real-world natural language smell datasets.
These contain REAL human descriptions, not scientific notation.

Sources:
1. Fragrantica (Kaggle) - 36,969 perfume reviews with accords & notes
2. Perfume Recommendation (Kaggle) - 2,191 natural text descriptions
3. Laymen Olfactory Perception (Zenodo) - 1,227 people x 74 odors, free-text
4. Aromo.ru (Kaggle) - 78,410 perfumes with fragrance families
"""
import os, sys, urllib.request, ssl, zipfile, json
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

BASE = Path('data/raw/natural_language')
BASE.mkdir(parents=True, exist_ok=True)


def download_file(url, dest):
    """Download a file with progress."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return True
    print(f"  Downloading: {url[:80]}...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"  ✓ Saved: {dest} ({dest.stat().st_size / 1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_zenodo_laymen():
    """3. Laymen Olfactory Perception - Zenodo (no auth needed)."""
    print("\n--- Dataset 3: Laymen Olfactory Perception (Zenodo) ---")
    out_dir = BASE / 'laymen_olfactory'
    out_dir.mkdir(exist_ok=True)
    
    # Zenodo DOI: 10.5281/zenodo.14727277
    # Try the Zenodo API to get download links
    api_url = "https://zenodo.org/api/records/14727277"
    try:
        with urllib.request.urlopen(api_url) as resp:
            data = json.loads(resp.read())
            files = data.get('files', [])
            print(f"  Found {len(files)} files on Zenodo")
            for f in files:
                fname = f['key']
                dl_url = f['links']['self']
                dest = out_dir / fname
                download_file(dl_url, dest)
    except Exception as e:
        print(f"  ✗ Zenodo API failed: {e}")
        # Fallback: direct download
        fallback = f"https://zenodo.org/records/14727277/files/data.zip?download=1"
        download_file(fallback, out_dir / 'data.zip')
    
    # Extract if zip
    for z in out_dir.glob('*.zip'):
        print(f"  Extracting {z.name}...")
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(out_dir)
        print(f"  ✓ Extracted")


def download_kaggle_datasets():
    """Download Kaggle datasets using opendatasets (prompts for credentials)."""
    try:
        import opendatasets as od
    except ImportError:
        print("  ✗ opendatasets not installed. Run: pip install opendatasets")
        return
    
    datasets = [
        {
            'name': 'Fragrantica Perfume Reviews',
            'url': 'https://www.kaggle.com/datasets/miufana1/fragranticacom-fragrance-dataset',
            'dir': BASE / 'fragrantica',
        },
        {
            'name': 'Perfume Recommendation Dataset',
            'url': 'https://www.kaggle.com/datasets/nandini1999/perfume-recommendation-dataset',
            'dir': BASE / 'perfume_recommendation',
        },
        {
            'name': 'Aromo.ru Fragrance Dataset',
            'url': 'https://www.kaggle.com/datasets/olgagmiufana1/aromoRu-fragrance-dataset',
            'dir': BASE / 'aromo',
        },
    ]
    
    for ds in datasets:
        print(f"\n--- Dataset: {ds['name']} ---")
        if ds['dir'].exists() and any(ds['dir'].glob('*.csv')):
            print(f"  Already downloaded: {ds['dir']}")
            continue
        try:
            od.download(ds['url'], data_dir=str(BASE))
            print(f"  ✓ Downloaded to {ds['dir']}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print(f"  Manual download: {ds['url']}")


def main():
    print("=" * 60)
    print("Downloading Natural Language Smell Datasets")
    print("=" * 60)
    
    # 1. Zenodo (no auth needed)
    download_zenodo_laymen()
    
    # 2. Kaggle datasets (may need credentials)
    print("\n" + "=" * 60)
    print("Kaggle Datasets")
    print("If prompted, enter your Kaggle username and API key.")
    print("Get your API key from: https://www.kaggle.com/settings")
    print("=" * 60)
    download_kaggle_datasets()
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for d in BASE.iterdir():
        if d.is_dir():
            files = list(d.glob('*'))
            csvs = list(d.glob('*.csv'))
            print(f"  {d.name}: {len(files)} files ({len(csvs)} CSVs)")
    
    print("\n✓ Download complete!")


if __name__ == '__main__':
    main()

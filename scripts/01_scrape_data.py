#!/usr/bin/env python3
"""
Script 01: Generate / scrape smell-to-molecule dataset.

Because web scraping depends on external site availability, this script
includes a comprehensive synthetic dataset generator that creates
realistic smell-description → chemical-compound pairs based on domain
knowledge.  It can also attempt live scraping if requested.

Usage:
    python scripts/01_scrape_data.py              # Generate synthetic data
    python scripts/01_scrape_data.py --live        # Attempt live scraping
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import random
import argparse
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------------
# Domain knowledge: smell descriptors → chemicals
# ---------------------------------------------------------------------------

SMELL_CHEMICAL_MAP = {
    # Citrus family
    'citrus': [
        {'name': 'Limonene', 'formula': 'C10H16'},
        {'name': 'Citral', 'formula': 'C10H16O'},
        {'name': 'Linalool', 'formula': 'C10H18O'},
    ],
    'lemon': [
        {'name': 'Limonene', 'formula': 'C10H16'},
        {'name': 'Citral', 'formula': 'C10H16O'},
        {'name': 'Neral', 'formula': 'C10H16O'},
    ],
    'orange': [
        {'name': 'Limonene', 'formula': 'C10H16'},
        {'name': 'Myrcene', 'formula': 'C10H16'},
        {'name': 'Linalool', 'formula': 'C10H18O'},
    ],
    'bergamot': [
        {'name': 'Linalyl acetate', 'formula': 'C12H20O2'},
        {'name': 'Limonene', 'formula': 'C10H16'},
        {'name': 'Bergamot oil', 'formula': 'mixture'},
    ],
    'grapefruit': [
        {'name': 'Nootkatone', 'formula': 'C15H22O'},
        {'name': 'Limonene', 'formula': 'C10H16'},
        {'name': 'Myrcene', 'formula': 'C10H16'},
    ],
    # Floral family
    'floral': [
        {'name': 'Linalool', 'formula': 'C10H18O'},
        {'name': 'Geraniol', 'formula': 'C10H18O'},
        {'name': 'Phenylethyl alcohol', 'formula': 'C8H10O'},
    ],
    'rose': [
        {'name': 'Geraniol', 'formula': 'C10H18O'},
        {'name': 'Citronellol', 'formula': 'C10H20O'},
        {'name': 'Phenylethyl alcohol', 'formula': 'C8H10O'},
    ],
    'jasmine': [
        {'name': 'Benzyl acetate', 'formula': 'C9H10O2'},
        {'name': 'Jasmone', 'formula': 'C11H16O'},
        {'name': 'Indole', 'formula': 'C8H7N'},
    ],
    'lavender': [
        {'name': 'Linalool', 'formula': 'C10H18O'},
        {'name': 'Linalyl acetate', 'formula': 'C12H20O2'},
        {'name': 'Camphor', 'formula': 'C10H16O'},
    ],
    'violet': [
        {'name': 'Ionone', 'formula': 'C13H20O'},
        {'name': 'Methyl ionone', 'formula': 'C14H22O'},
    ],
    'lily': [
        {'name': 'Hydroxycitronellal', 'formula': 'C10H20O2'},
        {'name': 'Linalool', 'formula': 'C10H18O'},
    ],
    # Woody family
    'woody': [
        {'name': 'Cedrene', 'formula': 'C15H24'},
        {'name': 'Santalol', 'formula': 'C15H24O'},
        {'name': 'Vetiverol', 'formula': 'C15H26O'},
    ],
    'cedar': [
        {'name': 'Cedrene', 'formula': 'C15H24'},
        {'name': 'Cedrol', 'formula': 'C15H26O'},
        {'name': 'Cedarwood oil', 'formula': 'mixture'},
    ],
    'sandalwood': [
        {'name': 'Santalol', 'formula': 'C15H24O'},
        {'name': 'Sandalwood oil', 'formula': 'mixture'},
        {'name': 'Santene', 'formula': 'C9H14'},
    ],
    'pine': [
        {'name': 'Pinene', 'formula': 'C10H16'},
        {'name': 'Bornyl acetate', 'formula': 'C12H20O2'},
        {'name': 'Limonene', 'formula': 'C10H16'},
    ],
    # Sweet family
    'vanilla': [
        {'name': 'Vanillin', 'formula': 'C8H8O3'},
        {'name': 'Ethyl vanillin', 'formula': 'C9H10O3'},
    ],
    'caramel': [
        {'name': 'Furaneol', 'formula': 'C6H8O3'},
        {'name': 'Maltol', 'formula': 'C6H6O3'},
        {'name': 'Cyclotene', 'formula': 'C6H8O2'},
    ],
    'honey': [
        {'name': 'Phenylacetic acid', 'formula': 'C8H8O2'},
        {'name': 'Methylphenylacetate', 'formula': 'C9H10O2'},
    ],
    # Spicy family
    'spicy': [
        {'name': 'Eugenol', 'formula': 'C10H12O2'},
        {'name': 'Cinnamaldehyde', 'formula': 'C9H8O'},
        {'name': 'Carvone', 'formula': 'C10H14O'},
    ],
    'cinnamon': [
        {'name': 'Cinnamaldehyde', 'formula': 'C9H8O'},
        {'name': 'Eugenol', 'formula': 'C10H12O2'},
    ],
    'clove': [
        {'name': 'Eugenol', 'formula': 'C10H12O2'},
        {'name': 'Beta-caryophyllene', 'formula': 'C15H24'},
    ],
    'pepper': [
        {'name': 'Piperine', 'formula': 'C17H19NO3'},
        {'name': 'Rotundone', 'formula': 'C15H22O'},
    ],
    'ginger': [
        {'name': 'Gingerol', 'formula': 'C17H26O4'},
        {'name': 'Zingiberene', 'formula': 'C15H24'},
    ],
    # Fresh / aquatic
    'fresh': [
        {'name': 'Dihydromyrcenol', 'formula': 'C10H20O'},
        {'name': 'Calone', 'formula': 'C11H14O3'},
        {'name': 'Linalool', 'formula': 'C10H18O'},
    ],
    'marine': [
        {'name': 'Calone', 'formula': 'C11H14O3'},
        {'name': 'Helional', 'formula': 'C9H10O3'},
    ],
    'aquatic': [
        {'name': 'Calone', 'formula': 'C11H14O3'},
        {'name': 'Floralozone', 'formula': 'C12H15NO'},
    ],
    # Herbal
    'mint': [
        {'name': 'Menthol', 'formula': 'C10H20O'},
        {'name': 'Menthone', 'formula': 'C10H18O'},
    ],
    'herbal': [
        {'name': 'Thymol', 'formula': 'C10H14O'},
        {'name': 'Carvacrol', 'formula': 'C10H14O'},
        {'name': 'Linalool', 'formula': 'C10H18O'},
    ],
    'rosemary': [
        {'name': 'Camphor', 'formula': 'C10H16O'},
        {'name': 'Cineole', 'formula': 'C10H18O'},
        {'name': 'Pinene', 'formula': 'C10H16'},
    ],
    # Musky / amber
    'musk': [
        {'name': 'Muscone', 'formula': 'C16H30O'},
        {'name': 'Galaxolide', 'formula': 'C18H26O'},
        {'name': 'Ambrettolide', 'formula': 'C16H28O2'},
    ],
    'amber': [
        {'name': 'Ambroxide', 'formula': 'C16H28O'},
        {'name': 'Labdanum', 'formula': 'mixture'},
    ],
    'powdery': [
        {'name': 'Heliotropin', 'formula': 'C8H6O3'},
        {'name': 'Ionone', 'formula': 'C13H20O'},
        {'name': 'Coumarin', 'formula': 'C9H6O2'},
    ],
    # Fruity
    'fruity': [
        {'name': 'Ethyl butyrate', 'formula': 'C6H12O2'},
        {'name': 'Isoamyl acetate', 'formula': 'C7H14O2'},
    ],
    'peach': [
        {'name': 'Gamma-decalactone', 'formula': 'C10H18O2'},
        {'name': 'Gamma-undecalactone', 'formula': 'C11H20O2'},
    ],
    'apple': [
        {'name': 'Ethyl-2-methylbutyrate', 'formula': 'C7H14O2'},
        {'name': 'Hexyl acetate', 'formula': 'C8H16O2'},
    ],
    'banana': [
        {'name': 'Isoamyl acetate', 'formula': 'C7H14O2'},
        {'name': 'Amyl acetate', 'formula': 'C7H14O2'},
    ],
    'tropical': [
        {'name': 'Allyl hexanoate', 'formula': 'C9H16O2'},
        {'name': 'Ethyl butyrate', 'formula': 'C6H12O2'},
    ],
}

# Description templates (slot-based generation)
TEMPLATES = [
    "A {adj1}, {adj2} fragrance with {note1} and {note2} notes",
    "{adj1} {note1} with {adj2} {note2} undertones",
    "Bright and {adj1}, featuring {note1}, {note2}, and a touch of {note3}",
    "A {adj1} scent blending {note1} with {note2}",
    "{adj1} and {adj2}, reminiscent of {note1} with hints of {note2}",
    "Warm {note1} combined with {adj1} {note2} and {note3}",
    "{note1} forward, with a {adj1} {note2} dry-down",
    "A complex {adj1} aroma of {note1}, {note2}, and {note3}",
    "Delicate {note1} paired with {adj1} {note2}",
    "Rich {note1} and {note2} with a {adj1} character",
    "An elegant blend of {note1} and {note2}, {adj1} and {adj2}",
    "{adj1} {note1} opening, settling into {adj2} {note2}",
    "A {adj1} composition featuring {note1}, layered with {note2}",
    "Sparkling {note1} with {adj1} {note2} in the heart",
    "Smooth {note1} and {adj1} {note2} intertwined beautifully",
    "A {adj1} interpretation of {note1} and {note2}",
    "{adj1}, {adj2} {note1} with a {note2} base",
    "Classic {note1} enriched with {adj1} {note2} accents",
    "An invigorating blend of {adj1} {note1} and {note2}",
    "Subtle {note1} with {adj1} {note2} overtones and {note3} finish",
]

ADJECTIVES = {
    'citrus':    ['zesty', 'bright', 'sparkling', 'crisp', 'tangy', 'vibrant'],
    'floral':    ['delicate', 'soft', 'elegant', 'romantic', 'lush', 'blooming'],
    'woody':     ['warm', 'earthy', 'deep', 'smooth', 'rich', 'grounding'],
    'sweet':     ['sweet', 'creamy', 'luscious', 'indulgent', 'decadent', 'gourmand'],
    'spicy':     ['warm', 'spicy', 'bold', 'fiery', 'exotic', 'aromatic'],
    'fresh':     ['fresh', 'clean', 'crisp', 'airy', 'light', 'invigorating'],
    'herbal':    ['green', 'aromatic', 'herbal', 'leafy', 'natural', 'crisp'],
    'musk':      ['sensual', 'warm', 'skin-like', 'intimate', 'soft', 'enveloping'],
    'fruity':    ['juicy', 'ripe', 'sweet', 'tropical', 'succulent', 'vibrant'],
}

# Map note names to their adjective family
NOTE_TO_FAMILY = {}
for family_notes in ['citrus', 'lemon', 'orange', 'bergamot', 'grapefruit']:
    NOTE_TO_FAMILY[family_notes] = 'citrus'
for family_notes in ['floral', 'rose', 'jasmine', 'lavender', 'violet', 'lily']:
    NOTE_TO_FAMILY[family_notes] = 'floral'
for family_notes in ['woody', 'cedar', 'sandalwood', 'pine']:
    NOTE_TO_FAMILY[family_notes] = 'woody'
for family_notes in ['vanilla', 'caramel', 'honey']:
    NOTE_TO_FAMILY[family_notes] = 'sweet'
for family_notes in ['spicy', 'cinnamon', 'clove', 'pepper', 'ginger']:
    NOTE_TO_FAMILY[family_notes] = 'spicy'
for family_notes in ['fresh', 'marine', 'aquatic']:
    NOTE_TO_FAMILY[family_notes] = 'fresh'
for family_notes in ['mint', 'herbal', 'rosemary']:
    NOTE_TO_FAMILY[family_notes] = 'herbal'
for family_notes in ['musk', 'amber', 'powdery']:
    NOTE_TO_FAMILY[family_notes] = 'musk'
for family_notes in ['fruity', 'peach', 'apple', 'banana', 'tropical']:
    NOTE_TO_FAMILY[family_notes] = 'fruity'


def _get_adj(note):
    family = NOTE_TO_FAMILY.get(note, 'fresh')
    return random.choice(ADJECTIVES.get(family, ['pleasant']))


def generate_sample(notes_combo):
    """Generate one training sample from a combination of notes."""
    template = random.choice(TEMPLATES)
    
    # Gather unique chemicals
    chemicals_dict = {}  # name -> formula
    for note in notes_combo:
        for chem in SMELL_CHEMICAL_MAP[note]:
            chemicals_dict[chem['name']] = chem['formula']
    
    # Assign weights that sum to ~1
    n = len(chemicals_dict)
    raw = [random.random() for _ in range(n)]
    total = sum(raw)
    weights = [round(w / total, 2) for w in raw]
    # Adjust last weight for rounding
    weights[-1] = round(1.0 - sum(weights[:-1]), 2)
    
    chemical_list = []
    for (name, formula), weight in zip(chemicals_dict.items(), weights):
        chemical_list.append({
            'name': name,
            'formula': formula,
            'weight': max(weight, 0.01),
        })
    
    # Build description
    adj1 = _get_adj(notes_combo[0])
    adj2 = _get_adj(notes_combo[-1]) if len(notes_combo) > 1 else _get_adj(notes_combo[0])
    note_names = list(notes_combo)
    description = template.format(
        adj1=adj1, adj2=adj2,
        note1=note_names[0],
        note2=note_names[1 % len(note_names)],
        note3=note_names[2 % len(note_names)],
    )
    
    return {
        'description': description,
        'chemicals': chemical_list,
        'notes': list(notes_combo),
    }


def generate_dataset(n_samples=5000, seed=42):
    """Generate a synthetic smell-to-molecule dataset."""
    random.seed(seed)
    all_notes = list(SMELL_CHEMICAL_MAP.keys())
    samples = []
    
    # Generate pairs (2-note combos)
    for combo in combinations(all_notes, 2):
        for _ in range(3):
            samples.append(generate_sample(combo))
    
    # Generate triples (3-note combos)
    for combo in combinations(all_notes, 3):
        samples.append(generate_sample(combo))
        if len(samples) >= n_samples * 2:
            break
    
    # Generate singles
    for note in all_notes:
        for _ in range(5):
            samples.append(generate_sample([note]))
    
    # Shuffle and limit
    random.shuffle(samples)
    samples = samples[:n_samples]
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate smell-to-molecule dataset")
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--live', action='store_true', help='Attempt live scraping')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'fragrantica').mkdir(exist_ok=True)
    
    if args.live:
        print("Attempting live scraping from Fragrantica...")
        try:
            from src.data.scrapers.fragrantica_scraper import FragranticaScraper
            scraper = FragranticaScraper()
            scraper.scrape_batch(
                start_id=1, end_id=100,
                output_file=str(output_dir / 'fragrantica/perfumes_raw.json')
            )
        except Exception as e:
            print(f"Live scraping failed: {e}")
            print("Falling back to synthetic data generation...")
    
    # Always generate synthetic data
    print(f"\nGenerating {args.n_samples} synthetic samples...")
    samples = generate_dataset(n_samples=args.n_samples, seed=args.seed)
    
    output_file = output_dir / 'fragrantica' / 'perfumes_raw.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated {len(samples)} samples → {output_file}")
    
    # Print statistics
    all_chems = set()
    for s in samples:
        for c in s['chemicals']:
            all_chems.add(c['name'])
    
    all_notes = set()
    for s in samples:
        all_notes.update(s['notes'])
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples:      {len(samples)}")
    print(f"  Unique chemicals:   {len(all_chems)}")
    print(f"  Unique note types:  {len(all_notes)}")
    print(f"  Avg description len: {sum(len(s['description']) for s in samples)/len(samples):.0f} chars")
    print(f"  Avg chemicals/sample: {sum(len(s['chemicals']) for s in samples)/len(samples):.1f}")


if __name__ == '__main__':
    main()

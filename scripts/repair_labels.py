import pandas as pd
import json
import numpy as np
from pathlib import Path
from src.data.chemical_vocab import CHEMICAL_TO_IDX, NUM_CHEMICALS

def main():
    print("="*60)
    print("NeoBERT Label Repair & Augmentation Pipeline")
    print("="*60)
    
    # 1. Load data
    train_path = Path('data/processed/train.csv')
    map_path = Path('data/processed/family_chemicals.json')
    
    if not train_path.exists() or not map_path.exists():
        print("ERROR: Missing data files.")
        return

    df = pd.read_csv(train_path)
    with open(map_path, 'r') as f:
        family_map = json.load(f)
    
    # 2. Build reverse map: CAS -> List of families
    reverse_map = {}
    for family, chems in family_map.items():
        for chem in chems:
            cas = chem.get('cas')
            if cas:
                if cas not in reverse_map:
                    reverse_map[cas] = set()
                # Normalize family name to match vocab
                reverse_map[cas].add(family)
    
    print(f"Total reverse-map entries: {len(reverse_map)} CAS numbers")

    # 3. Repair SILENT labels
    print("\nRepairing silent samples...")
    repaired_count = 0
    
    def repair_row(row):
        nonlocal repaired_count
        current_labels = json.loads(row['labels'])
        
        # If already has labels, keep them
        if sum(current_labels) > 0:
            return row['labels']
            
        # Try to find labels via CAS
        cas = str(row.get('cas_number', ''))
        if cas in reverse_map:
            new_labels = [0] * NUM_CHEMICALS
            for family in reverse_map[cas]:
                if family in CHEMICAL_TO_IDX:
                    new_labels[CHEMICAL_TO_IDX[family]] = 1
                    
            if sum(new_labels) > 0:
                repaired_count += 1
                return json.dumps(new_labels)
        
        return row['labels']

    df['labels'] = df.apply(repair_row, axis=1)

    # 4. Filter out any remaining silent samples (Optionally)
    # The user suggested "Balanced Dataset" - we should either label or remove
    final_labels = df['labels'].apply(lambda x: sum(json.loads(x)))
    silent_count = (final_labels == 0).sum()
    
    print(f"Successfully repaired {repaired_count} samples.")
    print(f"Remaining silent samples: {silent_count}")
    
    # 5. Semantic Merging (Phase 2, #6)
    print("\nApplying Semantic Merging...")
    
    merge_map = {
        'fresh_clean': 'fresh', 'fresh_citrus': 'fresh', 'fresh_green': 'fresh', 
        'fresh_ozonic': 'fresh', 'fresh_watery': 'fresh',
        'earthy_mushroom': 'earthy',
        'sweet_vanilla': 'sweet', 'sweet_caramel': 'sweet', 'sweet_honey': 'sweet',
        'spicy_warm': 'spicy',
        'woody_mossy': 'woody'
    }
    
    def merge_labels(label_str):
        labels = json.loads(label_str)
        new_labels = list(labels)
        for src, dst in merge_map.items():
            if src in CHEMICAL_TO_IDX and dst in CHEMICAL_TO_IDX:
                src_idx = CHEMICAL_TO_IDX[src]
                dst_idx = CHEMICAL_TO_IDX[dst]
                if labels[src_idx] == 1:
                    new_labels[dst_idx] = 1
                    # We can keep the src or zero it out. Keeping for multi-label.
        return json.dumps(new_labels)
    
    df['labels'] = df['labels'].apply(merge_labels)

    # 6. Save repaired data
    out_path = Path('data/processed/train_repaired.csv')
    df.to_csv(out_path, index=False)
    print(f"\n✓ Repaired dataset saved to {out_path}")
    
    # Generate statistics for weighted loss
    all_labels = np.array([json.loads(x) for x in df['labels']])
    pos_counts = all_labels.sum(axis=0)
    total = len(df)
    
    stats = {
        'pos_counts': pos_counts.tolist(),
        'pos_weights': ((total - pos_counts) / (pos_counts + 1e-6)).tolist(),
        'total_samples': total
    }
    with open('data/processed/repaired_stats.json', 'w') as f:
        json.dump(stats, f)
    print("✓ Weighted loss stats saved to repaired_stats.json")

if __name__ == "__main__":
    main()

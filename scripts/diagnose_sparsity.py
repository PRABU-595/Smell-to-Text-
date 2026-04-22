import pandas as pd
import json
from pathlib import Path

def main():
    train_path = Path('data/processed/train.csv')
    map_path = Path('data/processed/family_chemicals.json')
    
    if not train_path.exists() or not map_path.exists():
        print("Missing data files.")
        return

    df = pd.read_csv(train_path)
    with open(map_path, 'r') as f:
        family_map = json.load(f)
    
    # Build reverse map: CAS -> List of families
    reverse_map = {}
    for family, chems in family_map.items():
        for chem in chems:
            cas = chem.get('cas')
            if cas:
                if cas not in reverse_map:
                    reverse_map[cas] = []
                reverse_map[cas].append(family)
    
    total_samples = len(df)
    silent_samples = df[df['labels'] == '[]']
    silent_count = len(silent_samples)
    
    recoverable = 0
    for i, row in silent_samples.iterrows():
        cas = str(row.get('cas_number', ''))
        if cas in reverse_map:
            recoverable += 1
            
    print(f"Total training samples: {total_samples}")
    print(f"Silent samples (no labels): {silent_count} ({silent_count/total_samples*100:.1f}%)")
    print(f"Recoverable via internal mapping: {recoverable}")
    print(f"Remaining silent: {silent_count - recoverable}")

if __name__ == "__main__":
    main()

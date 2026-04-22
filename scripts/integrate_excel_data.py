#!/usr/bin/env python3
import pandas as pd
import json
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add root to path
sys.path.insert(0, os.path.abspath('.'))

from src.data.chemical_vocab import CHEMICAL_LIST, CHEMICAL_TO_IDX

def normalize_label(label):
    """Normalize label strings to match CHEMICAL_LIST."""
    # [x] Implement `scripts/repair_labels.py` to fix the 95.5% label sparsity (Integrated into integrate_excel_data.py)
    if pd.isna(label): return []
    label = str(label).lower().strip()
    # Handle synonyms or sub-mappings
    mapping = {
        'refreshing': 'fresh',
        'masculine': ['musky', 'woody'],
        'cool': 'fresh',
        'warm': 'spicy_warm',
        'deep': ['amber', 'balsamic'],
        'wood': 'woody',
        'spice': 'spicy',
        'fruit': 'fruity',
        'flower': 'floral',
    }
    
    results = []
    # Check direct match or split by comma/semicolon
    parts = [p.strip() for p in label.replace(',', ';').split(';')]
    for p in parts:
        if p in CHEMICAL_TO_IDX:
            results.append(p)
        elif p in mapping:
            m = mapping[p]
            if isinstance(m, list):
                results.extend(m)
            else:
                results.append(m)
        else:
            # Try partial matching for common prefixes
            for chem in CHEMICAL_LIST:
                if p in chem or chem in p:
                    results.append(chem)
                    break
                    
    return list(set(results))

def main():
    print("🚀 Starting Dataset Integration (Phase 17 - Revision 2)")
    
    # 1. Load Data 1.xlsx
    data1_path = r'C:\Users\iampr\Downloads\Data 1.xlsx'
    df1 = pd.read_excel(data1_path)
    print(f"Loaded Data 1.xlsx: {len(df1)} samples")
    
    new_rows = []
    for _, row in df1.iterrows():
        desc = row.get('First Impression', '')
        raw_fams = row.get('Odor Family ', '')
        fams = normalize_label(raw_fams)
        if desc and fams:
            new_rows.append({
                'description': desc,
                'families': json.dumps(fams),
                'molecule_id': f"Data1_{_}"
            })
            
    # 2. Load smell_dataset_300.xlsx
    smell300_path = r'C:\Users\iampr\Downloads\smell_dataset_300.xlsx'
    df2 = pd.read_excel(smell300_path, header=2)
    print(f"Loaded smell_dataset_300.xlsx: {len(df2)} samples")
    
    for idx, row in df2.iterrows():
        desc = row.get('NLP Description', '')
        if pd.isna(desc): desc = row.get('Smell Name', '')
        raw_fams = row.get('Odor Family Tags', '')
        fams = normalize_label(raw_fams)
        molecule = str(row.get('Chemical Compound (if known)', f"Smell300_{idx}"))
        
        if desc and fams:
            new_rows.append({
                'description': desc,
                'families': json.dumps(fams),
                'molecule_id': molecule
            })
            
    # 3. Load engage_survey_v2.xlsx
    engage_path = r'C:\Users\iampr\Downloads\engage_survey_v2.xlsx'
    if Path(engage_path).exists():
        df3 = pd.read_excel(engage_path, header=2)
        print(f"Loaded engage_survey_v2.xlsx: {len(df3)} samples")
        for idx, row in df3.iterrows():
            # Combine impression and comments
            imp = str(row.get('First Impression', '')).strip()
            comm = str(row.get('Comments / Free Response', '')).strip()
            desc = f"{imp}. {comm}".strip()
            
            raw_fams = row.get('Odor Family Detected', '')
            fams = normalize_label(raw_fams)
            
            if desc and fams:
                new_rows.append({
                    'description': desc,
                    'families': json.dumps(fams),
                    'molecule_id': f"Engage_{idx}"
                })

    # 4. Load existing dataset
    orig_path = Path('data/processed/real_smell_dataset.csv')
    if orig_path.exists():
        orig_df = pd.read_csv(orig_path)
        print(f"Existing dataset: {len(orig_df)} samples")
        
        # 5. Integrated Repair & Semantic Merging (Phase 17 - Revised)
        print("\n🛠️  Applying Hybrid Label Repair & Semantic Merging...")
        
        # Build reverse map for repair (expert database)
        reverse_map = {}
        map_path = Path('data/processed/family_chemicals.json')
        if map_path.exists():
            with open(map_path, 'r') as f:
                family_map = json.load(f)
            for family, chems in family_map.items():
                for chem in chems:
                    cas = chem.get('cas')
                    if cas:
                        if cas not in reverse_map: reverse_map[cas] = set()
                        reverse_map[cas].add(family)
                        
        def hybrid_labeling(row):
            current_fams = set()
            
            # 1. Expert Match via CAS (Extract from 'chemicals' JSON if possible)
            try:
                chem_data = json.loads(row.get('chemicals', '[]'))
                for item in chem_data:
                    name = item.get('name', '')
                    if 'Compound_' in name:
                        cas = name.split('Compound_')[1]
                        if cas in reverse_map:
                            current_fams.update(reverse_map[cas])
            except:
                pass

            # 2. Keyword Fallback (Critical for the 8,000 silent samples)
            desc = str(row.get('description', '')).lower()
            for chem_name in CHEMICAL_LIST:
                simple_name = chem_name.replace('_', ' ')
                if (len(simple_name) > 3 and simple_name in desc) or chem_name in desc:
                    current_fams.add(chem_name)
            
            # 3. Add any already identified families
            try:
                if 'families' in row and not pd.isna(row['families']):
                    current_fams.update(json.loads(row['families']))
            except:
                pass

            # 4. Semantic Merging (Phase 2, #6)
            merge_map = {
                'fresh_clean': 'fresh', 'fresh_citrus': 'fresh', 'fresh_green': 'fresh', 
                'fresh_ozonic': 'fresh', 'fresh_watery': 'fresh',
                'earthy_mushroom': 'earthy',
                'sweet_vanilla': 'sweet', 'sweet_caramel': 'sweet', 'sweet_honey': 'sweet',
                'spicy_warm': 'spicy',
                'woody_mossy': 'woody'
            }
            final_fams = set()
            for f in current_fams:
                final_fams.add(merge_map.get(f, f))
                
            return json.dumps(list(final_fams))

        orig_df['families'] = orig_df.apply(hybrid_labeling, axis=1)
        if 'molecule_id' not in orig_df.columns:
            orig_df['molecule_id'] = [f"orig_{i}" for i in range(len(orig_df))]
            
        orig_transformed = orig_df[['description', 'families', 'molecule_id']]
    else:
        print("Warning: Existing dataset not found.")
        orig_transformed = pd.DataFrame(columns=['description', 'families', 'molecule_id'])
        
    # 4. Combine
    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([orig_transformed, new_df], ignore_index=True)
    
    # 5. Integrated Repair & Semantic Merging (Phase 17 - Revised)
    print("\n🛠️  Applying Label Repair & Semantic Merging...")
    
    # Build reverse map for repair
    map_path = Path('data/processed/family_chemicals.json')
    if map_path.exists():
        with open(map_path, 'r') as f:
            family_map = json.load(f)
        reverse_map = {}
        for family, chems in family_map.items():
            for chem in chems:
                cas = chem.get('cas')
                if cas:
                    if cas not in reverse_map: reverse_map[cas] = set()
                    reverse_map[cas].add(family)
                    
        def repair_and_merge(row):
            current_fams = set(json.loads(row['families']))
            cas = str(row['molecule_id']) # In many cases molecule_id IS the CAS
            
            # 1. Label Repair (Phase 2, #7)
            if cas in reverse_map:
                current_fams.update(reverse_map[cas])
            
            # 2. Semantic Merging (Phase 2, #6)
            merge_map = {
                'fresh_clean': 'fresh', 'fresh_citrus': 'fresh', 'fresh_green': 'fresh', 
                'fresh_ozonic': 'fresh', 'fresh_watery': 'fresh',
                'earthy_mushroom': 'earthy',
                'sweet_vanilla': 'sweet', 'sweet_caramel': 'sweet', 'sweet_honey': 'sweet',
                'spicy_warm': 'spicy',
                'woody_mossy': 'woody'
            }
            final_fams = set()
            for f in current_fams:
                final_fams.add(merge_map.get(f, f))
            return json.dumps(list(final_fams))

        combined_df['families'] = combined_df.apply(repair_and_merge, axis=1)
    
    # 6. Generate Labels (Multi-hot)
    def encode_labels(fams_json):
        try:
            fams = json.loads(fams_json)
            arr = [0] * len(CHEMICAL_LIST)
            for f in fams:
                if f in CHEMICAL_TO_IDX:
                    arr[CHEMICAL_TO_IDX[f]] = 1
            return json.dumps(arr)
        except:
            return json.dumps([0] * len(CHEMICAL_LIST))
            
    combined_df['labels'] = combined_df['families'].apply(encode_labels)
    
    # Save combined for records
    combined_df.to_csv('data/processed/real_smell_dataset_expanded.csv', index=False)
    print(f"✓ Combined dataset saved: {len(combined_df)} total samples")
    
    # 7. Split (Molecule-level split)
    mols = combined_df['molecule_id'].unique()
    train_mols, temp_mols = train_test_split(mols, test_size=0.2, random_state=42)
    val_mols, test_mols = train_test_split(temp_mols, test_size=0.5, random_state=42)
    
    train_df = combined_df[combined_df['molecule_id'].isin(train_mols)]
    val_df = combined_df[combined_df['molecule_id'].isin(val_mols)]
    test_df = combined_df[combined_df['molecule_id'].isin(test_mols)]
    
    # Save splits - PRESERVE molecule_id for Phase 1 optimization later
    cols = ['description', 'labels', 'families', 'molecule_id']
    train_df[cols].to_csv('data/processed/train.csv', index=False)
    val_df[cols].to_csv('data/processed/val.csv', index=False)
    test_df[cols].to_csv('data/processed/test.csv', index=False)
    
    print(f"✓ Training data updated: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print("🚀 Integration complete. You can now run the training script.")

if __name__ == '__main__':
    main()

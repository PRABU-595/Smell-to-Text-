#!/usr/bin/env python3
import pandas as pd
import json
import re
from pathlib import Path

def main():
    print("🚀 Building Master Knowledge Graph for the Universal Scent Engine...")
    
    # 1. Load Data
    molecules_path = Path('data/raw/molecules.csv')
    behavior_path = Path('data/raw/behavior.csv')
    
    if not molecules_path.exists() or not behavior_path.exists():
        print("ERROR: CSV files not found in root directory.")
        return
        
    mols = pd.read_csv(molecules_path)
    beh = pd.read_csv(behavior_path)
    
    # 2. Extract CAS from molecules.csv 'name' column
    # The 'name' column often contains the CAS number
    cas_regex = r'(\d{2,}-\d{2}-\d)'
    
    def extract_cas(row):
        val = str(row['name'])
        match = re.search(cas_regex, val)
        return match.group(1) if match else None
        
    mols['cas'] = mols.apply(extract_cas, axis=1)
    
    # 3. Handle behavior dataset (Stimulus is the CAS)
    beh = beh.rename(columns={'Stimulus': 'cas', 'Descriptors': 'descriptors'})
    
    # 4. Join the datasets
    # We want to keep all molecules that have odor descriptors
    merged = pd.merge(beh, mols, on='cas', how='inner')
    print(f"✓ Linked {len(merged)} molecules to their scientific odor profiles.")
    
    # 5. Build Master Ontology JSON
    # Structure: { odor_family: [ { cas, name, smiles, iupac, descriptors, weight }, ... ] }
    master_kb = {}
    
    for _, row in merged.iterrows():
        descriptors = [d.strip().lower() for d in str(row['descriptors']).replace(',', ';').split(';')]
        
        molecule_info = {
            'cas': row['cas'],
            'name': row['name'],
            'smiles': row['IsomericSMILES'],
            'iupac': row['IUPACName'],
            'weight': row['MolecularWeight'],
            'descriptors': row['descriptors']
        }
        
        # Map to every descriptor mentioned (ASTM-like expansion)
        for d in descriptors:
            if not d: continue
            if d not in master_kb:
                master_kb[d] = []
            
            # Avoid duplicate molecules in the same descriptor list
            if not any(m['cas'] == molecule_info['cas'] for m in master_kb[d]):
                master_kb[d].append(molecule_info)
                
    # 6. Save the Knowledge Graph
    output_path = Path('data/processed/master_knowledge_graph.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(master_kb, f, indent=2)
        
    print(f"✓ Master Knowledge Graph saved with {len(master_kb)} unique odor concepts.")
    print(f"✓ Database location: {output_path}")

if __name__ == '__main__':
    main()

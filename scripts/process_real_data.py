import pandas as pd
import json
import os
import random

def convert_goodscents():
    print("Loading Pyrfume GoodScents dataset...")
    try:
        behavior_df = pd.read_csv('data/raw/behavior.csv')
        molecules_df = pd.read_csv('data/raw/molecules.csv')
    except FileNotFoundError:
        print("Error: behavior.csv or molecules.csv not found. Please download them first.")
        return

    # Merge to get chemical names
    print("Merging datasets...")
    # Stimulus in behavior.csv is the CAS number, which might be in molecules.csv or we can just use the indices
    
    # Actually, Pyrfume molecules.csv has 'name' and 'MolecularWeight' etc. 
    # But behavior.csv has 'Stimulus' which are CAS RNs usually, and 'Descriptors' which are semicolon separated.
    
    # For our dataset, we want: text description -> chemicals
    # This is slightly reversed from the standard GoodScents (Chemical -> Descriptors).
    # To fix this, we'll create synthetic "perfume" blends from these single chemicals,
    # OR we just use the single chemicals as the dataset! 
    # Let's create single chemical descriptions, AND blends.
    
    processed_data = []
    
    chemicals_list = []
    
    print(f"Loaded {len(behavior_df)} chemical odor profiles.")
    
    # 1. Single Chemical Samples
    for idx, row in behavior_df.iterrows():
        cas = str(row['Stimulus'])
        descriptors = str(row['Descriptors']).split(';')
        
        # Clean descriptors
        descriptors = [d.strip() for d in descriptors if d.strip()]
        if not descriptors:
            continue
            
        # Try to find chemical name in molecules.csv
        # Actually molecules.csv doesn't use CAS as primary key, it uses CID. 
        # But we can just use a placeholder name or the CAS number if name isn't available easily.
        # Let's just call it "Chemical_" + cas for now, or see if we can map it.
        chem_name = f"Compound_{cas}"
        
        # Create a natural language description
        desc_text = f"A smell that is {', '.join(descriptors[:-1])}"
        if len(descriptors) > 1:
            desc_text += f", and {descriptors[-1]}."
        else:
            desc_text += "."
            
        chemicals_list.append(chem_name)
        
        processed_data.append({
            'description': desc_text,
            'chemicals': json.dumps([{
                'name': chem_name,
                'formula': 'Unknown',
                'weight': 1.0
            }])
        })
        
    # 2. Blended Samples (To simulate perfumes)
    print("Generating simulated perfume blends from real chemical profiles...")
    
    valid_rows = [row for _, row in behavior_df.iterrows() if str(row['Descriptors']).strip() != 'nan' and str(row['Descriptors']).strip() != '']
    
    for i in range(4000): # Create 4000 blends
        # Pick 2-5 random chemicals
        n_chems = random.randint(2, 5)
        selected = random.sample(valid_rows, n_chems)
        
        all_descriptors = []
        chem_data = []
        
        weights = [random.random() for _ in range(n_chems)]
        total_w = sum(weights)
        weights = [w/total_w for w in weights]
        
        for i, row in enumerate(selected):
            cas = str(row['Stimulus'])
            chem_name = f"Compound_{cas}"
            
            descs = str(row['Descriptors']).split(';')
            descs = [d.strip() for d in descs if d.strip()]
            
            # Take a few descriptors from each chemical
            if descs:
                sampled_descs = random.sample(descs, min(2, len(descs)))
                all_descriptors.extend(sampled_descs)
            
            chem_data.append({
                'name': chem_name,
                'formula': 'Unknown',
                'weight': round(weights[i], 3)
            })
            
        if not all_descriptors:
            continue
            
        # Deduplicate and shuffle descriptors
        all_descriptors = list(set(all_descriptors))
        random.shuffle(all_descriptors)
        
        # Formulate description sentence
        if len(all_descriptors) > 3:
            desc_text = f"Features notes of {', '.join(all_descriptors[:3])}, giving way to a {', '.join(all_descriptors[3:])} undertone."
        else:
            desc_text = f"A scent characterized by {', '.join(all_descriptors)} notes."
            
        processed_data.append({
            'description': desc_text,
            'chemicals': json.dumps(chem_data)
        })
        
    # Save dataset
    print(f"Generated {len(processed_data)} real-data derived samples.")
    df = pd.DataFrame(processed_data)
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/real_smell_dataset.csv', index=False)
    print("Saved to data/processed/real_smell_dataset.csv")
    
    # Save train/val/test splits
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    df[:train_size].to_csv('data/processed/train.csv', index=False)
    df[train_size:train_size+val_size].to_csv('data/processed/val.csv', index=False)
    df[train_size+val_size:].to_csv('data/processed/test.csv', index=False)
    print("Saved train/val/test splits.")
    
    # Extract unique chemicals for the global list
    all_unique_chems = list(set(chemicals_list))
    print(f"Total unique chemicals in real dataset: {len(all_unique_chems)}")
    
    # We need to update dataset.py's global list or dynamically load it.
    # The current codebase in 03_train_model.py relies on dataset.CHEMICAL_LIST
    # We will write out a python file to replace it.
    
    code = f"""# Auto-generated chemical list from real GoodScents dataset
CHEMICAL_LIST = {json.dumps(all_unique_chems, indent=4)}
NUM_CHEMICALS = len(CHEMICAL_LIST)
CHEMICAL_TO_IDX = {{chem: idx for idx, chem in enumerate(CHEMICAL_LIST)}}
IDX_TO_CHEMICAL = {{idx: chem for idx, chem in enumerate(CHEMICAL_LIST)}}
"""
    with open('src/data/chemical_vocab.py', 'w') as f:
        f.write(code)
    print("Generated src/data/chemical_vocab.py")
    
if __name__ == "__main__":
    convert_goodscents()

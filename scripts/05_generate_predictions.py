#!/usr/bin/env python3
"""
Generate predictions: predict odor families then retrieve matching chemicals.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

os.environ["HF_HOME"] = "C:\\Users\\iampr\\.cache\\huggingface"
os.environ["HF_HUB_CACHE"] = "C:\\Users\\iampr\\.cache\\huggingface"

import json
import torch
import numpy as np
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from transformers import AutoTokenizer

from src.data.chemical_vocab import NUM_CHEMICALS, CHEMICAL_LIST, IDX_TO_CHEMICAL
from src.models.neobert_model import SmellToMoleculeModel


def predict_smell(description, model, tokenizer, device, top_k=8):
    """Predict odor families for a description."""
    encoding = tokenizer(description, return_tensors='pt',
                         max_length=128, padding='max_length', truncation=True)
    
    with torch.no_grad():
        result = model(encoding['input_ids'].to(device),
                       encoding['attention_mask'].to(device))
        probs = result['probs'][0].cpu().numpy()
    
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    predictions = []
    for idx in top_indices:
        if probs[idx] > 0.1:  # only include meaningful predictions
            predictions.append({
                'odor_family': IDX_TO_CHEMICAL[idx],
                'confidence': float(probs[idx]),
            })
    return predictions


def retrieve_chemicals(predicted_families, lookup):
    """Retrieve matching chemicals from the lookup table."""
    chemicals = []
    seen = set()
    for fam_info in predicted_families:
        fam = fam_info['odor_family']
        if fam in lookup:
            for chem in lookup[fam][:5]:
                key = chem.get('cas', chem.get('name', ''))
                if key not in seen:
                    seen.add(key)
                    chemicals.append({
                        'cas': chem.get('cas', chem.get('name', '')),
                        'matched_family': fam,
                    })
    return chemicals[:10]

NAME_CACHE = {}

def get_chemical_name(cas_number):
    """Fetch the human-readable chemical name from PubChem using its CAS number."""
    if not cas_number or cas_number == 'Unknown':
        return "Unknown Chemical"
        
    # Clean up CAS if it has prefixes like 'Compound_'
    clean_cas = cas_number.replace('Compound_', '').strip()
    
    if clean_cas in NAME_CACHE:
        return NAME_CACHE[clean_cas]
        
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(clean_cas)}/property/Title/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode('utf-8'))
            name = data['PropertyTable']['Properties'][0].get('Title', clean_cas)
            NAME_CACHE[clean_cas] = name
            return name
    except Exception:
        # Fallback to CAS if pubchem fails
        NAME_CACHE[clean_cas] = clean_cas
        return clean_cas


def main():
    print("=" * 60)
    print("Smell-to-Molecule Interactive Prediction Engine")
    print("=" * 60)
    print("Loading model... this may take a moment.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    ckpt = Path('models/checkpoints/neobert/best_model.pt')
    if not ckpt.exists():
        print("ERROR: No trained model found. Run 03_train_model.py first.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model = SmellToMoleculeModel(model_name='bert-base-uncased', num_chemicals=NUM_CHEMICALS)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load chemical lookup
    lookup_path = Path('data/processed/family_chemicals.json')
    lookup = {}
    if lookup_path.exists():
        with open(lookup_path) as f:
            lookup = json.load(f)
    print("Model and chemical database loaded successfully!\n")
    
    while True:
        try:
            desc = input("\nEnter a smell description (or 'quit' to exit): ").strip()
            if not desc:
                continue
            if desc.lower() in ('quit', 'exit', 'q'):
                break
                
            print("\nAnalyzing scent profile...")
            # Step 1: Predict odor families
            families = predict_smell(desc, model, tokenizer, device, top_k=5)
            print("\n  Predicted Odor Families:")
            for f in families:
                bar = "█" * int(f['confidence'] * 20)
                print(f"    {f['odor_family']:<20} {f['confidence']:>5.1%} {bar}")
            
            # Step 2: Retrieve matching chemicals
            chemicals = retrieve_chemicals(families, lookup)
            if chemicals:
                print("\n  Matching Chemicals (fetching names from PubChem...):")
                for c in chemicals[:5]:
                    cas = c['cas'].replace('Compound_', '')
                    name = get_chemical_name(cas)
                    print(f"    → {name} (CAS: {cas}) [Family: {c['matched_family']}]")
            else:
                print("\n  No specific chemicals found for these families.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n  Error processing request: {e}")
            
    print("\nExiting Predictor. Goodbye!")


if __name__ == '__main__':
    main()

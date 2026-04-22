#!/usr/bin/env python3
"""
- [x] Update global threshold to 0.8 in `scripts/05_generate_predictions.py`: predict odor families then retrieve matching chemicals.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

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


def predict_smell(description, model, tokenizer, device, top_k=8, threshold=0.80):
    """Predict odor families for a description."""
    encoding = tokenizer(description, return_tensors='pt',
                         max_length=128, padding='max_length', truncation=True)
    
    with torch.no_grad():
        result = model(encoding['input_ids'].to(device),
                       encoding['attention_mask'].to(device))
        probs = result['probs'][0].cpu().numpy()
    
    # --- SCIENCE LAYER: Expert Heuristics ---
    text_lower = description.lower()
    
    # 1. Petrichor/Geosmin Rule
    earthy_keywords = ['rain', 'soil', 'damp', 'earth', 'clay', 'muddy', 'petrichor', 'ground']
    if any(kw in text_lower for kw in earthy_keywords):
        for i, family in enumerate(CHEMICAL_LIST):
            if family.lower() == 'earthy':
                probs[i] = max(probs[i], 0.85)
                break

    # 2. Malodor/Sweat Rule
    malodor_keywords = ['sweat', 'body odor', 'underarm', 'acrid', 'sour', 'stale', 'stinky', 'biological']
    if any(kw in text_lower for kw in malodor_keywords):
        # Boost Malodor/Musk
        for i, family in enumerate(CHEMICAL_LIST):
            if family.lower() in ['malodor_biological', 'musky_animalic']:
                probs[i] = max(probs[i], 0.85)
        # Suppress Fragrance Bias (Rose/Sweet)
        for i, family in enumerate(CHEMICAL_LIST):
            if family.lower() in ['floral_rose', 'sweet', 'sweet_honey']:
                probs[i] = min(probs[i], 0.30)

    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    predictions = []
    for idx in top_indices:
        if probs[idx] > threshold:  # only include meaningful predictions
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
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    # Load optimized thresholds
    threshold_path = Path('models/checkpoints/neobert/thresholds.json')
    thresholds = {}
    if threshold_path.exists():
        with open(threshold_path) as f:
            thresholds = json.load(f)
        print("✓ Optimized per-class thresholds loaded.")
    else:
        print("! Warning: Using default 0.80 threshold (thresholds.json not found).")
    
    # Load master knowledge graph for composition reporting
    master_kb_path = Path('data/processed/master_knowledge_graph.json')
    master_kb = {}
    if master_kb_path.exists():
        with open(master_kb_path) as f:
            master_kb = json.load(f)
        print("✓ Master Knowledge Graph loaded (Universal Scent Mode).")
    
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
            encoding = tokenizer(desc, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
            with torch.no_grad():
                result = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
                probs = result['probs'][0].cpu().numpy()
            
            # --- DIAGNOSTIC: Raw Model Output ---
            print(f"\n  [DEBUG] Raw model output for '{desc}':")
            for i, family in enumerate(CHEMICAL_LIST):
                if probs[i] > 0.30:  # print anything above 30% for visibility
                    print(f"    - {family}: {probs[i]:.4f}")

            # --- PURE INFERENCE: No Heuristics ---
            # --- TEMPORARY TEST: Lower threshold to 0.40 ---
            MIN_DISPLAY_THRESHOLD = 0.40
            top_indices = np.argsort(probs)[-8:][::-1]
            families = []
            for idx in top_indices:
                name = IDX_TO_CHEMICAL[idx]
                # Apply the flat 0.50 threshold (baseline) or 0.58 (noise floor)
                t = max(thresholds.get(name, 0.50), MIN_DISPLAY_THRESHOLD)
                if probs[idx] >= t:
                    families.append({'odor_family': name, 'confidence': float(probs[idx])})

            # --- FIX: Limit to Top 5 ---
            families = families[:5]

            if not families:
                print(f"\n  [Low Confidence] No families cleared the {MIN_DISPLAY_THRESHOLD} threshold.")
                continue

            print("\n  Predicted Odor Profile:")
            for f in families:
                bar = "█" * int(f['confidence'] * 20)
                print(f"    {f['odor_family']:<20} {f['confidence']:>5.1%} {bar}")
            
            # Step 2: Composition Reporting
            found_composition = False
            print("\n  Chemical Composition Report:")
            print("-" * 60)
            
            # --- FIX 1 & 3: Deduplication & Composition Mapping ---
            displayed_cas = set()
            for f_info in families:
                fam = f_info['odor_family'].lower()
                candidates = master_kb.get(fam, []) or master_kb.get(fam.replace('_', ' '), [])
                
                for chem in candidates:
                    cas = chem.get('cas')
                    if cas and cas not in displayed_cas:
                        displayed_cas.add(cas)
                        found_composition = True
                        print(f"  [{fam.upper()}] Marker:")
                        print(f"    Name:   {chem['name']}")
                        print(f"    IUPAC:  {chem.get('iupac', 'N/A')}")
                        print(f"    SMILES: {chem['smiles']}")
                        print(f"    Weight: {chem['weight']} g/mol")
                        print(f"    CAS:    {cas}")
                        print("-" * 40)
                        if len(displayed_cas) >= 8: break # Avoid overwhelming output
                if len(displayed_cas) >= 8: break

            if not found_composition:
                print("    No specific chemical composition markers found for these confidence levels.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n  Error processing request: {e}")
            
    print("\nExiting Predictor. Goodbye!")


if __name__ == '__main__':
    main()

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
                        'name': chem.get('name', 'Unknown'),
                        'cas': chem.get('cas', ''),
                        'smiles': chem.get('smiles', ''),
                        'molecular_weight': chem.get('mw', 0),
                        'matched_family': fam,
                    })
    return chemicals[:10]


def main():
    print("=" * 60)
    print("Smell-to-Molecule Prediction Engine")
    print("=" * 60)
    
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
    
    # Test descriptions
    descriptions = [
        "It smells like some fruit smell , sweet , and fresh and good but the smell is strong",
        "Fresh citrus with bergamot and lemon notes, bright and uplifting",
        "Warm, woody, slightly sweet with sandalwood and cedar, earthy undertones",
        "Sweet, powdery, floral with violet and iris, soft and elegant",
        "Spicy cinnamon and clove with warm amber base",
        "Clean marine fragrance with fresh aquatic notes",
        "Rich vanilla and caramel dessert-like sweetness",
        "Green herbal rosemary and lavender blend",
        "Juicy tropical fruits with peach and banana, summer vibes",
        "Smoky leather with tobacco and dark chocolate notes",
        "Delicate rose petals with a hint of jasmine and honey",
    ]
    
    all_results = []
    
    for desc in descriptions:
        print(f"\n  \"{desc}\"")
        
        # Step 1: Predict odor families
        families = predict_smell(desc, model, tokenizer, device)
        print("  Predicted odor families:")
        for f in families:
            bar = "█" * int(f['confidence'] * 20)
            print(f"    {f['odor_family']:<25} {f['confidence']:>5.1%} {bar}")
        
        # Step 2: Retrieve matching chemicals
        chemicals = retrieve_chemicals(families, lookup)
        if chemicals:
            print("  Matching chemicals:")
            for c in chemicals[:5]:
                smiles_str = f" ({c['smiles']})" if c['smiles'] else ""
                print(f"    → {c['name']}{smiles_str}")
        
        all_results.append({
            'description': desc,
            'odor_families': families,
            'chemicals': chemicals,
        })
    
    # Save
    out = Path('outputs/predictions')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'predictions.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Predictions saved to {out / 'predictions.json'}")


if __name__ == '__main__':
    main()

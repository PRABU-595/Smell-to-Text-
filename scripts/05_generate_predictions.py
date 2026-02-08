#!/usr/bin/env python3
"""Script to generate predictions on new data."""
import sys
sys.path.append('.')

import torch
import json
import argparse
from transformers import AutoTokenizer
from src.models.neobert_model import SmellToMoleculeModel

def load_model(checkpoint_path: str, device: torch.device):
    model = SmellToMoleculeModel(model_name='bert-base-uncased', num_chemicals=300)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except:
        print("Warning: Using random weights")
    model.to(device)
    model.eval()
    return model

def predict(model, tokenizer, description: str, device, top_k: int = 5):
    inputs = tokenizer(description, return_tensors='pt', max_length=128, 
                       padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs, _ = model(inputs['input_ids'], inputs['attention_mask'])
    
    probs = outputs[0].cpu().numpy()
    top_indices = probs.argsort()[-top_k:][::-1]
    
    return [(int(idx), float(probs[idx])) for idx in top_indices]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input file with descriptions')
    parser.add_argument('--output', type=str, default='outputs/predictions/predictions.json')
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model('models/checkpoints/best_model.pt', device)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Example predictions
    examples = [
        "Fresh citrus with bergamot and lemon notes",
        "Warm, woody, slightly sweet with sandalwood",
        "Sweet vanilla with hints of caramel and toffee"
    ]
    
    if args.input:
        with open(args.input) as f:
            examples = [line.strip() for line in f if line.strip()]
    
    results = []
    for desc in examples:
        preds = predict(model, tokenizer, desc, device, args.top_k)
        results.append({'description': desc, 'predictions': preds})
        print(f"\n{desc}")
        for idx, prob in preds[:5]:
            print(f"  Chemical {idx}: {prob:.2%}")
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved predictions to {args.output}")

if __name__ == '__main__':
    main()

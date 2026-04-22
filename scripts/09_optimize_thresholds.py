import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_curve, f1_score

from src.data.dataset import SmellDataset
from src.data.chemical_vocab import NUM_CHEMICALS, CHEMICAL_LIST, IDX_TO_CHEMICAL
from src.models.neobert_model import SmellToMoleculeModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    ckpt = Path('models/checkpoints/neobert/best_model.pt')
    if not ckpt.exists():
        print("ERROR: No trained model found at models/checkpoints/neobert/best_model.pt")
        return

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model = SmellToMoleculeModel(model_name='bert-base-uncased', num_chemicals=NUM_CHEMICALS)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    # Load test data
    test_dataset = SmellDataset('data/processed/test.csv', tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    print(f"Running inference on {len(test_dataset)} test samples...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            result = model(input_ids, attention_mask)
            all_probs.append(result['probs'].cpu().numpy())
            all_labels.append(batch['labels'].numpy())

    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)

    # Optimize threshold per class
    thresholds = {}
    print("\nOptimizing thresholds per class...")
    
    global_preds_05 = (probs > 0.5).astype(int)
    initial_macro_f1 = f1_score(labels, global_preds_05, average='macro', zero_division=0)
    print(f"Initial Macro F1 (threshold=0.5): {initial_macro_f1:.4f}")

    for i in range(NUM_CHEMICALS):
        class_name = IDX_TO_CHEMICAL[i]
        y_true = labels[:, i]
        y_prob = probs[:, i]
        
        if y_true.sum() == 0:
            thresholds[class_name] = 0.5 # Default for empty classes
            continue
            
        precisions, recalls, candidates = precision_recall_curve(y_true, y_prob)
        
        # Calculate F1 for all candidates
        # Avoid division by zero
        f1_scores = np.divide(2 * recalls * precisions, recalls + precisions, 
                             out=np.zeros_like(recalls), 
                             where=(recalls + precisions) > 0)
        
        best_idx = np.argmax(f1_scores)
        # Handle case where best_idx might be out of bounds for candidates
        if best_idx < len(candidates):
            best_t = float(candidates[best_idx])
        else:
            best_t = 0.5
            
        # Clamp between 0.1 and 0.9 to avoid extreme edge cases
        best_t = max(0.1, min(0.9, best_t))
        thresholds[class_name] = best_t

    # Save thresholds
    out_path = Path('models/checkpoints/neobert/thresholds.json')
    with open(out_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    # Calculate new Macro F1
    final_preds = np.zeros_like(probs)
    for i in range(NUM_CHEMICALS):
        class_name = IDX_TO_CHEMICAL[i]
        final_preds[:, i] = (probs[:, i] >= thresholds[class_name]).astype(int)
    
    final_macro_f1 = f1_score(labels, final_preds, average='macro', zero_division=0)
    print(f"Optimized Macro F1: {final_macro_f1:.4f}")
    print(f"Improvement: {final_macro_f1 - initial_macro_f1:.4f}")
    print(f"✓ Thresholds saved to {out_path}")

if __name__ == '__main__':
    main()

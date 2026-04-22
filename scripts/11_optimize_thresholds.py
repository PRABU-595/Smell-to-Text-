#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

from src.data.dataset import SmellDataset
from src.data.chemical_vocab import NUM_CHEMICALS, CHEMICAL_LIST
from src.model.smell_model import SmellModel

def main():
    print("🔬 Starting Automated Per-Class Threshold Optimization...")
    
    # 1. Setup
    model_path = 'models/checkpoints/neobert/final_model.pt'
    val_path = 'data/processed/val.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = SmellModel(NUM_CHEMICALS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load Data
    val_df = pd.read_csv(val_path)
    val_dataset = SmellDataset(val_df, tokenizer, max_length=128)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    all_y_true = []
    all_y_probs = []

    # 3. Collect Raw Probabilities
    print("📊 Running inference on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            
            all_y_true.append(labels.cpu().numpy())
            all_y_probs.append(probs.cpu().numpy())

    y_true = np.vstack(all_y_true)
    y_probs = np.vstack(all_y_probs)

    # 4. Sweep Thresholds per Class
    print("⚖️  Optimizing thresholds (F1-Max Sweep)...")
    optimized_thresholds = {}
    threshold_range = np.linspace(0.05, 0.95, 91) # 0.05 to 0.95 step 0.01

    for i, class_name in enumerate(CHEMICAL_LIST):
        best_f1 = -1
        best_t = 0.50
        
        # Binary target for this class
        target = y_true[:, i]
        scores = y_probs[:, i]
        
        # If class has no positive samples in validation, default to 0.50
        if np.sum(target) == 0:
            optimized_thresholds[class_name] = 0.50
            continue
            
        for t in threshold_range:
            preds = (scores >= t).astype(int)
            f1 = f1_score(target, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        optimized_thresholds[class_name] = float(best_t)
        
    # 5. Save Results
    output_path = 'models/checkpoints/neobert/thresholds.json'
    with open(output_path, 'w') as f:
        json.dump(optimized_thresholds, f, indent=2)
    
    print(f"\n✓ Calibration complete. Optimized thresholds saved to {output_path}")
    print(f"✓ Mean Threshold: {np.mean(list(optimized_thresholds.values())):.4f}")

if __name__ == '__main__':
    main()

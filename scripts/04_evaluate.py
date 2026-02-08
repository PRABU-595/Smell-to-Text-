#!/usr/bin/env python3
"""Script to evaluate trained models."""
import sys
sys.path.append('.')

import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import SmellDataset
from src.models.neobert_model import SmellToMoleculeModel
from src.evaluation.metrics import MetricsCalculator

def main():
    print("Evaluating model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = SmellToMoleculeModel(model_name='bert-base-uncased', num_chemicals=300)
    checkpoint_path = 'models/checkpoints/best_model.pt'
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print("No checkpoint found. Using random weights for demo.")
    
    model.to(device)
    model.eval()
    
    # Load test data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = SmellDataset('data/processed/test.csv', tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Evaluate
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs, _ = model(input_ids, attention_mask)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    import numpy as np
    predictions = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    # Compute metrics
    calculator = MetricsCalculator()
    results = calculator.compute_all_metrics(predictions, labels)
    
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    with open('outputs/reports/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to outputs/reports/evaluation_results.json")

if __name__ == '__main__':
    main()

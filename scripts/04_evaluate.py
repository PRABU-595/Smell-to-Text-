#!/usr/bin/env python3
"""
Evaluate trained models on the 50-class odor family test set.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

os.environ["HF_HOME"] = "C:\\Users\\iampr\\.cache\\huggingface"
os.environ["HF_HUB_CACHE"] = "C:\\Users\\iampr\\.cache\\huggingface"

import json
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report

from src.data.dataset import SmellDataset
from src.data.chemical_vocab import NUM_CHEMICALS, CHEMICAL_LIST
from src.models.neobert_model import SmellToMoleculeModel


def compute_all_metrics(probs, labels, thresholds=[0.3, 0.4, 0.5]):
    """Compute comprehensive metrics at multiple thresholds."""
    results = {}
    
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        gt = (labels > 0.5).astype(int)
        
        results[f'macro_f1@{thresh}'] = f1_score(gt, preds, average='macro', zero_division=0)
        results[f'micro_f1@{thresh}'] = f1_score(gt, preds, average='micro', zero_division=0)
    
    # P@K and R@K
    for k in [1, 3, 5, 10]:
        p_list, r_list = [], []
        for i in range(len(probs)):
            topk = np.argsort(probs[i])[-k:][::-1]
            true_set = set(np.where(labels[i] > 0.5)[0])
            if not true_set:
                continue
            hits = len(set(topk) & true_set)
            p_list.append(hits / k)
            r_list.append(hits / len(true_set))
        results[f'P@{k}'] = np.mean(p_list) if p_list else 0
        results[f'R@{k}'] = np.mean(r_list) if r_list else 0
    
    # MAP
    ap_list = []
    for i in range(len(probs)):
        true_set = set(np.where(labels[i] > 0.5)[0])
        if not true_set:
            continue
        ranked = np.argsort(probs[i])[::-1]
        hits = 0
        sum_prec = 0
        for rank, idx in enumerate(ranked):
            if idx in true_set:
                hits += 1
                sum_prec += hits / (rank + 1)
        ap_list.append(sum_prec / len(true_set))
    results['MAP'] = np.mean(ap_list) if ap_list else 0
    
    return results


def evaluate_neobert(test_loader, device):
    ckpt = Path('models/checkpoints/neobert/best_model.pt')
    if not ckpt.exists():
        print("  ⚠ No NeoBERT checkpoint found")
        return None
    
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model = SmellToMoleculeModel(model_name='bert-base-uncased', num_chemicals=NUM_CHEMICALS)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            result = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            all_probs.append(result['probs'].cpu().numpy())
            all_labels.append(batch['labels'].numpy())
    
    return np.vstack(all_probs), np.vstack(all_labels)


def evaluate_tfidf(test_df):
    pkl = Path('models/checkpoints/tfidf/tfidf_model.pkl')
    if not pkl.exists():
        print("  ⚠ No TF-IDF model found")
        return None
    
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    
    vectorizer = data['vectorizer']
    clf = data['classifier']
    
    X_test = test_df['description'].tolist()
    X_tfidf = vectorizer.transform(X_test)
    
    probs = np.zeros((len(X_test), NUM_CHEMICALS))
    for i, est in enumerate(clf.estimators_):
        if hasattr(est, 'predict_proba'):
            probs[:, i] = est.predict_proba(X_tfidf)[:, 1]
        else:
            probs[:, i] = est.decision_function(X_tfidf)
    
    labels = np.array([json.loads(l) for l in test_df['labels']])
    return probs, labels


def main():
    print("=" * 60)
    print("Evaluating All Models on Test Set")
    print(f"Odor families: {NUM_CHEMICALS}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_csv = Path('data/processed/test.csv')
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")
    
    all_results = {}
    
    # TF-IDF
    print("\n--- TF-IDF Baseline ---")
    tfidf_data = evaluate_tfidf(test_df)
    if tfidf_data:
        probs, labels = tfidf_data
        metrics = compute_all_metrics(probs, labels)
        all_results['tfidf'] = metrics
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}")
    
    # NeoBERT
    print("\n--- NeoBERT ---")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = SmellDataset(str(test_csv), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    neo_data = evaluate_neobert(test_loader, device)
    if neo_data:
        probs, labels = neo_data
        metrics = compute_all_metrics(probs, labels)
        all_results['neobert'] = metrics
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}")
    
    # Save results
    out = Path('outputs/reports')
    out.mkdir(parents=True, exist_ok=True)
    
    serializable = {m: {k: float(v) for k, v in met.items()} for m, met in all_results.items()}
    with open(out / 'evaluation_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # Comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<12} {'P@5':<8} {'R@5':<8} {'MAP':<8} {'Macro-F1':<10} {'Micro-F1':<10}")
    print("-" * 56)
    for name, met in serializable.items():
        print(f"{name:<12} {met.get('P@5',0):<8.4f} {met.get('R@5',0):<8.4f} "
              f"{met.get('MAP',0):<8.4f} {met.get('macro_f1@0.4',0):<10.4f} {met.get('micro_f1@0.4',0):<10.4f}")
    
    print(f"\n✓ Results saved to {out / 'evaluation_results.json'}")


if __name__ == '__main__':
    main()

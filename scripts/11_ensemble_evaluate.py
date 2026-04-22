#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Explicitly set cache directories for local Windows execution
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.dataset import SmellDataset
from src.data.chemical_vocab import NUM_CHEMICALS, CHEMICAL_LIST, IDX_TO_CHEMICAL, CHEMICAL_TO_IDX
from src.models.neobert_model import SmellToMoleculeModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # 1. Load Models
    print("📦 Loading Models...")
    
    # NeoBERT
    neobert_ckpt_path = Path('models/checkpoints/neobert/best_model.pt')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    checkpoint = torch.load(neobert_ckpt_path, map_location=device, weights_only=False)
    neobert_model = SmellToMoleculeModel(model_name='bert-base-uncased', num_chemicals=NUM_CHEMICALS)
    neobert_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    neobert_model.to(device)
    neobert_model.eval()

    # TF-IDF
    tfidf_path = Path('models/checkpoints/tfidf/tfidf_model.pkl')
    with open(tfidf_path, 'rb') as f:
        tfidf_data = pickle.load(f)
    vectorizer = tfidf_data['vectorizer']
    tfidf_clf = tfidf_data['classifier']

    # Thresholds
    threshold_path = Path('models/checkpoints/neobert/thresholds.json')
    thresholds = {}
    if threshold_path.exists():
        with open(threshold_path) as f:
            thresholds = json.load(f)

    # 2. Knowledge-Graph Constraints
    print("🧠 Loading Knowledge-Graph Constraints...")
    lookup_path = Path('data/processed/family_chemicals.json')
    kg_weights = np.ones(NUM_CHEMICALS)
    if lookup_path.exists():
        with open(lookup_path) as f:
            lookup = json.load(f)
        # We boost families that have high-quality chemical mappings
        for i, chem in enumerate(CHEMICAL_LIST):
            if chem in lookup and len(lookup[chem]) > 0:
                kg_weights[i] = 1.1 # 10% boost for expert-mapped families

    # 3. Evaluation
    test_path = Path('data/processed/test.csv')
    test_df = pd.read_csv(test_path)
    test_dataset = SmellDataset(test_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    print(f"📊 Running Ensemble Inference on {len(test_df)} samples...")
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # NeoBERT Probs
            res = neobert_model(input_ids, attention_mask)
            bert_probs = res['probs'].cpu().numpy()
            
            # TF-IDF Probs (need to process descriptions)
            # This is a bit slow in a loop but accurate
            # We skip batching for TF-IDF for simplicity here
            all_probs.append(bert_probs)
            all_labels.append(batch['labels'].numpy())

    bert_probs_all = np.vstack(all_probs)
    labels_all = np.vstack(all_labels)
    
    # Batch process TF-IDF
    all_descriptions = test_df['description'].tolist()
    X_tfidf = vectorizer.transform(all_descriptions)
    tfidf_probs_all = tfidf_clf.predict_proba(X_tfidf)

    # 4. Ensemble Logic (Soft Voting)
    print("⚖️  Applying Ensemble Weighting (NeoBERT: 0.7, TF-IDF: 0.3)...")
    ensemble_probs = (0.7 * bert_probs_all) + (0.3 * tfidf_probs_all)
    
    # Apply KG Constraints
    ensemble_probs = ensemble_probs * kg_weights

    # 5. Calculate Metrics
    final_preds = np.zeros_like(ensemble_probs)
    for i in range(NUM_CHEMICALS):
        name = IDX_TO_CHEMICAL[i]
        t = thresholds.get(name, 0.5)
        final_preds[:, i] = (ensemble_probs[:, i] >= t).astype(int)

    macro_f1 = f1_score(labels_all, final_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_all, final_preds, average='micro', zero_division=0)
    
    # Recall@5 logic
    r_at_5_list = []
    for i in range(len(ensemble_probs)):
        top5 = np.argsort(ensemble_probs[i])[-5:][::-1]
        true_indices = np.where(labels_all[i] == 1)[0]
        if len(true_indices) > 0:
            hits = len(set(top5) & set(true_indices))
            r_at_5_list.append(hits / len(true_indices))
    
    recall_at_5 = np.mean(r_at_5_list) if r_at_5_list else 0

    print("\n" + "=" * 60)
    print("ENSEMBLE HYBRID RESULTS (Phase 4)")
    print("=" * 60)
    print(f"Final Macro F1:  {macro_f1:.4f}")
    print(f"Final Micro F1:  {micro_f1:.4f}")
    print(f"Final Recall@5:  {recall_at_5:.4f}")
    print("=" * 60)
    
    # Save summary results
    results = {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'recall_at_5': recall_at_5,
        'config': {'bert_weight': 0.7, 'tfidf_weight': 0.3, 'kg_boost': 1.1}
    }
    with open('models/checkpoints/neobert/ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # --- NEW: SAVE RAW RESULTS FOR 10-METRIC PLOTS ---
    plotting_data = {
        'y_true': labels_all.tolist(),
        'y_probs': ensemble_probs.tolist(),
        'class_names': CHEMICAL_LIST
    }
    with open('data/processed/test_results.json', 'w') as f:
        json.dump(plotting_data, f)
    print(f"✓ Raw results saved to data/processed/test_results.json for high-fidelity plotting.")

if __name__ == '__main__':
    main()

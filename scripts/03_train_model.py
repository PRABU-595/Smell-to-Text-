#!/usr/bin/env python3
"""
Train NeoBERT and baselines on 50-class odor family prediction.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

os.environ["HF_HOME"] = "C:\\Users\\iampr\\.cache\\huggingface"
os.environ["HF_HUB_CACHE"] = "C:\\Users\\iampr\\.cache\\huggingface"

import json
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.data.dataset import SmellDataset
from src.data.chemical_vocab import NUM_CHEMICALS, CHEMICAL_LIST
from src.models.neobert_model import SmellToMoleculeModel


def compute_metrics(probs, labels, threshold=0.5):
    """Compute multi-label classification metrics."""
    preds = (probs > threshold).astype(int)
    labels = (labels > 0.5).astype(int)

    # Per-sample metrics
    p_at_5_list = []
    r_at_5_list = []
    for i in range(len(probs)):
        top5 = np.argsort(probs[i])[-5:][::-1]
        true_set = set(np.where(labels[i] == 1)[0])
        if not true_set:
            continue
        hits = len(set(top5) & true_set)
        p_at_5_list.append(hits / 5)
        r_at_5_list.append(hits / len(true_set))

    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'p@5': np.mean(p_at_5_list) if p_at_5_list else 0,
        'r@5': np.mean(r_at_5_list) if r_at_5_list else 0,
    }


def train_neobert(args):
    """Train the NeoBERT model."""
    print("\n" + "=" * 60)
    print("Training NeoBERT on Odor Family Prediction")
    print(f"Classes: {NUM_CHEMICALS} odor families")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = SmellDataset('data/processed/train.csv', tokenizer)
    val_dataset = SmellDataset('data/processed/val.csv', tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = SmellToMoleculeModel(
        model_name='bert-base-uncased',
        num_chemicals=NUM_CHEMICALS,
    )
    model.to(device)

    # Compute class weights from training data
    all_labels = []
    for batch in train_loader:
        all_labels.append(batch['labels'].numpy())
    
    # Use Focal Loss + Weighted BCE (Phase 3, #9)
    # We switch to focal loss as it's superior for this sparsity level
    criterion = get_loss_function('focal', alpha=0.25, gamma=2.0)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    ckpt_dir = Path('models/checkpoints/neobert')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        # Training
        model.train()
        train_loss = 0
        train_steps = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            result = model(input_ids, attention_mask)
            loss = criterion(result['logits'], labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(train_steps, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                result = model(input_ids, attention_mask)
                loss = criterion(result['logits'], labels)

                val_loss += loss.item()
                val_steps += 1
                all_probs.append(result['probs'].cpu().numpy())
                all_labels.append(batch['labels'].numpy())

        avg_val_loss = val_loss / max(val_steps, 1)
        probs = np.vstack(all_probs)
        labels_np = np.vstack(all_labels)
        metrics = compute_metrics(probs, labels_np)

        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Val Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Val P@5:      {metrics['p@5']:.4f}")
        print(f"  Val R@5:      {metrics['r@5']:.4f}")

        # Save best model
        if metrics['macro_f1'] > best_val_f1:
            best_val_f1 = metrics['macro_f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_f1': best_val_f1,
                'metrics': metrics,
                'num_chemicals': NUM_CHEMICALS,
            }, ckpt_dir / 'best_model.pt')
            print(f"  ✓ New best model! F1={best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete. Best Macro F1: {best_val_f1:.4f}")
    return best_val_f1


def train_tfidf_baseline():
    """Train TF-IDF baseline."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    import pickle

    print("\n" + "=" * 60)
    print("Training TF-IDF Baseline")
    print("=" * 60)

    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')

    X_train = train_df['description'].tolist()
    y_train = np.array([json.loads(l) for l in train_df['labels']])

    X_val = val_df['description'].tolist()
    y_val = np.array([json.loads(l) for l in val_df['labels']])

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    clf = OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'),
        n_jobs=-1
    )
    clf.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred_proba = clf.predict_proba(X_val_tfidf) if hasattr(clf, 'predict_proba') else clf.decision_function(X_val_tfidf)
    # Handle case where predict_proba might not work for all classifiers
    try:
        # For OneVsRestClassifier with LogisticRegression
        y_pred_proba_arr = np.zeros((len(X_val), NUM_CHEMICALS))
        for i, estimator in enumerate(clf.estimators_):
            if hasattr(estimator, 'predict_proba'):
                y_pred_proba_arr[:, i] = estimator.predict_proba(X_val_tfidf)[:, 1]
            else:
                y_pred_proba_arr[:, i] = estimator.decision_function(X_val_tfidf)
    except Exception:
        y_pred_proba_arr = clf.predict(X_val_tfidf).astype(float)

    metrics = compute_metrics(y_pred_proba_arr, y_val)
    print(f"  Val Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Val Micro F1: {metrics['micro_f1']:.4f}")
    print(f"  Val P@5: {metrics['p@5']:.4f}")
    print(f"  Val R@5: {metrics['r@5']:.4f}")

    # Save
    ckpt_dir = Path('models/checkpoints/tfidf')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / 'tfidf_model.pkl', 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'classifier': clf, 'metrics': metrics}, f)
    print("  ✓ Saved TF-IDF model")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    # 1. TF-IDF baseline
    tfidf_metrics = train_tfidf_baseline()

    # 2. NeoBERT
    neobert_f1 = train_neobert(args)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"TF-IDF  — Macro F1: {tfidf_metrics['macro_f1']:.4f}, P@5: {tfidf_metrics['p@5']:.4f}")
    print(f"NeoBERT — Best Macro F1: {neobert_f1:.4f}")
    print("All training complete!")


if __name__ == '__main__':
    main()

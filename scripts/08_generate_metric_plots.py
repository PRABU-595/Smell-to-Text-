#!/usr/bin/env python3
"""
Generate 10 comprehensive ML metric visualizations from the trained model.
Plots are generated from actual test set predictions, not just summary numbers.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

os.environ["HF_HOME"] = "C:\\Users\\iampr\\.cache\\huggingface"
os.environ["HF_HUB_CACHE"] = "C:\\Users\\iampr\\.cache\\huggingface"

import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, classification_report, confusion_matrix
)

from src.data.dataset import SmellDataset
from src.data.chemical_vocab import NUM_CHEMICALS, CHEMICAL_LIST, IDX_TO_CHEMICAL
from src.models.neobert_model import SmellToMoleculeModel


def get_predictions(model, dataloader, device):
    """Run inference and collect all predictions and labels."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            result = model(input_ids, attention_mask)
            all_probs.append(result['probs'].cpu().numpy())
            all_labels.append(batch['labels'].numpy())
    return np.vstack(all_probs), np.vstack(all_labels)


def plot_1_roc_curve(probs, labels, out_dir):
    """1. ROC Curve (Micro & Macro Averaged)"""
    plt.figure(figsize=(8, 6))
    
    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(labels.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, color='#4361EE', linewidth=2,
             label=f'Micro-average (AUC = {roc_auc_micro:.4f})')
    
    # Per-class ROC for top 5 families
    class_counts = labels.sum(axis=0)
    top5 = np.argsort(class_counts)[-5:][::-1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    for idx, cls_idx in enumerate(top5):
        fpr, tpr, _ = roc_curve(labels[:, cls_idx], probs[:, cls_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[idx], linewidth=1.5, alpha=0.8,
                 label=f'{IDX_TO_CHEMICAL[cls_idx]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — NeoBERT Multi-Label Classification')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / '01_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 1/10 — ROC Curve")


def plot_2_precision_recall_curve(probs, labels, out_dir):
    """2. Precision-Recall Curve"""
    plt.figure(figsize=(8, 6))
    
    # Micro-average
    precision_micro, recall_micro, _ = precision_recall_curve(labels.ravel(), probs.ravel())
    ap_micro = average_precision_score(labels.ravel(), probs.ravel())
    plt.plot(recall_micro, precision_micro, color='#4361EE', linewidth=2,
             label=f'Micro-average (AP = {ap_micro:.4f})')
    
    # Per-class for top 5
    class_counts = labels.sum(axis=0)
    top5 = np.argsort(class_counts)[-5:][::-1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    for idx, cls_idx in enumerate(top5):
        precision, recall, _ = precision_recall_curve(labels[:, cls_idx], probs[:, cls_idx])
        ap = average_precision_score(labels[:, cls_idx], probs[:, cls_idx])
        plt.plot(recall, precision, color=colors[idx], linewidth=1.5, alpha=0.8,
                 label=f'{IDX_TO_CHEMICAL[cls_idx]} (AP = {ap:.3f})')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve — NeoBERT')
    plt.legend(loc='lower left', fontsize=8)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / '02_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 2/10 — Precision-Recall Curve")


def plot_3_f1_per_class(probs, labels, out_dir):
    """3. F1 Score per Odor Family (Top 25)"""
    preds = (probs > 0.5).astype(int)
    per_class_f1 = []
    for i in range(labels.shape[1]):
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        per_class_f1.append((IDX_TO_CHEMICAL[i], f1))
    
    per_class_f1.sort(key=lambda x: x[1], reverse=True)
    top25 = per_class_f1[:25]
    
    names = [x[0] for x in top25]
    scores = [x[1] for x in top25]
    
    plt.figure(figsize=(10, 7))
    bars = plt.barh(range(len(names)), scores, color='#4361EE', alpha=0.85)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel('F1 Score')
    plt.title('F1 Score per Odor Family (Top 25)')
    plt.xlim(0, 1.1)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center', fontsize=7)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / '03_f1_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 3/10 — F1 Score per Class")


def plot_4_confusion_heatmap(probs, labels, out_dir):
    """4. Multi-Label Confusion Matrix Heatmap (Top 15 classes)"""
    preds = (probs > 0.5).astype(int)
    class_counts = labels.sum(axis=0)
    top15 = np.argsort(class_counts)[-15:][::-1]
    
    # Build co-occurrence confusion
    cm = np.zeros((15, 4))  # TP, FP, FN, TN for each class
    for i, cls_idx in enumerate(top15):
        tp = ((preds[:, cls_idx] == 1) & (labels[:, cls_idx] == 1)).sum()
        fp = ((preds[:, cls_idx] == 1) & (labels[:, cls_idx] == 0)).sum()
        fn = ((preds[:, cls_idx] == 0) & (labels[:, cls_idx] == 1)).sum()
        tn = ((preds[:, cls_idx] == 0) & (labels[:, cls_idx] == 0)).sum()
        cm[i] = [tp, fp, fn, tn]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    class_names = [IDX_TO_CHEMICAL[idx] for idx in top15]
    im = ax.imshow(cm[:, :3], cmap='YlOrRd', aspect='auto')
    
    ax.set_yticks(range(15))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['True Pos', 'False Pos', 'False Neg'])
    ax.set_title('Confusion Breakdown — Top 15 Odor Families')
    
    for i in range(15):
        for j in range(3):
            ax.text(j, i, f'{int(cm[i, j])}', ha='center', va='center', fontsize=8, color='black')
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / '04_confusion_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 4/10 — Confusion Heatmap")


def plot_5_class_distribution(labels, out_dir):
    """5. Dataset Class Distribution"""
    class_counts = labels.sum(axis=0)
    sorted_idx = np.argsort(class_counts)[::-1]
    
    names = [IDX_TO_CHEMICAL[i] for i in sorted_idx[:25]]
    counts = [class_counts[i] for i in sorted_idx[:25]]
    
    plt.figure(figsize=(10, 7))
    bars = plt.barh(range(len(names)), counts, color='#7209B7', alpha=0.8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel('Number of Samples')
    plt.title('Dataset Class Distribution (Top 25 Odor Families)')
    plt.gca().invert_yaxis()
    for bar, count in zip(bars, counts):
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f'{int(count)}', va='center', fontsize=7)
    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / '05_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 5/10 — Class Distribution")


def plot_6_precision_at_k(probs, labels, out_dir):
    """6. Precision@K and Recall@K Curves"""
    ks = [1, 2, 3, 5, 8, 10]
    p_at_k = []
    r_at_k = []
    
    for k in ks:
        p_list, r_list = [], []
        for i in range(len(probs)):
            topk = np.argsort(probs[i])[-k:][::-1]
            true_set = set(np.where(labels[i] == 1)[0])
            if not true_set:
                continue
            hits = len(set(topk) & true_set)
            p_list.append(hits / k)
            r_list.append(hits / len(true_set))
        p_at_k.append(np.mean(p_list))
        r_at_k.append(np.mean(r_list))
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, p_at_k, marker='o', color='#4361EE', linewidth=2, label='Precision@K')
    ax1.plot(ks, r_at_k, marker='s', color='#F72585', linewidth=2, label='Recall@K')
    
    ax1.set_xlabel('K')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision@K and Recall@K')
    ax1.set_xticks(ks)
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    for i, k in enumerate(ks):
        ax1.annotate(f'{p_at_k[i]:.3f}', (k, p_at_k[i]), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=7, color='#4361EE')
        ax1.annotate(f'{r_at_k[i]:.3f}', (k, r_at_k[i]), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=7, color='#F72585')
    
    plt.tight_layout()
    plt.savefig(out_dir / '06_precision_recall_at_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 6/10 — Precision@K / Recall@K")


def plot_7_confidence_distribution(probs, labels, out_dir):
    """7. Prediction Confidence Distribution (Positive vs Negative)"""
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    
    plt.figure(figsize=(8, 5))
    plt.hist(neg_probs, bins=50, alpha=0.6, color='#A0C4FF', label='Negative (No Label)', density=True)
    plt.hist(pos_probs, bins=50, alpha=0.7, color='#4361EE', label='Positive (Has Label)', density=True)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Confidence Distribution — Positive vs Negative Labels')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / '07_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 7/10 — Confidence Distribution")


def plot_8_threshold_vs_f1(probs, labels, out_dir):
    """8. Threshold vs F1 Score Curve"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    macro_f1s = []
    micro_f1s = []
    
    for t in thresholds:
        preds = (probs > t).astype(int)
        macro_f1s.append(f1_score(labels, preds, average='macro', zero_division=0))
        micro_f1s.append(f1_score(labels, preds, average='micro', zero_division=0))
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, macro_f1s, marker='o', color='#4361EE', linewidth=2, label='Macro F1')
    plt.plot(thresholds, micro_f1s, marker='s', color='#F72585', linewidth=2, label='Micro F1')
    plt.xlabel('Decision Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Decision Threshold')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.ylim(0, 1.05)
    
    best_t_macro = thresholds[np.argmax(macro_f1s)]
    best_t_micro = thresholds[np.argmax(micro_f1s)]
    plt.axvline(x=best_t_macro, color='#4361EE', linestyle=':', alpha=0.4)
    plt.axvline(x=best_t_micro, color='#F72585', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(out_dir / '08_threshold_vs_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 8/10 — Threshold vs F1")


def plot_9_labels_per_sample(labels, out_dir):
    """9. Labels per Sample Distribution"""
    labels_per_sample = labels.sum(axis=1)
    
    plt.figure(figsize=(8, 5))
    plt.hist(labels_per_sample, bins=range(0, int(labels_per_sample.max()) + 2),
             color='#4361EE', alpha=0.8, edgecolor='white')
    plt.xlabel('Number of Odor Families per Sample')
    plt.ylabel('Frequency')
    plt.title('Distribution of Labels per Sample')
    plt.axvline(x=np.mean(labels_per_sample), color='red', linestyle='--',
                label=f'Mean = {np.mean(labels_per_sample):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / '09_labels_per_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 9/10 — Labels per Sample")


def plot_10_radar_chart(probs, labels, out_dir):
    """10. Radar Chart — Model Performance Overview"""
    preds = (probs > 0.5).astype(int)
    
    # Compute metrics
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    
    # P@5, R@5
    p5_list, r5_list = [], []
    for i in range(len(probs)):
        top5 = np.argsort(probs[i])[-5:][::-1]
        true_set = set(np.where(labels[i] == 1)[0])
        if not true_set:
            continue
        hits = len(set(top5) & true_set)
        p5_list.append(hits / 5)
        r5_list.append(hits / len(true_set))
    p5 = np.mean(p5_list)
    r5 = np.mean(r5_list)
    
    # MAP
    ap_scores = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:
            ap_scores.append(average_precision_score(labels[:, i], probs[:, i]))
    map_score = np.mean(ap_scores)
    
    # ROC AUC
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(labels, probs, average='micro')
    
    categories = ['Macro F1', 'Micro F1', 'P@5', 'R@5', 'MAP', 'ROC AUC']
    values = [macro_f1, micro_f1, p5, r5, map_score, roc_auc]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, color='#4361EE')
    ax.fill(angles, values, alpha=0.25, color='#4361EE')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title('NeoBERT — Performance Radar', y=1.08, fontsize=13)
    
    for angle, value, cat in zip(angles[:-1], values[:-1], categories):
        ax.annotate(f'{value:.3f}', xy=(angle, value), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, fontweight='bold', color='#2341CE')
    
    plt.tight_layout()
    plt.savefig(out_dir / '10_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 10/10 — Radar Chart")


def main():
    print("=" * 60)
    print("Generating 10 Comprehensive ML Metric Visualizations")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    ckpt = Path('models/checkpoints/neobert/best_model.pt')
    if not ckpt.exists():
        print("ERROR: No trained model found.")
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
    
    print(f"Test samples: {len(test_dataset)}")
    print("Running inference on test set...")
    
    probs, labels = get_predictions(model, test_loader, device)
    print(f"Predictions shape: {probs.shape}")
    
    out_dir = Path('outputs/visualizations/metrics')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    plot_1_roc_curve(probs, labels, out_dir)
    plot_2_precision_recall_curve(probs, labels, out_dir)
    plot_3_f1_per_class(probs, labels, out_dir)
    plot_4_confusion_heatmap(probs, labels, out_dir)
    plot_5_class_distribution(labels, out_dir)
    plot_6_precision_at_k(probs, labels, out_dir)
    plot_7_confidence_distribution(probs, labels, out_dir)
    plot_8_threshold_vs_f1(probs, labels, out_dir)
    plot_9_labels_per_sample(labels, out_dir)
    plot_10_radar_chart(probs, labels, out_dir)
    
    print(f"\n✓ All 10 metric visualizations saved to {out_dir}")


if __name__ == '__main__':
    main()

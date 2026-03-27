#!/usr/bin/env python3
"""
Generate visualizations from the evaluation results.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    results_path = Path('outputs/reports/evaluation_results.json')
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.")
        return
        
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    out_dir = Path('outputs/visualizations')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Macro F1 and Micro F1 @ 0.4 Comparison
    metrics = ['macro_f1@0.4', 'micro_f1@0.4', 'MAP', 'P@5', 'R@5']
    labels = ['Macro F1', 'Micro F1', 'MAP', 'P@5', 'R@5']
    
    tfidf_scores = [results['tfidf'][m] for m in metrics]
    neobert_scores = [results['neobert'][m] for m in metrics]
    
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, tfidf_scores, marker='o', linestyle='--', label='TF-IDF Baseline', color='#A0C4FF', linewidth=2, markersize=8)
    ax.plot(x, neobert_scores, marker='s', linestyle='-', label='NeoBERT', color='#4361EE', linewidth=2, markersize=8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison (Test Set: 2,885 samples)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.grid(True, alpha=0.3)
    
    for i, score in enumerate(tfidf_scores):
        ax.annotate(f"{score:.3f}", (x[i], score), textcoords="offset points", xytext=(0, -15), ha='center', color='#6084DF', fontweight='bold')
    for i, score in enumerate(neobert_scores):
        ax.annotate(f"{score:.3f}", (x[i], score), textcoords="offset points", xytext=(0, 10), ha='center', color='#2341CE', fontweight='bold')
    
    fig.tight_layout()
    
    plot_path = out_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved performance comparison to {plot_path}")
    
    # 2. Confidence thresholds F1
    thresholds = [0.3, 0.4, 0.5]
    macro_tfidf = [results['tfidf'][f'macro_f1@{t}'] for t in thresholds]
    macro_neobert = [results['neobert'][f'macro_f1@{t}'] for t in thresholds]
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, macro_tfidf, marker='o', linestyle='--', label='TF-IDF Macro F1', color='#A0C4FF')
    plt.plot(thresholds, macro_neobert, marker='s', linestyle='-', label='NeoBERT Macro F1', color='#4361EE')
    
    plt.xlabel('Probability Threshold')
    plt.ylabel('Macro F1 Score')
    plt.title('Macro F1 Sensitivity to Threshold')
    plt.xticks(thresholds)
    plt.ylim(0.7, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = out_dir / 'threshold_sensitivity.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved threshold sensitivity plot to {plot_path}")

if __name__ == '__main__':
    main()

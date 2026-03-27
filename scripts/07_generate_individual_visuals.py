#!/usr/bin/env python3
"""
Generate 10 individual metric visualizations from the evaluation results.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    print("=" * 60)
    print("Generating 10 Individual Visualizations")
    print("=" * 60)
    
    results_path = Path('outputs/reports/evaluation_results.json')
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.")
        return
        
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    out_dir = Path('outputs/visualizations/individual')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 10 metrics to plot individually
    metrics = [
        ('macro_f1@0.5', 'Macro F1 (Threshold 0.5)'),
        ('micro_f1@0.5', 'Micro F1 (Threshold 0.5)'),
        ('MAP', 'Mean Average Precision (MAP)'),
        ('P@1', 'Precision @ 1'),
        ('R@1', 'Recall @ 1'),
        ('P@5', 'Precision @ 5'),
        ('R@5', 'Recall @ 5'),
        ('P@10', 'Precision @ 10'),
        ('R@10', 'Recall @ 10'),
        ('macro_f1@0.3', 'Macro F1 (Threshold 0.3)')
    ]
    
    models = ['TF-IDF Baseline', 'NeoBERT']
    colors = ['#A0C4FF', '#4361EE']
    
    for metric_key, metric_name in metrics:
        tfidf_score = results['tfidf'][metric_key]
        neobert_score = results['neobert'][metric_key]
        scores = [tfidf_score, neobert_score]
        
        plt.figure(figsize=(6, 5))
        bars = plt.bar(models, scores, color=colors, width=0.5)
        
        plt.ylabel('Score')
        plt.title(f'Comparison: {metric_name}\n(Test Set: 2,885 samples)')
        plt.ylim(0, 1.1)
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
            
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = metric_name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_').replace('@', 'at').lower() + '.png'
        plot_path = out_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {metric_name} to {plot_path}")

if __name__ == '__main__':
    main()

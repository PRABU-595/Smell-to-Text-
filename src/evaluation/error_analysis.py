"""
Error analysis utilities for smell-to-molecule predictions
"""
import json
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np


class ErrorAnalyzer:
    """Analyze prediction errors to identify patterns."""
    
    def __init__(self, predictions: List[Dict], ground_truth: List[Dict]):
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.errors = []
        self._analyze()
    
    def _analyze(self):
        for pred, gt in zip(self.predictions, self.ground_truth):
            pred_set = set(pred.get('chemicals', []))
            gt_set = set(gt.get('chemicals', []))
            
            self.errors.append({
                'id': pred.get('id', ''),
                'description': pred.get('description', ''),
                'false_positives': list(pred_set - gt_set),
                'false_negatives': list(gt_set - pred_set),
                'true_positives': list(pred_set & gt_set),
                'precision': len(pred_set & gt_set) / len(pred_set) if pred_set else 0,
                'recall': len(pred_set & gt_set) / len(gt_set) if gt_set else 0
            })
    
    def get_common_false_positives(self, top_n: int = 10) -> List[Tuple[str, int]]:
        fp_counts = defaultdict(int)
        for e in self.errors:
            for chem in e['false_positives']:
                fp_counts[chem] += 1
        return sorted(fp_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_common_false_negatives(self, top_n: int = 10) -> List[Tuple[str, int]]:
        fn_counts = defaultdict(int)
        for e in self.errors:
            for chem in e['false_negatives']:
                fn_counts[chem] += 1
        return sorted(fn_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_hardest_samples(self, n: int = 10) -> List[Dict]:
        sorted_errors = sorted(self.errors, key=lambda x: x['precision'] + x['recall'])
        return sorted_errors[:n]
    
    def get_easiest_samples(self, n: int = 10) -> List[Dict]:
        sorted_errors = sorted(self.errors, key=lambda x: x['precision'] + x['recall'], reverse=True)
        return sorted_errors[:n]
    
    def categorize_errors(self) -> Dict[str, List]:
        categories = {'missing_key_note': [], 'over_prediction': [], 'confusion': [], 'good': []}
        for e in self.errors:
            if e['precision'] > 0.8 and e['recall'] > 0.8:
                categories['good'].append(e)
            elif len(e['false_positives']) > len(e['true_positives']):
                categories['over_prediction'].append(e)
            elif len(e['false_negatives']) > len(e['true_positives']):
                categories['missing_key_note'].append(e)
            else:
                categories['confusion'].append(e)
        return categories
    
    def generate_report(self) -> str:
        report = "# Error Analysis Report\n\n"
        report += f"Total samples analyzed: {len(self.errors)}\n\n"
        
        avg_precision = np.mean([e['precision'] for e in self.errors])
        avg_recall = np.mean([e['recall'] for e in self.errors])
        report += f"Average Precision: {avg_precision:.3f}\nAverage Recall: {avg_recall:.3f}\n\n"
        
        report += "## Common False Positives\n"
        for chem, count in self.get_common_false_positives():
            report += f"- {chem}: {count}\n"
        
        report += "\n## Common False Negatives\n"
        for chem, count in self.get_common_false_negatives():
            report += f"- {chem}: {count}\n"
        
        categories = self.categorize_errors()
        report += f"\n## Error Categories\n"
        for cat, items in categories.items():
            report += f"- {cat}: {len(items)} samples\n"
        
        return report

"""
Evaluation metrics for smell-to-molecule task
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class MetricsCalculator:
    def __init__(self, top_k=[1, 3, 5, 10]):
        self.top_k = top_k
    
    def precision_at_k(self, predictions, labels, k):
        """Precision@K for multi-label ranking"""
        precisions = []
        for pred, label in zip(predictions, labels):
            top_k_indices = np.argsort(pred)[-k:]
            relevant = label[top_k_indices].sum()
            precisions.append(relevant / k)
        return np.mean(precisions)
    
    def recall_at_k(self, predictions, labels, k):
        """Recall@K"""
        recalls = []
        for pred, label in zip(predictions, labels):
            top_k_indices = np.argsort(pred)[-k:]
            relevant = label[top_k_indices].sum()
            total_relevant = label.sum()
            if total_relevant > 0:
                recalls.append(relevant / total_relevant)
        return np.mean(recalls)
    
    def mean_average_precision(self, predictions, labels):
        """Mean Average Precision (MAP)"""
        aps = []
        for pred, label in zip(predictions, labels):
            indices = np.argsort(pred)[::-1]
            sorted_labels = label[indices]
            
            precision_sum = 0
            num_relevant = 0
            for i, relevant in enumerate(sorted_labels):
                if relevant:
                    num_relevant += 1
                    precision_sum += num_relevant / (i + 1)
            
            if num_relevant > 0:
                aps.append(precision_sum / num_relevant)
        
        return np.mean(aps)
    
    def ndcg_at_k(self, predictions, labels, k):
        """Normalized Discounted Cumulative Gain"""
        ndcgs = []
        for pred, label in zip(predictions, labels):
            indices = np.argsort(pred)[-k:][::-1]
            dcg = sum([label[i] / np.log2(pos + 2) 
                      for pos, i in enumerate(indices)])
            
            ideal_indices = np.argsort(label)[-k:][::-1]
            idcg = sum([label[i] / np.log2(pos + 2) 
                       for pos, i in enumerate(ideal_indices)])
            
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        
        return np.mean(ndcgs)
    
    def compute_all_metrics(self, predictions, labels):
        """Compute all metrics"""
        results = {}
        
        for k in self.top_k:
            results[f'P@{k}'] = self.precision_at_k(predictions, labels, k)
            results[f'R@{k}'] = self.recall_at_k(predictions, labels, k)
            results[f'NDCG@{k}'] = self.ndcg_at_k(predictions, labels, k)
        
        results['MAP'] = self.mean_average_precision(predictions, labels)
        
        # Multi-label F1
        binary_preds = (predictions > 0.5).astype(int)
        results['F1'] = f1_score(labels, binary_preds, average='micro')
        
        return results

"""
Visualization utilities for smell-to-molecule project
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """Plot training and validation loss/metrics curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Metrics curve
    for key in history:
        if 'f1' in key.lower() or 'map' in key.lower():
            axes[1].plot(history[key], label=key)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Metrics')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          labels: List[str], save_path: Optional[str] = None):
    """Plot confusion matrix for top chemicals."""
    from sklearn.metrics import confusion_matrix
    
    n_classes = min(20, len(labels))
    cm = confusion_matrix(y_true[:, :n_classes].argmax(1), y_pred[:, :n_classes].argmax(1))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels[:n_classes], yticklabels=labels[:n_classes])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Top 20 Chemicals)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_chemical_distribution(chemicals: List[str], counts: List[int], 
                               top_n: int = 20, save_path: Optional[str] = None):
    """Plot distribution of chemicals in dataset."""
    plt.figure(figsize=(10, 6))
    
    indices = np.argsort(counts)[-top_n:][::-1]
    top_chemicals = [chemicals[i] for i in indices]
    top_counts = [counts[i] for i in indices]
    
    plt.barh(range(len(top_chemicals)), top_counts, color='steelblue')
    plt.yticks(range(len(top_chemicals)), top_chemicals)
    plt.xlabel('Count')
    plt.title(f'Top {top_n} Chemicals by Frequency')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_precision_recall_at_k(results: Dict, save_path: Optional[str] = None):
    """Plot precision and recall at different K values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    k_values = sorted([int(k.split('@')[1]) for k in results.keys() if 'P@' in k])
    
    precision = [results.get(f'P@{k}', 0) for k in k_values]
    recall = [results.get(f'R@{k}', 0) for k in k_values]
    
    ax.plot(k_values, precision, 'o-', label='Precision@K', linewidth=2)
    ax.plot(k_values, recall, 's-', label='Recall@K', linewidth=2)
    
    ax.set_xlabel('K')
    ax.set_ylabel('Score')
    ax.set_title('Precision and Recall at K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results: Dict[str, Dict], metric: str = 'MAP',
                          save_path: Optional[str] = None):
    """Compare multiple models on a metric."""
    plt.figure(figsize=(8, 5))
    
    models = list(results.keys())
    scores = [results[m].get(metric, 0) for m in models]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    plt.bar(models, scores, color=colors)
    plt.ylabel(metric)
    plt.title(f'Model Comparison: {metric}')
    plt.xticks(rotation=45, ha='right')
    
    for i, (m, s) in enumerate(zip(models, scores)):
        plt.text(i, s + 0.01, f'{s:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_attention_heatmap(tokens: List[str], attention_weights: np.ndarray,
                             save_path: Optional[str] = None):
    """Visualize attention weights for a prediction."""
    plt.figure(figsize=(12, 3))
    
    weights = attention_weights[:len(tokens)]
    weights = weights / weights.max()
    
    plt.barh(range(len(tokens)), weights, color='coral')
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel('Attention Weight')
    plt.title('Token Attention Weights')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

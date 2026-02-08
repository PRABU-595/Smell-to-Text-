"""
Model utility functions and helpers
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Freeze specific layers in model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer name prefixes to freeze
    """
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in layer_names):
            param.requires_grad = False
            logger.info(f"Frozen layer: {name}")


def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """Unfreeze specific layers in model."""
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in layer_names):
            param.requires_grad = True
            logger.info(f"Unfrozen layer: {name}")


def get_layer_groups(model: nn.Module) -> Dict[str, List[str]]:
    """
    Get parameter names grouped by layer.
    
    Returns:
        Dictionary mapping layer names to parameter names
    """
    groups = {}
    for name, _ in model.named_parameters():
        layer = name.split('.')[0]
        if layer not in groups:
            groups[layer] = []
        groups[layer].append(name)
    return groups


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    **kwargs
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        **kwargs
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optional optimizer to restore
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from {filepath}")
    return checkpoint


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self, 
        patience: int = 5, 
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum improvement threshold
            mode: 'min' or 'max' for metric optimization
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class ModelMetadata:
    """Store and manage model metadata."""
    
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.metadata_path = self.model_path / 'metadata.json'
        self.metadata = {}
        
        if self.metadata_path.exists():
            self.load()
    
    def update(self, **kwargs) -> None:
        """Update metadata fields."""
        self.metadata.update(kwargs)
    
    def save(self) -> None:
        """Save metadata to file."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load(self) -> None:
        """Load metadata from file."""
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def __getitem__(self, key: str) -> Any:
        return self.metadata.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.metadata[key] = value


def get_attention_weights(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> List[torch.Tensor]:
    """
    Extract attention weights from BERT model.
    
    Args:
        model: BERT-based model
        input_ids: Token IDs
        attention_mask: Attention mask
        
    Returns:
        List of attention weight tensors per layer
    """
    model.eval()
    with torch.no_grad():
        outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
    return outputs.attentions


def calculate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calculate model size in different formats.
    
    Returns:
        Dictionary with size info
    """
    param_count = count_parameters(model)
    
    # Calculate size in bytes (assuming float32)
    size_bytes = param_count * 4
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    
    return {
        'parameters': param_count,
        'size_mb': round(size_mb, 2),
        'size_gb': round(size_gb, 4)
    }


def convert_to_onnx(
    model: nn.Module,
    sample_input: Dict[str, torch.Tensor],
    output_path: str,
    input_names: List[str] = ['input_ids', 'attention_mask'],
    output_names: List[str] = ['predictions']
) -> None:
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        sample_input: Sample input for tracing
        output_path: Path to save ONNX model
        input_names: Names for input tensors
        output_names: Names for output tensors
    """
    model.eval()
    
    torch.onnx.export(
        model,
        (sample_input['input_ids'], sample_input['attention_mask']),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'predictions': {0: 'batch_size'}
        },
        opset_version=12
    )
    logger.info(f"Exported model to ONNX: {output_path}")

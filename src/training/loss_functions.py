"""
Custom loss functions for smell-to-molecule prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-label classification.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted probabilities (after sigmoid)
            targets: Binary labels
            
        Returns:
            Loss value
        """
        # Clamp for numerical stability
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        
        # Compute focal weights
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weights
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification", 2020
    """
    
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05, reduction: str = 'mean'):
        """
        Initialize Asymmetric Loss.
        
        Args:
            gamma_neg: Focusing parameter for negatives
            gamma_pos: Focusing parameter for positives
            clip: Hard threshold for probability shifting
            reduction: Reduction method
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric loss."""
        # Clamp for numerical stability
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        
        # Asymmetric focusing
        xs_pos = inputs
        xs_neg = 1 - inputs
        
        # Probability shifting
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Compute losses
        los_pos = targets * torch.log(xs_pos)
        los_neg = (1 - targets) * torch.log(xs_neg)
        
        # Asymmetric focusing weights
        neg_weight = torch.pow(1 - xs_neg, self.gamma_neg)
        pos_weight = torch.pow(1 - xs_pos, self.gamma_pos)
        
        loss = -los_pos * pos_weight - los_neg * neg_weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss for molecule retrieval.
    
    Encourages correct chemicals to have higher scores than incorrect ones.
    """
    
    def __init__(self, margin: float = 0.1, reduction: str = 'mean'):
        """
        Initialize ranking loss.
        
        Args:
            margin: Margin between positive and negative scores
            reduction: Reduction method
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise ranking loss.
        
        Args:
            scores: Predicted scores for each chemical
            labels: Binary relevance labels
            
        Returns:
            Ranking loss
        """
        batch_size = scores.size(0)
        losses = []
        
        for i in range(batch_size):
            # Get positive and negative indices
            pos_mask = labels[i] == 1
            neg_mask = labels[i] == 0
            
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            
            pos_scores = scores[i][pos_mask]
            neg_scores = scores[i][neg_mask]
            
            # Compute pairwise hinge loss
            # Each positive should be higher than each negative by margin
            pos_expanded = pos_scores.unsqueeze(1)  # [n_pos, 1]
            neg_expanded = neg_scores.unsqueeze(0)  # [1, n_neg]
            
            pairwise_loss = F.relu(self.margin - pos_expanded + neg_expanded)
            losses.append(pairwise_loss.mean())
        
        if not losses:
            return torch.tensor(0.0, device=scores.device)
        
        total_loss = torch.stack(losses)
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        return total_loss


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with weights.
    """
    
    def __init__(self, losses: list, weights: Optional[list] = None):
        """
        Initialize combined loss.
        
        Args:
            losses: List of loss modules
            weights: Optional weights for each loss
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            self.weights = [1.0] * len(losses)
        else:
            self.weights = weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of losses."""
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        return total_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for multi-label classification.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor (0-1)
            reduction: Reduction method
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss."""
        # Smooth labels
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Binary cross entropy with smoothed labels
        loss = F.binary_cross_entropy(inputs, smoothed_targets, reduction=self.reduction)
        
        return loss


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss by name.
    
    Args:
        name: Loss function name
        **kwargs: Additional arguments for loss
        
    Returns:
        Loss module
    """
    loss_dict = {
        'bce': nn.BCELoss,
        'focal': FocalLoss,
        'asymmetric': AsymmetricLoss,
        'ranking': RankingLoss,
        'label_smoothing': LabelSmoothingLoss,
    }
    
    if name not in loss_dict:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(loss_dict.keys())}")
    
    return loss_dict[name](**kwargs)


if __name__ == '__main__':
    # Test loss functions
    batch_size = 4
    num_classes = 100
    
    predictions = torch.sigmoid(torch.randn(batch_size, num_classes))
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    print("Testing loss functions:")
    
    # Test each loss
    for name in ['bce', 'focal', 'asymmetric', 'ranking', 'label_smoothing']:
        loss_fn = get_loss_function(name)
        loss = loss_fn(predictions, targets)
        print(f"  {name}: {loss.item():.4f}")

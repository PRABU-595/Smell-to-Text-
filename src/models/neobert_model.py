"""
NeoBERT model for smell-to-molecule translation.
Multi-label classification using pre-trained transformer encoder.
"""
import torch
import torch.nn as nn
import os
from src.utils.logger import setup_logger

os.environ["HF_HOME"] = "C:\\Users\\iampr\\.cache\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "C:\\Users\\iampr\\.cache\\huggingface"

logger = setup_logger(__name__)

from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Any


class SmellToMoleculeModel(nn.Module):
    """
    NeoBERT-based model for smell description → chemical compound prediction.
    
    Architecture:
        Input Text → NeoBERT Encoder → [CLS] pooling → Dense → ReLU → Output → Sigmoid
    """
    
    def __init__(self, model_name='bert-base-uncased', num_chemicals=82,
                 hidden_dim=512, dropout=0.1, freeze_bert_layers=0):
        """
        Args:
            model_name: HuggingFace model name or path
            num_chemicals: Number of chemical output classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            freeze_bert_layers: Number of encoder layers to freeze (0 = none)
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_chemicals = num_chemicals
        
        # Load pre-trained encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Optionally freeze lower layers
        if freeze_bert_layers > 0:
            self._freeze_layers(freeze_bert_layers)
        
        # Classification head
        bert_dim = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_chemicals),
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights with Xavier uniform."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _freeze_layers(self, num_layers):
        """Freeze the first num_layers encoder layers."""
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        # Freeze encoder layers
        for layer in self.bert.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Optional target labels [batch, num_chemicals]
            
        Returns:
            dict with 'probs', 'logits', and optionally 'loss'
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_output)
        probs = torch.sigmoid(logits)
        
        result = {'probs': probs, 'logits': logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            result['loss'] = loss_fn(logits, labels)
        
        return result
    
    def predict(self, input_ids, attention_mask, threshold=0.5):
        """Predict with thresholding."""
        self.eval()
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
            predictions = (result['probs'] >= threshold).int()
        return predictions, result['probs']
    
    def get_attention_weights(self, input_ids, attention_mask):
        """Extract attention weights for visualization."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        return outputs.attentions
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, **kwargs):
        """Load model from checkpoint."""
        model = cls(**kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model

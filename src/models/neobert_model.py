"""
NeoBERT model for smell-to-molecule translation
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class SmellToMoleculeModel(nn.Module):
    def __init__(self, model_name='neobert-base', num_chemicals=300, 
                 hidden_dim=512, dropout=0.1):
        super().__init__()
        
        # Load pre-trained NeoBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_chemicals)
        
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        # Sigmoid for multi-label
        probs = torch.sigmoid(logits)
        
        return probs, outputs
    
    def get_attention_weights(self, input_ids, attention_mask):
        """Extract attention weights for visualization"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        return outputs.attentions

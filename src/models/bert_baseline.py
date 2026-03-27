"""
Standard BERT baseline for comparison with NeoBERT.
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertBaseline(nn.Module):
    """
    Standard BERT baseline (bert-base-uncased) for smell-to-molecule.
    Simpler architecture than NeoBERT model to demonstrate improvement.
    """
    
    def __init__(self, num_chemicals=82, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_chemicals)
        nn.init.xavier_uniform_(self.classifier.weight)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)
        
        result = {'probs': probs, 'logits': logits}
        
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            result['loss'] = loss_fn(logits, labels)
        
        return result
    
    def predict(self, input_ids, attention_mask, threshold=0.5):
        self.eval()
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
            predictions = (result['probs'] >= threshold).int()
        return predictions, result['probs']

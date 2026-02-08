"""
Standard BERT baseline for comparison
"""
import torch.nn as nn
from transformers import BertModel

class BertBaseline(nn.Module):
    def __init__(self, num_chemicals=300):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_chemicals)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return torch.sigmoid(logits)

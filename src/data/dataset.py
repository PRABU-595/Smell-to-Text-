"""
PyTorch Dataset for smell-to-molecule task
"""
import torch
from torch.utils.data import Dataset
import pandas as pd

class SmellDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenize description
        encoding = self.tokenizer(
            row['description'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Parse chemical labels
        chemicals = eval(row['chemicals'])  # JSON string to list
        labels = torch.zeros(self.num_chemicals)
        for chem in chemicals:
            labels[chem['id']] = chem['weight']
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels,
            'description': row['description']
        }

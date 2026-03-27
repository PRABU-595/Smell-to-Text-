"""
PyTorch Dataset for smell-to-molecule multi-label classification task.

Supports the new 50-class odor family format where labels are stored
as JSON-encoded multi-hot vectors.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import numpy as np
from typing import Optional, List

from src.data.chemical_vocab import CHEMICAL_LIST, CHEMICAL_TO_IDX, IDX_TO_CHEMICAL, NUM_CHEMICALS


class SmellDataset(Dataset):
    """PyTorch Dataset for odor family prediction."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 128,
                 num_chemicals: int = None):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_chemicals = num_chemicals or NUM_CHEMICALS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        encoding = self.tokenizer(
            str(row['description']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # New format: labels is a JSON-encoded list of 0/1 values
        labels = torch.zeros(self.num_chemicals)
        try:
            label_data = json.loads(str(row['labels']))
            if isinstance(label_data, list):
                for i, v in enumerate(label_data):
                    if i < self.num_chemicals:
                        labels[i] = float(v)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
        }

    @staticmethod
    def get_chemical_names() -> List[str]:
        return CHEMICAL_LIST.copy()

    @staticmethod
    def get_num_chemicals() -> int:
        return NUM_CHEMICALS
